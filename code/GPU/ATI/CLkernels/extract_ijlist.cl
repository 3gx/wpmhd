#ifndef _EXTRACT_IJLIST_CL_
#define _EXTRACT_IJLIST_CL_

#define HUGE 1.0e10f

__kernel void extract_ijlist(__global int2        *out_ijlist,    //  0
			     __global float4      *out_drij,      //  1
			     __global int         *leaf_list,     //  2
			     __global int2        *body_list,     //  3
			     __global int         *ilist,         //  4
			     __global float4      *ppos,          //  5
			     __global int         *jlist,         //  6
			     __global int         *joffset,       //  7
			     __global int         *ijlist_offset, //  8
			     __global int         *leaf_ngb_max,  //  9
			              float4       domain_hsize,  // 10
			     __local  float       *shmem) {       // 11

  // Get threadIdx, blockIdx & blockDim
  const int threadIdx = get_local_id(0);
  const int blockIdx  = get_group_id(0);
  const int blockDim  = get_local_size(0);

  // Get bodyId
  const int leaf_id  = leaf_list[blockIdx];
  const int2 ifirst  = body_list[leaf_id];
  const int  ni      = ifirst.y;
  const int  bodyIdx = ifirst.x + min(threadIdx, ni-1); 
  const int  bodyId  = ilist[bodyIdx];
  
  // Get i-body data
  const float4 ipos  = ppos[bodyId];
  const float  hi2   = ipos.w * ipos.w;
  
  // Compute number of j-particles
  const int offset = joffset[blockIdx    ];
  const int nj     = joffset[blockIdx + 1] - offset;

  // Compute offset where to write neighbours idx
  const int ijoffset = ijlist_offset[leaf_id] + min(threadIdx, ni-1);

  // Redistribute shared memory
  __local float *sh_posx = &shmem  [0];
  __local float *sh_posy = &sh_posx[blockDim];
  __local float *sh_posz = &sh_posy[blockDim];
  __local float *sh_posh = &sh_posz[blockDim];
  __local float *sh_jidx = &sh_posh[blockDim];

  // Extract ijlist
  int ngb_count = 0;
  for (int i = 0; i < nj; i += blockDim) {
    const int  jp  = i + threadIdx;
    const int jidx = (jp < nj) ? jlist[offset + jp] : -1;
    
    const float4 jpos  = (jidx >= 0) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[threadIdx] = jpos.x;
    sh_posy[threadIdx] = jpos.y;
    sh_posz[threadIdx] = jpos.z;
    sh_posh[threadIdx] = jpos.w;
    sh_jidx[threadIdx] = jidx;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int j = 0; j < blockDim; j++) {
      const float dx0 = sh_posx[j] - ipos.x;
      const float dy0 = sh_posy[j] - ipos.y;
      const float dz0 = sh_posz[j] - ipos.z;

#ifdef _PERIODIC_BOUNDARIES_
      const float dx = dx0 + ((fabs(dx0) > domain_hsize.x) ? -2.0f*sign(dx0)*domain_hsize.x : 0.0f);
      const float dy = dy0 + ((fabs(dy0) > domain_hsize.y) ? -2.0f*sign(dy0)*domain_hsize.y : 0.0f);
      const float dz = dz0 + ((fabs(dz0) > domain_hsize.z) ? -2.0f*sign(dz0)*domain_hsize.z : 0.0f);
#else
      const float dx = dx0;
      const float dy = dy0;
      const float dz = dz0;
#endif
      const float  s2 = dx*dx + dy*dy + dz*dz;      
      const float hj2 = sh_posh[j]*sh_posh[j];
      
      if (s2 > 0.0f && s2 <= fmax(hi2, hj2)) {
	if (threadIdx < ni) {
	  out_ijlist[ijoffset + ngb_count] = (int2){bodyId, sh_jidx[j]};
	  out_drij  [ijoffset + ngb_count] = (float4){dx, dy, dz, sqrt(s2)};
	}
	ngb_count += ni;
      }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  const int nbeg  = ngb_count/ni;
  const int n_ngb = leaf_ngb_max[leaf_id];
  for (int i = nbeg; i < n_ngb; i++) {
    if (threadIdx < ni) {
      out_ijlist[ijoffset + ngb_count] = (int2){bodyId, bodyId};
      out_drij  [ijoffset + ngb_count] = (float4){0.0f, 0.0f, 0.0f, 0.0f};
    }
    ngb_count += ni;
  }
}

///////

__kernel void extract_gather_list(__global int2        *out_ijlist,    //  0
				  __global float4      *out_drij,      //  1
				  __global int         *leaf_list,     //  2
				  __global int2        *body_list,     //  3
				  __global int         *ilist,         //  4
				  __global float4      *ppos,          //  5
				  __global int         *jlist,         //  6
				  __global int         *joffset,       //  7
				  __global int         *ijlist_offset, //  8
				  __global int         *leaf_ngb_max,  //  9
				  const    float4       domain_hsize,  // 10
				  __local  float       *shmem)  {      // 11

  // Get threadIdx, blockIdx & blockDim
  const int threadIdx = get_local_id(0);
  const int blockIdx  = get_group_id(0);
  const int blockDim  = get_local_size(0);

  // Get bodyId
  const int leaf_id  = leaf_list[blockIdx];
  const int2 ifirst  = body_list[leaf_id];
  const int  ni      = ifirst.y;
  const int  bodyIdx = ifirst.x + min(threadIdx, ni-1); 
  const int  bodyId  = ilist[bodyIdx];
  
  // Get i-body data
  const float4 ipos  = ppos[bodyId];
  const float  hi2   = ipos.w * ipos.w;
  
  // Compute number of j-particles
  const int offset = joffset[blockIdx    ];
  const int nj     = joffset[blockIdx + 1] - offset;

  // Compute offset where to write neighbours idx
  const int ijoffset = ijlist_offset[leaf_id] + min(threadIdx, ni-1);

  // Redistribute shared memory
  __local float *sh_posx = &shmem  [0];
  __local float *sh_posy = &sh_posx[blockDim];
  __local float *sh_posz = &sh_posy[blockDim];
  __local float *sh_jidx = &sh_posz[blockDim];

  // Extract ijlist
  int ngb_count = 0;
  for (int i = 0; i < nj; i += blockDim) {
    const int  jp  = i + threadIdx;
    const int jidx = (jp < nj) ? jlist[offset + jp] : -1;
    
    const float4 jpos  = (jidx >= 0) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[threadIdx] = jpos.x;
    sh_posy[threadIdx] = jpos.y;
    sh_posz[threadIdx] = jpos.z;
    sh_jidx[threadIdx] = jidx;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int j = 0; j < blockDim; j++) {
      const float dx0 = sh_posx[j] - ipos.x;
      const float dy0 = sh_posy[j] - ipos.y;
      const float dz0 = sh_posz[j] - ipos.z;

#ifdef _PERIODIC_BOUNDARIES_
      const float dx = dx0 + ((fabs(dx0) > domain_hsize.x) ? -2.0f*sign(dx0)*domain_hsize.x : 0.0f);
      const float dy = dy0 + ((fabs(dy0) > domain_hsize.y) ? -2.0f*sign(dy0)*domain_hsize.y : 0.0f);
      const float dz = dz0 + ((fabs(dz0) > domain_hsize.z) ? -2.0f*sign(dz0)*domain_hsize.z : 0.0f);
#else
      const float dx = dx0;
      const float dy = dy0;
      const float dz = dz0;
#endif
      const float  s2 = dx*dx + dy*dy + dz*dz;      
      
      if (s2 > 0.0f && s2 <= hi2) {
	if (threadIdx < ni) {
	  out_ijlist[ijoffset + ngb_count] = (int2){bodyId, sh_jidx[j]};
	  out_drij  [ijoffset + ngb_count] = (float4){dx, dy, dz, sqrt(s2)};
	}
	ngb_count += ni;
      }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  const int nbeg  = ngb_count/ni;
  const int n_ngb = leaf_ngb_max[leaf_id];
  for (int i = nbeg; i < n_ngb; i++) {
    if (threadIdx < ni) {
      out_ijlist[ijoffset + ngb_count] = (int2){bodyId, bodyId};
      out_drij  [ijoffset + ngb_count] = (float4){0.0f, 0.0f, 0.0f, 0.0f};
    }
    ngb_count += ni;
  }

}

#endif // _EXTRACT_IJLIST_CL_
