#ifndef _COMPUTE_NNGB_CL_
#define _COMPUTE_NNGB_CL_

#define HUGE 1.0e10f

__kernel void compute_nngb(__global int2        *out_nngb,     //  0
			   __global int         *leaf_list,    //  1
			   __global int2        *body_list,    //  2
			   __global int         *ilist,        //  3
			   __global float4      *ppos,         //  4
			   __global int         *jlist,        //  5
			   __global int         *joffset,      //  6
			   const    float4       domain_hsize, //  7
			   __local  float        *shmem) {     //  8


  // Get threadIdx, blockIdx & blockDim
  const int threadIdx = get_local_id(0);
  const int blockIdx  = get_group_id(0);
  const int blockDim  = get_local_size(0);
  const int globIdx   = get_global_id(0);

  // Get bodyId
  const int leaf_id  = leaf_list[blockIdx];
  const int2 ifirst  = body_list[leaf_id];
  const int  ni      = ifirst.y;

  int ni2b = 1;
  while (ni2b < ni) ni2b = ni2b << 1;
  ni2b = max(ni2b, NMAXPERLANE);

  const int nlanes = blockDim /ni2b;
  const int  lane  = threadIdx/ni2b;
  const int  tidx  = threadIdx%ni2b;

  const int  bodyIdx = ifirst.x + min(tidx, ni-1);
  const int  bodyId  = ilist[bodyIdx];

  // Get i-body data
  const float4 ipos = ppos[bodyId];
  const float hi    = ipos.w;
  const float hi2   = hi*hi;  

  // Compute number of j-particles
  const int offset = joffset[blockIdx    ];
  const int nj     = joffset[blockIdx + 1] - offset;
  
  // Compute work per lane
  const int nj_lane      = nj/nlanes;
  const int nj_lane_last = nj - nj_lane*(nlanes - 1);

  const int nj0  = nj_lane*lane;
  const int nj1  = nj0 + nj_lane_last;
  const int nj1e = nj0 + ((lane == nlanes - 1) ? nj_lane_last : nj_lane);

  const int tid0    = ni2b * lane;
  const int tid1    = ni2b + tid0;
  const int tid     = tidx + tid0;

  // Redistribute shared memory
  __local float *sh_posx = &shmem  [0];
  __local float *sh_posy = &sh_posx[blockDim];
  __local float *sh_posz = &sh_posy[blockDim];
  __local float *sh_posh = &sh_posz[blockDim];
  
  // Compute renormalisation matrix & count number of neighbours
  int   ngb_gather = 0;
  int   ngb_both   = 0;
  
  for (int i = nj0; i < nj1; i += ni2b) {
    const int  jp  = i + tidx;
    const int jidx = (jp < nj1e) ? jlist[offset + jp] : -1;
    
    const float4 jpos  = (jidx >= 0) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[tid] = jpos.x;
    sh_posy[tid] = jpos.y;
    sh_posz[tid] = jpos.z;
    sh_posh[tid] = jpos.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int j = tid0; j < tid1; j++) {
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
      const float s2 = dx*dx + dy*dy + dz*dz;
      const float hj = sh_posh[j];
      const float hj2 = hj*hj;
      
      ngb_gather += (s2 > 0.0f && s2 <     hi2      ) ? 1 : 0;
      ngb_both   += (s2 > 0.0f && s2 < max(hi2, hj2)) ? 1 : 0;
      
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  // Reduce data between lanes
  __local int   *sh_ngb1 = (__local int*)&sh_posx[0];
  __local int   *sh_ngb2 = (__local int*)&sh_posy[0];
  sh_ngb1[threadIdx] = ngb_gather;
  sh_ngb2[threadIdx] = ngb_both;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane == 0) 
    for (int j = ni2b; j < blockDim; j += ni2b) {
      ngb_gather += sh_ngb1[j + tidx];
      ngb_both   += sh_ngb2[j + tidx];
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (threadIdx < ni) {
    out_nngb[bodyId] = (int2){ngb_gather, ngb_both};
  }

}

#endif //
