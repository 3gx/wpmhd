#ifndef _EXTRACT_IJLIST_CL_
#define _EXTRACT_IJLIST_CL_

#define HUGE 1.0e10f

///////

__kernel void extract_ijlist(__global int           *out_nj,          //  0
			     __global int           *out_nj_all,      //  1
			     __global int           *out_jidx,        //  2
			     const __global int4    *in_group_bodies, //  3
			     const __global int     *in_ilist,        //  4
			     const __global int     *in_jlist,        //  5
			     const __global float4  *ppos,            //  6
			     const          float4   domain_hsize,    //  7
			     __local        float   *shmem)        {  //  8

  // Get threadIdx, blockIdx & blockDim
  const int localIdx = get_local_id(0);
  const int groupIdx = get_group_id(0);
  const int localDim = NBLOCKDIM;

  // Get i- & j- boides info in a group
  const int4 ifirst  = in_group_bodies[groupIdx];
  const int  ni      = ifirst.y;
  
  const int globalIdx = ifirst.x + min(localIdx, ni-1);
  const int  jbeg     = ifirst.z;
  const int  jend     = ifirst.w;

  // Get bodyId
  const int  bodyId  = in_ilist[globalIdx];
  const int  blockId = globalIdx/localDim;
  const int threadId = globalIdx%localDim;
  const int ijoffset = blockId * (NGBMAX*localDim) + threadId;
  
  // Redistribute shared memory
  __local float *sh_posx =               &shmem  [0];
  __local float *sh_posy =               &sh_posx[localDim];
  __local float *sh_posz =               &sh_posy[localDim];
  __local float *sh_posh =               &sh_posz[localDim];
  __local int   *sh_jidx = (__local int*)&sh_posh[localDim];

  // Get i-body data
  const float4 ipos  = ppos[bodyId];
  const float  hi2   = ipos.w * ipos.w;

  // Extract jidx_list
  int nj_count = 0;
  for (int i = jbeg; i < jend; i += localDim) {
    const int    jidx = (i + localIdx < jend) ? in_jlist[i + localIdx] : -1;
    const float4 jpos = (        jidx >= 0  ) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[localIdx] = jpos.x;
    sh_posy[localIdx] = jpos.y;
    sh_posz[localIdx] = jpos.z;
    sh_posh[localIdx] = jpos.w;
    sh_jidx[localIdx] = jidx;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int j = 0; j < localDim; j++) {
      const float4 dr0  = (float4){sh_posx[j], sh_posy[j], sh_posz[j], 0.0f} - ipos;
      const float4 dr   = dr0 -
#ifdef _PERIODIC_BOUNDARIES_
	(float4){(fabs(dr0.x) > domain_hsize.x) ? 2.0f*sign(dr0.x)*domain_hsize.x : 0.0f,
		 (fabs(dr0.y) > domain_hsize.y) ? 2.0f*sign(dr0.y)*domain_hsize.y : 0.0f,
		 (fabs(dr0.z) > domain_hsize.z) ? 2.0f*sign(dr0.z)*domain_hsize.z : 0.0f,
		 0.0f};
#else
      (float4){0.0f, 0.0f, 0.0f, 0.0f};
#endif

      const float  s2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      const float hj2 = sh_posh[j]*sh_posh[j];
      
      if (s2 > 0.0f && s2 <= fmax(hi2,hj2) && localIdx < ni) {
	out_jidx[ijoffset + nj_count] = sh_jidx[j];
	nj_count += localDim;
      }
      
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  const int nj_ngb = nj_count/localDim;

  if (localIdx < ni) {
    out_nj [globalIdx] = nj_ngb;
    out_nj_all[bodyId] = nj_ngb;
  }
}

/////////

__kernel void extract_gather_list(__global int           *out_nj,          //  0
				  __global int           *out_nj_all,      //  1
				  __global int           *out_jidx,        //  2
				  const __global int4    *in_group_bodies, //  3
				  const __global int     *in_ilist,        //  4
				  const __global int     *in_jlist,        //  5
				  const __global float4  *ppos,            //  6
				  const          float4   domain_hsize,    //  7
				  __local        float   *shmem)        {  //  8

  // Get threadIdx, blockIdx & blockDim
  const int localIdx = get_local_id(0);
  const int groupIdx = get_group_id(0);
  const int localDim = NBLOCKDIM;

  // Get i- & j- boides info in a group
  const int4 ifirst  = in_group_bodies[groupIdx];
  const int  ni      = ifirst.y;
  
  const int globalIdx = ifirst.x + min(localIdx, ni-1);
  const int  jbeg     = ifirst.z;
  const int  jend     = ifirst.w;

  // Get bodyId
  const int  bodyId  = in_ilist[globalIdx];
  const int  blockId = globalIdx/localDim;
  const int threadId = globalIdx%localDim;
  const int ijoffset = blockId * (NGBMAX*localDim) + threadId;
  
  // Redistribute shared memory
  __local float *sh_posx =               &shmem  [0];
  __local float *sh_posy =               &sh_posx[localDim];
  __local float *sh_posz =               &sh_posy[localDim];
  __local int   *sh_jidx = (__local int*)&sh_posz[localDim];

  // Get i-body data
  const float4 ipos  = ppos[bodyId];
  const float  hi2   = ipos.w * ipos.w;

  // Extract jidx_list
  int nj_count = 0;
  for (int i = jbeg; i < jend; i += localDim) {
    const int    jidx = (i + localIdx < jend) ? in_jlist[i + localIdx] : -1;
    const float4 jpos = (        jidx >= 0  ) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[localIdx] = jpos.x;
    sh_posy[localIdx] = jpos.y;
    sh_posz[localIdx] = jpos.z;
    sh_jidx[localIdx] = jidx;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int j = 0; j < localDim; j++) {
      const float4 dr0  = (float4){sh_posx[j], sh_posy[j], sh_posz[j], 0.0f} - ipos;
      const float4 dr   = dr0 -
#ifdef _PERIODIC_BOUNDARIES_
	(float4){(fabs(dr0.x) > domain_hsize.x) ? 2.0f*sign(dr0.x)*domain_hsize.x : 0.0f,
		 (fabs(dr0.y) > domain_hsize.y) ? 2.0f*sign(dr0.y)*domain_hsize.y : 0.0f,
		 (fabs(dr0.z) > domain_hsize.z) ? 2.0f*sign(dr0.z)*domain_hsize.z : 0.0f,
		 0.0f};
#else
      (float4){0.0f, 0.0f, 0.0f, 0.0f};
#endif

      const float s2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;      
      
      if (s2 > 0.0f && s2 <= hi2 && localIdx < ni) {
	out_jidx[ijoffset + nj_count] = sh_jidx[j];
	nj_count += localDim;
      }
      
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  const int nj_ngb = nj_count/localDim;

  if (localIdx < ni) {
    out_nj [globalIdx] = nj_ngb;
    out_nj_all[bodyId] = nj_ngb;
  }
}

#endif // _EXTRACT_IJLIST_CL_
