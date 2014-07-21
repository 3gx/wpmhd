#include "kernels.clh"

#ifndef _COMPUTE_WNGB_CL_
#define _COMPUTE_WNGB_CL_

__kernel void compute_wngb(      __global float2  *in_h_out_wngb,   //  0
			   const __global int4    *in_group_bodies, //  1
			   const __global int     *in_ilist,        //  2
			   const __global int     *in_jlist,        //  3
			   const __global float4  *ppos,            //  4
			   const          float4   domain_hsize,    //  5
			   __local        float   *shmem)        {  //  6
  
  // Get threadIdx, blockIdx & blockDim
  const int localIdx = get_local_id(0);
  const int groupIdx = get_group_id(0);
  const int localDim = 128; //NBLOCKDIM;

  // Get i- & j- boides info in a group
  const int4 ifirst  = in_group_bodies[groupIdx];
  const int  ni      = ifirst.y;

  // Use multiple threads per body

  int ni2b = 1;
  while (ni2b < ni) ni2b = ni2b << 1;
  ni2b = max(ni2b, NMAXPERLANE);
  const int nlanes = localDim/ni2b;
  const int  lane  = localIdx/ni2b;
  const int  tidx  = localIdx%ni2b;

  
  const int globalIdx = ifirst.x + min(tidx, ni-1);
  const int  jbeg0    = ifirst.z;
  const int  jend0    = ifirst.w;
  const int  nj       = jend0 - jbeg0;

  const int nj_lane      = nj/nlanes;
  const int nj_lane_last = nj - nj_lane*(nlanes - 1);

  const int jbeg  = jbeg0 + nj_lane*lane;
  const int jend  = jbeg  + nj_lane_last;
  const int jend1 = jbeg  + ((lane == nlanes - 1) ? nj_lane_last : nj_lane);

  const int tid0    = ni2b * lane;
  const int tid1    = ni2b + tid0;
  const int tid     = tidx + tid0;
  

  // Get bodyId
  const int  bodyId  = in_ilist[globalIdx];
  
  // Redistribute shared memory
  __local float *sh_posx = &shmem  [0];
  __local float *sh_posy = &sh_posx[localDim];
  __local float *sh_posz = &sh_posy[localDim];

  // Get i-body data
  const float4 ipos  = ppos[bodyId];
  const float2 wngb2 = in_h_out_wngb[bodyId];
  const float hi     = wngb2.x;
  const float inv_hi = 1.0f/hi;

  // Extract jidx_list
  float wngb = 0.0f;
  int   ingb = 0;
  for (int i = jbeg; i < jend; i += ni2b) {
    const int    jidx = (i + tidx < jend1) ? in_jlist[i + tidx] : -1;
    const float4 jpos = (    jidx >= 0   ) ?     ppos[    jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[tid] = jpos.x;
    sh_posy[tid] = jpos.y;
    sh_posz[tid] = jpos.z;

    barrier(CLK_LOCAL_MEM_FENCE);
    
#pragma unroll NMAXPERLANE
    for (int j = tid0; j < tid1; j++) {
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

      const float s  = (s2 > 0.0f) ? 1.0f/rsqrt(s2) : 0.0f;

      const float wk = kernel_w4(s*inv_hi) * NORM_COEFF;
      wngb += wk;
      ingb += (s <= hi) ? 1 : 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Reduce data between lanes
  __local float *sh_wngb = &shmem[0];
  __local int   *sh_ingb = &sh_wngb[localDim];
  sh_wngb[localIdx] = wngb;
  sh_ingb[localIdx] = ingb;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (lane == 0) 
    for (int j = ni2b; j < localDim; j += ni2b) {
      wngb += sh_wngb[j + tidx];
      ingb += sh_ingb[j + tidx];
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (localIdx < ni) {
    in_h_out_wngb[bodyId] = (float2){wngb, ingb};
  }
}


#endif  // _COMPUTE_WNGB_CL_
