#include "kernels.clh"

#pragma OPENCL EXTENSION cl_nv_compiler_options

#ifndef _COMPUTE_WNGB_CL_
#define _COMPUTE_WNGB_CL_


// #define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)
// #define blockIdx_x  get_group_id(0)
// #define blockIdx_y  get_group_id(1)
// #define threadIdx_x get_local_id(0)
// #define threadIdx_y get_local_id(1)
// #define gridDim_x   get_num_groups(0)
// #define gridDim_y   get_num_groups(1)
// #define blockDim_x  get_local_size(0)
// #define blockDim_y  get_local_size(1)

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
  for (int i = jbeg; i < jend; i += localDim) {
    const int    jidx = (i + localIdx < jend) ? in_jlist[i + localIdx] : -1;
    const float4 jpos = (        jidx >= 0  ) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[localIdx] = jpos.x;
    sh_posy[localIdx] = jpos.y;
    sh_posz[localIdx] = jpos.z;

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

      const float s  = (s2 > 0.0f) ? 1.0f/rsqrt(s2) : 0.0f;

      const float wk = kernel_w4(s*inv_hi) * NORM_COEFF;
      wngb += wk;
      ingb += (s <= hi) ? 1 : 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (localIdx < ni) {
    in_h_out_wngb[bodyId] = (float2){wngb, ingb};
  }
}


#endif  // _COMPUTE_WNGB_CL_
