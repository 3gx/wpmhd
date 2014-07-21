#pragma OPENCL EXTENSION cl_nv_compiler_options

#ifndef _COMPUTE_WNGB_CL_
#define _COMPUTE_WNGB_CL_

#if NDIM == 2
#define KERNEL_COEFF_1  (5.0f/7.0f*2.546479089470f )
#define KERNEL_COEFF_2  (5.0f/7.0f*15.278874536822f)
#define KERNEL_COEFF_3  (5.0f/7.0f*45.836623610466f)
#define KERNEL_COEFF_4  (5.0f/7.0f*30.557749073644f)
#define KERNEL_COEFF_5  (5.0f/7.0f*5.092958178941f)
#define KERNEL_COEFF_6  (5.0f/7.0f*(-15.278874536822f))
#define NORM_COEFF      3.14159265358979f
#elif NDIM == 3
#define KERNEL_COEFF_1   2.546479089470f
#define KERNEL_COEFF_2  15.278874536822f
#define KERNEL_COEFF_3  45.836623610466f
#define KERNEL_COEFF_4  30.557749073644f
#define KERNEL_COEFF_5   5.092958178941f
#define KERNEL_COEFF_6 (-15.278874536822f)
#define NORM_COEFF     (4.0f/3.0f*3.14159265358979)
#endif

#define HUGE 1.0e10f

__inline float kernel_w(const float u) {
  const float w1 = KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1.0f) * u * u;
  const float w2 = KERNEL_COEFF_5 * (1.0f - u) * (1.0f - u) * (1.0f- u);
  const float w  = (u < 0.5f) ? w1 : ((u < 1.0f) ?  w2 : 0.0f);
  return w;
}


__kernel void compute_wngb(__global float2      *in_h_out_wngb, //  0
			   __global int         *leaf_list,     //  1
			   __global int2        *body_list,     //  2
			   __global int         *ilist,         //  3
			   __global float4      *ppos,          //  4
			   __global int         *jlist,         //  5
			   __global int         *joffset,       //  6
			   const    float4       domain_hsize,  //  7
			   __local  float        *shmem) {      //  8
  

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
  const float4 ipos  = ppos[bodyId];
  const float2 wngb2 = in_h_out_wngb[bodyId];
  const float hi     = wngb2.x;
  const float inv_hi = 1.0f/hi;
  
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
  
  // Compute renormalisation matrix & count number of neighbours
  float wngb = 0.0f;
  int   ingb = 0;
  
  for (int i = nj0; i < nj1; i += ni2b) {
    const int  jp  = i + tidx;
    const int jidx = (jp < nj1e) ? jlist[offset + jp] : -1;
    
    const float4 jpos  = (jidx >= 0) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[tid] = jpos.x;
    sh_posy[tid] = jpos.y;
    sh_posz[tid] = jpos.z;
    
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
      const float s  = (s2 > 0.0f) ? 1.0f/rsqrt(s2) : 0.0f;

      const float wk = kernel_w(s*inv_hi) * NORM_COEFF;
      wngb += wk;
      ingb += (s <= hi) ? 1 : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Reduce data between lanes
  __local float *sh_wngb = &shmem[0];
  __local int   *sh_ingb = &sh_wngb[blockDim];
  sh_wngb[threadIdx] = wngb;
  sh_ingb[threadIdx] = ingb;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (lane == 0) {
    const int ni2b_max = blockDim;       // nlanes * ni2b;
    for (int j = ni2b; j < ni2b_max; j += ni2b) {
      wngb += sh_wngb[j + tidx];
      ingb += sh_ingb[j + tidx];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (threadIdx < ni) {
    in_h_out_wngb[bodyId] = (float2){wngb, ingb};
  }

}




__kernel void compute_wngb(__global float2      *in_h_out_wngb, //  0
			   __global int         *leaf_list,     //  1
			   __global int2        *body_list,     //  2
			   __global int         *ilist,         //  3
			   __global float4      *ppos,          //  4
			   __global int         *jlist,         //  5
			   __global int         *joffset,       //  6
			   const    float4       domain_hsize,  //  7
			   __local  float        *shmem) {      //  8
  

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
  const float4 ipos  = ppos[bodyId];
  const float2 wngb2 = in_h_out_wngb[bodyId];
  const float hi     = wngb2.x;
  const float inv_hi = 1.0f/hi;
  
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
  
  // Compute renormalisation matrix & count number of neighbours
  float wngb = 0.0f;
  int   ingb = 0;
  
  for (int i = nj0; i < nj1; i += ni2b) {
    const int  jp  = i + tidx;
    const int jidx = (jp < nj1e) ? jlist[offset + jp] : -1;
    
    const float4 jpos  = (jidx >= 0) ? ppos[jidx] : (float4){+HUGE, +HUGE, +HUGE, 0.0f};
    sh_posx[tid] = jpos.x;
    sh_posy[tid] = jpos.y;
    sh_posz[tid] = jpos.z;
    
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
      const float s  = (s2 > 0.0f) ? 1.0f/rsqrt(s2) : 0.0f;

      const float wk = kernel_w(s*inv_hi) * NORM_COEFF;
      wngb += wk;
      ingb += (s <= hi) ? 1 : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Reduce data between lanes
  __local float *sh_wngb = &shmem[0];
  __local int   *sh_ingb = &sh_wngb[blockDim];
  sh_wngb[threadIdx] = wngb;
  sh_ingb[threadIdx] = ingb;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (lane == 0) {
    const int ni2b_max = blockDim;       // nlanes * ni2b;
    for (int j = ni2b; j < ni2b_max; j += ni2b) {
      wngb += sh_wngb[j + tidx];
      ingb += sh_ingb[j + tidx];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (threadIdx < ni) {
    in_h_out_wngb[bodyId] = (float2){wngb, ingb};
  }

}


#endif  // _COMPUTE_WNGB_CL_
