#ifndef _COMPUTE_BMATRIX_CL_
#define _COMPUTE_BMATRIX_CL_

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

__inline float kernel_dw(const float u) {
  const float w1 = 0.0f*KERNEL_COEFF_1 + KERNEL_COEFF_2 * (3.0f * u*u - 2.0f * u);
  const float w2 = -3.0f*KERNEL_COEFF_5 * (1.0f - u) * (1.0f - u);
  const float w  = (u < 0.5f) ? w1 : ((u < 1.0f) ?  w2 : 0.0f);
  return w;
}

// #define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)
// #define blockIdx_x  get_group_id(0)
// #define blockIdx_y  get_group_id(1)
// #define threadIdx_x get_local_id(0)
// #define threadIdx_y get_local_id(1)
// #define gridDim_x   get_num_groups(0)
// #define gridDim_y   get_num_groups(1)
// #define blockDim_x  get_local_size(0)
// #define blockDim_y  get_local_size(1)

__kernel void compute_Bmatrix(__global float       *out_Bxx,       //  0
			      __global float       *out_Bxy,       //  1
			      __global float       *out_Bxz,       //  2
			      __global float       *out_Byy,       //  3
			      __global float       *out_Byz,       //  4
			      __global float       *out_Bzz,       //  5
			      __global int         *leaf_list,     //  6
			      __global int2        *body_list,     //  7
			      __global int         *ilist,         //  8
			      __global int         *ijlist_offset, //  9
			      __global int         *leaf_ngb_max,  // 10
			      __global float4      *ppos,          // 11
			      __global float4      *pvel,          // 12
			      __global float4      *drij) {        // 13
  

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
  
  // Compute offset where to write neighbours idx
  const int ijoffset    = ijlist_offset[leaf_id] + min(threadIdx, ni-1);
  const int nj          = leaf_ngb_max [leaf_id];

  const int ngb_max = ni * nj;

  //
  const float4 ipos = ppos[bodyId];
  const float4 ivel = pvel[bodyId];

  const float wi      = ivel.w;
  const float hi      = ipos.w;
  const float inv_hi  = 1.0f/hi;
#if NDIM == 3
  const float wi_inv_hidim  = wi * inv_hi*inv_hi*inv_hi;
#elif NDIM == 2
  const float wi_inv_hidim  = wi * inv_hi*inv_hi;
#endif

  ////

  float Exx = 0.0f;
  float Exy = 0.0f;
  float Exz = 0.0f;
  float Eyy = 0.0f;
  float Eyz = 0.0f;
  float Ezz = 0.0f;
  
  for (int i = 0; i < ngb_max; i += ni) {
    const int iaddr = ijoffset + i;
    const float4 dr = drij  [iaddr];

    const float wk = kernel_w(dr.w*inv_hi) * wi_inv_hidim;
    
    Exx += wk * dr.x*dr.x;
    Exy += wk * dr.x*dr.y;
    Exz += wk * dr.x*dr.z;
    Eyy += wk * dr.y*dr.y;
    Eyz += wk * dr.y*dr.z;
    Ezz += wk * dr.z*dr.z;

  }
  
  
  const float A11 = Exx;
  const float A12 = Exy;
  const float A22 = Eyy;
  const float A13 = Exz;
  const float A23 = Eyz;
  const float A33 = Ezz;

  float Bxx, Bxy, Bxz, Byy, Byz, Bzz;

#if NDIM == 2
  const float det  = A11*A22 - A12*A12;
  const float idet = (det != 0.0f) ? 1.0f/det : 0.0f;
  
  Bxx =  A22*idet;
  Bxy = -A12*idet;
  Bxz =  0.0f;
  Byy =  A11*idet;
  Byz =  0.0f;
  Bzz =  0.0f;
  
#elif NDIM == 3
  const float det = -A13*A13*A22 + 2.0f*A12*A13*A23 - A11*A23*A23 - A12*A12*A33 + A11*A22*A33;
  const float idet = (det != 0.0f) ? 1.0f/det : 0.0f;

  Bxx = (-A23*A23 + A22*A33)*idet;      
  Bxy = (+A13*A23 - A12*A33)*idet;      
  Bxz = (-A13*A22 + A12*A23)*idet;      
  Byy = (-A13*A13 + A11*A33)*idet;      
  Byz = (+A12*A13 - A11*A23)*idet;      
  Bzz = (-A12*A12 + A11*A22)*idet;      

#endif

  if (threadIdx < ni) {
    out_Bxx[bodyId] = Bxx;
    out_Bxy[bodyId] = Bxy;
    out_Bxz[bodyId] = Bxz;
    out_Byy[bodyId] = Byy;
    out_Byz[bodyId] = Byz;
    out_Bzz[bodyId] = Bzz;
  }

}


#endif  // _COMPUTE_BMATRIX_CL_
