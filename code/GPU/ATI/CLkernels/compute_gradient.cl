#ifndef _COMPUTE_GRADIENT_CL_
#define _COMPUTE_GRADIENT_CL_

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
#define TINY 1.0e-10f

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


__inline float slope_limiter(const float vi,
			     const float vi_min,
			     const float vi_max,
			     const float vmin,
			     const float vmax,
			     const float psi) {
  
  const float xmax = (vmax - vi)/(vi_max - vi + TINY);
  const float xmin = (vi - vmin)/(vi - vi_min + TINY);
  
  return fmin(1.0f, psi*fmin(xmin, xmax));
}

__kernel void compute_gradient(__global float4      *out_grad_x,    //  0
			       __global float4      *out_grad_y,    //  1
			       __global float4      *out_grad_z,    //  2
			       __global float4      *data,          //  3
			       __global int         *leaf_list,     //  4
			       __global int2        *body_list,     //  5
			       __global int         *ilist,         //  6
			       __global int         *ijlist_offset, //  7
			       __global int         *leaf_ngb_max,  //  8
			       __global int2        *ijlist,        //  9 
			       __global float4      *ppos,          // 10
			       __global float4      *pvel,          // 11
			       __global float4      *drij,          // 12
			       __global float       *Bxx,           // 13
			       __global float       *Bxy,           // 14
			       __global float       *Bxz,           // 15
			       __global float       *Byy,           // 16
			       __global float       *Byz,           // 17
			       __global float       *Bzz) {         // 18

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

  // Get i-ptcl data
  const float4 ipos  = ppos[bodyId];
  const float4 ivel  = pvel[bodyId];
  const float hi     = ipos.w;
  const float wi     = ivel.w;
  const float inv_hi = 1.0f/hi;
#if NDIM == 3
  const float wi_inv_hidim = inv_hi*inv_hi*inv_hi * wi;
#else
  const float wi_inv_hidim = inv_hi*inv_hi * wi;
#endif

  const float4 idata = data[bodyId];
  float4 grad0_x = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 grad0_y = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 grad0_z = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int i = 0; i < ngb_max; i += ni) {
    const int iaddr = ijoffset + i;
    const float4 dr = drij  [iaddr];
    const int2 ij   = ijlist[iaddr];
    const int  j    = ij.y;

    const float wk = kernel_w(dr.w*inv_hi) * wi_inv_hidim;
    
    const float4 jdata = data[j];
    const float4 diff = wk * (jdata - idata);
    grad0_x += dr.x * diff;
    grad0_y += dr.y * diff;
    grad0_z += dr.z * diff;
    
  }

  const float Axx = Bxx[bodyId];
  const float Axy = Bxy[bodyId];
  const float Axz = Bxz[bodyId];
  const float Ayy = Byy[bodyId];
  const float Ayz = Byz[bodyId];
  const float Azz = Bzz[bodyId];

  const float4 grad_x = Axx*grad0_x + Axy*grad0_y + Axz*grad0_z;
  const float4 grad_y = Axy*grad0_x + Ayy*grad0_y + Ayz*grad0_z;
  const float4 grad_z = Axz*grad0_x + Ayz*grad0_y + Azz*grad0_z;

  if (threadIdx < ni) {
    out_grad_x[bodyId] = grad_x;
    out_grad_y[bodyId] = grad_y;
    out_grad_z[bodyId] = grad_z;
  }
}


__kernel void limit_gradient(__global float4      *out_psi,       //  0
			     __global float4      *data,          //  1
			     __global float4      *in_grad_x,     //  2
			     __global float4      *in_grad_y,     //  3
			     __global float4      *in_grad_z,     //  4
			     __global int         *leaf_list,     //  5
			     __global int2        *body_list,     //  6
			     __global int         *ilist,         //  7
			     __global int         *ijlist_offset, //  8
			     __global int         *leaf_ngb_max,  //  9
			     __global int2        *ijlist,        // 10
			     __global float4      *drij,          // 11
			     const    float4       fc) {          // 12

			     
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

  ///////

  const float4 idata = data[bodyId];

  float4 jdata_min = idata;
  float4 jdata_max = idata;
  float4 idata_min = idata;
  float4 idata_max = idata;


  const float4 grad_x = in_grad_x[bodyId];
  const float4 grad_y = in_grad_y[bodyId];
  const float4 grad_z = in_grad_z[bodyId];

  for (int i = 0; i < ngb_max; i += ni) {
    const int iaddr = ijoffset + i;
    const float4 dr = drij  [iaddr];
    const int2 ij   = ijlist[iaddr];
    const int  j    = ij.y;
    const float4 ds = {0.5f*dr.x, 0.5f*dr.y, 0.5f*dr.z, 0.0f};
    
    const float4 jdata = data[j];
    jdata_min = fmin(jdata_min, jdata);
    jdata_max = fmax(jdata_max, jdata);

    const float4 idata_ij = idata + ds.x * grad_x + ds.y * grad_y + ds.z * grad_z;
    idata_min = fmin(idata_min, idata_ij);
    idata_max = fmax(idata_max, idata_ij);
  }

  const float4 psi = {slope_limiter(idata.x, idata_min.x, idata_max.x, jdata_min.x, jdata_max.x, fc.x),
		      slope_limiter(idata.y, idata_min.y, idata_max.y, jdata_min.y, jdata_max.y, fc.y),
		      slope_limiter(idata.z, idata_min.z, idata_max.z, jdata_min.z, jdata_max.z, fc.z),
		      slope_limiter(idata.w, idata_min.w, idata_max.w, jdata_min.w, jdata_max.w, fc.w)};
  if (threadIdx < ni) out_psi[bodyId] = psi;

}

__kernel void limit_gradient2(__global float4   *inout_mhd1_grad_x,    //  0
			      __global float4   *inout_mhd1_grad_y,    //  1
			      __global float4   *inout_mhd1_grad_z,    //  2
			      __global float4   *inout_mhd2_grad_x,    //  3
			      __global float4   *inout_mhd2_grad_y,    //  4
			      __global float4   *inout_mhd2_grad_z,    //  5
			      __global float4   *inout_mhd3_grad_x,    //  6
			      __global float4   *inout_mhd3_grad_y,    //  7
			      __global float4   *inout_mhd3_grad_z,    //  8
			      __global float4   *inmhd1_psi,           //  9
			      __global float4   *inmhd2_psi,           // 10
			      __global float4   *inmhd3_psi,           // 11
			      const    int       n_particles) {        // 12
  
  // Compute idx of the element
  const int gidx = get_global_id(0) + get_global_id(1) * get_global_size(0);
  const int idx  = min(gidx, n_particles - 1);
#if 0
  inout_mhd1_grad_x[idx] = inmhd1_psi[idx];
#else
  
  const float4 mhd1_psi    = inmhd1_psi[idx];
  const float4 mhd1_grad_x = inout_mhd1_grad_x[idx];
  const float4 mhd1_grad_y = inout_mhd1_grad_y[idx];
  const float4 mhd1_grad_z = inout_mhd1_grad_z[idx];
  if (gidx < n_particles) {
    inout_mhd1_grad_x[idx] = mhd1_grad_x * mhd1_psi;
    inout_mhd1_grad_y[idx] = mhd1_grad_y * mhd1_psi;
    inout_mhd1_grad_z[idx] = mhd1_grad_z * mhd1_psi;
  }

  const float4 mhd2_psi    = inmhd2_psi[idx];
  const float4 mhd2_grad_x = inout_mhd2_grad_x[idx];
  const float4 mhd2_grad_y = inout_mhd2_grad_y[idx];
  const float4 mhd2_grad_z = inout_mhd2_grad_z[idx];
  if (gidx < n_particles) {
    inout_mhd2_grad_x[idx] = mhd2_grad_x * mhd2_psi;
    inout_mhd2_grad_y[idx] = mhd2_grad_y * mhd2_psi;
    inout_mhd2_grad_z[idx] = mhd2_grad_z * mhd2_psi;
  }

  const float4 mhd3_psi    = inmhd3_psi[idx];
  const float4 mhd3_grad_x = inout_mhd3_grad_x[idx];
  const float4 mhd3_grad_y = inout_mhd3_grad_y[idx];
  const float4 mhd3_grad_z = inout_mhd3_grad_z[idx];
  if (gidx < n_particles) {
    inout_mhd3_grad_x[idx] = mhd3_grad_x * mhd3_psi;
    inout_mhd3_grad_y[idx] = mhd3_grad_y * mhd3_psi;
    inout_mhd3_grad_z[idx] = mhd3_grad_z * mhd3_psi;
  }
#endif  
}
			      

#endif  // _COMPUTE_GRADIENT_CL_
