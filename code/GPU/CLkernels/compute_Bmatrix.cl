#include "kernels.clh"

#ifndef _COMPUTE_BMATRIX_CL_
#define _COMPUTE_BMATRIX_CL_

__kernel void compute_Bmatrix(__global float         *out_Bxx,       //  0
			      __global float         *out_Bxy,       //  1
			      __global float         *out_Bxz,       //  2
			      __global float         *out_Byy,       //  3
			      __global float         *out_Byz,       //  4
			      __global float         *out_Bzz,       //  5
			      __global float         *out_dwdt,      //  6
			      const __global int     *in_ilist,      //  7
			      const __global int     *in_jlist,      //  8
			      const __global int     *in_nj,         //  9
			      const __global float4  *ppos,          // 10
			      const __global float4  *pvel,          // 11
			      const          float4   domain_hsize,  // 12
			      const          int      Ni) {          // 13
  
  // Get thread info: localIdx & globalIdx, etc
  const int localIdx  = get_local_id(0);
  const int localDim  = NBLOCKDIM; //get_local_size(0);
  const int groupIdx  = get_group_id(0);
  const int globalIdx = get_global_id(0);
  if (globalIdx >= Ni) return;
  

  // Get bodyId

  const int bodyId = in_ilist[globalIdx];         // alligned to 128-byte boundary
#if 0
//   const int NGBMIN = 16;
//   const int NGBMAX = 256;
  const int njm    = in_nj   [globalIdx] - 1;         // number of ngb for a particle
  const int nj     = ((njm / NGBMIN) + 1) * NGBMIN;      // nj%16 = 0, nj >= njm + 1
#else
  const int nj = in_nj[globalIdx];
#endif

  // compute Idx of first and last j-particle
  
  const int jbeg = groupIdx * (NGBMAX*localDim) + localIdx;
  const int jend = jbeg + nj * localDim;

  // compute renormalisation matrix
  
  const float4 ipos = ppos[bodyId];       // Not necessary aligned to 128-byte boundary
  const float4 ivel = pvel[bodyId];       // Not necessary aligned ....
  
  const float wi      = ivel.w;
  const float hi      = ipos.w;
  const float inv_hi  = 1.0f/hi;
#if NDIM == 3
  const float inv_hidim = inv_hi*inv_hi*inv_hi;
#elif NDIM == 2
  const float inv_hidim = inv_hi*inv_hi;
#endif
  const float wi_inv_hidim  = wi * inv_hidim;

  float Exx = 0.0f;
  float Exy = 0.0f;
  float Exz = 0.0f;
  float Eyy = 0.0f;
  float Eyz = 0.0f;
  float Ezz = 0.0f;
  float omega = 0.0f;
  float dnidt = 0.0f;
  
  // pragma unroll 16
  for (int j = jbeg; j < jend; j += localDim) {
    const int    jidx = in_jlist[j];
    const float4 jpos = ppos[jidx];
    const float4 dr = jpos - ipos;
#ifdef _PERIODIC_BOUNDARIES_
    const float dx = dr.x + ((fabs(dr.x) > domain_hsize.x) ? -2.0f*sign(dr.x)*domain_hsize.x : 0.0f);
    const float dy = dr.y + ((fabs(dr.y) > domain_hsize.y) ? -2.0f*sign(dr.y)*domain_hsize.y : 0.0f);
    const float dz = dr.z + ((fabs(dr.z) > domain_hsize.z) ? -2.0f*sign(dr.z)*domain_hsize.z : 0.0f);
#else
    const float dx = dr.x;
    const float dy = dr.y;
    const float dz = dr.z;
#endif
    const float s = sqrt(dx*dx + dy*dy + dz*dz);
    const float wk = kernel_w(s*inv_hi) * wi_inv_hidim;
    
    Exx += wk * dx*dx;
    Exy += wk * dx*dy;
    Exz += wk * dx*dz;
    Eyy += wk * dy*dy;
    Eyz += wk * dy*dz;
    Ezz += wk * dz*dz;

    const float4 jvel = pvel[jidx];
    const float4 dv = jvel - ivel;
    const float drdv = dx*dv.x + dy*dv.y + dz*dv.z;
    const float is2 = (s > 0.0f) ? 1.0f/(s*s) : 0.0f;
    const float qij = s * inv_hi;
    const float dwk = kernel_dw4(qij);

    omega += qij * dwk;
    dnidt += qij * dwk * drdv * is2;
  }
  
  // compute dwdt

  const float ni = 1.0f/wi;
  const float dhdn = -(hi/ni)/NDIM;
  omega  = 1.0f  + dhdn * inv_hi * (NDIM*ni + omega*inv_hidim);
  omega *= 1.0f/inv_hidim;
  dnidt *= 1.0f/omega;
  
  out_dwdt[bodyId] = -dnidt/(ni*ni);

  // invert E-matrix
  
  const float A11 = Exx;
  const float A12 = Exy;
  const float A22 = Eyy;
  const float A13 = Exz;
  const float A23 = Eyz;
  const float A33 = Ezz;

#if NDIM == 2
  const float det  = A11*A22 - A12*A12;
  const float idet = (fabs(det) > SMALLF) ? 1.0f/det : 0.0f;
  
  const float Bxx =  A22*idet;
  const float Bxy = -A12*idet;
  const float Bxz =  0.0f;
  const float Byy =  A11*idet;
  const float Byz =  0.0f;
  const float Bzz =  0.0f;
  
#elif NDIM == 3
  const float det = -A13*A13*A22 + 2.0f*A12*A13*A23 - A11*A23*A23 - A12*A12*A33 + A11*A22*A33;
  const float idet = (fabs(det) > SMALLF) ? 1.0f/det : 0.0f;

  const float Bxx = (-A23*A23 + A22*A33)*idet;      
  const float Bxy = (+A13*A23 - A12*A33)*idet;      
  const float Bxz = (-A13*A22 + A12*A23)*idet;      
  const float Byy = (-A13*A13 + A11*A33)*idet;      
  const float Byz = (+A12*A13 - A11*A23)*idet;      
  const float Bzz = (-A12*A12 + A11*A22)*idet;      

#endif

  out_Bxx[bodyId] = Bxx;
  out_Bxy[bodyId] = Bxy;
  out_Bxz[bodyId] = Bxz;
  out_Byy[bodyId] = Byy;
  out_Byz[bodyId] = Byz;
  out_Bzz[bodyId] = Bzz;
  
}


#endif  // _COMPUTE_BMATRIX_CL_
