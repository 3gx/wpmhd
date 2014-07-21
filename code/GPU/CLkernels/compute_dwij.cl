#include "kernels.clh"

#ifndef _COMPUTE_DWIJ_
#define _COMPUTE_DWIJ_

__kernel void compute_dwij(__global       float4      *out_dwij,          //  0
			   const __global int         *in_ilist,          //  1
			   const __global int         *in_jlist,          //  2
			   const __global int         *in_nj,             //  3
			   const __global float4      *ppos,              //  4
			   const __global float4      *pvel,              //  5
			   const __global float       *Bxx,               //  6
			   const __global float       *Bxy,               //  7
			   const __global float       *Bxz,               //  8
			   const __global float       *Byy,               //  9
			   const __global float       *Byz,               // 10
			   const __global float       *Bzz,               // 11
			   const          int          Ni,                // 12
			   const          float4       domain_hsize) {    // 13
  
  
  // Get thread info: localIdx & globalIdx, etc

  const int localIdx  = get_local_id(0);
  const int localDim  = get_local_size(0);
  const int groupIdx  = get_group_id(0);
  const int globalIdx = get_global_id(0);
  if (globalIdx >= Ni) return;

  // Get bodyId

  const int bodyId = in_ilist[globalIdx];   
  const int nj     = in_nj   [globalIdx];
  
  // compute Idx of first and last j-particle
  
  const int jbeg = groupIdx * (NGBMAX*localDim) + localIdx;
  const int jend = jbeg + nj * localDim;

  // Get i-ptcl data

  const float4 ipos  = ppos[bodyId];
  const float4 ivel  = pvel[bodyId];
  
  const float iBxx = Bxx[bodyId];
  const float iBxy = Bxy[bodyId];
  const float iBxz = Bxz[bodyId];
  const float iByy = Byy[bodyId];
  const float iByz = Byz[bodyId];
  const float iBzz = Bzz[bodyId];
    

  const float hi     = ipos.w;
  const float wi     = ivel.w;
  const float inv_hi = 1.0f/hi;
#if NDIM == 3
  const float wi_inv_hidim = inv_hi*inv_hi*inv_hi * wi;
#elif NDIM == 2
  const float wi_inv_hidim = inv_hi*inv_hi * wi;
#endif


  for (int jidx = jbeg; jidx < jend; jidx += localDim) {
    const int j = in_jlist[jidx];

    const float4 jpos = ppos[j];
    const float  hj   = jpos.w;
    
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


    const float4 jvel  = pvel[j];
    const float wj     = jvel.w;
    const float inv_hj = 1.0f/hj;
#if NDIM == 3
    const float wj_inv_hjdim = wj * inv_hj*inv_hj*inv_hj;
#elif NDIM == 2
    const float wj_inv_hjdim = wj * inv_hj*inv_hj;
#endif
    
    const float wki = kernel_w(s * inv_hi) * wi_inv_hidim;
    const float wkj = kernel_w(s * inv_hj) * wj_inv_hjdim;

    const float4 dwi = {wki * (iBxx*dx + iBxy*dy + iBxz*dz),
			wki * (iBxy*dx + iByy*dy + iByz*dz),
			wki * (iBxz*dx + iByz*dy + iBzz*dz), 0.0f}; 

    const float jBxx = Bxx[j];
    const float jBxy = Bxy[j];
    const float jBxz = Bxz[j];
    const float jByy = Byy[j];
    const float jByz = Byz[j];
    const float jBzz = Bzz[j];
    
    const float4 dwj = {wkj * (jBxx*dx + jBxy*dy + jBxz*dz),
			wkj * (jBxy*dx + jByy*dy + jByz*dz),
			wkj * (jBxz*dx + jByz*dy + jBzz*dz), 0.0f};
    
    const float4 dwij = wi * dwi + wj * dwj;

    
    const float  dw2 = dwij.x*dwij.x + dwij.y*dwij.y + dwij.z*dwij.z;
    const float iwij = (dw2 > 0.0f) ? rsqrt(dw2) : 0.0f;
    const float  wij = (dw2 > 0.0f) ? 1.0f/iwij  : 0.0f;
  
    const float4 e   = dwij * iwij;
    
    const float dR2   = e.x*e.x + e.y*e.y;
    const float idR   = (dR2 > 0.0f) ? rsqrt(dR2) : 0.0f;
    const float cosph = (idR == 0.0f) ? 1.0f  : e.x*idR;
    const float sinph = (idR == 0.0f) ? 0.0f  : e.y*idR;
    const float costh = e.z;
    const float sinth = (dR2 > 0.0f) ? 1.0f/idR : 0.0f;
    
    const float Axx =  cosph*sinth;
    const float Axy =  sinth*sinph;
    const float Axz =  costh;
    const float Ayx = -sinph;
    const float Ayy =  cosph;
    const float Ayz =  0.0f;
    const float Azx = -costh*cosph;
    const float Azy = -costh*sinph;
    const float Azz =  sinth;

    const float4 W  = 0.5f*(ivel + jvel);
    const float  Wx = Axx*W.x + Axy*W.y + Axz*W.z;
    
    out_dwij[jidx] = (float4){dwij.x, dwij.y, dwij.z, Wx};
  }
  
			      
}


#endif //  _COMPUTE_DWIJ_
