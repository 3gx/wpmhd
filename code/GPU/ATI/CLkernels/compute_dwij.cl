#ifndef _COMPUTE_DWIJ_
#define _COMPUTE_DWIJ_

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

__inline float kernel_w(const float u) {
  const float w1 = KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1.0f) * u * u;
  const float w2 = KERNEL_COEFF_5 * (1.0f - u) * (1.0f - u) * (1.0f- u);
  const float w  = (u < 0.5f) ? w1 : ((u < 1.0f) ?  w2 : 0.0f);
  return w;
}

__kernel void compute_dwij(__global float4      *out_dwij,      //  0
			   __global int2        *ijlist,        //  1
			   __global float4      *drij,          //  2 
			   __global float4      *ppos,          //  3
			   __global float4      *pvel,          //  4
			   __global float       *Bxx,           //  5
			   __global float       *Bxy,           //  6
			   __global float       *Bxz,           //  7
			   __global float       *Byy,           //  8
			   __global float       *Byz,           //  9
			   __global float       *Bzz,           // 10
			   int          n_states, // 11
			   float4  domain_hsize) {    // 12
  
  // Compute idx of the element
  const int gidx = get_global_id(0) + get_global_id(1) * get_global_size(0);
  const int idx  = min(gidx, n_states - 1);
  
  // Get i- & j- bodies idx
  const int2 ij = ijlist[idx];
  const int  i  = ij.x;
  const int  j  = ij.y;
  
  const float4 ipos = ppos[i];
  const float4 ivel = pvel[i];
  const float4 jpos = ppos[j];
  const float4 jvel = pvel[j];

  const float4 dr = drij[idx];
  const float   s = dr.w;
  
  const float hi = ipos.w;
  const float hj = jpos.w;
  const float wi = ivel.w;
  const float wj = jvel.w;
  const float inv_hi = 1.0f/hi;
  const float inv_hj = 1.0f/hj;
#if NDIM == 3
  const float inv_hidim = inv_hi*inv_hi*inv_hi;
  const float inv_hjdim = inv_hj*inv_hj*inv_hj;
#else
  const float inv_hidim = inv_hi*inv_hi;
  const float inv_hjdim = inv_hj*inv_hj;
#endif

  const float wki = kernel_w(s * inv_hi) * inv_hidim * wi;
  const float wkj = kernel_w(s * inv_hj) * inv_hjdim * wj;
  
  const float iBxx = Bxx[i];
  const float iBxy = Bxy[i];
  const float iBxz = Bxz[i];
  const float iByy = Byy[i];
  const float iByz = Byz[i];
  const float iBzz = Bzz[i];

  const float4 dwi = {wki * (iBxx*dr.x + iBxy*dr.y + iBxz*dr.z),
		      wki * (iBxy*dr.x + iByy*dr.y + iByz*dr.z),
		      wki * (iBxz*dr.x + iByz*dr.y + iBzz*dr.z), 0.0f}; 

  const float jBxx = Bxx[j];
  const float jBxy = Bxy[j];
  const float jBxz = Bxz[j];
  const float jByy = Byy[j];
  const float jByz = Byz[j];
  const float jBzz = Bzz[j];

  const float4 dwj = {wkj * (jBxx*dr.x + jBxy*dr.y + jBxz*dr.z),
		      wkj * (jBxy*dr.x + jByy*dr.y + jByz*dr.z),
		      wkj * (jBxz*dr.x + jByz*dr.y + jBzz*dr.z), 0.0f};
  
  const float4 dwij = {wi*dwi.x + wj*dwj.x,
		       wi*dwi.y + wj*dwj.y,
		       wi*dwi.z + wj*dwj.z, 0.0f};
  const float  dw2 = dwij.x*dwij.x + dwij.y*dwij.y + dwij.z*dwij.z;
  const float iwij = (dw2 > 0.0f) ? rsqrt(dw2) : 0.0f;
  const float  wij = (dw2 > 0.0f) ? 1.0f/iwij  : 0.0f;

  const float4 e = {dwij.x*iwij, dwij.y*iwij, dwij.z*iwij, 0.0f};
  
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
  
  const float4 W = {0.5f*(ivel.x + jvel.x),
		    0.5f*(ivel.y + jvel.y),
		    0.5f*(ivel.z + jvel.z), 0.0f};
  const float Wx = Axx*W.x + Axy*W.y + Axz*W.z;

  if (gidx < n_states) 
    out_dwij[idx] = (i != j) ? (float4){dwij.x, dwij.y, dwij.z, Wx} : (float4){0.0f, 0.0f, 0.0f, 0.0f};
			      
}


#endif //  _COMPUTE_DWIJ_
