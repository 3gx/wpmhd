#ifndef _COMPUTE_STATES_CL_
#define _COMPUTE_STATES_CL_

__inline float2 monotonicity(const float fLin, const float fRin, const float fi, const float fj) {
  float fL = fLin;
  float fR = fRin;
  const float Fmin = fmin(fi, fj);
  const float Fmax = fmax(fi, fj);
  if ((fL < Fmin) || (fL > Fmax) || (fR < Fmin) || (fR > Fmax)) {
    fL = fR = fi;
    fL = fi;
    fR = fj;
  }
  return (float2){fL, fR};
}

__kernel void compute_states(__global       float4      *out_stateL,      //  0
			     __global       float4      *out_stateR,      //  1
			     const __global float4      *data,            //  2
			     const __global float4      *grad_x,          //  3
			     const __global float4      *grad_y,          //  4
			     const __global float4      *grad_z,          //  5
			     const __global int         *in_ilist,        //  6
			     const __global int         *in_jlist,        //  7
			     const __global int         *in_nj,           //  8
			     const __global float4      *in_dwij,         //  9
			     const __global float4      *ppos,            // 10
			     const          int          Ni,              // 11
			     const          float4       domain_hsize,    // 12
			     const          int          reconstruct)  {  // 13
  
  // Get thread info: localIdx & globalIdx, etc

  const int localIdx  = get_local_id(0);
  const int localDim  = NBLOCKDIM; //get_local_size(0);
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

  const float4 ipos    = ppos  [bodyId];
  const float4 idata0  = data  [bodyId];
  const float4 igrad_x = grad_x[bodyId];
  const float4 igrad_y = grad_y[bodyId];
  const float4 igrad_z = grad_z[bodyId];

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

    const float4 jdata0 = data[j];
    
    float4 idata = idata0;
    float4 jdata = jdata0;

    if ((reconstruct & 1) == 1) {
      const float4 ds = {0.5f*dx, 0.5f*dy, 0.5f*dz, 0.0f};
      
      ///////// ij-state mhd1
      
      idata += ds.x * igrad_x;
      idata += ds.y * igrad_y;
      idata += ds.z * igrad_z;
    
      jdata -= ds.x * grad_x[j];
      jdata -= ds.y * grad_y[j];
      jdata -= ds.z * grad_z[j];
      
//       float2 v;
//       v = monotonicity(idata.x, jdata.x, idata0.x, jdata0.x); idata.x = v.x; jdata.x = v.y;
//       v = monotonicity(idata.y, jdata.y, idata0.y, jdata0.y); idata.y = v.x; jdata.y = v.y;
//       v = monotonicity(idata.z, jdata.z, idata0.z, jdata0.z); idata.z = v.x; jdata.z = v.y;
//       v = monotonicity(idata.w, jdata.w, idata0.w, jdata0.w); idata.w = v.x; jdata.w = v.y;
    }
    
    if ((reconstruct & 2) == 2) {
      const float4 dwij = in_dwij[jidx];
      
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
      
      const float idatx = Axx*idata.x + Axy*idata.y + Axz*idata.z;
      const float idaty = Ayx*idata.x + Ayy*idata.y + Ayz*idata.z;
      const float idatz = Azx*idata.x + Azy*idata.y + Azz*idata.z;
      
      const float jdatx = Axx*jdata.x + Axy*jdata.y + Axz*jdata.z;
      const float jdaty = Ayx*jdata.x + Ayy*jdata.y + Ayz*jdata.z;
      const float jdatz = Azx*jdata.x + Azy*jdata.y + Azz*jdata.z;
      
      idata = (float4){idatx, idaty, idatz, idata.w};
      jdata = (float4){jdatx, jdaty, jdatz, jdata.w};
    }

    out_stateL[jidx] = idata;
    out_stateR[jidx] = jdata;

  }

}

#endif
