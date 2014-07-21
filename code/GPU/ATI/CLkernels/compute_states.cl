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

__kernel void compute_states(__global float4      *out_stateL,         //  0
			     __global float4      *out_stateR,         //  1
			     __global int2        *ijlist,             //  2
			     __global float4      *drij,               //  3
			     __global float4      *data,               //  4
			     __global float4      *grad_x,             //  5
			     __global float4      *grad_y,             //  6
			     __global float4      *grad_z,             //  7
			     const    int          n_states,           //  8
			     const    int          do_first_order) {   //  9
  
  // Compute idx of the element
  const int gidx = get_global_id(0) + get_global_id(1) * get_global_size(0);
  const int idx  = min(gidx, n_states - 1);

  // Get i- & j- bodies idx
  const int2 ij = ijlist[idx];
  const int  i = ij.x;
  const int  j = ij.y;

  // Compute distance between two particles
  
  const float4 dr = drij[idx];
  const float4 ds = {0.5f*dr.x, 0.5f*dr.y, 0.5f*dr.z, 0.0f};

  // Read state of both i- & j-particle

  const float4 idata = data[i];
  const float4 jdata = data[j];

  if (do_first_order == 1) {
    if (gidx < n_states) {
      out_stateL[idx] = idata;
      out_stateR[idx] = jdata;
    }
    return;
  }
  // Reconstruct states to the interface
  
  float4 idata_ij = idata;

  const float4 igrad_x = grad_x[i];
  idata_ij.x += ds.x * igrad_x.x;
  idata_ij.y += ds.x * igrad_x.y;
  idata_ij.z += ds.x * igrad_x.z;
  idata_ij.w += ds.x * igrad_x.w;

  const float4 igrad_y = grad_y[i];
  idata_ij.x += ds.y * igrad_y.x;
  idata_ij.y += ds.y * igrad_y.y;
  idata_ij.z += ds.y * igrad_y.z;
  idata_ij.w += ds.y * igrad_y.w;

  const float4 igrad_z = grad_z[i];
  idata_ij.x += ds.z * igrad_z.x;
  idata_ij.y += ds.z * igrad_z.y;
  idata_ij.z += ds.z * igrad_z.z;
  idata_ij.w += ds.z * igrad_z.w;

  float4 jdata_ij = jdata;

  const float4 jgrad_x = grad_x[j];
  jdata_ij.x -= ds.x * jgrad_x.x;
  jdata_ij.y -= ds.x * jgrad_x.y;
  jdata_ij.z -= ds.x * jgrad_x.z;
  jdata_ij.w -= ds.x * jgrad_x.w;

  const float4 jgrad_y = grad_y[j];
  jdata_ij.x -= ds.y * jgrad_y.x;
  jdata_ij.y -= ds.y * jgrad_y.y;
  jdata_ij.z -= ds.y * jgrad_y.z;
  jdata_ij.w -= ds.y * jgrad_y.w;

  const float4 jgrad_z = grad_z[j];
  jdata_ij.x -= ds.z * jgrad_z.x;
  jdata_ij.y -= ds.z * jgrad_z.y;
  jdata_ij.z -= ds.z * jgrad_z.z;
  jdata_ij.w -= ds.z * jgrad_z.w;

  float2 v;
  v = monotonicity(idata_ij.x, jdata_ij.x, idata.x, jdata.x); idata_ij.x = v.x; jdata_ij.x = v.y;
  v = monotonicity(idata_ij.y, jdata_ij.y, idata.y, jdata.y); idata_ij.y = v.x; jdata_ij.y = v.y;
  v = monotonicity(idata_ij.z, jdata_ij.z, idata.z, jdata.z); idata_ij.z = v.x; jdata_ij.z = v.y;
  v = monotonicity(idata_ij.w, jdata_ij.w, idata.w, jdata.w); idata_ij.w = v.x; jdata_ij.w = v.y;

//   idata_ij = idata;
//   jdata_ij = jdata;
  if (gidx < n_states) {
    out_stateL[idx] = idata_ij;
    out_stateR[idx] = jdata_ij;
  }
}

#endif
