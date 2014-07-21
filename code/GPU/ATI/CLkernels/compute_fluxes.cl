#ifndef _COMPUTE_FLUXES_CL_
#define _COMPUTE_FLUXES_CL_

// #pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define sqr(x) ((x)*(x))
#define SQRT(x) ((x != 0.0f) ? 1.0f/rsqrt(x) : 0.0f)

struct ptcl_mhd_cl {
  float dens, etot, momx, momy, momz, by, bz;
};


// __inline float2 monotonicity(const float fLin, const float fRin, const float fi, const float fj) {
//   float fL = fLin;
//   float fR = fRin;
//   const float Fmin = fmin(fi, fj);
//   const float Fmax = fmax(fi, fj);
//   if ((fL < Fmin) || (fL > Fmax) || (fR < Fmin) || (fR > Fmax)) {
//     fL = fR = fi;
//     fL = fi;
//     fR = fj;
//   }
//   return (float2){fL, fR};
// }

__inline struct ptcl_mhd_cl riemann_solver(const float w, 
					   const float b,
					   const float gamma_gas,
					   const float dens_L, 
					   const float pres_L, 
					   const float velx_L,
					   const float vely_L,
					   const float velz_L,
					   const float By_L,
					   const float Bz_L,
					   const float dens_R, 
					   const float pres_R, 
					   const float velx_R,
					   const float vely_R,
					   const float velz_R,
					   const float By_R,
					   const float Bz_R) {
#if 1
  const float Bx     = b;
  const float signBx = (fabs(Bx) > 0.0f) ? Bx/fabs(Bx) : 0.0f;
  
  const float momx_L = dens_L*velx_L;
  const float momy_L = dens_L*vely_L;
  const float momz_L = dens_L*velz_L;
  
  const float momx_R = dens_R*velx_R;
  const float momy_R = dens_R*vely_R;
  const float momz_R = dens_R*velz_R;
  
  const float B2_L   = sqr(Bx)     + sqr(By_L)   + sqr(Bz_L);
  const float v2_L   = sqr(velx_L) + sqr(vely_L) + sqr(velz_L);
  const float etot_L = (gamma_gas == 1.0f) ? 
    pres_L                    + 0.5f*dens_L*v2_L + 0.5f*B2_L :
    pres_L/(gamma_gas - 1.0f) + 0.5f*dens_L*v2_L + 0.5f*B2_L;

  const float B2_R   = sqr(Bx)     + sqr(By_R)   + sqr(Bz_R);
  const float v2_R   = sqr(velx_R) + sqr(vely_R) + sqr(velz_R);
  const float etot_R = (gamma_gas == 1.0f) ? 
    pres_R                    + 0.5f*dens_R*v2_R + 0.5f*B2_R : 
    pres_R/(gamma_gas - 1.0f) + 0.5f*dens_R*v2_R + 0.5f*B2_R;
  
  const float gpl  = gamma_gas * pres_L;
  const float gpr  = gamma_gas * pres_R;
  const float gpbl = gpl + B2_L;
  const float gpbr = gpr + B2_R;

  //////////

  const float cfl = SQRT((gpbl + SQRT( sqr(gpbl) - 4.0f*gpl*sqr(Bx) ))/(2.0f*dens_L));
  const float cfr = SQRT((gpbr + SQRT( sqr(gpbr) - 4.0f*gpr*sqr(Bx) ))/(2.0f*dens_R));
  const float cfmax = fmax(cfl,cfr);

  const float S_L = fmin(velx_L, velx_R) - cfmax;
  const float S_R = fmax(velx_L, velx_R) + cfmax;

  //////////

  
  const float pT_L = pres_L + 0.5f * B2_L;
  const float pT_R = pres_R + 0.5f * B2_R;
  
  const float iSM = 1.0f/((S_R - velx_R)*dens_R - (S_L - velx_L)*dens_L);
  const float S_M  = ((S_R - velx_R)*momx_R - (S_L - velx_L)*momx_L - pT_R + pT_L)*iSM;
  
  const float ipTs = 1.0f/((S_R - velx_R)*dens_R - (S_L - velx_L)*dens_L);
  const float pT_s = ipTs * ((S_R - velx_R)*dens_R*pT_L - (S_L - velx_L)*dens_L*pT_R +
			    dens_L*dens_R*(S_R - velx_R)*(S_L - velx_L)*(velx_R - velx_L));
  
  const float velx_L_s  = S_M;
  const float velx_L_ss = S_M;
  const float velx_R_s  = S_M;
  const float velx_R_ss = S_M;
  const float B2x       = Bx*Bx;
  
  const float dens_L_s = dens_L * (S_L - velx_L)/(S_L - S_M);
  const float divL     = dens_L*(S_L - velx_L)*(S_L - S_M) - B2x;
  const float idivL    = (divL != 0.0f) ? 1.0f/divL : 0.0f;
  const float vely_L_s = vely_L - Bx*By_L*(S_M - velx_L) * idivL;
  const float velz_L_s = velz_L - Bx*Bz_L*(S_M - velx_L) * idivL;
  const float   By_L_s = By_L * (dens_L*sqr(S_L - velx_L) - B2x) * idivL;
  const float   Bz_L_s = Bz_L * (dens_L*sqr(S_L - velx_L) - B2x) * idivL;

  const float dens_R_s = dens_R * (S_R - velx_R)/(S_R - S_M);
  const float divR     = dens_R*(S_R - velx_R)*(S_R - S_M) - B2x;
  const float idivR    = (divR != 0.0f) ? 1.0f/divR : 0.0f;
  const float vely_R_s = vely_R - Bx*By_R*(S_M - velx_R) * idivR;
  const float velz_R_s = velz_R - Bx*Bz_R*(S_M - velx_R) * idivR;
  const float   By_R_s = By_R * (dens_R*sqr(S_R - velx_R) - B2x) * idivR;
  const float   Bz_R_s = Bz_R * (dens_R*sqr(S_R - velx_R) - B2x) * idivR;

  const float   vB_L   = velx_L  *Bx + vely_L  *By_L   + velz_L  *Bz_L;
  const float   vB_L_s = velx_L_s*Bx + vely_L_s*By_L_s + velz_L_s*Bz_L_s;
  const float etot_L_s = ((S_L - velx_L)*etot_L - pT_L*velx_L + pT_s*S_M + Bx*(vB_L - vB_L_s))/(S_L - S_M);

  const float   vB_R   = velx_R  *Bx + vely_R  *By_R   + velz_R  *Bz_R;
  const float   vB_R_s = velx_R_s*Bx + vely_R_s*By_R_s + velz_R_s*Bz_R_s;
  const float etot_R_s = ((S_R - velx_R)*etot_R - pT_R*velx_R + pT_s*S_M + Bx*(vB_R - vB_R_s))/(S_R - S_M);

  const float dens_L_ss = dens_L_s;
  const float dens_R_ss = dens_R_s;
  const float    S_L_s  = S_M - fabs(Bx)/SQRT(dens_L_s);
  const float    S_R_s  = S_M + fabs(Bx)/SQRT(dens_R_s);

  const float idSQRT  = 1.0f/(SQRT(dens_L_s) + SQRT(dens_R_s));
  const float  vely_ss = idSQRT*(SQRT(dens_L_s)*vely_L_s + SQRT(dens_R_s)*vely_R_s + (By_R_s - By_L_s)*signBx);
  const float  velz_ss = idSQRT*(SQRT(dens_L_s)*velz_L_s + SQRT(dens_R_s)*velz_R_s + (Bz_R_s - Bz_L_s)*signBx);

  const float By_ss = idSQRT*(SQRT(dens_L_s)*By_R_s + SQRT(dens_R_s)*By_L_s + SQRT(dens_L_s*dens_R_s)*(vely_R_s - vely_L_s)*signBx);
  const float Bz_ss = idSQRT*(SQRT(dens_L_s)*Bz_R_s + SQRT(dens_R_s)*Bz_L_s + SQRT(dens_L_s*dens_R_s)*(velz_R_s - velz_L_s)*signBx);
  
  const float vely_L_ss = vely_ss;
  const float velz_L_ss = velz_ss;
  const float   By_L_ss = By_ss;
  const float   Bz_L_ss = Bz_ss;

  const float vely_R_ss = vely_ss;
  const float velz_R_ss = velz_ss;
  const float   By_R_ss = By_ss;
  const float   Bz_R_ss = Bz_ss;

  const float vB_L_ss   = velx_L_ss*Bx + vely_L_ss*By_L_ss + velz_L_ss*Bz_L_ss;
  const float etot_L_ss = etot_L_s - SQRT(dens_L_s)*(vB_L_s - vB_L_ss)*signBx;

  const float vB_R_ss   = velx_R_ss*Bx + vely_R_ss*By_R_ss + velz_R_ss*Bz_R_ss;
  const float etot_R_ss = etot_R_s + SQRT(dens_R_s)*(vB_R_s - vB_R_ss)*signBx;

  const float Fdens_L = dens_L*velx_L;
  const float Fmomx_L = momx_L*velx_L + pT_L - B2x;
  const float Fmomy_L = momy_L*velx_L        - Bx*By_L;
  const float Fmomz_L = momz_L*velx_L        - Bx*Bz_L;
  const float Fetot_L = etot_L*velx_L + pT_L*velx_L - Bx*vB_L; 

  const float Fdens_R = dens_R*velx_R;
  const float Fmomx_R = momx_R*velx_R + pT_R - B2x;
  const float Fmomy_R = momy_R*velx_R        - Bx*By_R;
  const float Fmomz_R = momz_R*velx_R        - Bx*Bz_R;
  const float Fetot_R = etot_R*velx_R + pT_R*velx_R - Bx*vB_R;

  const float momx_L_s  = dens_L_s *velx_L_s;
  const float momy_L_s  = dens_L_s *vely_L_s;
  const float momz_L_s  = dens_L_s *velz_L_s;
  
  const float momx_L_ss = dens_L_ss*velx_L_ss;
  const float momy_L_ss = dens_L_ss*vely_L_ss;
  const float momz_L_ss = dens_L_ss*velz_L_ss;

  const float momx_R_s  = dens_R_s *velx_R_s;
  const float momy_R_s  = dens_R_s *vely_R_s;
  const float momz_R_s  = dens_R_s *velz_R_s;
  
  const float momx_R_ss = dens_R_ss*velx_R_ss;
  const float momy_R_ss = dens_R_ss*vely_R_ss;
  const float momz_R_ss = dens_R_ss*velz_R_ss;

  const float Fby_L  = By_L*velx_L - Bx * vely_L;
  const float Fbz_L  = Bz_L*velx_L - Bx * velz_L;
  
  const float Fby_R  = By_R*velx_R - Bx * vely_R;
  const float Fbz_R  = Bz_R*velx_R - Bx * velz_R;

  struct ptcl_mhd_cl flux;

  if (S_L > w) {
    flux.dens = Fdens_L - w * dens_L;
    flux.momx = Fmomx_L - w * momx_L;
    flux.momy = Fmomy_L - w * momy_L;
    flux.momz = Fmomz_L - w * momz_L;
    flux.etot = Fetot_L - w * etot_L;
    flux.by = Fby_L - w * By_L;
    flux.bz = Fbz_L - w * Bz_L;
  } else if (S_L <= w && w <= S_L_s) {
    flux.dens = Fdens_L + (S_L - w)*dens_L_s - S_L*dens_L;
    flux.momx = Fmomx_L + (S_L - w)*momx_L_s - S_L*momx_L;
    flux.momy = Fmomy_L + (S_L - w)*momy_L_s - S_L*momy_L;
    flux.momz = Fmomz_L + (S_L - w)*momz_L_s - S_L*momz_L;
    flux.etot = Fetot_L + (S_L - w)*etot_L_s - S_L*etot_L;
    flux.by = Fby_L + (S_L - w)*By_L_s - S_L*By_L;
    flux.bz = Fbz_L + (S_L - w)*Bz_L_s - S_L*Bz_L;
  } else if (S_L_s <= w && w <= S_M) {
    flux.dens = Fdens_L + (S_L_s - w)*dens_L_ss - (S_L_s - S_L)*dens_L_s - S_L*dens_L;
    flux.momx = Fmomx_L + (S_L_s - w)*momx_L_ss - (S_L_s - S_L)*momx_L_s - S_L*momx_L;
    flux.momy = Fmomy_L + (S_L_s - w)*momy_L_ss - (S_L_s - S_L)*momy_L_s - S_L*momy_L;
    flux.momz = Fmomz_L + (S_L_s - w)*momz_L_ss - (S_L_s - S_L)*momz_L_s - S_L*momz_L;
    flux.etot = Fetot_L + (S_L_s - w)*etot_L_ss - (S_L_s - S_L)*etot_L_s - S_L*etot_L;
    flux.by = Fby_L + (S_L_s - w)*By_L_ss - (S_L_s - S_L)*By_L_s - S_L*By_L;
    flux.bz = Fbz_L + (S_L_s - w)*Bz_L_ss - (S_L_s - S_L)*Bz_L_s - S_L*Bz_L;
  } else if (S_M <= w && w <= S_R_s) {
    flux.dens = Fdens_R + (S_R_s - w)*dens_R_ss - (S_R_s - S_R)*dens_R_s - S_R*dens_R;
    flux.momx = Fmomx_R + (S_R_s - w)*momx_R_ss - (S_R_s - S_R)*momx_R_s - S_R*momx_R;
    flux.momy = Fmomy_R + (S_R_s - w)*momy_R_ss - (S_R_s - S_R)*momy_R_s - S_R*momy_R;
    flux.momz = Fmomz_R + (S_R_s - w)*momz_R_ss - (S_R_s - S_R)*momz_R_s - S_R*momz_R;
    flux.etot = Fetot_R + (S_R_s - w)*etot_R_ss - (S_R_s - S_R)*etot_R_s - S_R*etot_R;
    flux.by = Fby_R + (S_R_s - w)*By_R_ss - (S_R_s - S_R)*By_R_s - S_R*By_R;
    flux.bz = Fbz_R + (S_R_s - w)*Bz_R_ss - (S_R_s - S_R)*Bz_R_s - S_R*Bz_R;
  } else if (S_R_s <= w && w <= S_R) {
    flux.dens = Fdens_R + (S_R - w)*dens_R_s - S_R*dens_R;
    flux.momx = Fmomx_R + (S_R - w)*momx_R_s - S_R*momx_R;
    flux.momy = Fmomy_R + (S_R - w)*momy_R_s - S_R*momy_R;
    flux.momz = Fmomz_R + (S_R - w)*momz_R_s - S_R*momz_R;
    flux.etot = Fetot_R + (S_R - w)*etot_R_s - S_R*etot_R;
    flux.by = Fby_R + (S_R - w)*By_R_s - S_R*By_R;
    flux.bz = Fbz_R + (S_R - w)*Bz_R_s - S_R*Bz_R;
  } else {
    flux.dens = Fdens_R - w * dens_R;
    flux.momx = Fmomx_R - w * momx_R;
    flux.momy = Fmomy_R - w * momy_R;
    flux.momz = Fmomz_R - w * momz_R;
    flux.etot = Fetot_R - w * etot_R;
    flux.by = Fby_R - w * By_R;
    flux.bz = Fbz_R - w * Bz_R;
  }
#else
  struct ptcl_mhd_cl flux = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f ,0.0f,0.0f};
#endif
  return flux;
}

__kernel void compute_fluxes(__global float4      *out_fluxes1,       //  0
			     __global float4      *out_fluxes2,       //  1
			     __global float4      *out_Fscalar,       //  2
			     __global float4      *out_divB,          //  3
			     __global int2        *ijlist,            //  4
			     __global float4      *in_drij,           //  5
			     __global float4      *in_dwij,           //  6
			     __global float4      *mhd1,              //  7
			     __global float4      *mhd2,              //  8
			     __global float4      *mhd3,              //  9
			     __global float4      *mhd1_grad_x,       // 10
			     __global float4      *mhd1_grad_y,       // 11
			     __global float4      *mhd1_grad_z,       // 12
			     __global float4      *mhd2_grad_x,       // 13
			     __global float4      *mhd2_grad_y,       // 14
			     __global float4      *mhd2_grad_z,       // 15
			     __global float4      *mhd3_grad_x,       // 16
			     __global float4      *mhd3_grad_y,       // 17
			     __global float4      *mhd3_grad_z,       // 18
			     const int            n_states,           // 19
			     const float          gamma_gas,          // 20
			     const float          ch_global,          // 21
			     const int            reconstruct) {   // 22

  // Compute idx of the element
  const int gidx = get_global_id(0) + get_global_id(1) * get_global_size(0);
  const int idx = min(gidx, n_states - 1);

  //////////////

  //////////////

  // Get i- & j- bodies idx
  const int2 ij = ijlist[idx];
  const int  i = ij.x;
  const int  j = ij.y;

  const float4 imhd1 = mhd1[i];
  const float4 imhd2 = mhd2[i];

  const float4 jmhd1 = mhd1[j];
  const float4 jmhd2 = mhd2[j];

  const float velx_L = imhd1.x;
  const float vely_L = imhd1.y;
  const float velz_L = imhd1.z;
  const float dens_L = imhd1.w;
  const float By_L   = imhd2.y;
  const float Bz_L   = imhd2.z;
  const float pres_L = imhd2.w;

  const float velx_R = jmhd1.x;
  const float vely_R = jmhd1.y;
  const float velz_R = jmhd1.z;
  const float dens_R = jmhd1.w;
  const float By_R   = jmhd2.y;
  const float Bz_R   = jmhd2.z;
  const float pres_R = jmhd2.w;

  const float Bx = imhd1.x;
  const float Wx = jmhd1.x;

  const struct ptcl_mhd_cl flux = riemann_solver(Wx, Bx, gamma_gas,
						 dens_L, pres_L, velx_L, vely_L, velz_L, By_L, Bz_L,
						 dens_R, pres_R, velx_R, vely_R, velz_R, By_R, Bz_R);
  
  const float flux_bx  = -Wx*Bx;
  
 if (gidx < n_states) {
    out_fluxes1[idx] = (float4){flux.momx, flux.momy, flux.momz , flux.dens};
    out_fluxes2[idx] = (float4){flux_bx ,  flux.by , flux.bz, flux.etot};
  }
  
  
}


#endif  
