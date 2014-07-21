#include "kernels.clh"

#ifndef _COMPUTE_FLUXES_CL_
#define _COMPUTE_FLUXES_CL_

#if 1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define REAL double
#else
#define REAL float
#endif

#define SQRT sqrt

__inline float compute_pressure(const float dens, const float ethm, const float gamma_gas) {
  return (gamma_gas > 1.0f) ? ethm*(gamma_gas - 1.0f) : ethm;
}

struct ptcl_mhd_cl {
  float dens, etot, momx, momy, momz, by, bz;
};

__inline REAL sqr(const REAL x) {return x*x;}

#define HLLD

#ifdef HLLD

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
  const float Bx     = b;
  const float signBx = (fabs(Bx) > 0.0f) ? Bx/fabs(Bx) : 0.0f;
  
  const REAL momx_L = dens_L*velx_L;
  const REAL momy_L = dens_L*vely_L;
  const REAL momz_L = dens_L*velz_L;
  
  const REAL momx_R = dens_R*velx_R;
  const REAL momy_R = dens_R*vely_R;
  const REAL momz_R = dens_R*velz_R;
  
  const REAL B2_L   = sqr(Bx)     + sqr(By_L)   + sqr(Bz_L);
  const REAL v2_L   = sqr(velx_L) + sqr(vely_L) + sqr(velz_L);
  const REAL etot_L = (gamma_gas == 1.0f) ? 
    pres_L                    + 0.5f*dens_L*v2_L + 0.5f*B2_L :
    pres_L/(gamma_gas - 1.0f) + 0.5f*dens_L*v2_L + 0.5f*B2_L;

  const REAL B2_R   = sqr(Bx)     + sqr(By_R)   + sqr(Bz_R);
  const REAL v2_R   = sqr(velx_R) + sqr(vely_R) + sqr(velz_R);
  const REAL etot_R = (gamma_gas == 1.0f) ? 
    pres_R                    + 0.5f*dens_R*v2_R + 0.5f*B2_R : 
    pres_R/(gamma_gas - 1.0f) + 0.5f*dens_R*v2_R + 0.5f*B2_R;
  
  const REAL gpl  = gamma_gas * pres_L;
  const REAL gpr  = gamma_gas * pres_R;
  const REAL gpbl = gpl + B2_L;
  const REAL gpbr = gpr + B2_R;

  //////////

#if 1
  const REAL cfl = SQRT((gpbl + SQRT( sqr(gpbl) - 4.0f*gpl*sqr(Bx) ))/(2.0f*dens_L));
  const REAL cfr = SQRT((gpbr + SQRT( sqr(gpbr) - 4.0f*gpr*sqr(Bx) ))/(2.0f*dens_R));
  const REAL cfmax = 1.2f*fmax(cfl,cfr);

  const REAL S_L = fmin(velx_L, velx_R) - cfmax;
  const REAL S_R = fmax(velx_L, velx_R) + cfmax;
#else
  const REAL dsl = sqrt(dens_L);
  const REAL dsr = sqrt(dens_R);
  const REAL ids = 1.0/(dsl + dsr);
  const REAL droe = dsl * dsr;
  
  const REAL uxroe = (dsl*velx_L + dsr*velx_R)*ids;
  const REAL uyroe = (dsl*vely_L + dsr*vely_R)*ids;
  const REAL uzroe = (dsl*velz_L + dsr*velz_R)*ids;
    
  const REAL byroe = (dsl*By_L + dsr*By_R)*ids;
  const REAL bzroe = (dsl*Bz_L + dsr*Bz_R)*ids;

  const REAL x = 0.5 * (sqr(By_L - By_R) + sqr(Bz_L - Bz_R))/sqr(dsl + dsl);
  const REAL y = 0.5 * (dens_L + dens_R)/droe;
    
  const REAL pbl = 0.5*B2_L;
  const REAL pbr = 0.5*B2_R;
    
  const REAL hl  = (etot_L + pres_L + pbl)/dens_L;
  const REAL hr  = (etot_R + pres_R + pbr)/dens_R;
  const REAL hroe  = (dsl*hl + dsr*hr)*ids;

   REAL di  = 1.0/droe;
   REAL vsq = sqr(uxroe) + sqr(uyroe) + sqr(uzroe);
   REAL btsq = sqr(byroe) + sqr(bzroe);
   REAL bt_startsq = ((gamma_gas - 1) - (gamma_gas - 2)*y)*btsq;
   REAL vaxsq = Bx*Bx*di;
   REAL hp = hroe - (vaxsq + btsq*di);
   REAL twid_asq = ((gamma_gas - 1)*(hp - 0.5*vsq) - (gamma_gas - 2)*x);

//   twid_asq = (hp - 0.5*vsq);
//   twid_asq = np.where(twid_asq > 1.0e-10, twid_asq, 1.0e-10)
   
   REAL ct2  = bt_startsq*di;
   REAL tsum = vaxsq + ct2 + twid_asq;
   REAL tdif = vaxsq + ct2 - twid_asq;
   REAL cf2_cs2 = sqrt(sqr(tdif) + 4.0*twid_asq*ct2);
   REAL cfsq = 0.5*(tsum + cf2_cs2);
   REAL cf = sqrt(cfsq);

  const float fsig = 1.2f;
  
  REAL S_L = uxroe - fsig*cf;
  REAL S_R = uxroe + fsig*cf;
  
   REAL asq = sqrt(gamma_gas * pres_L/dens_L);
  vaxsq = sqr(Bx)/dens_L;
  ct2   = (sqr(By_L) + sqr(Bz_L))/dens_L;
   REAL qsq = vaxsq + ct2 + asq;
   REAL tmp = vaxsq + ct2 - asq;
  cfsq = 0.5*(qsq + sqrt(sqr(tmp) + 4.0*asq*ct2));
  REAL cfl = fsig*sqrt(cfsq);

  asq = sqrt(gamma_gas * pres_R/dens_R);
  vaxsq = sqr(Bx)/dens_R;
  ct2   = (sqr(By_R) + sqr(Bz_R))/dens_R;
  qsq = vaxsq + ct2 + asq;
  tmp = vaxsq + ct2 - asq;
  cfsq = 0.5*(qsq + sqrt(sqr(tmp) + 4.0*asq*ct2));
  const REAL cfr = fsig*sqrt(cfsq);

  if (velx_R + cfr > S_R) S_R = velx_R + cfr;
  if (velx_L - cfl < S_L) S_L = velx_L - cfl;
#endif

  //////////

  
  const REAL pT_L = pres_L + 0.5f * B2_L;
  const REAL pT_R = pres_R + 0.5f * B2_R;
  
  const REAL iSM = 1.0f/((S_R - velx_R)*dens_R - (S_L - velx_L)*dens_L);
  const REAL S_M  = ((S_R - velx_R)*momx_R - (S_L - velx_L)*momx_L - pT_R + pT_L)*iSM;
  
  const REAL ipTs = 1.0f/((S_R - velx_R)*dens_R - (S_L - velx_L)*dens_L);
  const REAL pT_s = ipTs * ((S_R - velx_R)*dens_R*pT_L - (S_L - velx_L)*dens_L*pT_R +
			    dens_L*dens_R*(S_R - velx_R)*(S_L - velx_L)*(velx_R - velx_L));
  
  const REAL velx_L_s  = S_M;
  const REAL velx_L_ss = S_M;
  const REAL velx_R_s  = S_M;
  const REAL velx_R_ss = S_M;
  const REAL B2x       = Bx*Bx;
  
  const REAL dens_L_s = dens_L * (S_L - velx_L)/(S_L - S_M);
  const REAL dens_R_s = dens_R * (S_R - velx_R)/(S_R - S_M);
  const REAL divL     = dens_L*(S_L - velx_L)*(S_L - S_M) - B2x;
  const REAL divR     = dens_R*(S_R - velx_R)*(S_R - S_M) - B2x;
#if 1
  const REAL idivL    = (divL != 0.0f) ? 1.0f/divL : 0.0f;
  const REAL idivR    = (divR != 0.0f) ? 1.0f/divR : 0.0f;
#else
  const REAL idivL = 1.0f/divL;
  const REAL idivR = 1.0f/divR;
  const REAL idivL = (fabs(divL) > 1.0e-8f) ? 1.0f/divL : 0.0f;
  const REAL idivR = (fabs(divR) > 1.0e-8f) ? 1.0f/divR : 0.0f;
#endif
  const REAL vely_L_s = vely_L - Bx*By_L*(S_M - velx_L) * idivL;
  const REAL velz_L_s = velz_L - Bx*Bz_L*(S_M - velx_L) * idivL;
  const REAL   By_L_s = By_L * (dens_L*sqr(S_L - velx_L) - B2x) * idivL;
  const REAL   Bz_L_s = Bz_L * (dens_L*sqr(S_L - velx_L) - B2x) * idivL;

  const REAL vely_R_s = vely_R - Bx*By_R*(S_M - velx_R) * idivR;
  const REAL velz_R_s = velz_R - Bx*Bz_R*(S_M - velx_R) * idivR;
  const REAL   By_R_s = By_R * (dens_R*sqr(S_R - velx_R) - B2x) * idivR;
  const REAL   Bz_R_s = Bz_R * (dens_R*sqr(S_R - velx_R) - B2x) * idivR;

  const REAL   vB_L   = velx_L  *Bx + vely_L  *By_L   + velz_L  *Bz_L;
  const REAL   vB_L_s = velx_L_s*Bx + vely_L_s*By_L_s + velz_L_s*Bz_L_s;
  const REAL etot_L_s = ((S_L - velx_L)*etot_L - pT_L*velx_L + pT_s*S_M + Bx*(vB_L - vB_L_s))/(S_L - S_M);

  const REAL   vB_R   = velx_R  *Bx + vely_R  *By_R   + velz_R  *Bz_R;
  const REAL   vB_R_s = velx_R_s*Bx + vely_R_s*By_R_s + velz_R_s*Bz_R_s;
  const REAL etot_R_s = ((S_R - velx_R)*etot_R - pT_R*velx_R + pT_s*S_M + Bx*(vB_R - vB_R_s))/(S_R - S_M);

  const REAL dens_L_ss = dens_L_s;
  const REAL dens_R_ss = dens_R_s;
  const REAL    S_L_s  = S_M - fabs(Bx)/SQRT(dens_L_s);
  const REAL    S_R_s  = S_M + fabs(Bx)/SQRT(dens_R_s);

  const REAL idSQRT  = 1.0f/(SQRT(dens_L_s) + SQRT(dens_R_s));
  const REAL  vely_ss = idSQRT*(SQRT(dens_L_s)*vely_L_s + SQRT(dens_R_s)*vely_R_s + (By_R_s - By_L_s)*signBx);
  const REAL  velz_ss = idSQRT*(SQRT(dens_L_s)*velz_L_s + SQRT(dens_R_s)*velz_R_s + (Bz_R_s - Bz_L_s)*signBx);

  const REAL By_ss = idSQRT*(SQRT(dens_L_s)*By_R_s + SQRT(dens_R_s)*By_L_s + SQRT(dens_L_s*dens_R_s)*(vely_R_s - vely_L_s)*signBx);
  const REAL Bz_ss = idSQRT*(SQRT(dens_L_s)*Bz_R_s + SQRT(dens_R_s)*Bz_L_s + SQRT(dens_L_s*dens_R_s)*(velz_R_s - velz_L_s)*signBx);
  
  const REAL vely_L_ss = vely_ss;
  const REAL velz_L_ss = velz_ss;
  const REAL   By_L_ss = By_ss;
  const REAL   Bz_L_ss = Bz_ss;

  const REAL vely_R_ss = vely_ss;
  const REAL velz_R_ss = velz_ss;
  const REAL   By_R_ss = By_ss;
  const REAL   Bz_R_ss = Bz_ss;

  const REAL vB_L_ss   = velx_L_ss*Bx + vely_L_ss*By_L_ss + velz_L_ss*Bz_L_ss;
  const REAL etot_L_ss = etot_L_s - SQRT(dens_L_s)*(vB_L_s - vB_L_ss)*signBx;

  const REAL vB_R_ss   = velx_R_ss*Bx + vely_R_ss*By_R_ss + velz_R_ss*Bz_R_ss;
  const REAL etot_R_ss = etot_R_s + SQRT(dens_R_s)*(vB_R_s - vB_R_ss)*signBx;

  const REAL Fdens_L = dens_L*velx_L;
  const REAL Fmomx_L = momx_L*velx_L + pT_L - B2x;
  const REAL Fmomy_L = momy_L*velx_L        - Bx*By_L;
  const REAL Fmomz_L = momz_L*velx_L        - Bx*Bz_L;
  const REAL Fetot_L = etot_L*velx_L + pT_L*velx_L - Bx*vB_L; 

  const REAL Fdens_R = dens_R*velx_R;
  const REAL Fmomx_R = momx_R*velx_R + pT_R - B2x;
  const REAL Fmomy_R = momy_R*velx_R        - Bx*By_R;
  const REAL Fmomz_R = momz_R*velx_R        - Bx*Bz_R;
  const REAL Fetot_R = etot_R*velx_R + pT_R*velx_R - Bx*vB_R;

  const REAL momx_L_s  = dens_L_s *velx_L_s;
  const REAL momy_L_s  = dens_L_s *vely_L_s;
  const REAL momz_L_s  = dens_L_s *velz_L_s;
  
  const REAL momx_L_ss = dens_L_ss*velx_L_ss;
  const REAL momy_L_ss = dens_L_ss*vely_L_ss;
  const REAL momz_L_ss = dens_L_ss*velz_L_ss;

  const REAL momx_R_s  = dens_R_s *velx_R_s;
  const REAL momy_R_s  = dens_R_s *vely_R_s;
  const REAL momz_R_s  = dens_R_s *velz_R_s;
  
  const REAL momx_R_ss = dens_R_ss*velx_R_ss;
  const REAL momy_R_ss = dens_R_ss*vely_R_ss;
  const REAL momz_R_ss = dens_R_ss*velz_R_ss;

  const REAL Fby_L  = By_L*velx_L - Bx * vely_L;
  const REAL Fbz_L  = Bz_L*velx_L - Bx * velz_L;
  
  const REAL Fby_R  = By_R*velx_R - Bx * vely_R;
  const REAL Fbz_R  = Bz_R*velx_R - Bx * velz_R;

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
  return flux;
}

#else // HLLE

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
  const float Bx     = b;
  const float signBx = (fabs(Bx) > 0.0f) ? Bx/fabs(Bx) : 0.0f;
  
  const REAL momx_L = dens_L*velx_L;
  const REAL momy_L = dens_L*vely_L;
  const REAL momz_L = dens_L*velz_L;
  
  const REAL momx_R = dens_R*velx_R;
  const REAL momy_R = dens_R*vely_R;
  const REAL momz_R = dens_R*velz_R;
  
  const REAL B2_L   = sqr(Bx)     + sqr(By_L)   + sqr(Bz_L);
  const REAL v2_L   = sqr(velx_L) + sqr(vely_L) + sqr(velz_L);
  const REAL etot_L = (gamma_gas == 1.0f) ? 
    pres_L                    + 0.5f*dens_L*v2_L + 0.5f*B2_L :
    pres_L/(gamma_gas - 1.0f) + 0.5f*dens_L*v2_L + 0.5f*B2_L;

  const REAL B2_R   = sqr(Bx)     + sqr(By_R)   + sqr(Bz_R);
  const REAL v2_R   = sqr(velx_R) + sqr(vely_R) + sqr(velz_R);
  const REAL etot_R = (gamma_gas == 1.0f) ? 
    pres_R                    + 0.5f*dens_R*v2_R + 0.5f*B2_R : 
    pres_R/(gamma_gas - 1.0f) + 0.5f*dens_R*v2_R + 0.5f*B2_R;

  const REAL pT_L = pres_L + 0.5f * B2_L;
  const REAL pT_R = pres_R + 0.5f * B2_R;


  REAL gpl  = gamma_gas * pres_L;
  REAL gpr  = gamma_gas * pres_R;
  REAL gpbl = gpl + B2_L;
  REAL gpbr = gpr + B2_R;

#if 1
  const REAL cfl = SQRT((gpbl + SQRT( sqr(gpbl) - 4.0f*gpl*sqr(Bx) ))/(2.0f*dens_L));
  const REAL cfr = SQRT((gpbr + SQRT( sqr(gpbr) - 4.0f*gpr*sqr(Bx) ))/(2.0f*dens_R));
  const REAL cfmax = 1.2f*fmax(cfl,cfr);

  const REAL S_L = fmin(velx_L, velx_R) - cfmax;
  const REAL S_R = fmax(velx_L, velx_R) + cfmax;
#else
  
  const REAL sqrtdl = sqrt(dens_L);
  const REAL sqrtdr = sqrt(dens_R);
  const REAL isdlpdr = 1.0f/(sqrtdl + sqrtdr);

  const REAL droe  = sqrtdl*sqrtdr;
  const REAL v1roe = (sqrtdl*velx_L + sqrtdr*velx_R)*isdlpdr;
  const REAL v2roe = (sqrtdl*vely_L + sqrtdr*vely_R)*isdlpdr;
  const REAL v3roe = (sqrtdl*velz_L + sqrtdr*velz_R)*isdlpdr;

  const REAL b2roe = (sqrtdr*By_L + sqrtdl*By_R)*isdlpdr;
  const REAL b3roe = (sqrtdr*Bz_L + sqrtdl*Bz_R)*isdlpdr;
  const REAL x = 0.5f*(sqr(By_L - By_R) + sqr(Bz_L - Bz_R))/(sqr(sqrtdl + sqrtdr));
  const REAL y = 0.5f*(dens_L + dens_R)/droe;
  const REAL pbl = 0.5f*(sqr(Bx) + sqr(By_L) + sqr(Bz_L));
  const REAL pbr = 0.5f*(sqr(Bx) + sqr(By_R) + sqr(Bz_R));
  
  const REAL hroe  = ((etot_L + pres_L + pbl)/sqrtdl + (etot_R + pres_R + pbr)/sqrtdr)*isdlpdr;

  REAL asq = gamma_gas*pres_L/dens_L;
  REAL vaxsq = Bx*Bx/dens_L;
  REAL ct2 = (By_L*By_L + Bz_L*Bz_L)/dens_L;
  REAL qsq = vaxsq + ct2 + asq;
  REAL tmp = vaxsq + ct2 - asq;
  REAL cfsq = 0.5f*(qsq + sqrt(tmp*tmp + 4.0f*asq*ct2));
  const REAL cfl = sqrt(cfsq);
  


  asq = gamma_gas*pres_R/dens_R;
  vaxsq = Bx*Bx/dens_R;
  ct2 = (By_R*By_R + Bz_R*Bz_R)/dens_R;
  qsq = vaxsq + ct2 + asq;
  tmp = vaxsq + ct2 - asq;
  cfsq = 0.5f*(qsq + sqrt(tmp*tmp + 4.0f*asq*ct2));
  const REAL cfr = sqrt(cfsq);

  REAL S_R0, S_L0;
  if (gamma_gas > 1.0f) {
    const REAL d  = droe;
    const REAL v1 = v1roe;
    const REAL v2 = v2roe;
    const REAL v3 = v3roe;
    const REAL b1 = Bx;
    const REAL b2 = b2roe;
    const REAL b3 = b3roe;
    const REAL h  = hroe;

    const REAL di = 1.0f/d;
    const REAL vsq = v1*v1 + v2*v2 + v3*v3;
    const REAL btsq = b2*b2 + b3*b3;
    const REAL bt_starsq = ((gamma_gas - 1.0f) - (gamma_gas - 2.0f)*y)*btsq;
    const REAL vaxsq = b1*b1*di;
    const REAL hp = h - (vaxsq + btsq*di);
    const REAL twid_asq = fmax(((gamma_gas - 1.0f)*(hp-0.5f*vsq)-(gamma_gas - 2.0f)*x), TINY);

    /* Compute fast- and slow-magnetosonic speeds (eq. B18) */

    const REAL ct2 = bt_starsq*di;
    const REAL tsum = vaxsq + ct2 + twid_asq;
    const REAL tdif = vaxsq + ct2 - twid_asq;
    const REAL cf2_cs2 = sqrt((tdif*tdif + 4.0f*twid_asq*ct2));

    const REAL cfsq = 0.5f*(tsum + cf2_cs2);
    const REAL cf = sqrt(cfsq);

    const REAL cssq = twid_asq*vaxsq/cfsq;
    const REAL cs = sqrt(cssq);

    /* Compute beta(s) (eqs. A17, B20, B28) */

    const REAL bt = sqrt(btsq);
    const REAL bt_star = sqrt(bt_starsq);
    const REAL bet2 = (bt == 0.0f) ? 1.0f : b2/bt;
    const REAL bet3 = (bt == 0.0f) ? 0.0f : b3/bt;
    const REAL bet2_star = bet2/sqrt((gamma_gas - 1.0f) - (gamma_gas - 2.0f)*y);
    const REAL bet3_star = bet3/sqrt((gamma_gas - 1.0f) - (gamma_gas - 2.0f)*y);
    const REAL bet_starsq = bet2_star*bet2_star + bet3_star*bet3_star;
    const REAL vbet = v2*bet2_star + v3*bet3_star;

    REAL alpha_f, alpha_s;
    if ((cfsq-cssq) == 0.0f) {
      alpha_f = 1.0f;
      alpha_s = 0.0f;
    } else if ( (twid_asq - cssq) <= 0.0f) {
      alpha_f = 0.0f;
      alpha_s = 1.0f;
    } else if ( (cfsq - twid_asq) <= 0.0f) {
      alpha_f = 1.0f;
      alpha_s = 0.0f;
    } else {
      alpha_f = sqrt((twid_asq - cssq)/(cfsq - cssq));
      alpha_s = sqrt((cfsq - twid_asq)/(cfsq - cssq));
    }
    
    /* Compute Q(s) and A(s) (eq. A14-15), etc. */

    const REAL sqrtd = sqrt(d);
    const REAL isqrtd = 1.0f/sqrtd;
    const REAL s = sign(b1);
    const REAL twid_a = sqrt(twid_asq);
    const REAL qf = cf*alpha_f*s;
    const REAL qs = cs*alpha_s*s;
    const REAL af_prime = twid_a*alpha_f*isqrtd;
    const REAL as_prime = twid_a*alpha_s*isqrtd;
    const REAL afpbb = af_prime*bt_star*bet_starsq;
    const REAL aspbb = as_prime*bt_star*bet_starsq;

    S_R0 = v1roe + cf;
    S_L0 = v1roe - cf;

  } else {
    const REAL d  = droe;
    const REAL v1 = v1roe;
    const REAL v2 = v2roe;
    const REAL v3 = v3roe;
    const REAL b1 = Bx;
    const REAL b2 = b2roe;
    const REAL b3 = b3roe;

    const REAL Iso_csound2 = (pres_L/dens_L + pres_R/dens_R)*0.5f;
    
    const REAL di = 1.0f/d;
    const REAL btsq = b2*b2 + b3*b3;
    const REAL bt_starsq  = btsq*y;
    const REAL vaxsq = b1*b1*di;
    const REAL twid_csq = Iso_csound2 + x;

    /* Compute fast- and slow-magnetosonic speeds (eq. B39) */

    const REAL ct2 = bt_starsq*di;
    const REAL tsum = vaxsq + ct2 + twid_csq;
    const REAL tdif = vaxsq + ct2 - twid_csq;
    const REAL cf2_cs2 = sqrt((tdif*tdif + 4.0f*twid_csq*ct2));

    const REAL cfsq = 0.5f*(tsum + cf2_cs2);
    const REAL cf = sqrt(cfsq);

    const REAL cssq = twid_csq*vaxsq/cfsq;
    const REAL cs = sqrt(cssq);

    /* Compute beta's (eqs. A17, B28, B40) */
    
    const REAL bt = sqrt(btsq);
    const REAL bt_star = sqrt(bt_starsq);
    const REAL bet2 = (bt == 0.0f) ? 1.0f : b2/bt;
    const REAL bet3 = (bt == 0.0f) ? 0.0f : b3/bt;
    
    const REAL bet2_star = bet2/sqrt(y);
    const REAL bet3_star = bet3/sqrt(y);
    const REAL bet_starsq = bet2_star*bet2_star + bet3_star*bet3_star;
    
    /* Compute alpha's (eq. A16) */
    
    REAL alpha_f, alpha_s;
    if ((cfsq-cssq) == 0.0f) {
      alpha_f = 1.0f;
      alpha_s = 0.0f;
    } else if ((twid_csq - cssq) <= 0.0f) {
      alpha_f = 0.0f;
      alpha_s = 1.0f;
    } else if ((cfsq - twid_csq) <= 0.0f) {
      alpha_f = 1.0f;
      alpha_s = 0.0f;
    } else {
      alpha_f = sqrt((twid_csq - cssq)/(cfsq - cssq));
      alpha_s = sqrt((cfsq - twid_csq)/(cfsq - cssq));
    }
    
    /* Compute Q's (eq. A14-15), etc. */
    
    const REAL sqrtd = sqrt(d);
    const REAL s = sign(b1);
    const REAL twid_c = sqrt(twid_csq);
    const REAL qf = cf*alpha_f*s;
    const REAL qs = cs*alpha_s*s;
    const REAL af_prime = twid_c*alpha_f/sqrtd;
    const REAL as_prime = twid_c*alpha_s/sqrtd;
    
    

    S_R0 = v1roe + cf;
    S_L0 = v1roe - cf;

  }
  
  /* take max/min of Roe eigenvalues and L/R state wave speeds */
  const REAL S_R1 = fmax(S_R0, velx_R + cfr);
  const REAL S_L1 = fmin(S_L0, velx_L - cfl);

  const REAL S_R = fmax(S_R1, w);
  const REAL S_L = fmin(S_L1, w);
  

#endif

  REAL vB_L = velx_L*Bx + vely_L*By_L   + velz_L*Bz_L;
  REAL vB_R = velx_R*Bx + vely_R*By_R   + velz_R*Bz_R;
  
  REAL Fdens_L = dens_L*velx_L;
  REAL Fmomx_L = momx_L*velx_L + pT_L - Bx*Bx;
  REAL Fmomy_L = momy_L*velx_L        - Bx*By_L; 
  REAL Fmomz_L = momz_L*velx_L        - Bx*Bz_L;
  REAL Fetot_L = etot_L*velx_L + pT_L*velx_L - Bx*vB_L;
  REAL Fby_L   = By_L  *velx_L - Bx*vely_L;
  REAL Fbz_L   = Bz_L  *velx_L - Bx*velz_L;
  
  REAL Fdens_R = dens_R*velx_R;
  REAL Fmomx_R = momx_R*velx_R + pT_R - Bx*Bx;
  REAL Fmomy_R = momy_R*velx_R        - Bx*By_R; 
  REAL Fmomz_R = momz_R*velx_R        - Bx*Bz_R; 
  REAL Fetot_R = etot_R*velx_R + pT_R*velx_R - Bx*vB_R;
  REAL Fby_R   = By_R  *velx_R        - Bx*vely_R;
  REAL Fbz_R   = Bz_R  *velx_R        - Bx*velz_R;

  
  REAL U_dens = (S_R*dens_R - S_L*dens_L + Fdens_L - Fdens_R)/(S_R - S_L);
  REAL U_momx = (S_R*momx_R - S_L*momx_L + Fmomx_L - Fmomx_R)/(S_R - S_L);
  REAL U_momy = (S_R*momy_R - S_L*momy_L + Fmomy_L - Fmomy_R)/(S_R - S_L);
  REAL U_momz = (S_R*momz_R - S_L*momz_L + Fmomz_L - Fmomz_R)/(S_R - S_L);
  REAL U_etot = (S_R*etot_R - S_L*etot_L + Fetot_L - Fetot_R)/(S_R - S_L);
  
  struct ptcl_mhd_cl flux;
  flux.dens = (S_R*Fdens_L - S_L*Fdens_R + S_L*S_R*(dens_R - dens_L))/(S_R - S_L);
  flux.momx = (S_R*Fmomx_L - S_L*Fmomx_R + S_L*S_R*(momx_R - momx_L))/(S_R - S_L);
  flux.momy = (S_R*Fmomy_L - S_L*Fmomy_R + S_L*S_R*(momy_R - momy_L))/(S_R - S_L);
  flux.momz = (S_R*Fmomz_L - S_L*Fmomz_R + S_L*S_R*(momz_R - momz_L))/(S_R - S_L);
  flux.etot = (S_R*Fetot_L - S_L*Fetot_R + S_L*S_R*(etot_R - etot_L))/(S_R - S_L);

  REAL U_by   = (S_R*By_R - S_L*By_L + Fby_L - Fby_R)/(S_R - S_L);
  REAL U_bz   = (S_R*Bz_R - S_L*Bz_L + Fbz_L - Fbz_R)/(S_R - S_L);

  flux.by = (S_R*Fby_L - S_L*Fby_R + S_L*S_R*(By_R - By_L))/(S_R - S_L);
  flux.bz = (S_R*Fbz_L - S_L*Fbz_R + S_L*S_R*(Bz_R - Bz_L))/(S_R - S_L);

  flux.dens -= w*U_dens;
  flux.momx -= w*U_momx;
  flux.momy -= w*U_momy;
  flux.momz -= w*U_momz;
  flux.etot -= w*U_etot;
  flux.by   -= w*U_by;
  flux.bz   -= w*U_bz;
  

  return flux;
}


#endif

__kernel void compute_fluxes(__global       float4      *out_dqdt1,         //  0
			     __global       float4      *out_dqdt2,         //  1
			     __global       float4      *out_dqdt3,         //  2
			     __global       float4      *out_divB,          //  3
			     const __global float4      *in_mhd1_L,         //  4
			     const __global float4      *in_mhd1_R,         //  5
			     const __global float4      *in_mhd2_L,         //  6
			     const __global float4      *in_mhd2_R,         //  7
			     const __global float4      *in_mhd3_L,         //  8
			     const __global float4      *in_mhd3_R,         //  9
			     const __global int         *in_ilist,          // 10
			     const __global int         *in_jlist,          // 11
			     const __global int         *in_nj,             // 12
			     const __global float4      *in_dwij,           // 13
			     const          int          Ni,                // 14
			     const float                 gamma_gas,         // 15
			     const float                 ch_global) {       // 16
  
  // Get thread info: localIdx & globalIdx, etc
  const int localIdx  = get_local_id(0);
  const int localDim  = NBLOCKDIM; //get_local_size(0);
  const int groupIdx  = get_group_id(0);
  const int globalIdx = get_global_id(0);
  if (globalIdx >= Ni) return;

  // Get bodyId

  const int bodyId = in_ilist[globalIdx];         // alligned to 128-byte boundary
  const int nj     = in_nj   [globalIdx];
  
  // compute Idx of first and last j-particle
  
  const int jbeg = groupIdx * (NGBMAX*localDim) + localIdx;
  const int jend = jbeg + nj * localDim;


  float4 dqdt1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 dqdt2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 dqdt3 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 divB  = {0.0f, 0.0f, 0.0f, 0.0f};

  // pragma unroll 16
  for (int jidx = jbeg; jidx < jend; jidx += localDim) {
    const float4 dwij = in_dwij[jidx];

    ///////////

    const float4 mhd1_L = in_mhd1_L[jidx];
    const float4 mhd1_R = in_mhd1_R[jidx];

    const float dens_L = mhd1_L.w;
    const float velx_L = mhd1_L.x;
    const float vely_L = mhd1_L.y;
    const float velz_L = mhd1_L.z;

    const float dens_R = mhd1_R.w;
    const float velx_R = mhd1_R.x;
    const float vely_R = mhd1_R.y;
    const float velz_R = mhd1_R.z;

    const float4 mhd2_L = in_mhd2_L[jidx];
    const float4 mhd2_R = in_mhd2_R[jidx];

    const float pres_L = compute_pressure(dens_L, mhd2_L.w, gamma_gas);
    const float Bx_L   = mhd2_L.x;
    const float By_L   = mhd2_L.y;
    const float Bz_L   = mhd2_L.z;

    const float pres_R = compute_pressure(dens_R, mhd2_R.w, gamma_gas);
    const float Bx_R   = mhd2_R.x;
    const float By_R   = mhd2_R.y;
    const float Bz_R   = mhd2_R.z;

    const float4 mhd3_L = in_mhd3_L[jidx];
    const float4 mhd3_R = in_mhd3_R[jidx];
    
    const float psiL = mhd3_L.x;
    const float psiR = mhd3_R.x;
    
    //////////////

     
    const float  Wx   = dwij.w;

    ////
#if 1
    const REAL cfl = SQRT((gamma_gas*pres_L + sqr(Bx_L) + sqr(By_L) + sqr(Bz_L))/dens_L);
    const REAL cfr = SQRT((gamma_gas*pres_R + sqr(Bx_R) + sqr(By_R) + sqr(Bz_R))/dens_R);
    //      const float ch = 0.5f*(cfl + cfr)/2.0f; //0.5f*fmin(cfl, cfr);
    //      const float ch = fmin(cfl, cfr);
    const float ch = fmax(cfl, cfr);
    //      const float ch = cfl;
    const float B_M1  = (Bx_L + Bx_R)*0.5f - 0.5f/ch*(psiR - psiL);
    const float psiM1 = (psiL + psiR)*0.5f - 0.5f*ch*(Bx_R - Bx_L);
    
    ////

    const float Sx  = (fabs(Bx_L - Bx_R) < SMALLF) ? HUGE*sign(Bx_L *velx_L - Bx_R *velx_R ) : (Bx_L *velx_L - Bx_R *velx_R)/(Bx_L  - Bx_R);
    const float B_M2 = (Sx >= Wx) ? Bx_L : Bx_R;

    const float Sp    = (fabs(psiL - psiR) < SMALLF) ? HUGE*sign(psiL *velx_L - psiR *velx_R ) : (psiL *velx_L - psiR *velx_R)/(psiL  - psiR);
    const float psiM2 = 0.0f; //(Sp >= Wx) ? psiL : psiR;
#if 0
    const float B_M  = (B_M1  > fmin(Bx_L, Bx_R) && B_M1  < fmax(Bx_L, Bx_R)) ? B_M1  : B_M2;
    const float psiM = (psiM1 > fmin(psiL, psiR) && psiM1 < fmax(psiL, psiR)) ? psiM1 : psiM2;
#else
    const float B_M  = B_M1;
    const float psiM = psiM1;
#endif
    //    crap
#else
    const float psiM = (psiL + psiR)*0.5f;
    const float B_M  = (Bx_L + Bx_R)*0.5f;

#endif
    
    const struct ptcl_mhd_cl flux = riemann_solver(Wx, B_M, gamma_gas,
						   dens_L, pres_L, velx_L, vely_L, velz_L, By_L, Bz_L,
						   dens_R, pres_R, velx_R, vely_R, velz_R, By_R, Bz_R);
    
    
    const float flux_bx  = -Wx*B_M + psiM*0.0f;
    
    //////////////

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
    
    const float iAxx =  cosph*sinth;
    const float iAxy = -sinph;
    const float iAxz = -costh*cosph;
    const float iAyx =  sinth*sinph;
    const float iAyy =  cosph;
    const float iAyz = -costh*sinph;
    const float iAzx =  costh;
    const float iAzy =  0.0f;
    const float iAzz =  sinth; 
    
    const float flux_mass  = flux.dens;
    const float flux_etot  = flux.etot;
    const float flux_momx  = iAxx*flux.momx + iAxy*flux.momy + iAxz*flux.momz;
    const float flux_momy  = iAyx*flux.momx + iAyy*flux.momy + iAyz*flux.momz;
    const float flux_momz  = iAzx*flux.momx + iAzy*flux.momy + iAzz*flux.momz;
    const float flux_wBx   = iAxx*flux_bx   + iAxy*flux.by   + iAxz*flux.bz;
    const float flux_wBy   = iAyx*flux_bx   + iAyy*flux.by   + iAyz*flux.bz;
    const float flux_wBz   = iAzx*flux_bx   + iAzy*flux.by   + iAzz*flux.bz;
    
    //////////////

    dqdt1 += wij * (float4){flux_momx, flux_momy, flux_momz, flux_mass};
    dqdt2 += wij * (float4){flux_wBx,  flux_wBy,  flux_wBz,  flux_etot};
    
    const float4 scal = flux_mass * ((flux_mass > 0.0f) ? mhd3_L : mhd3_R);
    dqdt3 += wij * scal;

//     const float  Fpsi = sqr(ch) * B_M;
//     dqdt3 += (float4){Fpsi * wij, scal.y, scal.z, scal.w};
    
    const float4 gradPsi = dwij * psiM;
    divB += (float4){gradPsi.x, gradPsi.y, gradPsi.z, B_M * wij};
  }

  out_dqdt1[bodyId] = dqdt1;
  out_dqdt2[bodyId] = dqdt2;
  out_dqdt3[bodyId] = dqdt3;
  out_divB [bodyId] = divB;

}


#endif  
