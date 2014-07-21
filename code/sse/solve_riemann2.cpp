#include "gn.h"

float system::compute_pressure(const float dens, const float ethm) {
  return ethm*(gamma_gas - 1.0f);
}

void  riemann_solver(const float w, 
		     const float b,
		     const float v, const float gamma_gas,
		     const float dens_L, 
		     const float pres_L, 
		     const float velx_L,
		     const float vely_L,
		     const float velz_L,
		     const float Bx_L,
		     const float By_L,
		     const float Bz_L,
		     const float dens_R, 
		     const float pres_R, 
		     const float velx_R,
		     const float vely_R,
		     const float velz_R,
		     const float Bx_R,
		     const float By_R,
		     const float Bz_R,
		     float &Fdens, float &Fetot, 
		     float &Fmomx, float &Fmomy, float &Fmomz,
		     float &Fbx,   float &Fby,   float &Fbz);

ptcl_mhd system::solve_riemann(const float3 &Wij, const float3 &Bij, 
			       const float3 &Vij,
			       const ptcl_mhd &Qi, 
			       const ptcl_mhd &Qj, 
			       const float3 &e) {

  const float  ds   = sqrt(e.x*e.x + e.y*e.y);
  const float ids   = ( ds  > 0.0f) ? 1.0f/ds : 0.0f;
  const float cosph = (ids == 0.0f) ? 1.0f    : e.x*ids;
  const float sinph = (ids == 0.0f) ? 0.0f    : e.y*ids;
  const float costh = e.z;
  const float sinth = ds;

  const float Axx =  cosph*sinth;
  const float Axy =  sinth*sinph;
  const float Axz =  costh;
  const float Ayx = -sinph;
  const float Ayy =  cosph;
  const float Ayz =  0.0f;
  const float Azx = -costh*cosph;
  const float Azy = -costh*sinph;
  const float Azz =  sinth;

  const float dens_L = Qi.dens;
  const float ethm_L = Qi.ethm;
  const float velx_L = Axx*Qi.vel.x + Axy*Qi.vel.y + Axz*Qi.vel.z;
  const float vely_L = Ayx*Qi.vel.x + Ayy*Qi.vel.y + Ayz*Qi.vel.z;
  const float velz_L = Azx*Qi.vel.x + Azy*Qi.vel.y + Azz*Qi.vel.z;
  const float Bx_L   = Axx*Qi.B.x   + Axy*Qi.B.y   + Axz*Qi.B.z;
  const float By_L   = Ayx*Qi.B.x   + Ayy*Qi.B.y   + Ayz*Qi.B.z;
  const float Bz_L   = Azx*Qi.B.x   + Azy*Qi.B.y   + Azz*Qi.B.z;
  
  const float dens_R = Qj.dens;
  const float ethm_R = Qj.ethm;
  const float velx_R = Axx*Qj.vel.x + Axy*Qj.vel.y + Axz*Qj.vel.z;
  const float vely_R = Ayx*Qj.vel.x + Ayy*Qj.vel.y + Ayz*Qj.vel.z;
  const float velz_R = Azx*Qj.vel.x + Azy*Qj.vel.y + Azz*Qj.vel.z;
  const float Bx_R   = Axx*Qj.B.x   + Axy*Qj.B.y   + Axz*Qj.B.z;
  const float By_R   = Ayx*Qj.B.x   + Ayy*Qj.B.y   + Ayz*Qj.B.z;
  const float Bz_R   = Azx*Qj.B.x   + Azy*Qj.B.y   + Azz*Qj.B.z;
  
  const float w = Axx*Wij.x + Axy*Wij.y + Axz*Wij.z;
  const float b = Axx*Bij.x + Axy*Bij.y + Axz*Bij.z;
  const float v = Axx*Vij.x + Axy*Vij.y + Axz*Vij.z;
  
  const float pres_L = compute_pressure(dens_L, ethm_L);
  const float pres_R = compute_pressure(dens_R, ethm_R);
  
  float Fdens, Fetot, Fmomx, Fmomy, Fmomz, Fbx, Fby, Fbz;
  riemann_solver(w, b, v, gamma_gas,
		 dens_L, pres_L, velx_L, vely_L, velz_L, Bx_L, By_L, Bz_L,
		 dens_R, pres_R, velx_R, vely_R, velz_R, Bx_R, By_R, Bz_R,
		 Fdens,  Fetot,  Fmomx,  Fmomy,  Fmomz,  Fbx,  Fby,  Fbz); 

  const float iAxx =  cosph*sinth;
  const float iAxy = -sinph;
  const float iAxz = -costh*cosph;
  const float iAyx =  sinth*sinph;
  const float iAyy =  cosph;
  const float iAyz = -costh*sinph;
  const float iAzx =  costh;
  const float iAzy =  0.0f;
  const float iAzz =  sinth; 

  ptcl_mhd F;
  F.mass  = Fdens;
  F.etot  = Fetot;
  F.mom.x = iAxx*Fmomx + iAxy*Fmomy + iAxz*Fmomz;
  F.mom.y = iAyx*Fmomx + iAyy*Fmomy + iAyz*Fmomz;
  F.mom.z = iAzx*Fmomx + iAzy*Fmomy + iAzz*Fmomz;
  F.wB.x  = iAxx*Fbx   + iAxy*Fby   + iAxz*Fbz;
  F.wB.y  = iAyx*Fbx   + iAyy*Fby   + iAyz*Fbz;
  F.wB.z  = iAzx*Fbx   + iAzy*Fby +   iAzz*Fbz;

  return F;
}


void  riemann_solver(const float w, const float b, const float v, const float gamma_gas,
		     const float dens_L, 
		     const float pres_L, 
		     const float velx_L,
		     const float vely_L,
		     const float velz_L,
		     const float Bx_L,
		     const float By_L,
		     const float Bz_L,
		     const float dens_R, 
		     const float pres_R, 
		     const float velx_R,
		     const float vely_R,
		     const float velz_R,
		     const float Bx_R,
		     const float By_R,
		     const float Bz_R,
		     float &Fdens, float &Fetot, 
		     float &Fmomx, float &Fmomy, float &Fmomz,
		     float &Fbx,   float &Fby,   float &Fbz) {


  // #if 1
//  const float Bx = b;
// #else
//  const real Bx = (std::abs(Bx_L) > std::abs(Bx_R)) ? Bx_L : Bx_R;
// #endif
 
  
  const double Bx = b;
  const double signBx = (std::abs(Bx) > 0.0) ? Bx/std::abs(Bx) : 0.0;
  
  const double momx_L = dens_L*velx_L;
  const double momy_L = dens_L*vely_L;
  const double momz_L = dens_L*velz_L;
  
  const double momx_R = dens_R*velx_R;
  const double momy_R = dens_R*vely_R;
  const double momz_R = dens_R*velz_R;
  
  const double B2_L   = sqr(Bx)     + sqr(By_L)   + sqr(Bz_L);
  const double v2_L   = sqr(velx_L) + sqr(vely_L) + sqr(velz_L);
  const double etot_L = pres_L/(gamma_gas - 1) + 0.5*dens_L*v2_L + 0.5*B2_L;

  const double B2_R   = sqr(Bx)     + sqr(By_R)   + sqr(Bz_R);
  const double v2_R   = sqr(velx_R) + sqr(vely_R) + sqr(velz_R);
  const double etot_R = pres_R/(gamma_gas - 1) + 0.5*dens_R*v2_R + 0.5*B2_R;

  const double gpl  = gamma_gas * pres_L;
  const double gpr  = gamma_gas * pres_R;
  const double gpbl = gpl + B2_L;
  const double gpbr = gpr + B2_R;
  
#if 1

  const double cfl = sqrt((gpbl + sqrt( sqr(gpbl) - 4*gpl*sqr(Bx) ))/(2.0*dens_L));
  const double cfr = sqrt((gpbr + sqrt( sqr(gpbr) - 4*gpr*sqr(Bx) ))/(2.0*dens_R));
  const double cfmax = sqrt(1.0)*std::max(cfl,cfr);

  const double S_L = std::min(velx_L, velx_R) - cfmax;
  const double S_R = std::max(velx_L, velx_R) + cfmax;

#else
#error "Please make sure this code is not compiled in solve_riemann.cpp"

  double dsl = sqrt(dens_L);
  double dsr = sqrt(dens_R);
  double ids = 1.0/(dsl + dsr);
  double droe = dsl * dsr;
  
  double uxroe = (dsl*velx_L + dsr*velx_R)*ids;
  double uyroe = (dsl*vely_L + dsr*vely_R)*ids;
  double uzroe = (dsl*velz_L + dsr*velz_R)*ids;
    
  double byroe = (dsl*By_L + dsr*By_R)*ids;
  double bzroe = (dsl*Bz_L + dsr*Bz_R)*ids;

  double x = 0.5 * (sqr(By_L - By_R) + sqr(Bz_L - Bz_R))/sqr(dsl + dsl);
  double y = 0.5 * (dens_L + dens_R)/droe;
    
  double pbl = 0.5*B2_L;
  double pbr = 0.5*B2_R;
    
  double hl  = (etot_L + pres_L + pbl)/dens_L;
  double hr  = (etot_R + pres_R + pbr)/dens_R;
  double hroe  = (dsl*hl + dsr*hr)*ids;

  double di  = 1.0/droe;
  double vsq = sqr(uxroe) + sqr(uyroe) + sqr(uzroe);
  double btsq = sqr(byroe) + sqr(bzroe);
  double bt_startsq = ((gamma_gas - 1) - (gamma_gas - 2)*y)*btsq;
  double vaxsq = Bx*Bx*di;
  double hp = hroe - (vaxsq + btsq*di);
  double twid_asq = ((gamma_gas - 1)*(hp - 0.5*vsq) - (gamma_gas - 2)*x);

  double ct2  = bt_startsq*di;
  double tsum = vaxsq + ct2 + twid_asq;
  double tdif = vaxsq + ct2 - twid_asq;
  double cf2_cs2 = sqrt(sqr(tdif) + 4.0*twid_asq*ct2);
  double cfsq = 0.5*(tsum + cf2_cs2);
  double cf = sqrt(cfsq);

  real fsig = 1;
  
  double S_L = uxroe - fsig*cf;
  double S_R = uxroe + fsig*cf;

  double asq = sqrt(gamma_gas * pres_L/dens_L);
  vaxsq = sqr(Bx)/dens_L;
  ct2   = (sqr(By_L) + sqr(Bz_L))/dens_L;
  double qsq = vaxsq + ct2 + asq;
  double tmp = vaxsq + ct2 - asq;
  cfsq = 0.5*(qsq + sqrt(sqr(tmp) + 4.0*asq*ct2));
  double cfl = fsig*sqrt(cfsq);

  asq = sqrt(gamma_gas * pres_R/dens_R);
  vaxsq = sqr(Bx)/dens_R;
  ct2   = (sqr(By_R) + sqr(Bz_R))/dens_R;
  qsq = vaxsq + ct2 + asq;
  tmp = vaxsq + ct2 - asq;
  cfsq = 0.5*(qsq + sqrt(sqr(tmp) + 4.0*asq*ct2));
  double cfr = fsig*sqrt(cfsq);

  if (velx_R + cfr > S_R) S_R = velx_R + cfr;
  if (velx_L - cfl < S_L) S_L = velx_L - cfl;
#endif
  
  const double pT_L = pres_L + 0.5 * B2_L;
  const double pT_R = pres_R + 0.5 * B2_R;

  const double iSM = 1.0/((S_R - velx_R)*dens_R - (S_L - velx_L)*dens_L);
  const double S_M  = ((S_R - velx_R)*momx_R - (S_L - velx_L)*momx_L - pT_R + pT_L)*iSM;

  const double ipTs = 1.0/((S_R - velx_R)*dens_R - (S_L - velx_L)*dens_L);
  const double pT_s = ipTs * ((S_R - velx_R)*dens_R*pT_L - (S_L - velx_L)*dens_L*pT_R +
			      dens_L*dens_R*(S_R - velx_R)*(S_L - velx_L)*(velx_R - velx_L));
  
  const double B2x = Bx*Bx;

  const double dens_L_s = dens_L * (S_L - velx_L)/(S_L - S_M);
  const double vely_L_s = vely_L - Bx*By_L*(S_M - velx_L)/(dens_L*(S_L - velx_L)*(S_L - S_M) - B2x);
  const double velz_L_s = velz_L - Bx*Bz_L*(S_M - velx_L)/(dens_L*(S_L - velx_L)*(S_L - S_M) - B2x);
  const double   By_L_s = By_L * (dens_L*sqr(S_L - velx_L) - B2x)/(dens_L*(S_L - velx_L)*(S_L - S_M) - B2x);
  const double   Bz_L_s = Bz_L * (dens_L*sqr(S_L - velx_L) - B2x)/(dens_L*(S_L - velx_L)*(S_L - S_M) - B2x);


  const double dens_R_s = dens_R * (S_R - velx_R)/(S_R - S_M);
  const double vely_R_s = vely_R - Bx*By_R*(S_M - velx_R)/(dens_R*(S_R - velx_R)*(S_R - S_M) - B2x);
  const double velz_R_s = velz_R - Bx*Bz_R*(S_M - velx_R)/(dens_R*(S_R - velx_R)*(S_R - S_M) - B2x);
  const double   By_R_s = By_R * (dens_R*sqr(S_R - velx_R) - B2x)/(dens_R*(S_R - velx_R)*(S_R - S_M) - B2x);
  const double   Bz_R_s = Bz_R * (dens_R*sqr(S_R - velx_R) - B2x)/(dens_R*(S_R - velx_R)*(S_R - S_M) - B2x);

  const double   vB_L   = velx_L  *Bx + vely_L  *By_L   + velz_L  *Bz_L;
  const double   vB_L_s = velx_L_s*Bx + vely_L_s*By_L_s + velz_L_s*Bz_L_s;
  const double etot_L_s = ((S_L - velx_L)*etot_L - pT_L*velx_L + pT_s*S_M + Bx*(vB_L - vB_L_s))/(S_L - S_M);

  const double   vB_R   = velx_R  *Bx + vely_R  *By_R   + velz_R  *Bz_R;
  const double   vB_R_s = velx_R_s*Bx + vely_R_s*By_R_s + velz_R_s*Bz_R_s;
  const double etot_R_s = ((S_R - velx_R)*etot_R - pT_R*velx_R + pT_s*S_M + Bx*(vB_R - vB_R_s))/(S_R - S_M);

  const double dens_L_ss = dens_L_s;
  const double dens_R_ss = dens_R_s;
  const double    S_L_s  = S_M - std::abs(Bx)/sqrt(dens_L_s);
  const double    S_R_s  = S_M + std::abs(Bx)/sqrt(dens_R_s);

  const double idsqrt  = 1.0/(sqrt(dens_L_s) + sqrt(dens_R_s));
  const double  vely_ss = idsqrt*(sqrt(dens_L_s)*vely_L_s + sqrt(dens_R_s)*vely_R_s + (By_R_s - By_L_s)*signBx);
  const double  velz_ss = idsqrt*(sqrt(dens_L_s)*velz_L_s + sqrt(dens_R_s)*velz_R_s + (Bz_R_s - Bz_L_s)*signBx);

  const double By_ss = idsqrt*(sqrt(dens_L_s)*By_R_s + sqrt(dens_R_s)*By_L_s + sqrt(dens_L_s*dens_R_s)*(vely_R_s - vely_L_s)*signBx);
  const double Bz_ss = idsqrt*(sqrt(dens_L_s)*Bz_R_s + sqrt(dens_R_s)*Bz_L_s + sqrt(dens_L_s*dens_R_s)*(velz_R_s - velz_L_s)*signBx);
  
  const double vely_L_ss = vely_ss;
  const double velz_L_ss = velz_ss;
  const double   By_L_ss = By_ss;
  const double   Bz_L_ss = Bz_ss;

  const double vely_R_ss = vely_ss;
  const double velz_R_ss = velz_ss;
  const double   By_R_ss = By_ss;
  const double   Bz_R_ss = Bz_ss;

  const double vB_L_ss   = velx_L_ss*Bx + vely_L_ss*By_L_ss + velz_L_ss*Bz_L_ss;
  const double etot_L_ss = etot_L_s - sqrt(dens_L_s)*(vB_L_s - vB_L_ss)*signBx;

  const double vB_R_ss   = velx_R_ss*Bx + vely_R_ss*By_R_ss + velz_R_ss*Bz_R_ss;
  const double etot_R_ss = etot_R_s + sqrt(dens_R_s)*(vB_R_s - vB_R_ss)*signBx;

  const double Fdens_L = dens_L*velx_L;
  const double Fmomx_L = momx_L*velx_L + pT_L - B2x;
  const double Fmomy_L = momy_L*velx_L        - Bx*By_L;
  const double Fmomz_L = momz_L*velx_L        - Bx*Bz_L;
  const double Fetot_L = etot_L*velx_L + pT_L*velx_L - Bx*vB_L; 

  const double Fdens_R = dens_R*velx_R;
  const double Fmomx_R = momx_R*velx_R + pT_R - B2x;
  const double Fmomy_R = momy_R*velx_R        - Bx*By_R;
  const double Fmomz_R = momz_R*velx_R        - Bx*Bz_R;
  const double Fetot_R = etot_R*velx_R + pT_R*velx_R - Bx*vB_R;

  const double momx_L_s  = dens_L_s *velx_L_s;
  const double momy_L_s  = dens_L_s *vely_L_s;
  const double momz_L_s  = dens_L_s *velz_L_s;
  
  const double momx_L_ss = dens_L_ss*velx_L_ss;
  const double momy_L_ss = dens_L_ss*vely_L_ss;
  const double momz_L_ss = dens_L_ss*velz_L_ss;

  const double momx_R_s  = dens_R_s *velx_R_s;
  const double momy_R_s  = dens_R_s *vely_R_s;
  const double momz_R_s  = dens_R_s *velz_R_s;
  
  const double momx_R_ss = dens_R_ss*velx_R_ss;
  const double momy_R_ss = dens_R_ss*vely_R_ss;
  const double momz_R_ss = dens_R_ss*velz_R_ss;

  if (S_L > w) {
    Fdens = Fdens_L - w * dens_L;
    Fmomx = Fmomx_L - w * momx_L;
    Fmomy = Fmomy_L - w * momy_L;
    Fmomz = Fmomz_L - w * momz_L;
    Fetot = Fetot_L - w * etot_L;
  } else if (S_L <= w && w <= S_L_s) {
    Fdens = Fdens_L + (S_L - w)*dens_L_s - S_L*dens_L;
    Fmomx = Fmomx_L + (S_L - w)*momx_L_s - S_L*momx_L;
    Fmomy = Fmomy_L + (S_L - w)*momy_L_s - S_L*momy_L;
    Fmomz = Fmomz_L + (S_L - w)*momz_L_s - S_L*momz_L;
    Fetot = Fetot_L + (S_L - w)*etot_L_s - S_L*etot_L;
  } else if (S_L_s <= w && w <= S_M) {
    Fdens = Fdens_L + (S_L_s - w)*dens_L_ss - (S_L_s - S_L)*dens_L_s - S_L*dens_L;
    Fmomx = Fmomx_L + (S_L_s - w)*momx_L_ss - (S_L_s - S_L)*momx_L_s - S_L*momx_L;
    Fmomy = Fmomy_L + (S_L_s - w)*momy_L_ss - (S_L_s - S_L)*momy_L_s - S_L*momy_L;
    Fmomz = Fmomz_L + (S_L_s - w)*momz_L_ss - (S_L_s - S_L)*momz_L_s - S_L*momz_L;
    Fetot = Fetot_L + (S_L_s - w)*etot_L_ss - (S_L_s - S_L)*etot_L_s - S_L*etot_L;
  } else if (S_M <= w && w <= S_R_s) {
    Fdens = Fdens_R + (S_R_s - w)*dens_R_ss - (S_R_s - S_R)*dens_R_s - S_R*dens_R;
    Fmomx = Fmomx_R + (S_R_s - w)*momx_R_ss - (S_R_s - S_R)*momx_R_s - S_R*momx_R;
    Fmomy = Fmomy_R + (S_R_s - w)*momy_R_ss - (S_R_s - S_R)*momy_R_s - S_R*momy_R;
    Fmomz = Fmomz_R + (S_R_s - w)*momz_R_ss - (S_R_s - S_R)*momz_R_s - S_R*momz_R;
    Fetot = Fetot_R + (S_R_s - w)*etot_R_ss - (S_R_s - S_R)*etot_R_s - S_R*etot_R;
  } else if (S_R_s <= w && w <= S_R) {
    Fdens = Fdens_R + (S_R - w)*dens_R_s - S_R*dens_R;
    Fmomx = Fmomx_R + (S_R - w)*momx_R_s - S_R*momx_R;
    Fmomy = Fmomy_R + (S_R - w)*momy_R_s - S_R*momy_R;
    Fmomz = Fmomz_R + (S_R - w)*momz_R_s - S_R*momz_R;
    Fetot = Fetot_R + (S_R - w)*etot_R_s - S_R*etot_R;
  } else {
    Fdens = Fdens_R - w * dens_R;
    Fmomx = Fmomx_R - w * momx_R;
    Fmomy = Fmomy_R - w * momy_R;
    Fmomz = Fmomz_R - w * momz_R;
    Fetot = Fetot_R - w * etot_R;
  }

  ///////////////////

  const real Sx = (Bx_L == Bx_R) ? HUGE*sign(Bx_L*velx_L - Bx_R*velx_R) : (Bx_L*velx_L - Bx_R*velx_R)/(Bx_L - Bx_R);
  const real Sy = (By_L == By_R) ? HUGE*sign(By_L*velx_L - By_R*velx_R) : (By_L*velx_L - By_R*velx_R)/(By_L - By_R);
  const real Sz = (Bz_L == Bz_R) ? HUGE*sign(Bz_L*velx_L - Bz_R*velx_R) : (Bz_L*velx_L - Bz_R*velx_R)/(Bz_L - Bz_R);
  
  Fbx = (Sx >= w) ? Bx_L*(velx_L - w) : Bx_R*(velx_R - w);
  Fby = (Sy >= w) ? By_L*(velx_L - w) : By_R*(velx_R - w);
  Fbz = (Sz >= w) ? Bz_L*(velx_L - w) : Bz_R*(velx_R - w);

}
