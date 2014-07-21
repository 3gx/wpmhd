#include "gn.h"

inline void  monotonicity(real &fL, real &fR, const real fi, const real fj) {
  const real Fmin = std::min(fi, fj);
  const real Fmax = std::max(fi, fj);
  if ((fL < Fmin) || (fL > Fmax) || (fR < Fmin) || (fR > Fmax)) {
    fL = fR = fi;
    fL = fi;
    fR = fj;
  }
}

void system::mhd_interaction() {
  double t0 = get_time();
  
  const int n_groups = group_list.size();
  
  std::vector<int>       local_idx(local_n);
  std::vector<ptcl_mhd>  local_dot(local_n);

  pmhd_dot.resize(pmhd.size());
  pmhd_dot0.resize(pmhd.size());

  dwdt_i.resize(pmhd.size());
  divB_i.resize(pmhd.size());

  int iloc = 0;
  
  for (int group = 0; group < n_groups; group++) {
    const octnode &inode              =   *group_list[group];
    const int n_leaves                = ngb_leaf_list_outer[group].size();
    std::vector<octnode*> &ileaf_list = ngb_leaf_list_outer[group];
    
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (ibp->isexternal()) continue;
      const particle &pi = *ibp->pp;
      const ptcl_mhd &mi = pmhd[pi.local_idx];
      
      const boundary  bi = boundary(pi.pos, pi.h);
      const pfloat3 ipos = pi.pos;
      
      const float   hi      = pi.h;
      const float   wi      = pi.wght;
      
      const float   hi2     = hi*hi;
      const float inv_hi    = 1.0f/hi;
      const float inv_hidim = kernel.pow(inv_hi);
      
      ptcl_mhd dQdt;
      dQdt.set(0.0f);

      float  divB   =  0.0f;
      float3 bgradv   = {0.0f, 0.0f, 0.0f};
      float  bgradpsi = 0.0f;
      float  gradpsi[3]  = {0.0f, 0.0f, 0.0f};
      
#if !((defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_))
      float omega = 0.0f;
      float dnidt = 0.0f;
#endif      
      for (int leaf = 0; leaf < n_leaves; leaf++) {
 	const octnode &jnode = *ileaf_list[leaf];
	
 	if (!bi.overlap(jnode.inner)) continue;
	
	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  const particle &pj = *jbp->pp;
	  const ptcl_mhd &mj = pmhd[pj.local_idx];
	  
	  const pfloat3 jpos = pj.pos; 
	  const float3  dr   = {jpos.x - ipos.x,
				jpos.y - ipos.y,
				jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  
 	  const float hj  = pj.h;
 	  const float hj2 = hj*hj;

  	  if (s2 <= hi2 || s2 <= hj2)  {

	    const int li = pi.local_idx;
	    const int lj = pj.local_idx;
	    
	    const float inv_hj    = 1.0f/hj;
	    const float inv_hjdim = kernel.pow(inv_hj);
	    
	    const float wj  = pj.wght;
	    
	    const float s   = std::sqrt(s2);
	    const float wki = kernel.w(s*inv_hi)*inv_hidim * wi;
	    const float wkj = kernel.w(s*inv_hj)*inv_hjdim * wj;
	    
	    const float dwi[3] = {wki * (Bxx[li]*dr.x + Bxy[li]*dr.y + Bxz[li]*dr.z),
				  wki * (Bxy[li]*dr.x + Byy[li]*dr.y + Byz[li]*dr.z),
				  wki * (Bxz[li]*dr.x + Byz[li]*dr.y + Bzz[li]*dr.z)};
	    
	    const float dwj[3] = {wkj * (Bxx[lj]*dr.x + Bxy[lj]*dr.y + Bxz[lj]*dr.z),
				  wkj * (Bxy[lj]*dr.x + Byy[lj]*dr.y + Byz[lj]*dr.z),
				  wkj * (Bxz[lj]*dr.x + Byz[lj]*dr.y + Bzz[lj]*dr.z)};
	    
	    const float Vi = weights[li];
	    const float Vj = weights[lj];
	    const float dwij[3] = {Vi*dwi[0] + Vj*dwj[0],
				   Vi*dwi[1] + Vj*dwj[1],
				   Vi*dwi[2] + Vj*dwj[2]};
	    
	    ptcl_mhd Qi = mi;
	    ptcl_mhd Qj = mj;
	    
#if 1
	    const float ds[3] = {0.5f*dr.x, 0.5f*dr.y, 0.5f*dr.z};
  	    for (int k = 0; k < kernel.ndim; k++) {
	      Qi += pmhd_grad[k][li] * ds[k];
	      Qj -= pmhd_grad[k][lj] * ds[k];
	    }

	    if (do_ppm) {

	      Qi +=  pmhd_cross[0][li]*ds[0]*ds[0] + pmhd_cross[1][li]*ds[1]*ds[1] + pmhd_cross[2][li]*ds[2]*ds[2] + 
		2.0*(pmhd_cross[3][li]*ds[0]*ds[1] + pmhd_cross[4][li]*ds[0]*ds[2] + pmhd_cross[5][li]*ds[1]*ds[2]); 

	      Qj +=  pmhd_cross[0][lj]*ds[0]*ds[0] + pmhd_cross[1][lj]*ds[1]*ds[1] + pmhd_cross[2][lj]*ds[2]*ds[2] + 
		2.0*(pmhd_cross[3][lj]*ds[0]*ds[1] + pmhd_cross[4][lj]*ds[0]*ds[2] + pmhd_cross[5][lj]*ds[1]*ds[2]);
	    
	    }

#if 1
	    monotonicity(Qi.dens,  Qj.dens,  mi.dens,  mj.dens);
	    monotonicity(Qi.ethm,  Qj.ethm,  mi.ethm,  mj.ethm);
	    monotonicity(Qi.vel.x, Qj.vel.x, mi.vel.x, mj.vel.x);
	    monotonicity(Qi.vel.y, Qj.vel.y, mi.vel.y, mj.vel.y);
	    monotonicity(Qi.vel.z, Qj.vel.z, mi.vel.z, mj.vel.z);
	    monotonicity(Qi.B.x,   Qj.B.x,   mi.B.x,   mj.B.x);
	    monotonicity(Qi.B.y,   Qj.B.y,   mi.B.y,   mj.B.y);
	    monotonicity(Qi.B.z,   Qj.B.z,   mi.B.z,   mj.B.z);
	    monotonicity(Qi.psi,   Qj.psi,   mi.psi,   mj.psi);
#endif
#endif

	    const float3 Wij = {0.5f*(pi.vel.x + pj.vel.x),
				0.5f*(pi.vel.y + pj.vel.y),
				0.5f*(pi.vel.z + pj.vel.z)};
	    
	    const float3 Vij = {0.5f*(mi.vel.x + mj.vel.x),
				0.5f*(mi.vel.y + mj.vel.y),
				0.5f*(mi.vel.z + mj.vel.z)};
	    
	    const float3 Bij = {0.5f*(mi.B.x + mj.B.x),
				0.5f*(mi.B.y + mj.B.y),
				0.5f*(mi.B.z + mj.B.z)};
	    
	    const float  wij = std::sqrt(dwij[0]*dwij[0] + dwij[1]*dwij[1] + dwij[2]*dwij[2]);
	    const float iwij = (wij > 0.0f) ? 1.0f/wij : 0.0f;
	    
	    const float3 eij = {dwij[0] * iwij, dwij[1] * iwij, dwij[2] * iwij};
	    
  	    const ptcl_mhd Fij = solve_riemann(Wij, Bij, Vij, Qi, Qj, eij);
	    
 	    const float Bi[3] = {mi.B.x, mi.B.y, mi.B.z};
 	    const float Bj[3] = {mj.B.x, mj.B.y, mj.B.z};
	    
	    for (int k = 0; k < kernel.ndim; k++) {
  	      divB       += Bi[k]  * (Vi*dwi[k]) + Bj[k]  * (Vj*dwj[k]);
	      gradpsi[k] += mi.psi * (Vi*dwi[k]) + mj.psi * (Vj*dwj[k]);
	    }
 	    dQdt -= Fij * wij;
	    
	    // bgradv & bgradpsi
	    const float f = Vi*(Bi[0]*dwi[0] + Bi[1]*dwi[1] + Bi[2]*dwi[2]);
	    bgradv.x += f * (mj.vel.x - mi.vel.x);
	    bgradv.y += f * (mj.vel.y - mi.vel.y);
	    bgradv.z += f * (mj.vel.z - mi.vel.z);
	    bgradpsi += f * (mj.psi   - mi.psi);
	    
	    // compute dwdt
#if !((defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_))
	    const float qij = s*inv_hi;
	    const float dwk = kernel.dw(qij);
	    const float3 dv = {pj.vel.x - pi.vel.x,
			       pj.vel.y - pi.vel.y,
			       pj.vel.z - pi.vel.z};
	    const float drdv = dr.x*dv.x + dr.y*dv.y + dr.z*dv.z;
	    
	    const float is2  = (s2 > 0.0f) ? 1.0f/s2 : 0.0f;
	    
	    omega += qij * dwk;
	    dnidt += qij * dwk * drdv*is2;
#endif

	  }
	  
	}
      }
	
      divB_i[pi.local_idx] = divB/pi.wght;

#if 0
      divB = 0;      // force divB = 0;
#endif
      
#if 1
      // compute dQdt.psi

      const float pres = compute_pressure(mi.dens, mi.ethm);
      const float ch   = 0.5f*std::sqrt((gamma_gas*pres + 0.5f*(sqr(mi.B.x)+sqr(mi.B.y)+sqr(mi.B.z)))/mi.dens);
      float L = 0;
      switch(kernel.ndim) {
      case 1:
	L = pi.wght;
	break;
      case 2:
	L = std::sqrt(pi.wght/M_PI);
	break;
      case 3:
	L = powf(pi.wght*3.0f/4.0f/M_PI, 1.0f/3.0f);
	break;
      default:
	assert(kernel.ndim > 0 && kernel.ndim <= 3);
      };


      
      const float tau  = L/ch;
      const float mass = mi.dens*weights[pi.local_idx];
      dQdt.psi  -= mass*(sqr(ch)*divB/weights[pi.local_idx] + mi.psi/tau);
      
      dQdt.ethm -= bgradpsi;
      if (kernel.ndim >= 1) dQdt.B.x -= gradpsi[0];
      if (kernel.ndim >= 2) dQdt.B.y -= gradpsi[1];
      if (kernel.ndim >= 3) dQdt.B.z -= gradpsi[2];
     
#else
      dQdt.psi = 0;
#endif 

      dQdt.B.x  += bgradv.x;
      dQdt.B.y  += bgradv.y;
      dQdt.B.z  += bgradv.z;

      
      // compute dQdt in primitives
      
      local_idx[iloc] = pi.local_idx;
      ptcl_mhd &dot = local_dot[iloc++];

#if !((defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_))
      if (eulerian_mode) assert(dwdt == 0.0f);

      // compute dwdt

      const float ni   = 1.0f/wi;
      const float dhdn = -(hi/ni)/kernel.ndim;

      omega  = 1.0f + dhdn*inv_hi*(kernel.ndim*ni + omega*inv_hidim);
      omega /= inv_hidim;
      
      dnidt  = dnidt /omega;
      
      const float dwdt = -dnidt /sqr(ni);
      dwdt_i[pi.local_idx] = dwdt;


      const float imass = 1.0f/(wi*mi.dens);
      const float iwght = 1.0f/ wi;
      dot.dens  = (dQdt.dens - dwdt * mi.dens ) * iwght;
      
      dot.B.x  = (dQdt.B.x - dwdt * mi.B.x) * iwght;
      dot.B.y  = (dQdt.B.y - dwdt * mi.B.y) * iwght;
      dot.B.z  = (dQdt.B.z - dwdt * mi.B.z) * iwght;
      
      // subtract contribution from thermal and magnetic energy
      
//       const float uB = mi.vel.x*mi.B.x + mi.vel.y*mi.B.y + mi.vel.z*mi.B.z;
//       dQdt.ethm  -= uB * divB;
      
      dot.ethm = dQdt.ethm 
	- (mi.vel.x*dQdt.vel.x + mi.vel.y*dQdt.vel.y + mi.vel.z*dQdt.vel.z)
	+ (sqr(mi.vel.x) + sqr(mi.vel.y) + sqr(mi.vel.z))*dQdt.dens*0.5f
	- (mi.B.x*dQdt.B.x + mi.B.y*dQdt.B.y + mi.B.z*dQdt.B.z)
	+ (sqr(mi.B.x) + sqr(mi.B.y) + sqr(mi.B.z))*dwdt*0.5f;
      
      dot.ethm  = (dot.ethm - dwdt * mi.ethm) * iwght;

      dQdt.vel.x -= mi.B.x*divB;
      dQdt.vel.y -= mi.B.y*divB;
      dQdt.vel.z -= mi.B.z*divB;
      
      dot.vel.x = (dQdt.vel.x - dQdt.dens * mi.vel.x) * imass;
      dot.vel.y = (dQdt.vel.y - dQdt.dens * mi.vel.y) * imass;
      dot.vel.z = (dQdt.vel.z - dQdt.dens * mi.vel.z) * imass;
      dot.psi   = (dQdt.psi   - dQdt.dens * mi.psi  ) * imass;

      
#elif defined _CONSERVATIVE_

      const float uB = mi.vel.x*mi.B.x + mi.vel.y*mi.B.y + mi.vel.z*mi.B.z;
      dQdt.ethm  -= uB * divB;
      
      dQdt.vel.x -= mi.B.x*divB;
      dQdt.vel.y -= mi.B.y*divB;
      dQdt.vel.z -= mi.B.z*divB;

      dot = dQdt;
      
#elif _SEMICONSERVATIVE_

      dot = dQdt;

      const real  m  = mi.dens   * pi.wght;
      const real3 mu = {mi.vel.x*m, 
			mi.vel.y*m, 
			mi.vel.z*m};
      

      dot.ethm = dQdt.ethm 
	- (mu.x*dQdt.vel.x + mu.y*dQdt.vel.y + mu.z*dQdt.vel.z)/m
	+ (sqr(mu.x) + sqr(mu.y) + sqr(mu.z))/sqr(m) * dQdt.dens*0.5f;

      dot.vel.x -= mi.B.x*divB;
      dot.vel.y -= mi.B.y*divB;
      dot.vel.z -= mi.B.z*divB;
      
#endif
    }
  }
  assert(iloc == local_n);

  for (int i = 0; i < local_n; i++) {
    pmhd_dot[local_idx[i]] = local_dot[i];
  }

  t_compute += get_time() - t0;
}
