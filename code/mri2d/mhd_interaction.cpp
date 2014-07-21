#include "gn.h"

inline void  monotonicity(real &fL, real &fR, const real fi, const real fj) {
  const real Fmin = std::min(fi, fj);
  const real Fmax = std::max(fi, fj);
  if ((fL < Fmin) || (fL > Fmax) || (fR < Fmin) || (fR > Fmax)) {
//     fprintf(stderr, "fL= %g fR= %g  fi= %g fj= %g Fmin= %g Fmax= %g\n",
// 	    fL, fR, fi, fj, Fmin, Fmax);
//     assert(false);
    fL = fR = fi;
    fL = fi;
    fR = fj;
  }
}

void system::mhd_interaction() {
  double t0 = get_time();
  
  const int n_leaves = local_tree.leaf_list.size();

  pmhd_dot.resize(pmhd.size());

  dwdt_i.resize(pmhd.size());
  divB_i.resize(pmhd.size());
  psi_tau.resize(pmhd.size());
  
  for (int leaf = 0; leaf < n_leaves; leaf++) {
    const       octnode<TREE_NLEAF>   &inode      = *local_tree.leaf_list[leaf];
    std::vector<octnode<TREE_NLEAF>*> &ileaf_list = ngb_leaf_list_outer[leaf];
    const int n_leaves  = ileaf_list.size();
    
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (ibp->isexternal()) continue;
      const particle &pi = *ibp->pp;
      const ptcl_mhd &mi = pmhd[pi.local_idx];
      
      const float Vi = pi.wght;
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
      
      float omega = 0.0f;
      float dnidt = 0.0f;
      for (int leaf = 0; leaf < n_leaves; leaf++) {
 	const octnode<TREE_NLEAF> &jnode = *ileaf_list[leaf];
	
  	if (!bi.overlap(jnode.outer)) continue;
	
	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  const particle &pj = *jbp->pp;
	  const ptcl_mhd &mj = pmhd[pj.local_idx];
	  
	  const pfloat3 jpos = pj.pos; 
	  const float3  dr   = {jpos.x - ipos.x,
				jpos.y - ipos.y,
				jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);

	  const float dx_abs = jpos.x.getu() - ipos.x.getu();
	  const int   shear  =  (dx_abs > 0.0f ? -2 : +2) *
	    (int)(std::abs(dx_abs) > global_domain.hsize.x.getu());
	  
 	  const float hj  = pj.h;
 	  const float hj2 = hj*hj;

  	  if (s2 <= std::max(hi2, hj2))  {

	    const int li = pi.local_idx;
	    const int lj = pj.local_idx;
	    
	    const float inv_hj    = 1.0f/hj;
	    const float inv_hjdim = kernel.pow(inv_hj);
	    
	    const float wj  = pj.wght;
	    
	    const float s   = sqrt(s2);
	    const float wki = kernel.w(s*inv_hi)*inv_hidim * wi;
	    const float wkj = kernel.w(s*inv_hj)*inv_hjdim * wj;
	    
	    const float dwi[3] = {wki * (Bxx[li]*dr.x + Bxy[li]*dr.y + Bxz[li]*dr.z),
				  wki * (Bxy[li]*dr.x + Byy[li]*dr.y + Byz[li]*dr.z),
				  wki * (Bxz[li]*dr.x + Byz[li]*dr.y + Bzz[li]*dr.z)};
	    
	    const float dwj[3] = {wkj * (Bxx[lj]*dr.x + Bxy[lj]*dr.y + Bxz[lj]*dr.z),
				  wkj * (Bxy[lj]*dr.x + Byy[lj]*dr.y + Byz[lj]*dr.z),
				  wkj * (Bxz[lj]*dr.x + Byz[lj]*dr.y + Bzz[lj]*dr.z)};
	    
	    const float Vj = pj.wght;
	    const float dwij[3] = {Vi*dwi[0] + Vj*dwj[0],
				   Vi*dwi[1] + Vj*dwj[1],
				   Vi*dwi[2] + Vj*dwj[2]};
	    
	    ptcl_mhd Qi = mi;
	    ptcl_mhd Qj = mj;


	    particle Pj = pj;
	    ptcl_mhd Mj = mj;
	    Mj.vel.z -= shear*qOmega*Omega*global_domain.hsize.x.getu();
	    Pj.vel.z -= shear*qOmega*Omega*global_domain.hsize.x.getu();
	    Qj = Mj;
	    

	    if (!do_first_order) {
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
	    monotonicity(Qi.vel.x, Qj.vel.x, mi.vel.x, Mj.vel.x);
	    monotonicity(Qi.vel.y, Qj.vel.y, mi.vel.y, Mj.vel.y);
	    monotonicity(Qi.vel.z, Qj.vel.z, mi.vel.z, Mj.vel.z);
	    monotonicity(Qi.B.x,   Qj.B.x,   mi.B.x,   mj.B.x);
	    monotonicity(Qi.B.y,   Qj.B.y,   mi.B.y,   mj.B.y);
	    monotonicity(Qi.B.z,   Qj.B.z,   mi.B.z,   mj.B.z);
	    monotonicity(Qi.psi,   Qj.psi,   mi.psi,   mj.psi);
#endif
#endif
	    }

	    const float3 Wij = {0.5f*(pi.vel.x + Pj.vel.x),
				0.5f*(pi.vel.y + Pj.vel.y),
				0.5f*(pi.vel.z + Pj.vel.z)};
	    
	    const float3 Vij = {0.5f*(mi.vel.x + Mj.vel.x),
				0.5f*(mi.vel.y + Mj.vel.y),
				0.5f*(mi.vel.z + Mj.vel.z)};
	    
	    const float3 Bij = {0.5f*(mi.B.x + mj.B.x),
				0.5f*(mi.B.y + mj.B.y),
				0.5f*(mi.B.z + mj.B.z)};
	    
	    const float  wij = sqrt(dwij[0]*dwij[0] + dwij[1]*dwij[1] + dwij[2]*dwij[2]);
	    const float iwij = (wij > 0.0f) ? 1.0f/wij : 0.0f;
	    
	    const float3 eij = {dwij[0] * iwij, dwij[1] * iwij, dwij[2] * iwij};
	    
	    float Bm, psiM;
  	    const ptcl_mhd Fij = solve_riemann(Wij, Bij, Vij, Qi, Qj, eij, Bm, psiM); 
	    
	    bgradpsi += psiM * (mi.B.x*dwij[0] + mi.B.y*dwij[1] + mi.B.z*dwij[2]);
	    divB += Bm * wij;
 	    dQdt -= Fij * wij;
	    gradpsi[0] += psiM * dwij[0];
	    gradpsi[1] += psiM * dwij[1];
	    gradpsi[2] += psiM * dwij[2];
	    
	    
	    // compute dwdt
	    
	    const float qij = s*inv_hi;
	    const float dwk = kernel.dw(qij);
	    const float3 dv = {Pj.vel.x - pi.vel.x,
			       Pj.vel.y - pi.vel.y,
			       Pj.vel.z - pi.vel.z};
	    const float drdv = dr.x*dv.x + dr.y*dv.y + dr.z*dv.z;
	    
	    const float is2  = (s2 > 0.0f) ? 1.0f/s2 : 0.0f;
	    
	    omega += qij * dwk;
	    dnidt += qij * dwk * drdv*is2;

	  }
	  
	}
      }
	
      // compute dwdt

      const float ni   = 1.0f/wi;
      const float dhdn = -(hi/ni)/kernel.ndim;
      
      omega  = 1.0f + dhdn*inv_hi*(kernel.ndim*ni + omega*inv_hidim);
      omega /= inv_hidim;
      
      dnidt  = dnidt /omega;
      
      const float dwdt = -dnidt /sqr(ni);

      if (eulerian_mode) assert(dwdt == 0.0f);

      /////

      dwdt_i[pi.local_idx] = dwdt;
      divB_i[pi.local_idx] = divB/pi.wght;

      dQdt.wB.x -= gradpsi[0];
      dQdt.wB.y -= gradpsi[1];
      dQdt.wB.z -= gradpsi[2];
//       dQdt.psi  += mi.psi * dwdt;

      dQdt.etot -= bgradpsi;
      const float pres = compute_pressure(mi.dens, mi.ethm);
      const float   B2 = sqr(mi.B.x) + sqr(mi.B.y) + sqr(mi.B.z);
      const float  cs = sqrt((gamma_gas*pres + B2)/mi.dens);
      
      dQdt.psi  -= 0.5f*sqr(cs)*divB*mi.dens;

      dQdt.wB.x -= mi.vel.x*divB;
      dQdt.wB.y -= mi.vel.y*divB;
      dQdt.wB.z -= mi.vel.z*divB;
      
      const float uB = mi.vel.x*mi.B.x + mi.vel.y*mi.B.y + mi.vel.z*mi.B.z;
      dQdt.mom.x -= mi.B.x*divB;
      dQdt.mom.y -= mi.B.y*divB;
      dQdt.mom.z -= mi.B.z*divB;
      dQdt.etot  -= uB * divB;

      // compute dQdt in primitive      
      ptcl_mhd &dot = pmhd_dot[pi.local_idx];
      dot = dQdt;
      
#ifdef _CONSERVATIVE_
      const real  m  = mi.dens   * pi.wght;
      const float4 acc = body_forces(pi.local_idx);
      dot.mom.x += acc.x*m;
      dot.mom.y += acc.y*m;
      dot.mom.z += acc.z*m;

#elif defined _SEMICONSERVATIVE_
      const real  m  = mi.dens   * pi.wght;
      const real3 mu = {mi.vel.x*m, 
			mi.vel.y*m, 
			mi.vel.z*m};
      
      dot.ethm = dQdt.etot 
	- (mu.x*dQdt.mom.x + mu.y*dQdt.mom.y + mu.z*dQdt.mom.z)/m
	+ (sqr(mu.x) + sqr(mu.y) + sqr(mu.z))/sqr(m) * dQdt.mass*0.5f;

      
      const float4 acc = body_forces(pi.local_idx);
      dot.mom.x += acc.x*m;
      dot.mom.y += acc.y*m;
      dot.mom.z += acc.z*m;

#else

      const real  m  = mi.dens   * pi.wght;
      const real3 mu = {mi.vel.x*m, 
			mi.vel.y*m, 
			mi.vel.z*m};
      const real3 B = {mi.B.x * Vi,
		       mi.B.y * Vi,
		       mi.B.z * Vi};
      
      dot.ethm = dQdt.etot 
	- (mu.x*dQdt.mom.x + mu.y*dQdt.mom.y + mu.z*dQdt.mom.z)/m
	+ (sqr(mu.x) + sqr(mu.y) + sqr(mu.z))/sqr(m) * dQdt.mass*0.5f
        - (B.x*dQdt.B.x + B.y*dQdt.B.y + B.z*dQdt.B.z)/Vi
	+ (sqr(B.x) + sqr(B.y) + sqr(B.z))/sqr(Vi) * dwdt*0.5f;
      
      const float4 acc = body_forces(pi.local_idx);
      dot.mom.x += acc.x*m;
      dot.mom.y += acc.y*m;
      dot.mom.z += acc.z*m;


#endif
      
    }
  }

  t_compute += get_time() - t0;
}
