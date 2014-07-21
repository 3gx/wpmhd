#include "gn.h"
#include "myMPI.h"

inline real slope_limiter1(const real vi,
			   const real vi_min,
			   const real vi_max,
			   const real vmin,
			   const real vmax,
			   const real psi = 0.5) {
  
  const real xmax = (vmax - vi)/(vi_max - vi + TINY);
  const real xmin = (vi - vmin)/(vi - vi_min + TINY);
  
  return std::min((real)1.0, psi*std::min(xmin, xmax));
}

inline real slope_limiter(const real psi, 
			  const real vij, 
			  const real vi, 
			  const real vmin, 
			  const real vmax, const real f = 1) {
  real p = 1;
  if      (vij > 0) p = f*(vmax - vi)/vij;
  else if (vij < 0) p = f*(vmin - vi)/vij;
  
  return std::min(psi, p);
}



void system::gradient() {
  double t0 = get_time();
  const int n_groups = group_list.size();


  pmhd_grad[0].resize(local_n);
  pmhd_grad[1].resize(local_n);
  pmhd_grad[2].resize(local_n);

  
  for (int group = 0; group < n_groups; group++) {
    const octnode &inode              = *group_list  [group];
    const int n_leaves                = ngb_leaf_list[group].size();
    std::vector<octnode*> &ileaf_list = ngb_leaf_list[group];
    
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (ibp->isexternal()) continue;

      particle &pi = *ibp->pp;             // *non-const* since I change later pi.vel
      const ptcl_mhd &mi = pmhd[pi.local_idx];
      const boundary bi = boundary(pi.pos, pi.h);
      
      const pfloat3 ipos    = pi.pos;
      const float   hi      = pi.h;
      const float   wi      = pi.wght;
      
      const float   hi2     = hi*hi;
      const float inv_hi    = 1.0f/hi;
      const float inv_hidim = kernel.pow(inv_hi);
      
//       float3 vel = {0.0f,0.0f,0.0f};
      float3 ds  = {0.0f,0.0f,0.0f};
      float3 R   = {0.0f,0.0f,0.0f};
      float  ri  =  0.0f;
      float  Mi  = 0.0f;

      real  sum = 0.0f;
      
      ptcl_mhd grad[3];
      grad[0].set(0);
      grad[1].set(0);
      grad[2].set(0);
      
      ptcl_mhd pmin, pmax;
      pmin.set(+HUGE);
      pmax.set(-HUGE);

      const int lidx = pi.local_idx;
      assert(lidx < local_n);
      
      const real Axx = Bxx[lidx];
      const real Axy = Bxy[lidx];
      const real Axz = Bxz[lidx];
      const real Ayy = Byy[lidx];
      const real Ayz = Byz[lidx];
      const real Azz = Bzz[lidx];
      
      for (int leaf = 0; leaf < n_leaves; leaf++) {
   	const octnode &jnode = *ileaf_list[leaf];

 	if (!boundary(bi).overlap(jnode.inner)) continue;

	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
 	  const particle &pj     = *jbp->pp;
	  const ptcl_mhd &mj     = pmhd[pj.local_idx];

	  const pfloat3 jpos = pj.pos; 
	  const float3 dr = {jpos.x - ipos.x,
			     jpos.y - ipos.y,
			     jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  
//  	  float    hj  = pj.h;
//  	  float    hj2 = hj*hj;
//  	  if (s2 <= hi2 && s2 <= hj2) {
   	  if (s2 <= hi2) {
//  	  if (true) {
	    const float s = sqrt(s2);
	    const real w  = kernel.w(s*inv_hi);
#if 0
	    const real wj = pj.wght;
	    const real wk = wj * w*inv_hidim * wi;
#else
	    const real wk = w * inv_hidim * wi;
#endif


	    const real dw[3] = {wk * (Axx*dr.x + Axy*dr.y + Axz*dr.z),
				wk * (Axy*dr.x + Ayy*dr.y + Ayz*dr.z),
				wk * (Axz*dr.x + Ayz*dr.y + Azz*dr.z)};
	    
	    const ptcl_mhd dp = mj - mi;
	    for (int k = 0; k < kernel.ndim; k++) {
	      grad[k] += dp * dw[k];
	    }
	    
	    pmin = min(pmin, mj);
	    pmax = max(pmax, mj);

#if 0
	    float Vj = pj.wght * w * inv_hidim;
#else
	    float Vj = pi.wght * (w * inv_hidim);
#endif

  	    sum  += Vj;
	    ds.x += Vj * (dr.x + 0*(mj.vel.x - mi.vel.x)*dt_global);
	    ds.y += Vj * (dr.y + 0*(mj.vel.y - mi.vel.y)*dt_global);
	    ds.z += Vj * (dr.z + 0*(mj.vel.z - mi.vel.z)*dt_global);

// 	    ds.x += Vj * (mj.vel.x - mi.vel.x);
// 	    ds.y += Vj * (mj.vel.y - mi.vel.y);
// 	    ds.z += Vj * (mj.vel.z - mi.vel.z);
	    
	    
	    if (s > 0) {
	      Mi += wk;
	      ri += wk*s;
	    }
	    
	    
	  }
	  
	}
      }
      t_grad = get_time() - t0;
      
      // ****** advection velocity *********
      
      ri *= 1.0f/Mi;

      sum = 1.0;
      const float f   = 0.5f;
      const float idt = (dt_global > 0.0) ? f/sum/dt_global : 0.0f;
      
      pi.vel = (float3){mi.vel.x - ds.x*idt,
			mi.vel.y - ds.y*idt,
			mi.vel.z - ds.z*idt};

//       const float eps = 0.5f;
//       pi.vel = (float3){mi.vel.x + eps * ds.x,
// 			mi.vel.y + eps * ds.y,
// 			mi.vel.z + eps * ds.z};
      

      /////// limit
      
      ptcl_mhd psi;
      psi.set(1.0f);

      ptcl_mhd pi_min, pi_max;
      pi_min.set(+HUGE);
      pi_max.set(-HUGE);


      
      for (int leaf = 0; leaf < n_leaves; leaf++) {
 	const octnode &jnode = *ileaf_list[leaf];
	
 	if (!boundary(bi).overlap(jnode.inner)) continue;
	
	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  const particle        &pj     = *jbp->pp;
	  const pfloat3 jpos = pj.pos; 
	  const float3 dr = {jpos.x - ipos.x,
			     jpos.y - ipos.y,
			     jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  
//  	  float    hj  = pj.h;
//  	  float    hj2 = hj*hj;
//   	  if (s2 <= hi2 && s2 <= hj2) {
    	  if (s2 <= hi2) {
	    // 	  if (true) {
	    
	    const float f = 1.0f;
	    const ptcl_mhd pij = 0.5f*grad[0]*dr.x + 0.5f*grad[1]*dr.y + 0.5f*grad[2]*dr.z;

	    pi_max = max(pi_max, mi + abs(pij));
	    pi_min = min(pi_min, mi - abs(pij));

	    psi.dens  = slope_limiter(psi.dens,  pij.dens,  mi.dens,  pmin.dens,  pmax.dens,  f);
	    psi.ethm  = slope_limiter(psi.ethm,  pij.ethm,  mi.ethm,  pmin.ethm,  pmax.ethm,  f);
	    psi.vel.x = slope_limiter(psi.vel.x, pij.vel.x, mi.vel.x, pmin.vel.x, pmax.vel.x, f);
	    psi.vel.y = slope_limiter(psi.vel.y, pij.vel.y, mi.vel.y, pmin.vel.y, pmax.vel.y, f);
	    psi.vel.z = slope_limiter(psi.vel.z, pij.vel.z, mi.vel.z, pmin.vel.z, pmax.vel.z, f);
	    psi.B.x   = slope_limiter(psi.B.x,   pij.vel.x, mi.B.x,   pmin.B.x,   pmax.B.x,   f);
	    psi.B.y   = slope_limiter(psi.B.y,   pij.vel.y, mi.B.y,   pmin.B.y,   pmax.B.y,   f);
	    psi.B.z   = slope_limiter(psi.B.z,   pij.vel.z, mi.B.z,   pmin.B.z,   pmax.B.z,   f);
	    psi.psi   = slope_limiter(psi.psi,   pij.psi,   mi.psi,   pmin.psi,   pmax.psi,   f);

	    const float s = sqrt(s2);
	    if (s > 0.0f) {
	      const real w  = kernel.w(s*inv_hi);
	      const real wk = w * inv_hidim * wi;
	      R.x -= wk * dr.x/(s*sqr(s/ri));
	      R.y -= wk * dr.y/(s*sqr(s/ri));
	      R.z -= wk * dr.z/(s*sqr(s/ri));
	    }
	  }
	  
	  
	}
	
	
      }
      
      const real fc = 1.0;
      const real fn = 0.5;
      psi.dens  = slope_limiter1(mi.dens,  pi_min.dens,  pi_max.dens,  pmin.dens,  pmax.dens,  fc);
      psi.ethm  = slope_limiter1(mi.ethm,  pi_min.ethm,  pi_max.ethm,  pmin.ethm,  pmax.ethm,  fn);
      psi.vel.x = slope_limiter1(mi.vel.x, pi_min.vel.x, pi_max.vel.x, pmin.vel.x, pmax.vel.x, fn);
      psi.vel.y = slope_limiter1(mi.vel.y, pi_min.vel.y, pi_max.vel.y, pmin.vel.y, pmax.vel.y, fn);
      psi.vel.z = slope_limiter1(mi.vel.z, pi_min.vel.z, pi_max.vel.z, pmin.vel.z, pmax.vel.z, fn);
      psi.B.x   = slope_limiter1(mi.B.x,   pi_min.B.x,   pi_max.B.x,   pmin.B.x,   pmax.B.x,   fc);
      psi.B.y   = slope_limiter1(mi.B.y,   pi_min.B.y,   pi_max.B.y,   pmin.B.y,   pmax.B.y,   fc);
      psi.B.z   = slope_limiter1(mi.B.z,   pi_min.B.z,   pi_max.B.z,   pmin.B.z,   pmax.B.z,   fc);
      psi.psi   = slope_limiter1(mi.psi,   pi_min.psi,   pi_max.psi,   pmin.psi,   pmax.psi,   fn);
      
      
      for (int k = 0; k < kernel.ndim; k++) {
	pmhd_grad[k][lidx] = grad[k]*psi;
      }


#if 0
      const float C      = 0.1f;
      const float inv_dt = (dt_global > 0.0) ? ri*C/Mi/dt_global : 0.0f;
      
      pi.vel = (float3){mi.vel.x + R.x*inv_dt,
			mi.vel.y + R.y*inv_dt,
			mi.vel.z + R.z*inv_dt};
#endif

      if (eulerian_mode) pi.vel = (float3){0.0f,0.0f,0.0f};

//       pi.vel.x *= 0.2;
//       pi.vel.y *= 0.2;
//       pi.vel.z *= 0.2;

   }

  }
  
  t_compute += get_time() - t0;

  
  import_boundary_pmhd_grad();
  import_boundary_pvel();

}
