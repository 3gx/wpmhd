#include "gn.h"
#include "myMPI.h"

inline real slope_limiter(const real vi,
			  const real vi_min,
			  const real vi_max,
			  const real vmin,
			  const real vmax,
			  const real psi = 0.5) {
  
  const real xmax = (vmax - vi)/(vi_max - vi + TINY);
  const real xmin = (vi - vmin)/(vi - vi_min + TINY);
  
  return std::min((real)1.0, psi*std::min(xmin, xmax));
}

void system::gradient() {
  MPI_Barrier(MPI_COMM_WORLD);
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
      
      float3 ds  = {0.0f,0.0f,0.0f};

      real  sum = 0.0f;
      
      ptcl_mhd grad0[3];
      grad0[0].set(0);
      grad0[1].set(0);
      grad0[2].set(0);
      
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
      
      
      std::vector<ptcl_mhd> jlist_pmhd;
      std::vector<float4  > jlist_dr;
      jlist_pmhd.reserve(256);
      jlist_dr.reserve(256);
      
      for (int leaf = 0; leaf < n_leaves; leaf++) {
 	const octnode &jnode = *ileaf_list[leaf];
	
 	if (!bi.overlap(jnode.inner)) continue;
	
 	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  const particle &pj  = *jbp->pp;
	  const pfloat3 jpos  = pj.pos; 
	  const float3 dr = {jpos.x - ipos.x,
			     jpos.y - ipos.y,
			     jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  
    	  if (s2 <= hi2) {
	    jlist_dr.push_back  (float4(dr.x, dr.y, dr.z, sqrt(s2)));
	    jlist_pmhd.push_back(pmhd[pi.local_idx]);
 	  }
 	}
      }
      
      ////////      

      const int nj = jlist_dr.size();
      for (int j = 0; j < nj; j++) {
	const ptcl_mhd mj = jlist_pmhd[j];
	const float4   dr = jlist_dr[j];
	const float     s = dr.w;
	
	const float w  = kernel.w(s*inv_hi);
	const float wk = w * inv_hidim * wi;
	
	
	const float drr[3] = {wk*dr.x, wk*dr.y, wk*dr.z};
	const ptcl_mhd dp  = mj - mi;
	for (int k = 0; k < kernel.ndim; k++) {
	  grad0[k] += dp * drr[k];
	}
	
	pmin = min(pmin, mj);
	pmax = max(pmax, mj);
	
	const float Vj = pi.wght * (w * inv_hidim);
	
	sum  += Vj;
	ds.x += Vj * dr.x;
	ds.y += Vj * dr.y;
	ds.z += Vj * dr.z;
	
	
      }
      
      const ptcl_mhd grad[3] = {Axx*grad0[0] + Axy*grad0[1] + Axz*grad0[2],
				Axy*grad0[0] + Ayy*grad0[1] + Ayz*grad0[2],
				Axz*grad0[0] + Ayz*grad0[1] + Azz*grad0[2]};
      
      // ****** advection velocity *********
      
      sum = 1.0;
      const float f   = 0.5f;
      const float idt = (dt_global > 0.0) ? f/sum/dt_global : 0.0f;
      
      pi.vel = (float3){mi.vel.x - ds.x*idt,
			mi.vel.y - ds.y*idt,
			mi.vel.z - ds.z*idt};
      

      /////// limit
      
      ptcl_mhd psi;
      psi.set(1.0f);

      ptcl_mhd pi_min, pi_max;
      pi_min.set(+HUGE);
      pi_max.set(-HUGE);
      
      for (int j = 0; j < nj; j++) {
	const float4   dr = jlist_dr[j];
	const ptcl_mhd grad1[3] = {grad[0]*(0.5f*dr.x),
				   grad[1]*(0.5f*dr.y),
				   grad[2]*(0.5f*dr.z)};
 	const ptcl_mhd pij = mi + grad1[0] + grad1[1] + grad1[2];
	
 	pi_max = max(pi_max, pij);
 	pi_min = min(pi_min, pij);
      }

      const real fc = 1.0;
      const real fn = 0.5;
      psi.dens  = slope_limiter(mi.dens,  pi_min.dens,  pi_max.dens,  pmin.dens,  pmax.dens,  fc);
      psi.ethm  = slope_limiter(mi.ethm,  pi_min.ethm,  pi_max.ethm,  pmin.ethm,  pmax.ethm,  fn);
      psi.vel.x = slope_limiter(mi.vel.x, pi_min.vel.x, pi_max.vel.x, pmin.vel.x, pmax.vel.x, fn);
      psi.vel.y = slope_limiter(mi.vel.y, pi_min.vel.y, pi_max.vel.y, pmin.vel.y, pmax.vel.y, fn);
      psi.vel.z = slope_limiter(mi.vel.z, pi_min.vel.z, pi_max.vel.z, pmin.vel.z, pmax.vel.z, fn);
      psi.B.x   = slope_limiter(mi.B.x,   pi_min.B.x,   pi_max.B.x,   pmin.B.x,   pmax.B.x,   fc);
      psi.B.y   = slope_limiter(mi.B.y,   pi_min.B.y,   pi_max.B.y,   pmin.B.y,   pmax.B.y,   fc);
      psi.B.z   = slope_limiter(mi.B.z,   pi_min.B.z,   pi_max.B.z,   pmin.B.z,   pmax.B.z,   fc);
      psi.psi   = slope_limiter(mi.psi,   pi_min.psi,   pi_max.psi,   pmin.psi,   pmax.psi,   fn);

      for (int k = 0; k < kernel.ndim; k++) {
	pmhd_grad[k][lidx] = grad[k]*psi;
      }

      if (eulerian_mode) pi.vel = (float3){0.0f,0.0f,0.0f};

   }

  }
  
  
  MPI_Barrier(MPI_COMM_WORLD);
  t_grad = get_time() - t0;
  t_compute += get_time() - t0;

  
  import_boundary_pmhd_grad();
  import_boundary_pvel();

}
