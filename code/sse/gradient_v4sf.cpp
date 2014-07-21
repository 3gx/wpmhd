#include "gn.h"
#include "myMPI.h"

inline float slope_limiter_float(const float vi,
			       const float vi_min,
			       const float vi_max,
			       const float vmin,
			       const float vmax,
			       const float psi = 0.5f) {
  
  const float xmax = (vmax - vi)/(vi_max - vi + TINY);
  const float xmin = (vi - vmin)/(vi - vi_min + TINY);
  
  return std::min((float)1.0, psi*std::min(xmin, xmax));
}


void system::gradient_v4sf() {
  double t0 = get_time();
  const int n_groups = group_list.size();

  pmhd_grad[0].resize(local_n);
  pmhd_grad[1].resize(local_n);
  pmhd_grad[2].resize(local_n);
  
  Bxx.resize(local_n);
  Bxy.resize(local_n);
  Bxz.resize(local_n);
  Byy.resize(local_n);
  Byz.resize(local_n);
  Bzz.resize(local_n);

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

      ptcl_mhd4 v4sf_pmin(mi,mi,mi,mi);
      ptcl_mhd4 v4sf_pmax(mi,mi,mi,mi);

      const int lidx = pi.local_idx;
      assert(lidx < local_n);

      // create list of j-particles

      std::vector<ptcl_mhd> jlist_pmhd;
      std::vector<float4  > jlist_dr;
      jlist_pmhd.reserve(256);
      jlist_dr.reserve(256);

      for (int leaf = 0; leaf < n_leaves; leaf++) {
   	const octnode &jnode = *ileaf_list[leaf];
 	if (!bi.overlap(jnode.inner)) continue;
	
 	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  const particle &pj = *jbp->pp;
	  
	  const pfloat3 jpos = pj.pos; 
	  const float3 dr = {jpos.x - ipos.x,
			     jpos.y - ipos.y,
			     jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  if (s2 <= hi2) {
	    jlist_dr.push_back  (float4(dr.x, dr.y, dr.z, std::sqrt(s2)));
	    jlist_pmhd.push_back(pmhd[pj.local_idx]);
	  }
	}
      }

      
      //////// compute gradient & renormalisation matrix
      
      int nj = jlist_dr.size();
      nj = (((nj - 1) >> 2) + 1) << 2;
      jlist_pmhd.resize(nj);
      jlist_dr.resize(nj);
      
      const v4sf v4sf_inv_hi    = v4sf(inv_hi);
      const v4sf v4sf_inv_hidim = v4sf(inv_hidim);

      ptcl_mhd4 v4sf_grad[3];
      v4sf v4sf_sum;
      v4sf v4sf_ds[3];

      v4sf Exx(0.0f), Exy(0.0f), Exz(0.0f), Eyy(0.0f), Eyz(0.0f), Ezz(0.0f);

      for (int j = 0; j < nj; j += 4) {
	ptcl_mhd list_mj[4];
	float4   list_dr[4];
	for (int k = 0; k < 4; k++) {
	  list_mj[k] = jlist_pmhd[j + k];
	  list_dr[k] = jlist_dr  [j + k];
	}
	  
	const ptcl_mhd4 mj(list_mj[0],   list_mj[1],   list_mj[2],   list_mj[3]);
	const v4sf      dx(list_dr[0].x, list_dr[1].x, list_dr[2].x, list_dr[3].x);
	const v4sf      dy(list_dr[0].y, list_dr[1].y, list_dr[2].y, list_dr[3].y);
	const v4sf      dz(list_dr[0].z, list_dr[1].z, list_dr[2].z, list_dr[3].z);
	const v4sf       s(list_dr[0].w, list_dr[1].w, list_dr[2].w, list_dr[3].w);
	
	v4sf q = s*v4sf_inv_hi;
	const v4sf w  = v4sf(kernel.w(q[0]), kernel.w(q[1]), kernel.w(q[2]), kernel.w(q[3]));
	const v4sf wk = w*v4sf_inv_hidim*wi;
	
	const v4sf dr[3]    = {wk*dx, wk*dy, wk*dz};
	const ptcl_mhd4 dp  = mj - mi;
	for (int k = 0; k < kernel.ndim; k++) {
	  v4sf_grad[k] += dp * dr[k];
	}
	
	v4sf_pmin.min(mj);
	v4sf_pmax.max(mj);
	
	const v4sf Vj = wk;
	
	v4sf_sum  += Vj;
	for (int k = 0; k < 3; k++)
	  v4sf_ds[k] += Vj * dr[k];
	
	Exx += wk * dr[0]*dr[0];
	Exy += wk * dr[0]*dr[1];
	Exz += wk * dr[0]*dr[2];
	Eyy += wk * dr[1]*dr[1];
	Eyz += wk * dr[1]*dr[2];
	Ezz += wk * dr[2]*dr[2];
	    
      }


      // invert matrix
      
      float Axx = Exx.reduce();
      float Axy = Exy.reduce();
      float Ayy = Eyy.reduce();
      float Axz = Exz.reduce();
      float Ayz = Eyz.reduce();
      float Azz = Ezz.reduce();
      float det = 0.0f;

      switch(kernel.ndim) {
      case 1:
	Bxx[lidx] = 1.0f/Axx;
	Bxy[lidx] = 0.0f;
	Bxz[lidx] = 0.0f;
	Byy[lidx] = 0.0f;
	Byz[lidx] = 0.0f;
	Bzz[lidx] = 0.0f;
	break;
	
      case 2:
	det = Axx*Ayy - Axy*Axy;
	assert(det != 0.0f);
	det = 1.0f/det;
	
	Bxx[lidx] =  Ayy*det;
	Bxy[lidx] = -Axy*det;
	Bxz[lidx] =  0.0f;
	Byy[lidx] =  Axx*det;
	Byz[lidx] =  0.0f;
	Bzz[lidx] =  0.0f;
	break;
	
      default:
	det = -Axz*Axz*Ayy + 2*Axy*Axz*Ayz - Axx*Ayz*Ayz - Axy*Axy*Azz + Axx*Ayy*Azz;
	assert (det != 0.0f);
	det = 1.0/det;
	
	Bxx[lidx] = (-Ayz*Ayz + Ayy*Azz)*det;      
	Bxy[lidx] = (+Axz*Ayz - Axy*Azz)*det;      
	Bxz[lidx] = (-Axz*Ayy + Axy*Ayz)*det;      
	Byy[lidx] = (-Axz*Axz + Axx*Azz)*det;      
	Byz[lidx] = (+Axy*Axz - Axx*Ayz)*det;      
	Bzz[lidx] = (-Axy*Axy + Axx*Ayy)*det;      
	
      }

      // compute gradient

      Axx = Bxx[lidx];
      Axy = Bxy[lidx];
      Axz = Bxz[lidx];
      Ayy = Byy[lidx];
      Ayz = Byz[lidx];
      Azz = Bzz[lidx];

      const ptcl_mhd grad_tmp[3] = {v4sf_grad[0].reduce(),
				    v4sf_grad[1].reduce(),
				    v4sf_grad[2].reduce()};
      const ptcl_mhd grad[3] = {Axx*grad_tmp[0] + Axy*grad_tmp[1] + Axz*grad_tmp[2],
				Axy*grad_tmp[0] + Ayy*grad_tmp[1] + Ayz*grad_tmp[2],
				Axz*grad_tmp[0] + Ayz*grad_tmp[1] + Azz*grad_tmp[2]};

      for (int k = 0; k < 3; k++) 
	v4sf_grad[k].set(grad[k], grad[k], grad[k], grad[k]);
      

      // ****** advection velocity *********
      
      const float sum = v4sf_sum.reduce();
      const float f   = 0.5f;
      const float idt = (dt_global > (real)0.0) ? f/sum/dt_global : 0.0f;
      
      const float3 ds = {v4sf_ds[0].reduce(),
			 v4sf_ds[1].reduce(),
			 v4sf_ds[2].reduce()};
      pi.vel = (float3){mi.vel.x - ds.x*idt,
			mi.vel.y - ds.y*idt,
			mi.vel.z - ds.z*idt};
      

      /////// limit gradient
      
      ptcl_mhd4 v4sf_pi_min(mi,mi,mi,mi);
      ptcl_mhd4 v4sf_pi_max(mi,mi,mi,mi);
      
      for (int j = 0; j < nj; j += 4) {
	float4   list_dr[4];
	for (int k = 0; k < 4; k++) {
	  list_dr[k] = jlist_dr  [j + k];
	}

	const v4sf dr[3] = {v4sf(list_dr[0].x, list_dr[1].x, list_dr[2].x, list_dr[3].x),
			    v4sf(list_dr[0].y, list_dr[1].y, list_dr[2].y, list_dr[3].y),
			    v4sf(list_dr[0].z, list_dr[1].z, list_dr[2].z, list_dr[3].z)};
	
	ptcl_mhd4 v4sf_pij(mi, mi, mi, mi);
	for (int k = 0; k < kernel.ndim; k++)
	  v4sf_pij += v4sf_grad[k]*dr[k];
	
	v4sf_pi_max.max(v4sf_pij);
	v4sf_pi_min.min(v4sf_pij);
      }


      const ptcl_mhd pmax   = v4sf_pmax.reduce();
      const ptcl_mhd pmin   = v4sf_pmin.reduce();
      const ptcl_mhd pi_min = v4sf_pi_min.reduce();
      const ptcl_mhd pi_max = v4sf_pi_max.reduce();

      ptcl_mhd psi;
      psi.set(1.0);
      
      const float fc = 1.0f;
      const float fn = 0.5f;
      psi.dens  = slope_limiter_float(mi.dens,  pi_min.dens,  pi_max.dens,  pmin.dens,  pmax.dens,  fc);
      psi.ethm  = slope_limiter_float(mi.ethm,  pi_min.ethm,  pi_max.ethm,  pmin.ethm,  pmax.ethm,  fn);
      psi.vel.x = slope_limiter_float(mi.vel.x, pi_min.vel.x, pi_max.vel.x, pmin.vel.x, pmax.vel.x, fn);
      psi.vel.y = slope_limiter_float(mi.vel.y, pi_min.vel.y, pi_max.vel.y, pmin.vel.y, pmax.vel.y, fn);
      psi.vel.z = slope_limiter_float(mi.vel.z, pi_min.vel.z, pi_max.vel.z, pmin.vel.z, pmax.vel.z, fn);
      psi.B.x   = slope_limiter_float(mi.B.x,   pi_min.B.x,   pi_max.B.x,   pmin.B.x,   pmax.B.x,   fc);
      psi.B.y   = slope_limiter_float(mi.B.y,   pi_min.B.y,   pi_max.B.y,   pmin.B.y,   pmax.B.y,   fc);
      psi.B.z   = slope_limiter_float(mi.B.z,   pi_min.B.z,   pi_max.B.z,   pmin.B.z,   pmax.B.z,   fc);
      psi.psi   = slope_limiter_float(mi.psi,   pi_min.psi,   pi_max.psi,   pmin.psi,   pmax.psi,   fn);
      
      
      for (int k = 0; k < kernel.ndim; k++) {
	pmhd_grad[k][lidx] = grad[k]*psi;
      }


      if (eulerian_mode) pi.vel = (float3){0.0f,0.0f,0.0f};


   }

  }

  t_grad     = get_time() - t0;
  t_compute += get_time() - t0;

  
  import_boundary_Bmatrix();
  import_boundary_pmhd_grad();
  import_boundary_pvel();

}

