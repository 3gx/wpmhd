#include "gn.h"
#include "myMPI.h"
#include "matrix.h"

static int n_failed_to_invert;

inline float slope_limiter(const float vi,
			   const float vi_min,
			   const float vi_max,
			   const float vmin,
			   const float vmax,
			   const float psi = 0.5f) {
  
  const float xmax = (vmax - vi)/(vi_max - vi + TINY);
  const float xmin = (vi - vmin)/(vi - vi_min + TINY);
  
  return std::min((float)1.0, psi*std::min(xmin, xmax));
}

void system::gradient_ppm() {
  double t0 = get_time();
  const int n_leafs = local_tree.leaf_list.size();


  n_failed_to_invert = 0;

  pmhd_grad[0].resize(local_n);
  pmhd_grad[1].resize(local_n);
  pmhd_grad[2].resize(local_n);
  for (int i = 0; i < 6; i++)
    pmhd_cross[i].resize(local_n);
  
  
  for (int leaf = 0; leaf < n_leafs; leaf++) {
    const octnode<TREE_NLEAF> &inode = *local_tree.leaf_list[leaf];
    const int n_leaves               = ngb_leaf_list[leaf].size();
    const std::vector<octnode<TREE_NLEAF>*> &ileaf_list = ngb_leaf_list[leaf];
    
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
      
      ptcl_mhd grad[3], cross[6];
      grad[0].set(0);
      grad[1].set(0);
      grad[2].set(0);
      for (int ii = 0; ii < 6; ii++)
 	cross[ii].set(0.0);
      
      ptcl_mhd pmin, pmax;
      pmin = pmax = mi;
      
      const int lidx = pi.local_idx;
      assert(lidx < local_n);
      
      const real Axx = Bxx[lidx];
      const real Axy = Bxy[lidx];
      const real Axz = Bxz[lidx];
      const real Ayy = Byy[lidx];
      const real Ayz = Byz[lidx];
      const real Azz = Bzz[lidx];

      // create list of j-particles

      std::vector<ptcl_mhd> jlist_pmhd;
      std::vector<float4  > jlist_dr;
      jlist_pmhd.reserve(256);
      jlist_dr.reserve(256);

      for (int leaf = 0; leaf < n_leaves; leaf++) {
   	const octnode<TREE_NLEAF> &jnode = *ileaf_list[leaf];
 	if (!boundary(bi).overlap(jnode.inner)) continue;

 	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  const particle &pj = *jbp->pp;
	  
	  const pfloat3 jpos = pj.pos; 
	  const float3 dr = {jpos.x - ipos.x,
			     jpos.y - ipos.y,
			     jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  if (s2 <= hi2) {
	    jlist_dr.push_back  (float4(dr.x, dr.y, dr.z, sqrt(s2)));
	    jlist_pmhd.push_back(pmhd[pj.local_idx]);
	  }
	}
      }

      ////////


      matrix<real>          S(9,9);
      std::vector<ptcl_mhd> Q(9);     // Qx, Qy, Qz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz
      for (int ii = 0; ii < 9; ii++)  {
	Q[ii].set(0.0);
	for (int jj = 0; jj < 9; jj++) {
	  S(ii,jj) = 0;
	}
      }
      
      ///////////

      const int nj = jlist_dr.size();
      for (int j = 0; j < nj; j++) {
	const ptcl_mhd mj = jlist_pmhd[j];
	const float4   dr = jlist_dr[j];
	const float     s = dr.w;

	const real w  = kernel.w(s*inv_hi);
	const real wk = w * inv_hidim * wi;
	
	
	const real dw[3] = {wk * (Axx*dr.x + Axy*dr.y + Axz*dr.z),
			    wk * (Axy*dr.x + Ayy*dr.y + Ayz*dr.z),
			    wk * (Axz*dr.x + Ayz*dr.y + Azz*dr.z)};
	
	const ptcl_mhd dp = mj - mi;
	for (int k = 0; k < kernel.ndim; k++) {
	  grad[k] += dp * dw[k];
	}
	
	pmin.min(mj);
	pmax.max(mj);
	
	const float Vj = pi.wght * (w * inv_hidim);
	
	sum  += Vj;
	ds.x += Vj * dr.x;
	ds.y += Vj * dr.y;
	ds.z += Vj * dr.z;
	
	////////////  using MLP to compute derivatives ...
	
	// S_{\gamma\alpha}
	
	S(0,0) += w * dr.x*dr.x;       // S_11
	S(0,1) += w * dr.x*dr.y;       // S_12
	S(0,2) += w * dr.x*dr.z;       // S_13
	
	S(1,0) += w * dr.y*dr.x;       // S_21
	S(1,1) += w * dr.y*dr.y;       // S_22
	S(1,2) += w * dr.y*dr.z;       // S_23
	
	S(2,0) += w * dr.z*dr.x;       // S_31
	S(2,1) += w * dr.z*dr.y;       // S_32
	S(2,2) += w * dr.z*dr.z;       // S_33
	
	// S_{\gamma\delta\alpha}
	    
	S(3,0) += w * dr.x*dr.x*dr.x;      // S_111
	S(3,1) += w * dr.x*dr.x*dr.y;      // S_112
	S(3,2) += w * dr.x*dr.x*dr.z;      // S_113
	
	S(4,0) += w * dr.y*dr.y*dr.x;      // S_221
	S(4,1) += w * dr.y*dr.y*dr.y;      // S_222
	S(4,2) += w * dr.y*dr.y*dr.z;      // S_223
	
	S(5,0) += w * dr.z*dr.z*dr.x;      // S_331
	S(5,1) += w * dr.z*dr.z*dr.y;      // S_332
	S(5,2) += w * dr.z*dr.z*dr.z;      // S_333
	
	S(6,0) += w * dr.x*dr.y*dr.x;      // S_121
	S(6,1) += w * dr.x*dr.y*dr.y;      // S_122
	S(6,2) += w * dr.x*dr.y*dr.z;      // S_123
	
	S(7,0) += w * dr.x*dr.z*dr.x;      // S_131
	S(7,1) += w * dr.x*dr.z*dr.y;      // S_132
	S(7,2) += w * dr.x*dr.z*dr.z;      // S_133
	
	S(8,0) += w * dr.y*dr.z*dr.x;      // S_231
	S(8,1) += w * dr.y*dr.z*dr.y;      // S_232
	S(8,2) += w * dr.y*dr.z*dr.z;      // S_233
	
	// S_{\gamma\alpha\beta}
	
	S(0,3) += w * dr.x*dr.x*dr.x;      // S_111
	S(1,3) += w * dr.y*dr.x*dr.x;      // S_211
	S(2,3) += w * dr.z*dr.x*dr.x;      // S_311
	
	S(0,4) += w * dr.x*dr.y*dr.y;      // S_122
	S(1,4) += w * dr.y*dr.y*dr.y;      // S_222
	S(2,4) += w * dr.z*dr.y*dr.y;      // S_322
	
	S(0,5) += w * dr.x*dr.z*dr.z;      // S_133
	S(1,5) += w * dr.y*dr.z*dr.z;      // S_233
	S(2,5) += w * dr.z*dr.z*dr.z;      // S_333
	
	S(0,6) += w * dr.x*dr.x*dr.y;      // S_112
	S(1,6) += w * dr.y*dr.x*dr.y;      // S_212
	S(2,6) += w * dr.z*dr.x*dr.y;      // S_312
	
	S(0,7) += w * dr.x*dr.x*dr.z;      // S_113
	S(1,7) += w * dr.y*dr.x*dr.z;      // S_213
	S(2,7) += w * dr.z*dr.x*dr.z;      // S_313
	
	S(0,8) += w * dr.x*dr.y*dr.z;      // S_123
	S(1,8) += w * dr.y*dr.y*dr.z;      // S_223
	S(2,8) += w * dr.z*dr.y*dr.z;      // S_323
	
	// S_{\gamma\delta\alpha\beta}
	
	S(3,3) += w * dr.x*dr.x*dr.x*dr.x;      // S_1111
	S(3,4) += w * dr.x*dr.x*dr.y*dr.y;      // S_1122
	S(3,5) += w * dr.x*dr.x*dr.z*dr.z;      // S_1133
	S(3,6) += w * dr.x*dr.x*dr.x*dr.y;      // S_1112
	S(3,7) += w * dr.x*dr.x*dr.x*dr.z;      // S_1113
	S(3,8) += w * dr.x*dr.x*dr.y*dr.z;      // S_1123
	
	S(4,3) += w * dr.y*dr.y*dr.x*dr.x;      // S_2211
	S(4,4) += w * dr.y*dr.y*dr.y*dr.y;      // S_2222
	S(4,5) += w * dr.y*dr.y*dr.z*dr.z;      // S_2233
	S(4,6) += w * dr.y*dr.y*dr.x*dr.y;      // S_2212
	S(4,7) += w * dr.y*dr.y*dr.x*dr.z;      // S_2213
	S(4,8) += w * dr.y*dr.y*dr.y*dr.z;      // S_2223
	
	S(5,3) += w * dr.z*dr.z*dr.x*dr.x;      // S_3311
	S(5,4) += w * dr.z*dr.z*dr.y*dr.y;      // S_3322
	S(5,5) += w * dr.z*dr.z*dr.z*dr.z;      // S_3333
	S(5,6) += w * dr.z*dr.z*dr.x*dr.y;      // S_3312
	S(5,7) += w * dr.z*dr.z*dr.x*dr.z;      // S_3313
	S(5,8) += w * dr.z*dr.z*dr.y*dr.z;      // S_3323
	
	S(6,3) += w * dr.x*dr.y*dr.x*dr.x;      // S_1211
	S(6,4) += w * dr.x*dr.y*dr.y*dr.y;      // S_1222
	S(6,5) += w * dr.x*dr.y*dr.z*dr.z;      // S_1233
	S(6,6) += w * dr.x*dr.y*dr.x*dr.y;      // S_1212
	S(6,7) += w * dr.x*dr.y*dr.x*dr.z;      // S_1213
	S(6,8) += w * dr.x*dr.y*dr.y*dr.z;      // S_1223
	
	S(7,3) += w * dr.x*dr.z*dr.x*dr.x;      // S_1311
	S(7,4) += w * dr.x*dr.z*dr.y*dr.y;      // S_1322
	S(7,5) += w * dr.x*dr.z*dr.z*dr.z;      // S_1333
	S(7,6) += w * dr.x*dr.z*dr.x*dr.y;      // S_1312
	S(7,7) += w * dr.x*dr.z*dr.x*dr.z;      // S_1313
	S(7,8) += w * dr.x*dr.z*dr.y*dr.z;      // S_1323
	
	S(8,3) += w * dr.y*dr.z*dr.x*dr.x;      // S_2311
	S(8,4) += w * dr.y*dr.z*dr.y*dr.y;      // S_2322
	S(8,5) += w * dr.y*dr.z*dr.z*dr.z;      // S_2333
	S(8,6) += w * dr.y*dr.z*dr.x*dr.y;      // S_2312
	S(8,7) += w * dr.y*dr.z*dr.x*dr.z;      // S_2313
	S(8,8) += w * dr.y*dr.z*dr.y*dr.z;      // S_2323
	
	/////////
	
	Q[0] += w * dp * dr.x;          /*Q1*/		
	Q[1] += w * dp * dr.y;          /*Q2*/		
	Q[2] += w * dp * dr.z;          /*Q3*/		
	Q[3] += w * dp * dr.x*dr.x;     /*Q11*/		
	Q[4] += w * dp * dr.y*dr.y;     /*Q22*/		
	Q[5] += w * dp * dr.z*dr.z;     /*Q33*/		
	Q[6] += w * dp * dr.x*dr.y;     /*Q12*/		
	Q[7] += w * dp * dr.x*dr.z;     /*Q13*/		
	Q[8] += w * dp * dr.y*dr.z;     /*Q23*/		
	
	
      }
      
      for (int k = 0; k < 9; k++) {
	S(k,6) *= 2.0;
	S(k,7) *= 2.0;
	S(k,8) *= 2.0;
      }
      
      ptcl_mhd ds_extr[3];
      ds_extr[0].set(0.0);
      ds_extr[1].set(0.0);
      ds_extr[2].set(0.0);
      
      // compute gradients ...

      if (kernel.ndim == 1) {
	assert(kernel.ndim >= 2 && kernel.ndim <= 3);
      } if (kernel.ndim == 2) {
	matrix<real> S2(5,5);
	std::vector<ptcl_mhd> Q2(5);
	
	Q2[0] = Q[0];
	Q2[1] = Q[1];
	Q2[2] = Q[3];
	Q2[3] = Q[4];
	Q2[4] = Q[6];
	
	// S_{\gamma\alpha}
	
	S2(0,0) = S(0,0);
	S2(0,1) = S(0,1);
	S2(1,0) = S(1,0);
	S2(1,1) = S(1,1);
	
	// S_{\gamma\delta\alpha}
	
	S2(2,0) = S(3,0);
	S2(2,1) = S(3,1);
	S2(3,0) = S(4,0);
	S2(3,1) = S(4,1);
	S2(4,0) = S(6,0);
	S2(4,1) = S(6,1);
	
	// S_{\gamma\alpha\beta}
	
	S2(0,2) = S(0,3);
	S2(1,2) = S(1,3);
	S2(0,3) = S(0,4);
	S2(1,3) = S(1,4);
	S2(0,4) = S(0,6);
	S2(1,4) = S(1,6);
	
	// S_{\gamma\delta\alpha\beta}
	
	S2(2,2) = S(3,3);
	S2(2,3) = S(3,4);
	S2(2,4) = S(3,6);
	
	S2(3,2) = S(4,3);
	S2(3,3) = S(4,4);
	S2(3,4) = S(4,6);
	
	S2(4,2) = S(6,3);
	S2(4,3) = S(6,4);
	S2(4,4) = S(6,6);
	
	if (S2.invert()) {
	  
	  for (int ii = 0; ii < 2; ii++) {
	    grad[ii].set(0.0);
	    for (int jj = 0; jj < 5; jj++) {
	      grad[ii] += S2(ii,jj)*Q2[jj];
	    }
	  }
	  grad[2].set(0.0);
	  
	  for (int ii = 0; ii < 3; ii++){
	    cross[ii].set(0.0);
	    for (int jj = 0; jj < 5; jj++) {
	      cross[ii] += S2(ii+2,jj)*Q2[jj];
	    }
	  }
	  
	  // 	0   1   2   3   4   5
	  // 	xx, yy, zz, xy, xz, yz
	  cross[3] = cross[2];
	  cross[2].set(0.0);
	  cross[4].set(0.0);
	  cross[5].set(0.0);
	} else {
	  n_failed_to_invert++;
	}
	
      } else if (kernel.ndim == 3) {
	if (S.invert()) {
	  
	  for (int ii = 0; ii < 3; ii++) {
	    grad[ii].set(0.0);
	    for (int jj = 0; jj < 9; jj++) {
	      grad[ii] += S(ii,jj)*Q[jj];
	    }
	  }
	  
	  for (int ii = 0; ii < 6; ii++){
	    cross[ii].set(0.0);
	    for (int jj = 0; jj < 9; jj++) {
	      cross[ii] += S(ii+3,jj)*Q[jj];
	    }
	  }
	} else {
	  n_failed_to_invert += 1;
	}
	
	
      } else {
	assert(kernel.ndim >= 1 && kernel.ndim <= 3);
      }

      
      // ****** advection velocity *********

//       sum = 1.0;
      float f   = 0.5f;
      if (gravity_mass > 0.0f) f = 0.0f;
      const float idt = (dt_global > 0.0) ? f/sum/dt_global : 0.0f;
      
      pi.vel = (float3){mi.vel.x - ds.x*idt,
			mi.vel.y - ds.y*idt,
			mi.vel.z - ds.z*idt};
      
      /////// limit
      
      ptcl_mhd psi;
      psi.set(1.0f);
      
      ptcl_mhd pi_min, pi_max;
      pi_min = pi_max = mi;
      
      for (int j = 0; j < nj; j++) {
	const float4   dr = jlist_dr[j];
	const float3   ds = {0.5f*dr.x , 0.5f*dr.y, 0.5f*dr.z};
 	const ptcl_mhd pij = mi + 
	  grad [0]*ds.x      + grad [1]*ds.y      + grad [2]*ds.z + 
	  cross[0]*ds.x*ds.x + cross[1]*ds.y*ds.y + cross[2]*ds.z*ds.z + 
	  2.0*(cross[3]*ds.x*ds.y + cross[4]*ds.x*ds.z + cross[5]*ds.y*ds.z); 
	
	
 	pi_max.max(pij);
 	pi_min.min(pij);
      }
    
      
      const float fc = 1.0f;
      const float fn = 0.5f;
      psi.dens  = slope_limiter(mi.dens,  pi_min.dens,  pi_max.dens,  pmin.dens,  pmax.dens,  fc);
      psi.ethm  = slope_limiter(mi.ethm,  pi_min.ethm,  pi_max.ethm,  pmin.ethm,  pmax.ethm,  fn);
      psi.vel.x = slope_limiter(mi.vel.x, pi_min.vel.x, pi_max.vel.x, pmin.vel.x, pmax.vel.x, fn);
      psi.vel.y = slope_limiter(mi.vel.y, pi_min.vel.y, pi_max.vel.y, pmin.vel.y, pmax.vel.y, fn);
      psi.vel.z = slope_limiter(mi.vel.z, pi_min.vel.z, pi_max.vel.z, pmin.vel.z, pmax.vel.z, fn);
      psi.B.x   = slope_limiter(mi.B.x,   pi_min.B.x,   pi_max.B.x,   pmin.B.x,   pmax.B.x,   fc);
      psi.B.y   = slope_limiter(mi.B.y,   pi_min.B.y,   pi_max.B.y,   pmin.B.y,   pmax.B.y,   fc);
      psi.B.z   = slope_limiter(mi.B.z,   pi_min.B.z,   pi_max.B.z,   pmin.B.z,   pmax.B.z,   fc);
      psi.psi   = slope_limiter(mi.psi,   pi_min.psi,   pi_max.psi,   pmin.psi,   pmax.psi,   fn);

      for (int k = 0; k < kernel.ndim; k++) 
	pmhd_grad[k][lidx] = grad[k]*psi;
      
      for (int k = 0; k < 6; k++)
	pmhd_cross[k][lidx] = cross[k]*psi;
      
      
      
      if (eulerian_mode) pi.vel = (float3){0.0f,0.0f,0.0f};
      
    }
  }

  t_grad    =  get_time() - t0;
  t_compute += get_time() - t0;

  int nfail_glob = 0;
  MPI_Allreduce(&n_failed_to_invert, &nfail_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (myid == 0) {
    fprintf(stderr,  " **** failed to_invert_matrix = %d  [%g %c] ******** \n",
	    nfail_glob, (100.0f*nfail_glob)/global_n, '%');
  }
  
  import_boundary_pmhd_grad();
  import_boundary_pmhd_cross();
  import_boundary_pvel();

}
