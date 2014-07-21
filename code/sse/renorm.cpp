#include "gn.h"
#include "myMPI.h"

void system::renorm() {
  double t0 = get_time();

  const int n_groups = group_list.size();
  
  int ncheck = 0;
  int niter  = 0;
  
  Bxx.resize(local_n);
  Bxy.resize(local_n);
  Bxz.resize(local_n);
  Byy.resize(local_n);
  Byz.resize(local_n);
  Bzz.resize(local_n);
  for (int group = 0; group < n_groups; group++) {
    const octnode &inode                    = *group_list  [group];
    const int n_leaves                      = ngb_leaf_list[group].size();
    const std::vector<octnode*> &ileaf_list = ngb_leaf_list[group];
    
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (ibp->isexternal()) continue;
      const particle &pi = *ibp->pp;
      const boundary bi = boundary(pi.pos, pi.h);
      
      const pfloat3 ipos = pi.pos;
      
      const float   hi      = pi.h;
      const float   hi2     = hi*hi;
      const float inv_hi    = 1.0f/hi;
      const float inv_hidim = kernel.pow(inv_hi);
      const float   wi      = pi.wght;
      
      real Exx, Exy, Exz, Eyy, Eyz, Ezz;
      Exx = Exy = Exz = Eyy = Eyz = Ezz = 0;
      
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
	  
	  ncheck++;

//   	  float    hj  = pj.h;
//   	  float    hj2 = hj*hj;
//   	  if (s2 <= hi2 && s2 <= hj2) {
	  if (s2 <= hi2) {
// 	  if (true) {
	    niter++;

#if 0	    
	    const real wj = pj.wght;
	    const real wk = wj * kernel.w(sqrt(s2)*inv_hi)*inv_hidim * wi;
#else
	    const real wk = kernel.w(sqrt(s2)*inv_hi) * inv_hidim * wi;
#endif
	    
	    Exx += wk * dr.x*dr.x;
	    Exy += wk * dr.x*dr.y;
	    Exz += wk * dr.x*dr.z;
	    Eyy += wk * dr.y*dr.y;
	    Eyz += wk * dr.y*dr.z;
	    Ezz += wk * dr.z*dr.z;
	    
	  }
	  
	}
      }
      /// store i-particle data
      const int  lidx = pi.local_idx;
      
      real A11 = Exx;
      real A12 = Exy;
      real A22 = Eyy;
      real A13 = Exz;
      real A23 = Eyz;
      real A33 = Ezz;
      real det = 0;
      switch(kernel.ndim) {
      case 1:
	Bxx[lidx] = 1.0/Exx;
	Bxy[lidx] = 0;
	Bxz[lidx] = 0;
	Byy[lidx] = 0;
	Byz[lidx] = 0;
	Bzz[lidx] = 0;
	break;
      case 2:
	det = A11*A22 - A12*A12;
	assert(det != 0.0f);
	det = 1.0f/det;
	
	Bxx[lidx] =  A22*det;
	Bxy[lidx] = -A12*det;
	Bxz[lidx] =  0.0f;
	Byy[lidx] =  A11*det;
	Byz[lidx] =  0.0f;
	Bzz[lidx] =  0.0f;
	break;
	
      default:
	det = -A13*A13*A22 + 2*A12*A13*A23 - A11*A23*A23 - A12*A12*A33 + A11*A22*A33;
	assert (det != 0);
	det = 1.0/det;
	
	Bxx[lidx] = (-A23*A23 + A22*A33)*det;      
	Bxy[lidx] = (+A13*A23 - A12*A33)*det;      
	Bxz[lidx] = (-A13*A22 + A12*A23)*det;      
	Byy[lidx] = (-A13*A13 + A11*A33)*det;      
	Byz[lidx] = (+A12*A13 - A11*A23)*det;      
	Bzz[lidx] = (-A12*A12 + A11*A22)*det;      
	
      }
      

//       fprintf(stderr, "proc= %d: lidx= %d: B= %g %g %g %g %g %g\n",
// 	      myid, lidx, 
// 	      Bxx[lidx],
// 	      Bxy[lidx],
// 	      Bxz[lidx],
// 	      Byy[lidx],
// 	      Byz[lidx],
// 	      Bzz[lidx]);
      
    }
  }
  
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "proc= %d: local_n= %d ncheck= %g niter= %g\n",
	  myid, local_n, 1.0*ncheck/local_n, 1.0*niter/local_n);
#endif
  t_compute += get_time() - t0;

  import_boundary_Bmatrix();


}
