#include "gn.h"

void system::vain() {
  const int n_groups = group_list.size();

  std::vector<particle> ip;
  std::vector<particle> jp;

  ip.reserve(128);
  jp.reserve(128);
  
  int ncheck = 0;
  int niter  = 0;
  for (int group = 0; group < n_groups; group++) {
    const octnode &inode              = *group_list  [group];
    const int n_leaves                = ngb_leaf_list[group].size();
    std::vector<octnode*> &ileaf_list = ngb_leaf_list[group];

#if 0    
    ip.clear();
    jp.clear();
    
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) 
      ip.push_back(*ibp->pp);
    
    for (int leaf = 0; leaf < n_leaves; leaf++) {
      octnode &jnode = *ileaf_list[leaf];
      for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) 
	jp.push_back(*jbp->pp);
    }

    ///
#if 1
    const int ni = ip.size();
    const int nj = jp.size();
    for (int i = 0; i < ni; i++) {
      const particle &pi= ip[i];
      const pfloat3 &ipos = pi.pos;
      const float   hi      = pi.h;
      const float   hi2     = hi*hi;
//       const boundary bi     = boundary(pi.pos, pi.h);
      for (int j = 0; j < nj; j++) {
	const particle &pj = jp[j];
	const pflmoat3 &jpos = pj.pos;
// 	if (!bi.overlap(jpos)) continue;
	ncheck++;
	
	
	const float3 dr = {jpos.x - ipos.x,
			   jpos.y - ipos.y,
			   jpos.z - ipos.z};
	const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	if (s2 <= hi2) {
	  niter++;
 	}
	
      }
    }
#endif

#else

    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (ibp->isexternal()) continue;
      particle        &pi     = *ibp->pp;
//       ptcl_mhd_struct &pi_mhd = ptcl_mhd[pi.local_idx];
      
      boundary bi = boundary(pi.pos, pi.h);
      
      pfloat3 ipos = pi.pos;
      
      float   hi      = pi.h;
      float   hi2     = hi*hi;
      
      for (int leaf = 0; leaf < n_leaves; leaf++) {
 	octnode &jnode = *ileaf_list[leaf];
	
 	if (!boundary(bi).overlap(jnode.inner)) continue;

	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  particle        &pj     = *jbp->pp;
// 	  ptcl_mhd_struct &pj_mhd = ptcl_mhd[pj.local_idx];

// 	  particle &pj = *jbp->pp;
	  pfloat3 jpos = pj.pos; 
	  float3 dr = {jpos.x - ipos.x,
		       jpos.y - ipos.y,
		       jpos.z - ipos.z};
	  float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  
	  ncheck++;

// 	  float    hj  = pj.h;
// 	  float    hj2 = hj*hj;
// 	  if (s2 <= hi2 && s2 <= hj2) {
 	  if (s2 <= hi2) {
// 	  if (true) {
	    niter++;

	    
	    
	  }
	}
      }
    }
#endif

  }

  fprintf(stderr, "proc= %d: local_n= %d ncheck= %g niter= %g\n",
	  myid, local_n, 1.0*ncheck/local_n, 1.0*niter/local_n);
}
