#include "gn.h"
#include "myMPI.h"

int nfail;

void system::compute_weights() {
  const double t0 = get_time();
  
  std::vector<octnode<TREE_NLEAF>*> leaf_list = local_tree.leaf_list;
  std::vector<particle> pvec_send[NMAXPROC];
  
  for (int p = 0; p < nproc; p++)  pvec_send[p].reserve(128);
  
  nfail = 0;
  const float scale_factor = 1.1f;
  
  while (true) {
    const double t1 = get_time();
    compute_weights(leaf_list, scale_factor);
    t_compute += get_time() - t1;
    
    for (int i = 0; i < local_n; i++) local_tree.body_list[i].update();
    
    int group_ngb_list_size_loc  = leaf_list.size();
    int group_ngb_list_size_glob;
    
    const double t2 = get_time();
    MPI_Allreduce(&group_ngb_list_size_loc, 
		  &group_ngb_list_size_glob, 
		  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    t_communicate += get_time() - t2;

//     assert(group_ngb_list_size_glob == 0);
    if (group_ngb_list_size_glob == 0) break;
    
    //**** if some blocks left, import more boundary particles particles
    
    if (myid == 0) fprintf(stderr, " nimport= %d [%d] group_ngb_list_size= %d\n", 
			   Nimport, NMAXIMPORT, group_ngb_list_size_glob);
    
    //**** update outer domin boundaries
    
    std::vector<boundary> outer_tiles_old[NMAXPROC];
    for (int proc = 0; proc < nproc; proc++) 
      outer_tiles_old[proc] = outer_tiles[proc];
    calculate_outer_domain_boundaries(scale_factor);
    
    for (int proc = 0; proc < nproc; proc++) {
      assert(outer_tiles[proc].size() == outer_tiles_old[proc].size());
      for (size_t tile = 0; tile < outer_tiles[proc].size(); tile++)
	outer_tiles[proc][tile].merge(outer_tiles_old[proc][tile]);
    }
    
    local_tree.root.calculate_outer_boundary(scale_factor);
    
    //***** import boundary particles
    
    for (int p = 0; p < nproc; p++) pvec_send[p].clear();
    
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      for (size_t tile = 0; tile < outer_tiles[p].size(); tile++) {
	assert(outer_tiles[p][tile].isinbox(outer_tiles_old[p][tile]));
	local_tree.root.walk_boundary(outer_tiles_old[p][tile],
				      outer_tiles    [p][tile],
				      pvec_send[p]);
      }
    }
    import_pvec_buffers(pvec_send);
  }
  
  local_tree.root.calculate_inner_boundary();
  local_tree.root.calculate_outer_boundary();

  int nfail_glob = 0;
  MPI_Allreduce(&nfail, &nfail_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (myid == 0) {
    fprintf(stderr,  " **** n_too_few_ngb = %d  [%g %c] ******** \n",
	    nfail_glob, (100.0f*nfail_glob)/global_n, '%');
  }
  

  t_compute_weights = get_time() - t0;
}

/////////////////////
/////////////////////
/////////////////////
/////////////////////
/////////////////////

void system::compute_weights(std::vector<octnode<TREE_NLEAF>*> &group_ngb_list, const float scale_factor) {
  
  std::vector<octnode<TREE_NLEAF>*> group_list = group_ngb_list;
  std::vector<octnode<TREE_NLEAF>*> ileaf_list;
  group_ngb_list.clear();
  ileaf_list.reserve(1024);
  
  const int n_groups = group_list.size();
  for (int group = 0; group < n_groups; group++) {
    octnode<TREE_NLEAF> &inode = *group_list[group];                             // current group to process
    boundary ibnd  = inode.calculate_outer_boundary(scale_factor);   // size of  the group
    
    //******** find gather neighbour list

    ileaf_list.clear();
    local_tree. root.find_leaves_inner(ibnd, ileaf_list);
    import_tree.root.find_leaves_inner(ibnd, ileaf_list);
    int n_leaves  = ileaf_list.size();
    
    //******** process each body in the group

    bool outside_local_domain = false;
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      assert(!ibp->isexternal());

      particle &pi = *ibp->pp;
      const pfloat3 ipos = pi.pos;
      
      int  niter  = 100;
      bool keep_looping = true;
      
      float wngb;
      int ingb  = 0;
      int ingbp = NGBmean;
      const int iNGBmin = (int)(NGBmean/4.0f);
      const int iNGBmax = (int)(NGBmean*4.0f);
      
      bool too_few_ngb = false;
      while (keep_looping && niter-- >= 0) {
	if (niter < 20) { too_few_ngb = true; }
	if (!(niter > 2)) {
	  fprintf(stderr, "proc= %d pi.local_idx= %d %d  ingb= %d\n",
		  myid, pi.local_idx, pi.global_idx, ingb);
	}
	if (niter < 15 && pi.local_idx == 152885) {
	  fprintf(stderr, "niter= %d pi.local_idx= %d ingb= %d pi.h= %g  \n",
		  niter, pi.local_idx, ingb, pi.h);
	}
	assert(niter > 2);
	boundary bi = boundary(pi.pos, pi.h);
	if (!ibnd.isinbox(bi)) {
	  ileaf_list.clear();
	  ibnd  = inode.calculate_outer_boundary(scale_factor);
	  
  	  if ((outside_local_domain = !box.isinproc(ibnd, myid))) break;
	  
	  local_tree. root.find_leaves_inner(ibnd, ileaf_list);
	  import_tree.root.find_leaves_inner(ibnd, ileaf_list);
	  n_leaves = ileaf_list.size();
	}
	
	////////

	const float   hi   = pi.h;
	const float   hi2  = hi*hi;
	const float inv_hi = 1.0f/hi;
	
	wngb = 0.0f;
	ingb = 0;
	
	for (int leaf = 0; leaf < n_leaves; leaf++) {
	  const octnode<TREE_NLEAF> &jnode = *ileaf_list[leaf];
	  
	  if (!bi.overlap(jnode.inner)) continue;
	  
	  for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	    const particle  pj = *jbp->pp;
	    const pfloat3 jpos = pj.pos; 
	    const float3 dr = {jpos.x - ipos.x,
			       jpos.y - ipos.y,
			       jpos.z - ipos.z};
	    const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	    if (s2 <= hi2) {
 	      wngb += kernel.w4(sqrt(s2)*inv_hi) * kernel.NORM_COEFF;
 	      ingb += 1;
	    }
	  }
	}
	
	////////

	if ((ingb < iNGBmin && ingbp < iNGBmin && ingb < ingbp) ||
	    (ingb > iNGBmax && ingbp > iNGBmax && ingb > ingbp)) {
 	  too_few_ngb = true;
//  	  fprintf(stderr, "too_few= i= %d\n", pi.local_idx);
	}
	ingbp = ingb;
	
	if ((wngb < NGBmin || wngb > NGBmax) && !too_few_ngb) {
	  const float dh = (NGBmean - wngb)/(wngb*kernel.ndim + TINY);
	  if      (fabs(dh) < 0.07f) pi.h *= 1.0f + dh;
	  else if (     dh  < 0.0f)  pi.h *= 0.93f;
	  else                       pi.h *= 1.03f;
	} else if (ingb < iNGBmin || ingb > iNGBmax) {
	  too_few_ngb = true;
	  const float dh = (NGBmean - ingb)/(ingb*kernel.ndim + TINY);
	  if      (fabs(dh) < 0.07f) pi.h *= 1.0f + dh;
	  else if (     dh  < 0.0f)  pi.h *= 0.93f;
	  else                       pi.h *= 1.03f;
// 	  pi.h *= 0.5f*(1 + std::pow(1.0f*NGBmean/(ingb + 1), 1.0f/kernel.ndim));
	} else {
	  keep_looping = false;
	}
	
	/////////
// 	if (pi.local_idx == 1667) {
// 	  fprintf(stderr, "i= %d keep_looping = %d niter= %d too_few= %d pi.h= %g  wngb= %g ingb= %d\n",
// 		  pi.local_idx, keep_looping, niter, too_few_ngb, pi.h, wngb, ingb);
// 	}
	pi.wght = wngb/kernel.NORM_COEFF * kernel.pow(inv_hi);
	pi.wght = 1.0f/pi.wght;
      
      }
      
      if (outside_local_domain) {
	group_ngb_list.push_back(&inode);
	break;
      }
      
      if (too_few_ngb) nfail++;
    }
  }
  
}

void system::build_ngb_leaf_lists() {
  const int n_leaves = local_tree.leaf_list.size();

  ngb_leaf_list.resize(n_leaves);
  ngb_leaf_list_outer.resize(n_leaves);
  
  import_tree.root.calculate_inner_boundary();
  import_tree.root.calculate_outer_boundary();

  for (int leaf = 0; leaf < n_leaves; leaf++) {
    octnode<TREE_NLEAF> &inode = *local_tree.leaf_list[leaf];
    
    ngb_leaf_list      [leaf].clear();
    ngb_leaf_list_outer[leaf].clear();
    ngb_leaf_list      [leaf].reserve(16);
    ngb_leaf_list_outer[leaf].reserve(16);
    
    const boundary ibnd = inode.calculate_outer_boundary();
    
    local_tree. root.find_leaves_inner(             ibnd, ngb_leaf_list[leaf]);
    import_tree.root.find_leaves_inner(             ibnd, ngb_leaf_list[leaf]);
    local_tree. root.find_leaves_outer(inode.inner, ibnd, ngb_leaf_list_outer[leaf]);
    import_tree.root.find_leaves_outer(inode.inner, ibnd, ngb_leaf_list_outer[leaf]);
    
  }

}
