#include "gn.h"
#include "myMPI.h"

extern int nfail;

void system::compute_weights_cl() {
  const double t0 = get_time();
  
  std::vector<octnode<TREE_NLEAF>*> leaf_list = local_tree.leaf_list;
  std::vector<particle> pvec_send[NMAXPROC];
  
  for (int p = 0; p < nproc; p++)  pvec_send[p].reserve(128);
  
  nfail = 0;
  const float scale_factor = 1.1f;
  
  while (true) {
    const double t1 = get_time();
    const double t00 = get_time();
    fprintf(stderr, " do ... %d  \n", (int)leaf_list.size());
    compute_weights_cl(leaf_list, scale_factor);
    fprintf(stderr, " done in %g sec  %d \n", get_time() - t00, (int)leaf_list.size());
//     exit(-1);
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

void system::compute_weights_cl(std::vector<octnode<TREE_NLEAF>*> &leaf_list_import, const float scale_factor) {

  double t_gpu = 0.0;
  double t_gpu0 = 0.0;
  double t_cpy = 0.0;
  double t_chk = 0.0;
  double t_inner = 0.0;
  double t_outer = 0.0;
  std::vector<octnode<TREE_NLEAF>*> leaf_list = leaf_list_import;
  std::vector<octnode<TREE_NLEAF>*> leaf_list_next, ileaf_list;
  leaf_list_import.clear();
  leaf_list_next.reserve(1024);
  ileaf_list.reserve(1024);

  gpu.joffset.cmalloc(NGPUBLOCKS + 1);
  gpu.leaf_list.cmalloc(NGPUBLOCKS);
  
  const int n_data = pvec.size();
  gpu.ppos.cmalloc(n_data);
  gpu.in_h_out_wngb.cmalloc(local_n);
  for (int i = 0; i < n_data; i++) {
    const particle &p = pvec[i];
    gpu.ppos[i] = float4(p.pos.x.getu(), p.pos.y.getu(), p.pos.z.getu(), p.h);
    if (i < local_n) gpu.in_h_out_wngb[i].x = p.h;
  }
  gpu.ppos.h2d();

  const int iNGBmin = (int)(NGBmean/4.0f);
  const int iNGBmax = (int)(NGBmean*4.0f);
  
  std::vector<bool> too_few_ngb(local_n);
  std::vector<int > ingbp      (local_n);
  for (int i = 0; i < local_n; i++) {
    ingbp      [i] = NGBmean;
    too_few_ngb[i] = false;
  }
  
  std::vector<int> jlist, joffset(NGPUBLOCKS + 1), cpu_leaf_list(NGPUBLOCKS);
  jlist.reserve(1024);
  
  int  niter  = 100;
  while(leaf_list.size() > 0) {
    const double dtouter = get_time();
    niter--;
    assert(niter > 2);
//     fprintf(stderr, "leaf_list.size= %d  leaf_list_next.size()= %d\n",
//  	    (int)leaf_list.size(), (int)leaf_list_next.size());
    leaf_list_next.clear();
    const double tcpy0 = get_time();
    gpu.in_h_out_wngb.h2d();
    t_cpy += get_time() - tcpy0;
    
    const int n_leaves = leaf_list.size();

    gpu.body_list.cmalloc(n_leaves);
    std::vector<int> ilist;
    ilist.reserve(1024);
  
    for (int leaf = 0; leaf < n_leaves; leaf++) {
      const octnode<TREE_NLEAF> &inode = *leaf_list[leaf];
      gpu.body_list[leaf] = (int2){ilist.size(), inode.nparticle};
      for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next)  
	ilist.push_back(ibp->pp->local_idx);
    }
    double tcpy1 = get_time();
    const int ni = ilist.size();
    gpu.ilist.cmalloc(ni);
    for (int i = 0; i < ni; i++) {
      gpu.ilist[i] = ilist[i];
    }
    gpu.body_list.h2d();
    gpu.ilist.h2d();
    t_cpy += get_time() - tcpy1;

    
    int leaf_beg = 0;
    const double dtinner = get_time();
    while(leaf_beg < n_leaves) {
      jlist.clear();
      int block = 0, nj = 0;
      joffset[0] = nj;
      const int leaf_end = std::min(leaf_beg + NGPUBLOCKS, n_leaves);
      for (int leaf = leaf_beg; leaf < leaf_end; leaf++) {
	octnode<TREE_NLEAF> &inode = *leaf_list[leaf];
// 	const boundary ibnd = inode.calculate_outer_boundary(scale_factor);
	const boundary ibnd = inode.calculate_outer_boundary();
	
	ileaf_list.clear();
	local_tree. root.find_leaves_inner(ibnd, ileaf_list);
	import_tree.root.find_leaves_inner(ibnd, ileaf_list);
	const int n_ngb_leaves = ileaf_list.size();
	for (int ngb_leaf = 0; ngb_leaf < n_ngb_leaves; ngb_leaf++) {
	  const octnode<TREE_NLEAF> &jnode = *ileaf_list[ngb_leaf];
	  for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next)  {
	    jlist.push_back(jbp->pp->local_idx);
	    nj++;
	  }
	}
	cpu_leaf_list[block] = leaf;
	joffset[++block] = nj;
	assert(nj == (int)jlist.size());
      }
      
      clFinish(gpu.context.get_command_queue());
      const double tcpy3 = get_time();
      gpu.jlist.cmalloc(joffset[block]);
      gpu.joffset[0] = 0;
      for (int i = 0; i < block; i++) {
	gpu.leaf_list[i] = cpu_leaf_list[i];
	gpu.joffset[i+1] = joffset[i+1];
      }
      for (int i = 0; i < gpu.joffset[block]; i++) {
	gpu.jlist[i] = jlist[i];
      }
//       fprintf(stderr, "size= %d\n", gpu.joffset[block]);
      gpu.jlist.h2d(CL_FALSE);
      gpu.joffset.h2d(CL_FALSE);
      gpu.leaf_list.h2d(CL_FALSE);
      t_cpy += get_time() - tcpy3;
      
      // compute wngb
      const double dtgpu = get_time();
      const int nthreads = NMAXTHREADS;
      gpu.compute_wngb.setWork(-1, nthreads, block);
      gpu.compute_wngb.set_arg<void* >(0, gpu.in_h_out_wngb.p());
      gpu.compute_wngb.set_arg<void* >(1, gpu.leaf_list.p());
      gpu.compute_wngb.set_arg<void* >(2, gpu.body_list.p());
      gpu.compute_wngb.set_arg<void* >(3, gpu.ilist.p());
      gpu.compute_wngb.set_arg<void* >(4, gpu.ppos.p());
      gpu.compute_wngb.set_arg<void* >(5, gpu.jlist.p());
      gpu.compute_wngb.set_arg<void* >(6, gpu.joffset.p());
      gpu.compute_wngb.set_arg<float4>(7, &gpu.domain_hsize);
      gpu.compute_wngb.set_arg<float >(8, NULL, 3*nthreads);
      gpu.compute_wngb.execute();
      t_gpu0 += get_time() - dtgpu;
//       clFinish(gpu.context.get_command_queue());
      t_gpu += get_time() - dtgpu;

      leaf_beg += NGPUBLOCKS;
    }
    clFinish(gpu.context.get_command_queue());
    t_inner += get_time() - dtinner;

    const double tcpy2 = get_time();
    gpu.in_h_out_wngb.d2h();
    t_cpy += get_time() - tcpy2;

    const double dtchk = get_time();
    for (int leaf = 0; leaf < n_leaves; leaf++) {
      octnode<TREE_NLEAF> &inode = *leaf_list[leaf];
      bool leaf_complete = true;
      for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
	const int    idx = ibp->pp->local_idx;
	const float wngb =      gpu.in_h_out_wngb[idx].x;
	const int   ingb = (int)gpu.in_h_out_wngb[idx].y;
//  	fprintf(stderr, "leaf %d: i= %d  wngb= %g  ingb= %d  hi= %g \n",
//  		leaf, idx, wngb, ingb, ibp->pp->h);
	
	if ((ingb < iNGBmin && ingbp[idx] < iNGBmin && ingb < ingbp[idx]) ||
	    (ingb > iNGBmax && ingbp[idx] > iNGBmax && ingb > ingbp[idx])) {
	  too_few_ngb[idx] = true;
	}
	ingbp[idx] = ingb;
	
	float hi = ibp->pp->h;
	if ((wngb < NGBmin || wngb > NGBmax) && !too_few_ngb[idx]) {
	  const float dh = (NGBmean - wngb)/(wngb*kernel.ndim + TINY);
	  if      (fabs(dh) < 0.07f) hi *= 1.0f + dh;
	  else if (     dh  < 0.0f)  hi *= 0.93f;
	  else                       hi *= 1.03f;
	  leaf_complete = false;
	} else if (ingb < iNGBmin || ingb > iNGBmax) {
	  too_few_ngb[idx] = true;
	  const float dh = (NGBmean - ingb)/(ingb*kernel.ndim + TINY);
	  if      (fabs(dh) < 0.07f) hi *= 1.0f + dh;
	  else if (     dh  < 0.0f)  hi *= 0.93f;
	  else                       hi *= 1.03f;
	  leaf_complete = false;
	} else {
	  if (too_few_ngb[idx]) nfail++;
	}
	ibp->pp->h    = hi;
// 	ibp->pp->wght = 1.0f/(wngb/kernel.NORM_COEFF * kernel.pow(inv_hi));
 	ibp->pp->wght = kernel.pow(hi) * kernel.NORM_COEFF/wngb;
	ibp->update();
	gpu.in_h_out_wngb[idx].x = hi;
	if (niter < 20) {too_few_ngb[idx] = true;}
      }
    
      octnode<TREE_NLEAF> inode_copy = inode;
      inode_copy.calculate_outer_boundary();
//       if(!box.isinproc(inode_copy.outer, myid)) {
// 	leaf_list_import.push_back(&inode);
      if (!leaf_complete) {
	leaf_list_next.push_back(&inode);
      }
    }
    t_chk += get_time() - dtchk;
    
    const int n0 = leaf_list.size();
    leaf_list = leaf_list_next;
    
    t_outer += get_time() - dtouter;
    fprintf(stderr, "_niter= %d (%d);  t_gpu= %g (%g) t_cpy= %g  t_chk= %g  %g %g\n",
	    niter, n0, t_gpu, t_gpu0, t_cpy, t_chk, t_inner, t_outer);
  }
  
  
  
}

