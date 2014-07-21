#include "gn.h"
#include "myMPI.h"


int nfail;

void system::adjust_positions(const int niter, const float eps) {
  do_adjust_particles = false;
  gpu.domain_hsize = float4(global_domain.hsize.x.getu(),
			    global_domain.hsize.y.getu(),
			    global_domain.hsize.z.getu(), 0.0f);
  

  distribute_particles();
  
  build_tree();

  import_boundary_pvec_into_a_tree();

  compute_weights_cl();
  compute_weights_cl();
  compute_weights_cl();
  
  do_adjust_particles = true;
  
  for (int iter = 0; iter < niter; iter++) {
    build_tree();
    import_boundary_pvec_into_a_tree();
    compute_weights_cl();
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    
    FILE *fout = fopen("out.dat", "w");
    for (int i = 0; i < local_n; i++) {
      fprintf(fout, "%g %g %g \n",
	      pvec[i].pos.x.getu(),
	      pvec[i].pos.y.getu(),
	      pvec[i].pos.z.getu());
      
      pvec[i].pos.x.add(-gpu.drmean[i].x*eps);
      pvec[i].pos.y.add(-gpu.drmean[i].y*eps);
      if (kernel.ndim > 2) pvec[i].pos.z.add(-gpu.drmean[i].z*eps);

      dx += std::abs(gpu.drmean[i].x);
      dy += std::abs(gpu.drmean[i].y);
      if (kernel.ndim > 2) dz += std::abs(gpu.drmean[i].z);
    }
    fclose(fout);
    fprintf(stderr, ">>> converge_iter= %d: dr = %g %g %g \n",
	    niter - iter,
	    dx/local_n,
	    dy/local_n,
	    dz/local_n);

    distribute_particles();
  }

  do_adjust_particles = false;
}

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

//   double t_gpu0 = 0.0;
  double t_cpy = 0.0;
//   double t_chk = 0.0;
//   double t_inner = 0.0;
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
  gpu.drmean.cmalloc(local_n);
  for (int i = 0; i < n_data; i++) {
    const particle &p = pvec[i];
    gpu.ppos[i] = float4(p.pos.x.getu(), p.pos.y.getu(), p.pos.z.getu(), p.h);
    if (i < local_n) gpu.in_h_out_wngb[i].x = p.h;
  }
  gpu.ppos.h2d();

  int iNGBmin = 14; //(int)(NGBmean/3.0f);
  if (kernel.ndim < 3) iNGBmin = std::min(iNGBmin, 1);
  const int iNGBmax = 128; //128;  // (int)(NGBmean*3.0f);
  
  std::vector<bool> too_few_ngb(local_n);
  std::vector<int > ingbp      (local_n);
  for (int i = 0; i < local_n; i++) {
    ingbp      [i] = NGBmean;
    too_few_ngb[i] = false;
  }
  
  std::vector<int> jlist, joffset(NGPUBLOCKS + 1), cpu_leaf_list(NGPUBLOCKS);
  jlist.reserve(1024);
  
  //////  Allocate CPU-arrays
  std::vector<int > cpu_ilist;
  std::vector<int > cpu_jlist;
  std::vector<int4> cpu_group_bodies;
  cpu_ilist.reserve(1024);
  cpu_jlist.reserve(1024);
  cpu_group_bodies.reserve(1024);
    

  double t_loop0 = 0.0f;
  double t_loop1 = 0.0f;
  double t_gpu = 0.0;
  const double t_loop_main = get_time();

#if 0
  const int NblockDIM = NBLOCKDIM;
#else
  const int NblockDIM = 128;
#endif
  
  int  niter  = 100;
  while(leaf_list.size() > 0) {
    const double dtouter = get_time();
    niter--;
    assert(niter > 2);
    leaf_list_next.clear();

    const double tcpy0 = get_time();
    gpu.in_h_out_wngb.h2d();
    t_cpy += get_time() - tcpy0;
    
    //////  Prepare CPU-arrays
    const double dt_loop0 = get_time();
    const int n_leaves = leaf_list.size();
    int leaf_beg = 0;
    boundary ibnd_prev;
    while (leaf_beg < n_leaves) {
      cpu_ilist.clear();
      cpu_jlist.clear();
      cpu_group_bodies.clear();
      
      const int leaf_end = std::min(leaf_beg + NGPUBLOCKS, n_leaves);
      for (int leaf = leaf_beg; leaf < leaf_end; leaf++) {
	
	// extract i-particles
	
	octnode<TREE_NLEAF> &inode  = *leaf_list[leaf];
	const int ibeg = cpu_ilist.size();
	for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next)  
	  cpu_ilist.push_back(ibp->pp->local_idx);
	const int iend = cpu_ilist.size();
	
	// extract j-particles

	const boundary ibnd  = inode.calculate_outer_boundary();
	ileaf_list.clear();
	local_tree. root.find_leaves_inner(ibnd, ileaf_list);
	import_tree.root.find_leaves_inner(ibnd, ileaf_list);
	const int n_ngb_leaves = ileaf_list.size();
	const int jbeg = cpu_jlist.size();
	for (int ngb_leaf = 0; ngb_leaf < n_ngb_leaves; ngb_leaf++) {
	  const octnode<TREE_NLEAF> &jnode = *ileaf_list[ngb_leaf];
	  for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next)  
	    cpu_jlist.push_back(jbp->pp->local_idx);
	}
	const int jend = cpu_jlist.size();
	
	// store jbeg & jend for each block
	
	for (int i = 0;  i < inode.nparticle; i += NblockDIM) {
 	  const int ni = std::min(NblockDIM, inode.nparticle - i);
	  cpu_group_bodies.push_back((int4){ibeg + i, ni, jbeg, jend});
	}
	assert(ibeg + inode.nparticle == iend);
      
      }
      
      clFinish(gpu.context.get_command_queue());
      
      gpu.in_ilist.cmalloc       (cpu_ilist.size()); 
      gpu.in_jlist.cmalloc       (cpu_jlist.size());
      gpu.in_group_bodies.cmalloc(cpu_group_bodies.size());
      
      for (size_t i = 0; i < cpu_ilist.size();        i++) gpu.in_ilist       [i] = cpu_ilist       [i];
      for (size_t i = 0; i < cpu_jlist.size();        i++) gpu.in_jlist       [i] = cpu_jlist       [i];
      for (size_t i = 0; i < cpu_group_bodies.size(); i++) gpu.in_group_bodies[i] = cpu_group_bodies[i];
      
      gpu.in_ilist.h2d       (CL_FALSE);
      gpu.in_jlist.h2d       (CL_FALSE);
      gpu.in_group_bodies.h2d(CL_FALSE);
      
      //////

      const double t0 = get_time();
      if (do_adjust_particles) {
	gpu.compute_wngb_dr.setWork_block1D(NblockDIM, cpu_group_bodies.size());
	gpu.compute_wngb_dr.set_arg<void* >( 0, gpu.drmean.p());
	gpu.compute_wngb_dr.set_arg<void* >( 1, gpu.in_h_out_wngb.p());
	gpu.compute_wngb_dr.set_arg<void* >( 2, gpu.in_group_bodies.p());
	gpu.compute_wngb_dr.set_arg<void* >( 3, gpu.in_ilist.p());
	gpu.compute_wngb_dr.set_arg<void* >( 4, gpu.in_jlist.p());
	gpu.compute_wngb_dr.set_arg<void* >( 5, gpu.ppos.p());
	gpu.compute_wngb_dr.set_arg<float4>( 6, &gpu.domain_hsize);
	gpu.compute_wngb_dr.set_arg<float >( 7, NULL, 3 * NblockDIM);
	gpu.compute_wngb_dr.execute();
      } else {
	gpu.compute_wngb.setWork_block1D(NblockDIM, cpu_group_bodies.size());
	gpu.compute_wngb.set_arg<void* >( 0, gpu.in_h_out_wngb.p());
	gpu.compute_wngb.set_arg<void* >( 1, gpu.in_group_bodies.p());
	gpu.compute_wngb.set_arg<void* >( 2, gpu.in_ilist.p());
	gpu.compute_wngb.set_arg<void* >( 3, gpu.in_jlist.p());
	gpu.compute_wngb.set_arg<void* >( 4, gpu.ppos.p());
	gpu.compute_wngb.set_arg<float4>( 5, &gpu.domain_hsize);
	gpu.compute_wngb.set_arg<float >( 6, NULL, 3 * NblockDIM);
	gpu.compute_wngb.execute();
      }
      //       clFinish(gpu.context.get_command_queue());
      t_gpu += get_time() - t0;

      leaf_beg += NGPUBLOCKS;
    }
    clFinish(gpu.context.get_command_queue());
    t_loop0 += get_time() - dt_loop0;

    const double tcpy2 = get_time();
    gpu.in_h_out_wngb.d2h();
    t_cpy += get_time() - tcpy2;

    const double dt_loop1 = get_time();
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

#if 0
	const float zzz = 0.2f; //191919f;
	const float zz1 = 0.1919191f;
#else
	const float zzz = 0.05f;
	const float zz1 = 0.05f;
#endif
	
	float hi = ibp->pp->h;
	if ((wngb < NGBmin || wngb > NGBmax) && !too_few_ngb[idx]) {
	  const float dh = (NGBmean - wngb)/(wngb*kernel.ndim + TINY);
	  if      (fabs(dh) < zzz) hi *= 1.0f + dh;
	  else if (     dh  < 0.0f)  hi *= 1.0f - zz1;
	  else                       hi *= 1.0f + zz1;
	  leaf_complete = false;
	} else if (ingb < iNGBmin || ingb > iNGBmax) {
	  too_few_ngb[idx] = true;
	  const float dh = (NGBmean - ingb)/(ingb*kernel.ndim + TINY);
	  if      (fabs(dh) < zzz) hi *= 1.0f + dh;
	  else if (     dh  < 0.0f)  hi *= 1.0f - zz1;
	  else                       hi *= 1.0f + zz1;
	  leaf_complete = false;
	} else {
// 	  if (too_few_ngb[idx]) nfail++;
	}
	ibp->pp->h    = hi;
// 	ibp->pp->wght = 1.0f/(wngb/kernel.NORM_COEFF * kernel.pow(inv_hi));
 	ibp->pp->wght = kernel.pow(hi) * kernel.NORM_COEFF/wngb;
	ibp->update();
	gpu.in_h_out_wngb[idx].x = hi;
	if (niter < 80) {too_few_ngb[idx] = true;}
      }
    
      if (!leaf_complete) {
	leaf_list_next.push_back(&inode);
      }
    }
    t_loop1 += get_time() - dt_loop1;
    
    nfail = 0;
    for (int idx = 0; idx < local_n; idx++) {
      if (too_few_ngb[idx]) {
	nfail++;
	pvec[idx].few = 1;
      } else {
	pvec[idx].few = 0;
      }
    }

    const int n0 = leaf_list.size();
    leaf_list = leaf_list_next;
    
    t_outer += get_time() - dtouter;
//     fprintf(stderr, "_niter= %d (%d);  t_gpu= %g (%g) t_cpy= %g  t_chk= %g  %g %g\n",
// 	    niter, n0, t_gpu, t_gpu0, t_cpy, t_chk, t_inner, t_outer);
    fprintf(stderr, "_niter= %d (%d): t_loop0= %g (t_gpu= %g) t_loop1= %g  t_loop_main= %g\n",
	    niter, n0, t_loop0, t_gpu, t_loop1, get_time() - t_loop_main);
  }

  if (do_adjust_particles) gpu.drmean.d2h();
  
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

