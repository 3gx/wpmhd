#include "gn.h"
#include "myMPI.h"

void system::compute_weights() {
  
  std::vector<boundary> export_list_loc;
  std::vector<particle> pvec_send[NMAXPROC];
  std::vector<particle> pvec_recv[NMAXPROC];
  
  for (int p = 0; p < nproc; p++) {
    pvec_send[p].reserve(128);
  }
  export_list_loc.reserve(128);
  
  const float scale_factor = 1.1f;
  
  while (true) {
#ifdef _DEBUG_PRINT_
    fprintf(stderr, "proc= %d: *** compute weights *** \n", myid);
#endif
    
    export_list_loc.clear();
    
    double t0 = get_time();
    compute_weights(export_list_loc, scale_factor);

    for (int i = 0;       i < local_n;            i++) octp[i].update();
    for (int i = local_n; i < local_n + import_n; i++) octp[i].update(true);
    t_compute += get_time() - t0;
    

    t0 = get_time();
    int export_list_size_loc  = export_list_loc.size();
    int export_list_size_glob;

    MPI_Allreduce(&export_list_size_loc, 
		  &export_list_size_glob, 
		  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    t_communicate += get_time() - t0;

    if (export_list_size_glob > 0)
      fprintf(stderr, "proc= %d: local_size= %d  global_size= %d\n", myid,
	      export_list_size_loc,
	      export_list_size_glob);
    if (export_list_size_glob == 0) break;
    
    t0 = get_time();
    
    /**** update domain boundaries *****/
    
    std::vector<boundary> box_outer_old = box.outer;
    boundary bnd = root_node.calculate_outer_boundary(scale_factor);
    bnd.merge(box_outer_old[myid]);
    myMPI_allgather<boundary>(bnd, box.outer, myid, nproc);
    
    /**** import boundary particles *****/
    
    for (int p = 0; p < nproc; p++) 
      pvec_send[p].clear();
    
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      assert(box.outer[p].isinbox(box_outer_old[p]));
      root_node.walk_boundary(box_outer_old[p], box.outer[p], pvec_send[p]);
      const int np = pvec_send[p].size();
      for (int i = 0; i < np; i++) 
	pidx_send[p].push_back(pvec_send[p][i].local_idx);
    }
    
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
    myMPI_all2all<particle>(pvec_send, pvec_recv, myid, nproc, debug_flag);
    
    /**** import boundary particles *****/
    
    const int import_n_old = import_n;
    
#if 0
    const particle *pvec_first = &pvec[0];
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pvec_recv[p].size(); i++) 
	pvec.push_back(pvec_recv[p][i]);
    assert(pvec_first == &pvec[0]);
#else
    int cntr = 0;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pvec_recv[p].size(); i++) 
	cntr++;
    
    assert(safe_resize(pvec, local_n + import_n_old + cntr));
    
    cntr = 0;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pvec_recv[p].size(); i++) 
	pvec[local_n + import_n_old + cntr++] = pvec_recv[p][i];

#endif
    
    import_n = pvec.size() - local_n;
#ifndef _DEBUG_PRINT_
    fprintf(stderr, "proc= %d: n_import_old= %d  new= %d\n",
	    myid, import_n_old, import_n);
#endif
    assert(safe_resize(octp, local_n + import_n));
    
    for (int i = local_n + import_n_old; i < local_n + import_n; i++) {
      octp[i] = octbody(pvec[i], true);
    }
    
    for (int i = 0;       i < local_n;            i++) octp[i].update();
    for (int i = local_n; i < local_n + import_n; i++) octp[i].update(true);

    for (int i = local_n + import_n_old; i < local_n + import_n; i++) {
      octnode::insert_octbody(root_node, octp[i], octn);
    }
    
    root_node.calculate_inner_boundary();

    for (int i = local_n + import_n_old; i < local_n + import_n; i++) {
      pvec[i].local_idx = i;
    }
    
    leaf_list.clear();
    group_list.clear();
    
    root_node.extract_leaves(leaf_list);
    root_node.extract_leaves(group_list);
    t_communicate += get_time() - t0;
  }
  
}

/////////////////////
/////////////////////
/////////////////////
/////////////////////
/////////////////////

void system::compute_weights(std::vector<boundary> &export_list, const float scale_factor) {
  
  int n_groups = group_list.size();
  std::vector<octnode*> ileaf_list;
  ileaf_list.reserve(1024);
  
  for (int group = 0; group < n_groups; group++) {
    octnode &inode = *group_list[group];
    
    ileaf_list.clear();
    boundary ibnd = inode.calculate_outer_boundary(scale_factor);
    root_node.find_leaves_inner(ibnd, ileaf_list);
    int n_leaves  = ileaf_list.size();
    
    bool outside_domain = false;

    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (outside_domain) {
	export_list.push_back(ibnd);
	break;
      }

      if (ibp->isexternal()) continue;
      particle &pi = *ibp->pp;
      
      pfloat3 ipos = pi.pos;
      
      int  niter        = 1000;
      bool keep_looping = true;
      
      real wngb;
      int ingb;
      while (keep_looping && niter-- >= 0) {
	
	boundary bi = boundary(pi.pos, pi.h);
	if (!ibnd.isinbox(bi)) {
	  ileaf_list.clear();
	  ibnd  = inode.calculate_outer_boundary(scale_factor);
	  
 	  if ((outside_domain = !box.outer[myid].isinbox(ibnd))) break;
	  
	  root_node.find_leaves_inner(ibnd, ileaf_list);
	  n_leaves = ileaf_list.size();
	}
	
	float   hi   = pi.h;
	float   hi2  = hi*hi;
	float inv_hi = 1.0f/hi;
	
	wngb = 0.0;
	ingb = 0;
	
	for (int leaf = 0; leaf < n_leaves; leaf++) {
	  octnode &jnode = *ileaf_list[leaf];
	  
	  if (!boundary(bi).overlap(jnode.inner)) continue;
	  
	  for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	    particle &pj = *jbp->pp;
	    pfloat3 jpos = pj.pos; 
	    float3 dr = {jpos.x - ipos.x,
			 jpos.y - ipos.y,
			 jpos.z - ipos.z};
	    float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	    if (s2 <= hi2) {
 	      wngb += kernel.w4(sqrt(s2)*inv_hi) * kernel.NORM_COEFF;
 	      ingb += 1;
	    }
	
	  }
	  
	}
	
	if (wngb < NGBmin || wngb > NGBmax) {
	  float dh = (NGBmean - wngb)/(wngb*kernel.ndim + TINY);
	  if      (fabs(dh) < 0.2f) pi.h *= 1.0f + dh;
	  else if (     dh  < 0.0f) pi.h *= 0.8f;
	  else                      pi.h *= 1.2f;
	} else {
	  keep_looping = false;
	}
	
	pi.wght = wngb/kernel.NORM_COEFF * kernel.pow(inv_hi);
	pi.wght = 1.0f/pi.wght;
      
      }
      
    }
  }

// return;  
//   ngb_leaf_list.resize(n_groups);
//   ngb_leaf_list_outer.resize(n_groups);
// //  ngb_leaf_list.clear();

//   int size = 0;
//   for (int group = 0; group < n_groups; group++) {
//     octnode &inode = *group_list[group];
    
//     ngb_leaf_list[group].clear();
//     ngb_leaf_list[group].reserve(16);
//     boundary ibnd = inode.calculate_outer_boundary();
//     root_node.find_leaves_inner(ibnd, ngb_leaf_list[group]);
//     root_node.find_leaves_outer(ibnd, ngb_leaf_list_outer[group]);
//     size += ngb_leaf_list[group].size();
//   }
// #ifdef _DEBUG_PRINT_
//   fprintf(stderr, "proc= %d: size= %d\n", myid, size);
// #endif

// #if _DEBUG_DUMP_INTERMEDIATE_
//   fprintf(stderr, "dump weights\n");
  
//   char fn[256];
//   sprintf(fn, "proc_%.2d_wght.dump", myid);
//   FILE *fout = fopen(fn, "w");
  
//   for (int i = 0; i < local_n; i++) {
//     particle &p = pvec[i];
    
//     fprintf(fout, "%d %d  %g %g %g  %g %g  \n", 
// 	    (int)p.global_idx,
// 	    (int)p.local_idx,
// 	    (float)p.pos.x.getu(),
// 	    (float)p.pos.y.getu(),
// 	    (float)p.pos.z.getu(),
// 	    p.h,
// 	    1.0/(p.wght)*kernel.NORM_COEFF/kernel.pow(1.0/p.h));
//   }
//   fclose(fout);
// #endif

}

/////////////////////
/////////////////////
/////////////////////
/////////////////////
/////////////////////

// void system::compute_weights_gpu(std::vector<boundary> &export_list, const float scale_factor) {

//   const int n_groups = group_list.size();
//   std::vector<octnode*> ileaf_list;
//   ileaf_list.reserve(1024);
  
//   std::vector<octnode*> groups0(n_groups);
//   std::vector<octnode*> groups1(n_groups);
//   int ngroups0 = 0;
//   int ngroups1 = 0;
//   for (int group = 0; group < n_groups; group++) 
//     groups0[ngroups0++] = group_list[group];
  
//   // repeat loop untill all groups are exhausted

//   while (ngroups0 >= 0) {
//     ngroups1 = 0;

//     // send jobs
    
//     for (int group = 0; group < ngroups0; group++) {
//       octnode &inode = *groups0[group];
//       const boundary ibnd = inode.calculate_outer_boundary(scale_factor);
//       if (!box.outer[myid].isinbox(ibnd)) {
// 	export_list.push_back(ibnd);
// 	continue;
//       }
      
//       ileaf_list.clear();
//       root_node.find_leaves_inner(ibnd, ileaf_list);
     
// //       gpu_wght_job.schedule(group, inode, ileaf_list);
// //       gpu_wght_job.execute_if_full();
//     }
    
// //     gpu_wght_job.execute();
// //     gpu_wght_job.copye_data_to_host();
    
//     // collect results
    
// //     const int njobs = gpu_wght_job.get_njobs();
// //     for (int job = 0; job < njobs; job++) 
// //       if ((group = gpu_wght_job_get_result(job, wght_list)) > 0)
// // 	groups1[ngroups1++] = groups0[group - 1];

//     std::swap( groups0,  groups1);
//     std::swap(ngroups0, ngroups1);
//   }

//   if ((int)export_list.size() > 0) return;

//   // extract neighbour leaf-list for interactions

//   ngb_leaf_list.resize(n_groups);

//   int size = 0;
//   for (int group = 0; group < n_groups; group++) {
//     octnode &inode = *group_list[group];
    
//     ngb_leaf_list[group].clear();
//     ngb_leaf_list[group].reserve(16);
//     boundary ibnd = inode.calculate_outer_boundary();
//     root_node.find_leaves_inner(ibnd, ngb_leaf_list[group]);
//     size += ngb_leaf_list[group].size();
//   }

// #ifdef _DEBUG_PRINT_
//   fprintf(stderr, "proc= %d: size= %d\n", myid, size);
// #endif

// #if _DEBUG_DUMP_INTERMEDIATE_
//   fprintf(stderr, "dump weights\n");
  
//   char fn[256];
//   sprintf(fn, "proc_%.2d_wght.dump", myid);
//   FILE *fout = fopen(fn, "w");
  
//   for (int i = 0; i < local_n; i++) {
//     particle &p = pvec[i];
    
//     fprintf(fout, "%d %d  %g %g %g  %g %g  \n", 
// 	    (int)p.global_idx,
// 	    (int)p.local_idx,
// 	    (float)p.pos.x.getu(),
// 	    (float)p.pos.y.getu(),
// 	    (float)p.pos.z.getu(),
// 	    p.h,
// 	    1.0/(p.wght)*kernel.NORM_COEFF/kernel.pow(1.0/p.h));
//   }
//   fclose(fout);
// #endif

// }

////////////

void system::build_ngb_leaf_lists() {
  const int n_groups = group_list.size();

  ngb_leaf_list.resize(n_groups);
  ngb_leaf_list_outer.resize(n_groups);
  
  int size = 0;
  for (int group = 0; group < n_groups; group++) {
    octnode &inode = *group_list[group];
    
    ngb_leaf_list[group].clear();
    ngb_leaf_list[group].reserve(16);
    ngb_leaf_list_outer[group].clear();
    ngb_leaf_list_outer[group].reserve(16);

    boundary ibnd = inode.calculate_outer_boundary();

    root_node.find_leaves_inner(ibnd, ngb_leaf_list[group]);
#if 1
    root_node.find_leaves_outer(inode.inner, ibnd, ngb_leaf_list_outer[group]);
#else
    root_node.find_leaves_inner(ibnd, ngb_leaf_list_outer[group]);
#endif
    
    size += ngb_leaf_list[group].size();
  }
}
