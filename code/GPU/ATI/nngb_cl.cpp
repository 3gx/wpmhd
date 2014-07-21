#include "gn.h"
#include "myMPI.h"

void system::nngb_cl() {
  const double t0 = get_time();
  
  // Allocate GPU arrays, both local & imported

  const int4 size4 = get_gpu_size();
  if (size4.x != gpu.nngb.get_size()) {
    gpu.nngb.cmalloc(size4.x);
  }

  // Compute neighbour number
  
  std::vector<int> cpu_jlist, cpu_joffset(NGPUBLOCKS + 1), cpu_leaf_list(NGPUBLOCKS);
  cpu_jlist.reserve(1024);

  const int n_leaves = local_tree.leaf_list.size();

  int leaf_beg = 0;
  while (leaf_beg < n_leaves) {
    cpu_jlist.clear();
    int block = 0, nj = 0;
    cpu_joffset[0] = nj;
    const int leaf_end = std::min(leaf_beg + NGPUBLOCKS, n_leaves);
    for (int leaf = leaf_beg; leaf < leaf_end; leaf++) {
      const std::vector<octnode<TREE_NLEAF>*> &ileaf_list = ngb_leaf_list_outer[leaf];
      const int n_ngb_leaves = ileaf_list.size();
      for (int ngb_leaf = 0; ngb_leaf < n_ngb_leaves; ngb_leaf++) {
	const octnode<TREE_NLEAF> &jnode = *ileaf_list[ngb_leaf];
	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next)  {
	  cpu_jlist.push_back(jbp->pp->local_idx);
	  nj++;
	}
      }
      cpu_leaf_list[block] = leaf;
      cpu_joffset[++block] = nj;
      assert(nj == (int)cpu_jlist.size());
    }

    clFinish(gpu.context.get_command_queue());
    gpu.jlist.cmalloc(cpu_joffset[block]);
    gpu.joffset[0] = 0;
    for (int i = 0; i < block; i++) {
      gpu.leaf_list[i] = cpu_leaf_list[i];
      gpu.joffset[i+1] = cpu_joffset[i+1];
    }
    for (int i = 0; i < gpu.joffset[block]; i++) {
      gpu.jlist[i] = cpu_jlist[i];
    }
    gpu.jlist.h2d(CL_FALSE);
    gpu.joffset.h2d(CL_FALSE);
    gpu.leaf_list.h2d(CL_FALSE);

    const int nthreads = NMAXTHREADS;
    gpu.compute_nngb.setWork(-1, nthreads, block);
    gpu.compute_nngb.set_arg<void* >( 0, gpu.nngb.p());
    gpu.compute_nngb.set_arg<void* >( 1, gpu.leaf_list.p());
    gpu.compute_nngb.set_arg<void* >( 2, gpu.body_list.p());
    gpu.compute_nngb.set_arg<void* >( 3, gpu.ilist.p());
    gpu.compute_nngb.set_arg<void* >( 4, gpu.ppos.p());
    gpu.compute_nngb.set_arg<void* >( 5, gpu.jlist.p());
    gpu.compute_nngb.set_arg<void* >( 6, gpu.joffset.p());
    gpu.compute_nngb.set_arg<float4>( 7, &gpu.domain_hsize);
    gpu.compute_nngb.set_arg<float >( 8, NULL, 4*nthreads);
    gpu.compute_nngb.execute();

    leaf_beg += NGPUBLOCKS;
  }
  clFinish(gpu.context.get_command_queue());
  gpu.nngb.d2h();


  int ngb_min = global_n;
  int ngb_max = 0;
  int ngb_sum = 0;
  int ngbs_min = global_n;
  int ngbs_max = 0;
  int ngbs_sum = 0;
  for (int i = 0; i < local_n; i++) {
    ngb_min = std::min(ngb_min, gpu.nngb[i].x);
    ngb_max = std::max(ngb_max, gpu.nngb[i].x);
    ngb_sum += gpu.nngb[i].x;
    ngbs_min = std::min(ngbs_min, gpu.nngb[i].y);
    ngbs_max = std::max(ngbs_max, gpu.nngb[i].y);
    ngbs_sum += gpu.nngb[i].y;
  }

  const double t1 = get_time();

  int ngb_min_glob, ngb_max_glob, ngb_sum_glob;
  MPI_Allreduce(&ngb_min, &ngb_min_glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ngb_max, &ngb_max_glob, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ngb_sum, &ngb_sum_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int ngbs_min_glob, ngbs_max_glob, ngbs_sum_glob;
  MPI_Allreduce(&ngbs_min, &ngbs_min_glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ngbs_max, &ngbs_max_glob, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ngbs_sum, &ngbs_sum_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  t_communicate += get_time() - t1;

  if (myid == 0) {
    fprintf(stderr, "ngb_min= %d %d  ngb_max= %d %d  ngb_mean= %g %g\n", 
	    ngb_min_glob,  ngbs_min_glob,
	    ngb_max_glob,  ngbs_max_glob,
	    ngb_sum_glob*1.0/global_n, ngbs_sum_glob*1.0/global_n);
  
  }
  
  t_nngb = get_time() - t0;
  
}
