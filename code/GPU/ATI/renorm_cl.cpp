#include "gn.h"
#include "myMPI.h"

void system::renorm_cl() {
  const double t0 = get_time();


  // Allocate GPU arrays, both local & imported

  ////////

  const int n_leaves = local_tree.leaf_list.size();
  
  /// compute ijlist_offset

  std::vector<int> ijlist_offset(n_leaves + 1);
  gpu.ijlist_offset.cmalloc(n_leaves + 1);
  gpu.leaf_ngb_max.cmalloc(n_leaves);
  int leaf_beg = 0;
  int max_size  = 0;
  ijlist_offset[0] = 0;
  while (leaf_beg < n_leaves) {
    const int leaf_end = std::min(leaf_beg + NGPUBLOCKS, n_leaves);
    gpu.ijlist_offset[leaf_beg] = 0;
    int max_ngb = 0;
    for (int leaf = leaf_beg; leaf < leaf_end; leaf++) {
      const octnode<TREE_NLEAF> &inode = *local_tree.leaf_list[leaf];
      for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
	max_ngb = std::max(max_ngb, gpu.nngb[ibp->pp->local_idx].x);
      }
      gpu.ijlist_offset[leaf + 1] = gpu.ijlist_offset[leaf] + max_ngb * inode.nparticle;
      ijlist_offset    [leaf + 1] =     ijlist_offset[leaf] + max_ngb * inode.nparticle;

      max_size = std::max(max_size, gpu.ijlist_offset[leaf + 1]);
      gpu.leaf_ngb_max[leaf] = max_ngb;
    }    
    leaf_beg += NGPUBLOCKS;
  }
  gpu.ijlist_offset.h2d();
  gpu.leaf_ngb_max.h2d();  
 
  fprintf(stderr, "max_size= %d\n", max_size);
  gpu.ijlist.cmalloc(max_size);
  gpu.drij.cmalloc(max_size);
  fprintf(stderr, "max_size= %d\n", max_size);

  const int n_data = pvec.size();

  const int4 size4 = get_gpu_size();
  const int size2b = size4.y;

  fprintf(stderr, "size2b= %d\n", size2b);
  
  if (size2b != gpu.Bxx.get_size()) {
    gpu.Bxx.cmalloc(size2b);
    gpu.Bxy.cmalloc(size2b);
    gpu.Bxz.cmalloc(size2b);
    gpu.Byy.cmalloc(size2b);
    gpu.Byz.cmalloc(size2b);
    gpu.Bzz.cmalloc(size2b);
  }
  fprintf(stderr, "size2b= %d\n", size2b);
  
  if (size2b != gpu.mhd1_grad_x.get_size()) {
    gpu.mhd1_grad_x.cmalloc(size2b);
    gpu.mhd1_grad_y.cmalloc(size2b);
    gpu.mhd1_grad_z.cmalloc(size2b);
    gpu.mhd2_grad_x.cmalloc(size2b);
    gpu.mhd2_grad_y.cmalloc(size2b);
    gpu.mhd2_grad_z.cmalloc(size2b);
    gpu.mhd3_grad_x.cmalloc(size2b);
    gpu.mhd3_grad_y.cmalloc(size2b);
    gpu.mhd3_grad_z.cmalloc(size2b);
  fprintf(stderr, "size2b= %d\n", size2b);
  }

  if (!do_first_order) {
    gpu.mhd1_psi.cmalloc(size4.x);
    gpu.mhd2_psi.cmalloc(size4.x);
    gpu.mhd3_psi.cmalloc(size4.x);
  }

  fprintf(stderr, "size2b= %d\n", size2b);

  /////////////////////

  std::vector<int> cpu_jlist, cpu_joffset(NGPUBLOCKS + 1), cpu_leaf_list(NGPUBLOCKS);
  cpu_jlist.reserve(1024);

  double t_gpu_time = 0;
  double t_loop_time0 = get_time();
  leaf_beg = 0;
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

    ////////

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

    //////
    
    const int nthreads = NMAXTHREADS;
    gpu.extract_gather_list.setWork(-1, nthreads, block);
    gpu.extract_gather_list.set_arg<void* >( 0, gpu.ijlist.p());
    gpu.extract_gather_list.set_arg<void* >( 1, gpu.drij.p());
    gpu.extract_gather_list.set_arg<void* >( 2, gpu.leaf_list.p());
    gpu.extract_gather_list.set_arg<void* >( 3, gpu.body_list.p());
    gpu.extract_gather_list.set_arg<void* >( 4, gpu.ilist.p());
    gpu.extract_gather_list.set_arg<void* >( 5, gpu.ppos.p());
    gpu.extract_gather_list.set_arg<void* >( 6, gpu.jlist.p());
    gpu.extract_gather_list.set_arg<void* >( 7, gpu.joffset.p());
    gpu.extract_gather_list.set_arg<void* >( 8, gpu.ijlist_offset.p());
    gpu.extract_gather_list.set_arg<void* >( 9, gpu.leaf_ngb_max.p());
    gpu.extract_gather_list.set_arg<float4>(10, &gpu.domain_hsize);
    gpu.extract_gather_list.set_arg<float >(11, NULL, 4*nthreads);
//      fprintf(stderr, "gather_list.exec\n");
    double tgpu = get_time();
    gpu.extract_gather_list.execute();
    clFinish(gpu.context.get_command_queue());
    //   fprintf(stderr, "extract gather in %g sec \n", get_time() - tgpu);
    t_gpu_time += get_time() - tgpu;


    gpu.compute_Bmatrix.setWork(-1, nthreads, block);
    gpu.compute_Bmatrix.set_arg<void* >( 0, gpu.Bxx.p());
    gpu.compute_Bmatrix.set_arg<void* >( 1, gpu.Bxy.p());
    gpu.compute_Bmatrix.set_arg<void* >( 2, gpu.Bxz.p());
    gpu.compute_Bmatrix.set_arg<void* >( 3, gpu.Byy.p());
    gpu.compute_Bmatrix.set_arg<void* >( 4, gpu.Byz.p());
    gpu.compute_Bmatrix.set_arg<void* >( 5, gpu.Bzz.p());
    gpu.compute_Bmatrix.set_arg<void* >( 6, gpu.leaf_list.p());
    gpu.compute_Bmatrix.set_arg<void* >( 7, gpu.body_list.p());
    gpu.compute_Bmatrix.set_arg<void* >( 8, gpu.ilist.p());
    gpu.compute_Bmatrix.set_arg<void* >( 9, gpu.ijlist_offset.p());
    gpu.compute_Bmatrix.set_arg<void* >(10, gpu.leaf_ngb_max.p());
    gpu.compute_Bmatrix.set_arg<void* >(11, gpu.ppos.p());
    gpu.compute_Bmatrix.set_arg<void* >(12, gpu.pvel.p());
    gpu.compute_Bmatrix.set_arg<void* >(13, gpu.drij.p());
//     fprintf(stderr, "gather_Bmatrix.exec\n");
    tgpu = get_time();
    gpu.compute_Bmatrix.execute();
    clFinish(gpu.context.get_command_queue());
    //  fprintf(stderr, "compute_Bmatrix in %g sec \n", get_time() - tgpu);
    t_gpu_time += get_time() - tgpu;

    if (!do_first_order) {
      gpu.compute_gradient.setWork(-1, nthreads, block);
      gpu.compute_gradient.set_arg<void* >( 4, gpu.leaf_list.p());
      gpu.compute_gradient.set_arg<void* >( 5, gpu.body_list.p());
      gpu.compute_gradient.set_arg<void* >( 6, gpu.ilist.p());
      gpu.compute_gradient.set_arg<void* >( 7, gpu.ijlist_offset.p());
      gpu.compute_gradient.set_arg<void* >( 8, gpu.leaf_ngb_max.p());
      gpu.compute_gradient.set_arg<void* >( 9, gpu.ijlist.p());
      gpu.compute_gradient.set_arg<void* >(10, gpu.ppos.p());
      gpu.compute_gradient.set_arg<void* >(11, gpu.pvel.p());
      gpu.compute_gradient.set_arg<void* >(12, gpu.drij.p());
      gpu.compute_gradient.set_arg<void* >(13, gpu.Bxx.p());
      gpu.compute_gradient.set_arg<void* >(14, gpu.Bxy.p());
      gpu.compute_gradient.set_arg<void* >(15, gpu.Bxz.p());
      gpu.compute_gradient.set_arg<void* >(16, gpu.Byy.p());
      gpu.compute_gradient.set_arg<void* >(17, gpu.Byz.p());
      gpu.compute_gradient.set_arg<void* >(18, gpu.Bzz.p());

      gpu.limit_gradient.setWork(-1, nthreads, block);
      gpu.limit_gradient.set_arg<void* >( 5, gpu.leaf_list.p());
      gpu.limit_gradient.set_arg<void* >( 6, gpu.body_list.p());
      gpu.limit_gradient.set_arg<void* >( 7, gpu.ilist.p());
      gpu.limit_gradient.set_arg<void* >( 8, gpu.ijlist_offset.p());
      gpu.limit_gradient.set_arg<void* >( 9, gpu.leaf_ngb_max.p());
      gpu.limit_gradient.set_arg<void* >(10, gpu.ijlist.p());
      gpu.limit_gradient.set_arg<void* >(11, gpu.drij.p());

      float4 fc;

      ///////// MHD1

      gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd1_grad_x.p());
      gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd1_grad_y.p());
      gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd1_grad_z.p());
      gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd1.p());
      tgpu = get_time();
      gpu.compute_gradient.execute();
      clFinish(gpu.context.get_command_queue());
      //      fprintf(stderr, "compute_gradient in %g sec \n", get_time() - tgpu);
      t_gpu_time += get_time() - tgpu;

      fc = float4(0.5f, 0.5f, 0.5f, 1.0f);
      gpu.limit_gradient.set_arg<void* >( 0, gpu.mhd1_psi.p());
      gpu.limit_gradient.set_arg<void* >( 1, gpu.mhd1.p());
      gpu.limit_gradient.set_arg<void* >( 2, gpu.mhd1_grad_x.p());
      gpu.limit_gradient.set_arg<void* >( 3, gpu.mhd1_grad_y.p());
      gpu.limit_gradient.set_arg<void* >( 4, gpu.mhd1_grad_z.p());
      gpu.limit_gradient.set_arg<float4>(12, &fc);
      tgpu = get_time();
      gpu.limit_gradient.execute();
      clFinish(gpu.context.get_command_queue());
      //fprintf(stderr, "limit_gradient in %g sec \n", get_time() - tgpu);
      t_gpu_time += get_time() - tgpu;

      ///////// MHD2

      gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd2_grad_x.p());
      gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd2_grad_y.p());
      gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd2_grad_z.p());
      gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd2.p());
      tgpu = get_time();
      gpu.compute_gradient.execute();
      clFinish(gpu.context.get_command_queue());
      t_gpu_time += get_time() - tgpu;

      fc = float4(1.0f, 1.0, 1.0, 0.5f);
      gpu.limit_gradient.set_arg<void* >( 0, gpu.mhd2_psi.p());
      gpu.limit_gradient.set_arg<void* >( 1, gpu.mhd2.p());
      gpu.limit_gradient.set_arg<void* >( 2, gpu.mhd2_grad_x.p());
      gpu.limit_gradient.set_arg<void* >( 3, gpu.mhd2_grad_y.p());
      gpu.limit_gradient.set_arg<void* >( 4, gpu.mhd2_grad_z.p());
      gpu.limit_gradient.set_arg<float4>(12, &fc);
      tgpu = get_time();
      gpu.limit_gradient.execute();
      clFinish(gpu.context.get_command_queue());
      t_gpu_time += get_time() - tgpu;

      ///////// MHD3

      gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd3_grad_x.p());
      gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd3_grad_y.p());
      gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd3_grad_z.p());
      gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd3.p());
      tgpu = get_time();
      gpu.compute_gradient.execute();
      clFinish(gpu.context.get_command_queue());
      t_gpu_time += get_time() - tgpu;

      fc = float4(1.0f, 1.0, 1.0, 1.0f);
      gpu.limit_gradient.set_arg<void* >( 0, gpu.mhd3_psi.p());
      gpu.limit_gradient.set_arg<void* >( 1, gpu.mhd3.p());
      gpu.limit_gradient.set_arg<void* >( 2, gpu.mhd3_grad_x.p());
      gpu.limit_gradient.set_arg<void* >( 3, gpu.mhd3_grad_y.p());
      gpu.limit_gradient.set_arg<void* >( 4, gpu.mhd3_grad_z.p());
      gpu.limit_gradient.set_arg<float4>(12, &fc);
      tgpu = get_time();
      gpu.limit_gradient.execute();
      clFinish(gpu.context.get_command_queue());
      t_gpu_time += get_time() - tgpu;

    }

    leaf_beg += NGPUBLOCKS;
  }
  clFinish(gpu.context.get_command_queue());
  double t_loop_time = get_time() - t_loop_time0;
  fprintf(stderr, "t_gpu_time= %g   t_loop_time= %g\n",
	  t_gpu_time, t_loop_time);

  gpu.Bxx.d2h(CL_FALSE);
  gpu.Bxy.d2h(CL_FALSE);
  gpu.Bxz.d2h(CL_FALSE);
  gpu.Byy.d2h(CL_FALSE);
  gpu.Byz.d2h(CL_FALSE);
  gpu.Bzz.d2h(CL_FALSE);

  if (!do_first_order) {
  clFinish(gpu.context.get_command_queue());
  const double tx0 = get_time();
    fprintf(stderr, "limit_gradient2 kernel call ... \n");
    gpu.limit_gradient2.setWork(local_n, 128);
    gpu.limit_gradient2.set_arg<void*>( 0, gpu.mhd1_grad_x.p());
    gpu.limit_gradient2.set_arg<void*>( 1, gpu.mhd1_grad_y.p());
    gpu.limit_gradient2.set_arg<void*>( 2, gpu.mhd1_grad_z.p());
    gpu.limit_gradient2.set_arg<void*>( 3, gpu.mhd2_grad_x.p());
    gpu.limit_gradient2.set_arg<void*>( 4, gpu.mhd2_grad_y.p());
    gpu.limit_gradient2.set_arg<void*>( 5, gpu.mhd2_grad_z.p());
    gpu.limit_gradient2.set_arg<void*>( 6, gpu.mhd3_grad_x.p());
    gpu.limit_gradient2.set_arg<void*>( 7, gpu.mhd3_grad_y.p());
    gpu.limit_gradient2.set_arg<void*>( 8, gpu.mhd3_grad_z.p());
    gpu.limit_gradient2.set_arg<void*>( 9, gpu.mhd1_psi.p());
    gpu.limit_gradient2.set_arg<void*>(10, gpu.mhd2_psi.p());
    gpu.limit_gradient2.set_arg<void*>(11, gpu.mhd3_psi.p());
    gpu.limit_gradient2.set_arg<int  >(12, &local_n);
//     fprintf(stderr, "gather_limit2.exec\n");
    gpu.limit_gradient2.execute();
    clFinish(gpu.context.get_command_queue());
    fprintf(stderr, "limit_gradient2 kernel done ... %g sec\n", get_time() - tx0);
    
    gpu.mhd1_grad_x.d2h(CL_FALSE);
    gpu.mhd1_grad_y.d2h(CL_FALSE);
    gpu.mhd1_grad_z.d2h(CL_FALSE);
    gpu.mhd2_grad_x.d2h(CL_FALSE);
    gpu.mhd2_grad_y.d2h(CL_FALSE);
    gpu.mhd2_grad_z.d2h(CL_FALSE);
    gpu.mhd3_grad_x.d2h(CL_FALSE);
    gpu.mhd3_grad_y.d2h(CL_FALSE);
    gpu.mhd3_grad_z.d2h(CL_FALSE);
  } else {
    clFinish(gpu.context.get_command_queue());
  }
  Bxx.resize(local_n);
  Bxy.resize(local_n);
  Bxz.resize(local_n);
  Byy.resize(local_n);
  Byz.resize(local_n);
  Bzz.resize(local_n);

  for (int i = 0; i < local_n; i++) {
    Bxx[i] = gpu.Bxx[i];
    Bxy[i] = gpu.Bxy[i];
    Bxz[i] = gpu.Bxz[i];
    Byy[i] = gpu.Byy[i];
    Byz[i] = gpu.Byz[i];
    Bzz[i] = gpu.Bzz[i];
  }
  
  t_compute += get_time() - t0;

  const double t1 = get_time();  
  import_boundary_Bmatrix();  
  
  for (int i = local_n; i < n_data; i++) {
    gpu.Bxx[i] = Bxx[i];
    gpu.Bxy[i] = Bxy[i];
    gpu.Bxz[i] = Bxz[i];
    gpu.Byy[i] = Byy[i];
    gpu.Byz[i] = Byz[i];
    gpu.Bzz[i] = Bzz[i];
  }
  clFinish(gpu.context.get_command_queue());
  
  gpu.Bxx.h2d(CL_FALSE);
  gpu.Bxy.h2d(CL_FALSE);
  gpu.Bxz.h2d(CL_FALSE);
  gpu.Byy.h2d(CL_FALSE);
  gpu.Byz.h2d(CL_FALSE);
  gpu.Bzz.h2d(CL_FALSE);

  /////

  if (!do_first_order) {
    pmhd_grad[0].resize(local_n);
    pmhd_grad[1].resize(local_n);
    pmhd_grad[2].resize(local_n);
    for (int i = 0; i < local_n; i++) {
      float4 mhd1_grad, mhd2_grad, mhd3_grad;
      ptcl_mhd mhd_grad;
      
      mhd1_grad = gpu.mhd1_grad_x[i];
      mhd2_grad = gpu.mhd2_grad_x[i];
      mhd3_grad = gpu.mhd3_grad_x[i];
      mhd_grad.dens  = mhd1_grad.w;
      mhd_grad.ethm  = mhd2_grad.w;
      mhd_grad.psi   = mhd3_grad.w;
      mhd_grad.vel.x = mhd1_grad.x;
      mhd_grad.vel.y = mhd1_grad.y;
      mhd_grad.vel.z = mhd1_grad.z;
      mhd_grad.B.x   = mhd2_grad.x;
      mhd_grad.B.y   = mhd2_grad.y;
      mhd_grad.B.z   = mhd2_grad.z;
      pmhd_grad[0][i] = mhd_grad;
      
      mhd1_grad = gpu.mhd1_grad_y[i];
      mhd2_grad = gpu.mhd2_grad_y[i];
      mhd3_grad = gpu.mhd3_grad_y[i]; 
      mhd_grad.dens  = mhd1_grad.w;
      mhd_grad.ethm  = mhd2_grad.w;
      mhd_grad.psi   = mhd3_grad.w;
      mhd_grad.vel.x = mhd1_grad.x;
      mhd_grad.vel.y = mhd1_grad.y;
      mhd_grad.vel.z = mhd1_grad.z;
      mhd_grad.B.x   = mhd2_grad.x;
      mhd_grad.B.y   = mhd2_grad.y;
      mhd_grad.B.z   = mhd2_grad.z;
      pmhd_grad[1][i] = mhd_grad;

      mhd1_grad = gpu.mhd1_grad_z[i];
      mhd2_grad = gpu.mhd2_grad_z[i];
      mhd3_grad = gpu.mhd3_grad_z[i];
      mhd_grad.dens  = mhd1_grad.w;
      mhd_grad.ethm  = mhd2_grad.w;
      mhd_grad.psi   = mhd3_grad.w;
      mhd_grad.vel.x = mhd1_grad.x;
      mhd_grad.vel.y = mhd1_grad.y;
      mhd_grad.vel.z = mhd1_grad.z;
      mhd_grad.B.x   = mhd2_grad.x;
      mhd_grad.B.y   = mhd2_grad.y;
      mhd_grad.B.z   = mhd2_grad.z;
      pmhd_grad[2][i] = mhd_grad;
    }
  
    import_boundary_pmhd_grad();

  
    for (int i = local_n; i < n_data; i++) {
      ptcl_mhd mhd_grad;
      float4 mhd1_grad, mhd2_grad, mhd3_grad;
    
      mhd_grad = pmhd_grad[0][i];
      gpu.mhd1_grad_x[i] = float4(mhd_grad.vel.x, mhd_grad.vel.y, mhd_grad.vel.z, mhd_grad.dens);
      gpu.mhd2_grad_x[i] = float4(mhd_grad.B.x,   mhd_grad.B.y,   mhd_grad.B.z,   mhd_grad.ethm);
      gpu.mhd3_grad_x[i] = float4(mhd_grad.psi,  0.0f, 0.0f, 0.0f);
    
      mhd_grad = pmhd_grad[1][i]; 
      gpu.mhd1_grad_y[i] = float4(mhd_grad.vel.x, mhd_grad.vel.y, mhd_grad.vel.z, mhd_grad.dens);
      gpu.mhd2_grad_y[i] = float4(mhd_grad.B.x,   mhd_grad.B.y,   mhd_grad.B.z,   mhd_grad.ethm);
      gpu.mhd3_grad_y[i] = float4(mhd_grad.psi,  0.0f, 0.0f, 0.0f);

      mhd_grad = pmhd_grad[2][i];
      gpu.mhd1_grad_z[i] = float4(mhd_grad.vel.x, mhd_grad.vel.y, mhd_grad.vel.z, mhd_grad.dens);
      gpu.mhd2_grad_z[i] = float4(mhd_grad.B.x,   mhd_grad.B.y,   mhd_grad.B.z,   mhd_grad.ethm);
      gpu.mhd3_grad_z[i] = float4(mhd_grad.psi,  0.0f, 0.0f, 0.0f);
    }

    gpu.mhd1_grad_x.h2d();
    gpu.mhd1_grad_y.h2d();
    gpu.mhd1_grad_z.h2d();
    gpu.mhd2_grad_x.h2d();
    gpu.mhd2_grad_y.h2d();
    gpu.mhd2_grad_z.h2d();
    gpu.mhd3_grad_x.h2d();
    gpu.mhd3_grad_y.h2d();
    gpu.mhd3_grad_z.h2d();
  }

  clFinish(gpu.context.get_command_queue());
  t_communicate += get_time() - t1;

  if (!do_first_order) {
    gpu.mhd1_psi.ocl_free();
    gpu.mhd2_psi.ocl_free();
    gpu.mhd3_psi.ocl_free();
  }

  t_renorm = get_time() - t0;
  
}
