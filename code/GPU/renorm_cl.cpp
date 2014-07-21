#include "gn.h"
#include "myMPI.h"

int4 system::get_gpu_size() {
  const int n_data = pvec.size();
  
  int local_n2b = 1;
  while (local_n2b < local_n) local_n2b = local_n2b << 1;
  int size2b = 0;
  while (size2b < n_data) size2b += 2*local_n2b;

  return (int4){local_n2b, size2b, 0,0};
}


void system::renorm_cl() {
  const double t0 = get_time();
  const int NGBMAX = 192;

  // Allocate GPU arrays, both local & imported

  const int4 size4 = get_gpu_size();
  const int size2b = size4.y;
  
  if (size2b != (int)gpu.dwdt.size()) {
    gpu.dwdt.cmalloc(size2b);
  }

  if (size2b != (int)gpu.nj_gather.size()) {
    gpu.nj_gather.cmalloc(size2b);
  }

  if (size2b != (int)gpu.Bxx.size()) {
    gpu.Bxx.cmalloc(size2b);
    gpu.Bxy.cmalloc(size2b);
    gpu.Bxz.cmalloc(size2b);
    gpu.Byy.cmalloc(size2b);
    gpu.Byz.cmalloc(size2b);
    gpu.Bzz.cmalloc(size2b);
  }
  
  if (size2b != (int)gpu.mhd1_grad_x.size()) {
    gpu.mhd1_grad_x.cmalloc(size2b);
    gpu.mhd1_grad_y.cmalloc(size2b);
    gpu.mhd1_grad_z.cmalloc(size2b);
    gpu.mhd2_grad_x.cmalloc(size2b);
    gpu.mhd2_grad_y.cmalloc(size2b);
    gpu.mhd2_grad_z.cmalloc(size2b);
    gpu.mhd3_grad_x.cmalloc(size2b);
    gpu.mhd3_grad_y.cmalloc(size2b);
    gpu.mhd3_grad_z.cmalloc(size2b);
  }

  ////////  Allocate CPU-arrays

  std::vector<int > cpu_ilist;
  std::vector<int > cpu_jlist;
  std::vector<int4> cpu_group_bodies;
  cpu_ilist.reserve(1024);
  cpu_jlist.reserve(1024);
  cpu_group_bodies.reserve(1024);

  /////  Prepare CPU-arrays
  const double t_gpu_loop = get_time();
  
  const int n_leaves = local_tree.leaf_list.size();
  int leaf_beg = 0;
  while (leaf_beg < n_leaves) {
    cpu_ilist.clear();
    cpu_jlist.clear();
    cpu_group_bodies.clear();

    const int leaf_end = std::min(leaf_beg + NGPUBLOCKS, n_leaves);
    for (int leaf = leaf_beg; leaf < leaf_end; leaf++) {

      // extract i-particles

      const octnode<TREE_NLEAF> &inode  = *local_tree.leaf_list[leaf];
      const int ibeg = cpu_ilist.size();
      for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next)  
	cpu_ilist.push_back(ibp->pp->local_idx);
      const int iend = cpu_ilist.size();

      // extract j-particles

      const int jbeg = cpu_jlist.size();
#if 0
      const std::vector<octnode<TREE_NLEAF>*> &ileaf_list = ngb_leaf_list[leaf];
#else
      const std::vector<octnode<TREE_NLEAF>*> &ileaf_list = ngb_leaf_list_outer[leaf];
#endif
      const int n_ngb_leaves = ileaf_list.size();
      for (int ngb_leaf = 0; ngb_leaf < n_ngb_leaves; ngb_leaf++) {
	const octnode<TREE_NLEAF> &jnode = *ileaf_list[ngb_leaf];
	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next)  
	  cpu_jlist.push_back(jbp->pp->local_idx);
      }
      const int jend = cpu_jlist.size();

      // store jbeg & jend for each block

      for (int i = 0;  i < inode.nparticle; i += NBLOCKDIM) {
	const int ni = std::min(NBLOCKDIM, inode.nparticle - i);
	cpu_group_bodies.push_back((int4){ibeg + i, ni, jbeg, jend});
      }
      assert(ibeg + inode.nparticle == iend);
      
    }

    ////////
    
    clFinish(gpu.context.get_command_queue());
    
    gpu.in_ilist.cmalloc       (cpu_ilist.size()); 
    gpu.in_jlist.cmalloc       (cpu_jlist.size());
    gpu.in_group_bodies.cmalloc(cpu_group_bodies.size());
    
    int Ni  = cpu_ilist.size();
    int NiM = ((Ni - 1)/NBLOCKDIM + 1)*NBLOCKDIM;

    const int max_size = NiM * NGBMAX;
    if (max_size > (int)gpu.jidx.size() ) gpu.jidx.cmalloc(max_size);
    if (NiM      > (int)gpu.nj.size()   ) gpu.nj.cmalloc(NiM);
    
    for (size_t i = 0; i < cpu_ilist.size();        i++) gpu.in_ilist       [i] = cpu_ilist       [i];
    for (size_t i = 0; i < cpu_jlist.size();        i++) gpu.in_jlist       [i] = cpu_jlist       [i];
    for (size_t i = 0; i < cpu_group_bodies.size(); i++) gpu.in_group_bodies[i] = cpu_group_bodies[i];

    gpu.in_ilist.h2d       (CL_FALSE);
    gpu.in_jlist.h2d       (CL_FALSE);
    gpu.in_group_bodies.h2d(CL_FALSE);

    //////
    
#if 0
    gpu.extract_gather_list.setWork_block1D(NBLOCKDIM, cpu_group_bodies.size());
    gpu.extract_gather_list.set_arg<void* >( 0, gpu.nj.p());
    gpu.extract_gather_list.set_arg<void* >( 1, gpu.nj_gather.p());
    gpu.extract_gather_list.set_arg<void* >( 2, gpu.jidx.p());
    gpu.extract_gather_list.set_arg<void* >( 3, gpu.in_group_bodies.p());
    gpu.extract_gather_list.set_arg<void* >( 4, gpu.in_ilist.p());
    gpu.extract_gather_list.set_arg<void* >( 5, gpu.in_jlist.p());
    gpu.extract_gather_list.set_arg<void* >( 6, gpu.ppos.p());
    gpu.extract_gather_list.set_arg<float4>( 7, &gpu.domain_hsize);
    gpu.extract_gather_list.set_arg<float >( 8, NULL, 4 * NBLOCKDIM);
    gpu.extract_gather_list.execute();
#else
    gpu.extract_ijlist.setWork_block1D(NBLOCKDIM, cpu_group_bodies.size());
    gpu.extract_ijlist.set_arg<void* >( 0, gpu.nj.p());
    gpu.extract_ijlist.set_arg<void* >( 1, gpu.nj_gather.p());
    gpu.extract_ijlist.set_arg<void* >( 2, gpu.jidx.p());
    gpu.extract_ijlist.set_arg<void* >( 3, gpu.in_group_bodies.p());
    gpu.extract_ijlist.set_arg<void* >( 4, gpu.in_ilist.p());
    gpu.extract_ijlist.set_arg<void* >( 5, gpu.in_jlist.p());
    gpu.extract_ijlist.set_arg<void* >( 6, gpu.ppos.p());
    gpu.extract_ijlist.set_arg<float4>( 7, &gpu.domain_hsize);
    gpu.extract_ijlist.set_arg<float >( 8, NULL, 5 * NBLOCKDIM);
    gpu.extract_ijlist.execute();
#endif

//     gpu.nj.d2h();
//     gpu.jidx.d2h();
//     gpu.nj_gather.d2h();
//     for (size_t block = 0; block < cpu_group_bodies.size(); block++) {
//       const int4 ifirst = gpu.in_group_bodies[block];
//       fprintf(stderr, "block= %d:  group_block_bodies= (%d %d %d %d) \n", 
// 	      (int)block,
// 	      ifirst.x, ifirst.y, ifirst.z, ifirst.w);
//       for (int i = ifirst.x; i < ifirst.x + ifirst.y; i++) {
// 	const int nj = gpu.nj[i];
// 	const int bodyId = gpu.in_ilist[i];
// 	const int  blockId = i/NBLOCKDIM;
// 	const int ijoffset = blockId * (NGBMAX*NBLOCKDIM) + i%NBLOCKDIM;
// 	fprintf(stderr, " i= %d  bodyId= %d  nj= %d  ijoffset= cpu:%d  gpu:%d \n",
// 		i, bodyId, nj, ijoffset, gpu.nj_gather[bodyId]);
// 	fprintf(stderr, "   ");
// 	for (int j = 0; j < nj; j++) {
// 	  fprintf(stderr, "%d ", gpu.jidx[ijoffset + j * NBLOCKDIM]);
// 	}
// 	fprintf(stderr, "\n");
//       }
//     }

//     exit(-1);
    gpu.compute_Bmatrix.setWork_1D(Ni, NBLOCKDIM);
    gpu.compute_Bmatrix.set_arg<void* >( 0, gpu.Bxx.p());
    gpu.compute_Bmatrix.set_arg<void* >( 1, gpu.Bxy.p());
    gpu.compute_Bmatrix.set_arg<void* >( 2, gpu.Bxz.p());
    gpu.compute_Bmatrix.set_arg<void* >( 3, gpu.Byy.p());
    gpu.compute_Bmatrix.set_arg<void* >( 4, gpu.Byz.p());
    gpu.compute_Bmatrix.set_arg<void* >( 5, gpu.Bzz.p());
    gpu.compute_Bmatrix.set_arg<void* >( 6, gpu.dwdt.p());
    gpu.compute_Bmatrix.set_arg<void* >( 7, gpu.in_ilist.p());
    gpu.compute_Bmatrix.set_arg<void* >( 8, gpu.jidx.p());
    gpu.compute_Bmatrix.set_arg<void* >( 9, gpu.nj.p()); 
    gpu.compute_Bmatrix.set_arg<void* >(10, gpu.ppos.p());
    gpu.compute_Bmatrix.set_arg<void* >(11, gpu.pvel.p());
    gpu.compute_Bmatrix.set_arg<float4>(12, &gpu.domain_hsize);
    gpu.compute_Bmatrix.set_arg<int   >(13, &Ni);
    gpu.compute_Bmatrix.execute();

//     gpu.Bxx.d2h();
//     for (int i = 0; i < Ni; i++) {
//       const int bodyId = gpu.in_ilist[i];
//       fprintf(stderr, "i= %d  bodyId= %d  Bxx= %g\n",
//               i, bodyId, gpu.Bxx[bodyId]);
//     }
//     exit(-1);

    if (!do_first_order) {
      gpu.compute_gradient.setWork_1D(Ni, NBLOCKDIM);
      gpu.compute_gradient.set_arg<void* >( 4, gpu.in_ilist.p());
      gpu.compute_gradient.set_arg<void* >( 5, gpu.jidx.p());
      gpu.compute_gradient.set_arg<void* >( 6, gpu.nj.p()); 
      gpu.compute_gradient.set_arg<void* >( 7, gpu.ppos.p());
      gpu.compute_gradient.set_arg<void* >( 8, gpu.pvel.p());
      gpu.compute_gradient.set_arg<void* >( 9, gpu.Bxx.p());
      gpu.compute_gradient.set_arg<void* >(10, gpu.Bxy.p());
      gpu.compute_gradient.set_arg<void* >(11, gpu.Bxz.p());
      gpu.compute_gradient.set_arg<void* >(12, gpu.Byy.p());
      gpu.compute_gradient.set_arg<void* >(13, gpu.Byz.p());
      gpu.compute_gradient.set_arg<void* >(14, gpu.Bzz.p());
      gpu.compute_gradient.set_arg<float4>(15, &gpu.domain_hsize);
      gpu.compute_gradient.set_arg<int   >(16, &Ni);

      float4 fac;

      ///////// MHD1

#if 1
      const float fc = 0.9f; // 0.9f;
      const float fn = 0.9f; // 0.5f;
#elif 0
      const float fc = 0.5f; // 0.9f;
      const float fn = 0.5f; // 0.5f;
#else
      const float fc = 1.0f;
      const float fn = 1.0f;
#endif
      fac = float4(fn, fn, fn, fc);
      gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd1_grad_x.p());
      gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd1_grad_y.p());
      gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd1_grad_z.p());
      gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd1.p());
      gpu.compute_gradient.set_arg<float4>(17, &fac);
      gpu.compute_gradient.execute();
      
      fac = float4(fc, fc, fc, fn);
      gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd2_grad_x.p());
      gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd2_grad_y.p());
      gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd2_grad_z.p());
      gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd2.p());
      gpu.compute_gradient.set_arg<float4>(17, &fac);
      gpu.compute_gradient.execute();
      
      fac = float4(fn, fc, fc, fc);
      gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd3_grad_x.p());
      gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd3_grad_y.p());
      gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd3_grad_z.p());
      gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd3.p());
      gpu.compute_gradient.set_arg<float4>(17, &fac);
      gpu.compute_gradient.execute();

    }
    leaf_beg += NGPUBLOCKS;
  }
  clFinish(gpu.context.get_command_queue());
  fprintf(stderr, "t_gpu_loop= %g sec\n", get_time() - t_gpu_loop);

  gpu.dwdt.d2h(CL_FALSE);
  gpu.nj_gather.d2h(CL_FALSE);

  gpu.Bxx.d2h(CL_FALSE);
  gpu.Bxy.d2h(CL_FALSE);
  gpu.Bxz.d2h(CL_FALSE);
  gpu.Byy.d2h(CL_FALSE);
  gpu.Byz.d2h(CL_FALSE);
  gpu.Bzz.d2h(CL_FALSE);

  if (!do_first_order) {
//     fprintf(stderr, "limit_gradient2 kernel call ... \n");
//     gpu.limit_gradient2.setWork(local_n, 64);
//     gpu.limit_gradient2.set_arg<void*>( 0, gpu.mhd1_grad_x.p());
//     gpu.limit_gradient2.set_arg<void*>( 1, gpu.mhd1_grad_y.p());
//     gpu.limit_gradient2.set_arg<void*>( 2, gpu.mhd1_grad_z.p());
//     gpu.limit_gradient2.set_arg<void*>( 3, gpu.mhd2_grad_x.p());
//     gpu.limit_gradient2.set_arg<void*>( 4, gpu.mhd2_grad_y.p());
//     gpu.limit_gradient2.set_arg<void*>( 5, gpu.mhd2_grad_z.p());
//     gpu.limit_gradient2.set_arg<void*>( 6, gpu.mhd3_grad_x.p());
//     gpu.limit_gradient2.set_arg<void*>( 7, gpu.mhd3_grad_y.p());
//     gpu.limit_gradient2.set_arg<void*>( 8, gpu.mhd3_grad_z.p());
//     gpu.limit_gradient2.set_arg<void*>( 9, gpu.mhd1_psi.p());
//     gpu.limit_gradient2.set_arg<void*>(10, gpu.mhd2_psi.p());
//     gpu.limit_gradient2.set_arg<void*>(11, gpu.mhd3_psi.p());
//     gpu.limit_gradient2.set_arg<int  >(12, &local_n);
// //     fprintf(stderr, "gather_limit2.exec\n");
//     gpu.limit_gradient2.execute();
    
//     gpu.mhd1_grad_x.d2h(CL_FALSE);
//     gpu.mhd1_grad_y.d2h(CL_FALSE);
//     gpu.mhd1_grad_z.d2h(CL_FALSE);
//     gpu.mhd2_grad_x.d2h(CL_FALSE);
//     gpu.mhd2_grad_y.d2h(CL_FALSE);
//     gpu.mhd2_grad_z.d2h(CL_FALSE);
//     gpu.mhd3_grad_x.d2h(CL_FALSE);
//     gpu.mhd3_grad_y.d2h(CL_FALSE);
//     gpu.mhd3_grad_z.d2h(CL_FALSE);
  } else {
    clFinish(gpu.context.get_command_queue());
  }
  clFinish(gpu.context.get_command_queue());
//   fprintf(stderr, "limit_gradient2 kernel done ... \n");

#if 0  
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
#endif


  clFinish(gpu.context.get_command_queue());

  int ngb_min = global_n;
  int ngb_max = 0;
  int ngb_sum = 0;
  for (int i = 0; i < local_n; i++) {
    ngb_min = std::min(ngb_min, gpu.nj_gather[i]);
    ngb_max = std::max(ngb_max, gpu.nj_gather[i]);
    ngb_sum += gpu.nj_gather[i];
  }

      
  int ngb_min_glob, ngb_max_glob, ngb_sum_glob;
  MPI_Allreduce(&ngb_min, &ngb_min_glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ngb_max, &ngb_max_glob, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ngb_sum, &ngb_sum_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (myid == 0) {
    fprintf(stderr, " >>>>> gather: ngb_min= %d  ngb_max= %d  ngb_mean= %g\n", 
	    ngb_min_glob, ngb_max_glob, ngb_sum_glob*1.0/global_n);
  
  }
  assert(ngb_max < NGBMAX);


  t_renorm = get_time() - t0;
  
}
