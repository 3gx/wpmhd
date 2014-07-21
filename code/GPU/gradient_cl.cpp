#include "gn.h"
#include "myMPI.h"

void system::gradient_cl() {
  const double t0 = get_time();
  
  // Allocate GPU arrays: local & imported

  const int n_data = pvec.size();
  const int4 size4 = get_gpu_size();
  const int size2b = size4.y;
  
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
  }
  
  if (do_first_order) {
    t_gradient = get_time() - t0;
    return;
    for (int i = 0; i < n_data; i++) {
      gpu.mhd1_grad_x[i] = float4(0,0,0,0);
      gpu.mhd1_grad_y[i] = float4(0,0,0,0);
      gpu.mhd1_grad_z[i] = float4(0,0,0,0);
      gpu.mhd2_grad_x[i] = float4(0,0,0,0);
      gpu.mhd2_grad_y[i] = float4(0,0,0,0);
      gpu.mhd2_grad_z[i] = float4(0,0,0,0);
      gpu.mhd3_grad_x[i] = float4(0,0,0,0);
      gpu.mhd3_grad_y[i] = float4(0,0,0,0);
      gpu.mhd3_grad_z[i] = float4(0,0,0,0);
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
  
    t_gradient = get_time() - t0;
    return;
  }
  
  // Extract neighbour lists & call compute_gradient.cl
  
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
      const std::vector<octnode<TREE_NLEAF>*> &ileaf_list = ngb_leaf_list[leaf];
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
    gpu.compute_gradient.setWork(-1, nthreads, block);
    gpu.compute_gradient.set_arg<void* >( 5, gpu.Bxx.p());
    gpu.compute_gradient.set_arg<void* >( 6, gpu.Bxy.p());
    gpu.compute_gradient.set_arg<void* >( 7, gpu.Bxz.p());
    gpu.compute_gradient.set_arg<void* >( 8, gpu.Byy.p());
    gpu.compute_gradient.set_arg<void* >( 9, gpu.Byz.p());
    gpu.compute_gradient.set_arg<void* >(10, gpu.Bzz.p());
    gpu.compute_gradient.set_arg<void* >(11, gpu.leaf_list.p());
    gpu.compute_gradient.set_arg<void* >(12, gpu.body_list.p());
    gpu.compute_gradient.set_arg<void* >(13, gpu.ilist.p());
    gpu.compute_gradient.set_arg<void* >(14, gpu.ppos.p());
    gpu.compute_gradient.set_arg<void* >(15, gpu.pvel.p());
    gpu.compute_gradient.set_arg<void* >(16, gpu.jlist.p());
    gpu.compute_gradient.set_arg<void* >(17, gpu.joffset.p());
    gpu.compute_gradient.set_arg<float4>(18, &gpu.domain_hsize);
    gpu.compute_gradient.set_arg<float4>(19, NULL, nthreads * 7);
    float4 fc;

    fc = float4(0.5f, 0.5f, 0.5f, 1.0f);
    gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd1_grad_x.p());
    gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd1_grad_y.p());
    gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd1_grad_z.p());
    gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd1.p());
    gpu.compute_gradient.set_arg<float4>( 4, &fc);
    gpu.compute_gradient.execute();

    fc = float4(1.0f, 1.0f, 1.0f, 0.5f);
    gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd2_grad_x.p());
    gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd2_grad_y.p());
    gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd2_grad_z.p());
    gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd2.p());
    gpu.compute_gradient.set_arg<float4>( 4, &fc);
    gpu.compute_gradient.execute();

    fc = float4(0.5f, 1.0f, 1.0f, 1.0f);
    gpu.compute_gradient.set_arg<void* >( 0, gpu.mhd3_grad_x.p());
    gpu.compute_gradient.set_arg<void* >( 1, gpu.mhd3_grad_y.p());
    gpu.compute_gradient.set_arg<void* >( 2, gpu.mhd3_grad_z.p());
    gpu.compute_gradient.set_arg<void* >( 3, gpu.mhd3.p());
    gpu.compute_gradient.set_arg<float4>( 4, &fc);
    gpu.compute_gradient.execute();
    
    leaf_beg += NGPUBLOCKS;
  }
  clFinish(gpu.context.get_command_queue());

  gpu.mhd1_grad_x.d2h();
  gpu.mhd1_grad_y.d2h();
  gpu.mhd1_grad_z.d2h();
  gpu.mhd2_grad_x.d2h();
  gpu.mhd2_grad_y.d2h();
  gpu.mhd2_grad_z.d2h();
  gpu.mhd3_grad_x.d2h();
  gpu.mhd3_grad_y.d2h();
  gpu.mhd3_grad_z.d2h();

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
//     fprintf(stderr, "i= %d  grad= %g %g %g ; grad_cl= %g %g %g\n",
// 	    i, 
// 	    pmhd_grad[0][i].B.x, pmhd_grad[0][i].B.y, 0.0,
// 	    mhd_grad.B.x, mhd_grad.B.y, 0.0);
	    
	   
	    
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
  t_compute += get_time() - t0;

  import_boundary_pmhd_grad();
  import_boundary_pvel();
  

  const double t1 = get_time();
  
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

  t_communicate += get_time() - t1;

  t_gradient = get_time() - t0;
}
