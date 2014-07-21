#include "gn.h"

void system::mhd_interaction_cl() {
  const double t0 = get_time();
  const int NGBMAX = 192;

  const int4 size4 = get_gpu_size();

  const int size2b = size4.y;
  if (size2b != (int)gpu.nj_both.size()) {
    gpu.nj_both.cmalloc(size2b);
  }

  if (size4.x > (int)gpu.dqdt1.size()) {
    gpu.dqdt1.cmalloc(size4.x);
    gpu.dqdt2.cmalloc(size4.x);
    gpu.dqdt3.cmalloc(size4.x);
    gpu.divB.cmalloc (size4.x);
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
      const std::vector<octnode<TREE_NLEAF>*> &ileaf_list = ngb_leaf_list_outer[leaf];
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
    
    gpu.extract_ijlist.setWork_block1D(NBLOCKDIM, cpu_group_bodies.size());
//     for (int i = 0; i < gpu.nj.size(); i++) 
//       gpu.nj[i] = NGBmean;
//     gpu.nj.h2d();

    gpu.extract_ijlist.set_arg<void* >( 0, gpu.nj.p());
    gpu.extract_ijlist.set_arg<void* >( 1, gpu.nj_both.p());
    gpu.extract_ijlist.set_arg<void* >( 2, gpu.jidx.p());
    gpu.extract_ijlist.set_arg<void* >( 3, gpu.in_group_bodies.p());
    gpu.extract_ijlist.set_arg<void* >( 4, gpu.in_ilist.p());
    gpu.extract_ijlist.set_arg<void* >( 5, gpu.in_jlist.p());
    gpu.extract_ijlist.set_arg<void* >( 6, gpu.ppos.p());
    gpu.extract_ijlist.set_arg<float4>( 7, &gpu.domain_hsize);
    gpu.extract_ijlist.set_arg<float >( 8, NULL, 5 * NBLOCKDIM);
    gpu.extract_ijlist.execute();

    if (max_size > (int)gpu.dwij.size()) {
      gpu.dwij.cmalloc(max_size);
      gpu.mhd1_statesL.cmalloc(max_size);
      gpu.mhd1_statesR.cmalloc(max_size);
      gpu.mhd2_statesL.cmalloc(max_size);
      gpu.mhd2_statesR.cmalloc(max_size);
      gpu.mhd3_statesL.cmalloc(max_size);
      gpu.mhd3_statesR.cmalloc(max_size);
    }

//     gpu.nj.d2h();
//     int njmin = local_n;
//     int njmax = 0;
//     for (int i = 0; i < Ni; i++) {
//       fprintf(stderr, "i= %d [%d]: nj= %d\n", i, Ni, gpu.nj[i]);
//       njmin = std::min(njmin, gpu.nj[i]);
//       njmax = std::max(njmax, gpu.nj[i]);
//     }
//     fprintf(stderr, "  njmin= %d   njmax= %d\n", njmin, njmax);

    gpu.compute_dwij.setWork_1D(Ni, NBLOCKDIM);
    gpu.compute_dwij.set_arg<void* >( 0, gpu.dwij.p());
    gpu.compute_dwij.set_arg<void* >( 1, gpu.in_ilist.p());
    gpu.compute_dwij.set_arg<void* >( 2, gpu.jidx.p());
    gpu.compute_dwij.set_arg<void* >( 3, gpu.nj.p()); 
    gpu.compute_dwij.set_arg<void* >( 4, gpu.ppos.p());
    gpu.compute_dwij.set_arg<void* >( 5, gpu.pvel.p());
    gpu.compute_dwij.set_arg<void* >( 6, gpu.Bxx.p());
    gpu.compute_dwij.set_arg<void* >( 7, gpu.Bxy.p());
    gpu.compute_dwij.set_arg<void* >( 8, gpu.Bxz.p());
    gpu.compute_dwij.set_arg<void* >( 9, gpu.Byy.p());
    gpu.compute_dwij.set_arg<void* >(10, gpu.Byz.p());
    gpu.compute_dwij.set_arg<void* >(11, gpu.Bzz.p());
    gpu.compute_dwij.set_arg<int   >(12, &Ni);
    gpu.compute_dwij.set_arg<float4>(13, &gpu.domain_hsize);
    gpu.compute_dwij.execute();

    gpu.compute_states.setWork_1D(Ni, NBLOCKDIM);
    gpu.compute_states.set_arg<void* >( 6, gpu.in_ilist.p());
    gpu.compute_states.set_arg<void* >( 7, gpu.jidx.p());
    gpu.compute_states.set_arg<void* >( 8, gpu.nj.p()); 
    gpu.compute_states.set_arg<void* >( 9, gpu.dwij.p());
    gpu.compute_states.set_arg<void* >(10, gpu.ppos.p());
    gpu.compute_states.set_arg<int   >(11, &Ni);
    gpu.compute_states.set_arg<float4>(12, &gpu.domain_hsize);

    int reconstruct;
    gpu.compute_states.set_arg<void *>( 0, gpu.mhd1_statesL.p());
    gpu.compute_states.set_arg<void *>( 1, gpu.mhd1_statesR.p());
    gpu.compute_states.set_arg<void *>( 2, gpu.mhd1.p());
    gpu.compute_states.set_arg<void *>( 3, gpu.mhd1_grad_x.p());
    gpu.compute_states.set_arg<void *>( 4, gpu.mhd1_grad_y.p());
    gpu.compute_states.set_arg<void *>( 5, gpu.mhd1_grad_z.p());
    reconstruct = 2 + ((do_first_order) ? 0 : 1);
    gpu.compute_states.set_arg<int   >(13, &reconstruct);
    gpu.compute_states.execute();

    gpu.compute_states.set_arg<void *>( 0, gpu.mhd2_statesL.p());
    gpu.compute_states.set_arg<void *>( 1, gpu.mhd2_statesR.p());
    gpu.compute_states.set_arg<void *>( 2, gpu.mhd2.p());
    gpu.compute_states.set_arg<void *>( 3, gpu.mhd2_grad_x.p());
    gpu.compute_states.set_arg<void *>( 4, gpu.mhd2_grad_y.p());
    gpu.compute_states.set_arg<void *>( 5, gpu.mhd2_grad_z.p());
    reconstruct = 2 + ((do_first_order) ? 0 : 1);
    gpu.compute_states.set_arg<int   >(13, &reconstruct);
    gpu.compute_states.execute();

    gpu.compute_states.set_arg<void *>( 0, gpu.mhd3_statesL.p());
    gpu.compute_states.set_arg<void *>( 1, gpu.mhd3_statesR.p());
    gpu.compute_states.set_arg<void *>( 2, gpu.mhd3.p());
    gpu.compute_states.set_arg<void *>( 3, gpu.mhd3_grad_x.p());
    gpu.compute_states.set_arg<void *>( 4, gpu.mhd3_grad_y.p());
    gpu.compute_states.set_arg<void *>( 5, gpu.mhd3_grad_z.p());
    reconstruct = ((do_first_order) ? 0 : 1);
    gpu.compute_states.set_arg<int   >(13, &reconstruct);
    gpu.compute_states.execute();

    gpu.compute_fluxes.setWork_1D(Ni, NBLOCKDIM);
    gpu.compute_fluxes.set_arg<void* >( 0, gpu.dqdt1.p());
    gpu.compute_fluxes.set_arg<void* >( 1, gpu.dqdt2.p());
    gpu.compute_fluxes.set_arg<void* >( 2, gpu.dqdt3.p());
    gpu.compute_fluxes.set_arg<void* >( 3, gpu.divB.p());
    gpu.compute_fluxes.set_arg<void* >( 4, gpu.mhd1_statesL.p());
    gpu.compute_fluxes.set_arg<void* >( 5, gpu.mhd1_statesR.p());
    gpu.compute_fluxes.set_arg<void* >( 6, gpu.mhd2_statesL.p());
    gpu.compute_fluxes.set_arg<void* >( 7, gpu.mhd2_statesR.p());
    gpu.compute_fluxes.set_arg<void* >( 8, gpu.mhd3_statesL.p());
    gpu.compute_fluxes.set_arg<void* >( 9, gpu.mhd3_statesR.p());
    gpu.compute_fluxes.set_arg<void* >(10, gpu.in_ilist.p());
    gpu.compute_fluxes.set_arg<void* >(11, gpu.jidx.p());
    gpu.compute_fluxes.set_arg<void* >(12, gpu.nj.p()); 
    gpu.compute_fluxes.set_arg<void* >(13, gpu.dwij.p());
    gpu.compute_fluxes.set_arg<int   >(14, &Ni);
    gpu.compute_fluxes.set_arg<float >(15, &gamma_gas);
    gpu.compute_fluxes.set_arg<float >(16, &ch_glob);
    gpu.compute_fluxes.execute();

    leaf_beg += NGPUBLOCKS;
  }
  
  clFinish(gpu.context.get_command_queue());
  fprintf(stderr, "t_gpu_loop= %g sec\n", get_time() - t_gpu_loop);

  gpu.dqdt1.d2h();
  gpu.dqdt2.d2h();
  gpu.dqdt3.d2h();
  gpu.divB.d2h();
  gpu.nj_both.d2h(CL_FALSE);

//   fprintf(stderr, "n_states_tot = %g  t_gpu= %g \n", 1.0f*n_states_tot/local_n, t_gpu);

  pmhd_dot.resize(local_n);
  dwdt_i.resize(local_n);
  divB_i.resize(local_n);

  

#pragma omp parallel for
  for (int i = 0; i < local_n; i++) {
    const particle &pi = pvec[i];
    ptcl_mhd mi = pmhd[i];
    mi.B.x += constB.x;
    mi.B.y += constB.y;
    mi.B.z += constB.z;
    
    const float wi   = pi.wght;
    const float dwdt = gpu.dwdt[i];
    
    ptcl_mhd dQdt;
    dQdt.mass  = -gpu.dqdt1[i].w;
    dQdt.mom.x = -gpu.dqdt1[i].x;
    dQdt.mom.y = -gpu.dqdt1[i].y;
    dQdt.mom.z = -gpu.dqdt1[i].z;
    dQdt.etot  = -gpu.dqdt2[i].w;
    dQdt.wB.x  = -gpu.dqdt2[i].x;
    dQdt.wB.y  = -gpu.dqdt2[i].y;
    dQdt.wB.z  = -gpu.dqdt2[i].z;
    dQdt.psi   = -gpu.dqdt3[i].x; ///pi.wght;
    
    if (eulerian_mode) assert(dwdt == 0.0f);
    
    ///////

    assert(i == pi.local_idx);
    
    const float divB = gpu.divB[i].w;
    dwdt_i[i] = dwdt;
    divB_i[i] = divB/wi;
    
    //


    ptcl_mhd &dot = pmhd_dot[i];
    const ptcl_mhd dot0 = dot;
    dot = dQdt;

//      if (i < 100) {
    const float3 gradpsi = {gpu.divB[i].x, gpu.divB[i].y, gpu.divB[i].z};

//     dQdt.psi += mi.psi * dwdt;
#if 1
    dQdt.wB.x -= mi.vel.x*divB;
    dQdt.wB.y -= mi.vel.y*divB;
    dQdt.wB.z -= mi.vel.z*divB;
    
    const float uB = mi.vel.x*mi.B.x + mi.vel.y*mi.B.y + mi.vel.z*mi.B.z;
    dQdt.mom.x -= mi.B.x*divB;
    dQdt.mom.y -= mi.B.y*divB;
    dQdt.mom.z -= mi.B.z*divB;
    dQdt.etot  -= uB * divB;
#endif


#if 1
    const float pres = compute_pressure(mi.dens, mi.ethm);
    const float   B2 = sqr(mi.B.x) + sqr(mi.B.y) + sqr(mi.B.z);
    const float  cs = sqrt((gamma_gas*pres + B2)/mi.dens);
    
    dQdt.wB.x -= gradpsi.x;
    dQdt.wB.y -= gradpsi.y;
    dQdt.wB.z -= gradpsi.z;
    
    dQdt.psi  -= 0.5f*sqr(cs)*divB*mi.dens;
    dQdt.etot -= mi.B.x*gradpsi.x + mi.B.y*gradpsi.y + mi.B.z*gradpsi.z;
#endif


    // compute dQdt in primitive      
    dot = dQdt;

    const real  m  = mi.dens   * wi;
    const float4 acc = body_forces(pi.pos);

#ifdef _CONSERVATIVE_
#elif defined _SEMICONSERVATIVE_
    dot.ethm = dQdt.etot 
      - (mi.vel.x*dQdt.mom.x + mi.vel.y*dQdt.mom.y + mi.vel.z*dQdt.mom.z)
      + (sqr(mi.vel.x) + sqr(mi.vel.y) + sqr(mi.vel.z)) * dQdt.mass*0.5f;
    
    if (!do_kepler_drift) {
      dot.mom.x += acc.x*m;
      dot.mom.y += acc.y*m;
      dot.mom.z += acc.z*m;
     }
#else

    dot.ethm = dQdt.etot 
      - (mi.vel.x*dQdt.mom.x + mi.vel.y*dQdt.mom.y + mi.vel.z*dQdt.mom.z)
      + (sqr(mi.vel.x) + sqr(mi.vel.y) + sqr(mi.vel.z)) * dQdt.mass*0.5f
      - (mi.B.x*dQdt.wB.x + mi.B.y*dQdt.wB.y + mi.B.z*dQdt.wB.z)
      + (sqr(mi.B.x) + sqr(mi.B.y) + sqr(mi.B.z)) * dwdt * 0.5f;

    if (!do_kepler_drift) {
      dot.mom.x += acc.x*m;
      dot.mom.y += acc.y*m;
      dot.mom.z += acc.z*m;
     }
#endif


  }

  clFinish(gpu.context.get_command_queue());

  int ngb_min = global_n;
  int ngb_max = 0;
  int ngb_sum = 0;
  for (int i = 0; i < local_n; i++) {
    ngb_min = std::min(ngb_min, gpu.nj_both[i]);
    ngb_max = std::max(ngb_max, gpu.nj_both[i]);
    ngb_sum += gpu.nj_both[i];
  }


  int ngb_min_glob, ngb_max_glob, ngb_sum_glob;
  MPI_Allreduce(&ngb_min, &ngb_min_glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ngb_max, &ngb_max_glob, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ngb_sum, &ngb_sum_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (myid == 0) {
    fprintf(stderr, " >>>>> gather_scatter: ngb_min= %d  ngb_max= %d  ngb_mean= %g\n", 
	    ngb_min_glob, ngb_max_glob, ngb_sum_glob*1.0/global_n);
  
  }
  assert(ngb_max < NGBMAX);
  
  t_mhd_interaction = get_time() - t0;
  return;


}
