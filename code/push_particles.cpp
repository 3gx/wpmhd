#include "gn.h"

void system::push_particles() {

//   do_morton_order = true;
  do_morton_order = false;

  const real dt  =     dt_global;
  const real dth = 0.5*dt_global;
  const real cr  = 0.025f; //0.18;
  
  int idx = 0;
  while (idx < local_n) {
    remove_particles_within_racc(idx);
    idx++;
  }
  MPI_Allreduce(&local_n, &global_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (myid == 0) fprintf(stderr, "  global_n = %d\n" , global_n);

  // predict @ half-step

  distribute_particles();

//   do_first_order = true;
  dt_global = 0;
  derivs();
  dt_global = dt;
  
  for (int i = 0; i < local_n; i++) {
    const ptcl_mhd &mi = pmhd[i];
    const particle &pi = pvec[i];
    boundary_particles(i);

    if (kernel.ndim > 0) pvec[i].pos.x.add(pvec[i].vel.x * dth);
    if (kernel.ndim > 1) pvec[i].pos.y.add(pvec[i].vel.y * dth);
    if (kernel.ndim > 2) pvec[i].pos.z.add(pvec[i].vel.z * dth);
  }

  distribute_particles();
  
  std::vector<particle> pvec0 = pvec;
  std::vector<ptcl_mhd> pmhd0 = pmhd;
  
  int ethm_p = 0, dens_p = 0;
  for (int i = 0; i < local_n; i++) {
    ptcl_mhd prim = (pmhd[i] * (1.0/pvec[i].wght)).to_primitive();
    const float cs2 = prim.ethm/prim.dens;
    
    pmhd[i] += pmhd_dot[i] * dth;
//     pmhd[i].psi *= exp(-cdth / psi_tau[i]/cr);
    pmhd[i].psi *= exp(-0.5f*courant_no*cr);
    
    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i].dens -= pmhd_dot[i].dens*dth;
    }
    boundary_particles(i);
    
    prim = (pmhd[i] * (1.0/pvec[i].wght)).to_primitive();
    if (prim.ethm <= 0.0) {
      ethm_p++;
      prim.ethm = cs2*prim.dens;
    }
    pmhd[i] = prim.to_conservative() * pvec[i].wght;
    
  }

  int dens_p_glob, ethm_p_glob;
  MPI_Allreduce(&dens_p,  &dens_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ethm_p,  &ethm_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  
  if (dens_p_glob > 0 && myid == 0)  {
    fprintf(stderr, " **proc= %d: predict,  dens_p= %d  (%g %c)\n", myid, dens_p_glob, 100.0*dens_p_glob/global_n, '%');
  }
  if (ethm_p_glob > 0 && myid == 0)  {
    fprintf(stderr, " **proc= %d: predict,  ethm_p= %d  (%g %c)\n", myid, ethm_p_glob, 100.0*ethm_p_glob/global_n, '%');
  }
  assert(dens_p_glob == 0);

  // correct @ full step
  
  do_first_order = false;
  derivs();
  
  ethm_p = dens_p = 0;
  
  for (int i = 0; i < local_n; i++) {
    boundary_particles(i);
    const float3 dpos = {(pvec[i].vel.x - 0.5f*pvec0[i].vel.x) * dt,
 (pvec[i].vel.y - 0.5f*pvec0[i].vel.y) * dt,
			 (pvec[i].vel.z - 0.5f*pvec0[i].vel.z) * dt};
    if (kernel.ndim > 0) pvec[i].pos.x.add(dpos.x);
    if (kernel.ndim > 1) pvec[i].pos.y.add(dpos.y);
    if (kernel.ndim > 2) pvec[i].pos.z.add(dpos.z);
    
    ptcl_mhd prim   = (pmhd0[i] * (1.0/pvec[i].wght)).to_primitive();
    const float cs2 = prim.ethm/prim.dens;
    
    pmhd[i] = pmhd0[i] + pmhd_dot[i]*dt;
//     pmhd[i].psi *= exp(-dt / psi_tau[i]/cr);
    pmhd[i].psi *= exp(-courant_no*cr);
    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i].dens = pmhd0[i].dens;
    }
    boundary_particles(i);
    
    prim = (pmhd[i] * (1.0/pvec[i].wght)).to_primitive();
    if (pmhd[i].ethm <= 0.0) {
      ethm_p++;
      prim.ethm = cs2*prim.dens;
    }
    pmhd[i] = prim.to_conservative() * pvec[i].wght;

  }
  
  
  MPI_Allreduce(&dens_p,  &dens_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ethm_p,  &ethm_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  
  if (dens_p_glob > 0 && myid == 0)  {
    fprintf(stderr, " **proc= %d: correct,  dens_p= %d  (%g %c)\n", myid, dens_p_glob, 100.0*dens_p_glob/global_n, '%');
  }
  if (ethm_p_glob > 0 && myid == 0)  {
    fprintf(stderr, " **proc= %d: correct,  ethm_p= %d  (%g %c)\n", myid, ethm_p_glob, 100.0*ethm_p_glob/global_n, '%');
  }

  assert(dens_p_glob == 0);
//   assert(ethm_p == 0);
//   assert(dens_p == 0);
  
  // compute timestep
  
//   compute_dt();

//   distribute_particles();
  
  morton_order();

//   double t_communicate_local = t_communicate;
//   double t_compute_local     = t_compute;
//   MPI_Allreduce(&t_communicate_local, 
// 		&t_communicate, 
// 		1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//   MPI_Allreduce(&t_compute_local, &t_compute, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  

//   double t_interaction_local = t_interaction;
//   double t_renorm_local      = t_renorm;
//   double t_grad_local        = t_grad;
//   double t_tree_local        = t_tree;
//   double t_weight_local      = t_weight;
//   MPI_Allreduce(&t_interaction_local, &t_interaction, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//   MPI_Allreduce(&t_renorm_local,      &t_renorm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//   MPI_Allreduce(&t_weight_local,      &t_weight, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//   MPI_Allreduce(&t_grad_local,        &t_grad,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//   MPI_Allreduce(&t_tree_local,        &t_tree,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

}
