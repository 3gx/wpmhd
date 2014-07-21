#include "gn.h"

void system::push_particles() {

  do_morton_order = false;

  const real dt = dt_global;
  const real dth = 0.5*dt_global;

#if 0
  
  // predict

  int ethm_p = 0;
  int dens_p = 0;
  for (int i = 0; i < local_n; i++) {
    if (kernel.ndim > 0) pvec[i].pos.x.add(pvec[i].vel.x * dt_global);
    if (kernel.ndim > 1) pvec[i].pos.y.add(pvec[i].vel.y * dt_global);
    if (kernel.ndim > 2) pvec[i].pos.z.add(pvec[i].vel.z * dt_global);
    
    pmhd[i] += pmhd_dot[i] * dt_global;

    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i].dens -= pmhd_dot[i].dens*dt_global;
    }
    if (pmhd[i].ethm <= 0.0) {
      ethm_p++;
      pmhd[i].ethm -= pmhd_dot[i].ethm*dt_global;
    }
    
  }
  
  if (dens_p > 0)  {
    fprintf(stderr, " **proc= %d: predict,  dens_p= %d  (%g %c)\n", myid, dens_p, 100.0*dens_p/local_n, '%');
  }
  if (ethm_p > 0)  {
    fprintf(stderr, " **proc= %d: predict,  ethm_p= %d  (%g %c)\n", myid, ethm_p, 100.0*ethm_p/local_n, '%');
  }


  distribute_particles();

  // store original velocities & accelerations
  std::vector<particle> pvec0     = pvec;
  std::vector<ptcl_mhd> pmhd_dot0 = pmhd_dot;
  
  derivs();

  ethm_p = dens_p = 0;
  
  // correct
  
  for (int i = 0; i < local_n; i++) {
    const float3 drc = {(pvec[i].vel.x - pvec0[i].vel.x)*dth,
			(pvec[i].vel.y - pvec0[i].vel.y)*dth,
			(pvec[i].vel.z - pvec0[i].vel.z)*dth};
    
    if (kernel.ndim > 0) pvec[i].pos.x.add(drc.x);
    if (kernel.ndim > 1) pvec[i].pos.y.add(drc.y);
    if (kernel.ndim > 2) pvec[i].pos.z.add(drc.z);
    
    pmhd[i] += (pmhd_dot[i] - pmhd_dot0[i]) * dth;

    
    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i].dens -= (pmhd_dot[i].dens - pmhd_dot0[i].dens) * dth;
    }
    if (pmhd[i].ethm <= 0.0) {
      ethm_p++;
      pmhd[i].ethm -= (pmhd_dot[i].ethm - pmhd_dot0[i].ethm) * dth;
    }

  }
  
  if (dens_p > 0)  {
    fprintf(stderr, " **proc= %d: correct,  dens_p= %d  (%g %c)\n", myid, dens_p, 100.0*dens_p/local_n, '%');
  }
  if (ethm_p > 0)  {
    fprintf(stderr, " **proc= %d: correct,  ethm_p= %d  (%g %c)\n", myid, ethm_p, 100.0*ethm_p/local_n, '%');
  }
  
//   distribute_particles();


#else

  
  // predict @ full step
  
  distribute_particles();
  
  dt_global = 0;
  derivs();
  dt_global = dt;
  
  int ethm_p = 0;
  int dens_p = 0;
  for (int i = 0; i < local_n; i++) {
    if (kernel.ndim > 0) pvec[i].pos.x.add(pvec[i].vel.x * dt_global);
    if (kernel.ndim > 1) pvec[i].pos.y.add(pvec[i].vel.y * dt_global);
    if (kernel.ndim > 2) pvec[i].pos.z.add(pvec[i].vel.z * dt_global);
    
    pmhd[i] += pmhd_dot[i] * dt_global;

    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i].dens -= pmhd_dot[i].dens*dt_global;
    }
    if (pmhd[i].ethm <= 0.0) {
      ethm_p++;
      pmhd[i].ethm -= pmhd_dot[i].ethm*dt_global;
    }
    
  }
  
  if (dens_p > 0)  {
    fprintf(stderr, " **proc= %d: predict,  dens_p= %d  (%g %c)\n", myid, dens_p, 100.0*dens_p/local_n, '%');
  }
  if (ethm_p > 0)  {
    fprintf(stderr, " **proc= %d: predict,  ethm_p= %d  (%g %c)\n", myid, ethm_p, 100.0*ethm_p/local_n, '%');
  }


  // correct @ full step
  
  distribute_particles();
  
  // store original velocities & accelerations
  std::vector<particle> pvec0     = pvec;
  std::vector<ptcl_mhd> pmhd_dot0 = pmhd_dot;
  
  derivs();
  
  ethm_p = dens_p = 0;
  
  for (int i = 0; i < local_n; i++) {
    if (kernel.ndim > 0) pvec[i].pos.x.add((pvec[i].vel.x - pvec0[i].vel.x) * dth);
    if (kernel.ndim > 1) pvec[i].pos.y.add((pvec[i].vel.y - pvec0[i].vel.y) * dth);
    if (kernel.ndim > 2) pvec[i].pos.z.add((pvec[i].vel.z - pvec0[i].vel.z) * dth);
    
    pmhd[i] += (pmhd_dot[i] - pmhd_dot0[i]) * dth;

    
    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i].dens -= (pmhd_dot[i].dens - pmhd_dot0[i].dens) * dth;
    }
    if (pmhd[i].ethm <= 0.0) {
      ethm_p++;
      pmhd[i].ethm -= (pmhd_dot[i].ethm - pmhd_dot0[i].ethm) * dth;
    }

  }
  
  if (dens_p > 0)  {
    fprintf(stderr, " **proc= %d: correct,  dens_p= %d  (%g %c)\n", myid, dens_p, 100.0*dens_p/local_n, '%');
  }
  if (ethm_p > 0)  {
    fprintf(stderr, " **proc= %d: correct,  ethm_p= %d  (%g %c)\n", myid, ethm_p, 100.0*ethm_p/local_n, '%');
  }

#endif

  // compute timestep
  
  compute_dt();


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
