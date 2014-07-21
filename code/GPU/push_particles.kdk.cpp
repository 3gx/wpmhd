#include "gn.h"

#define DENSMIN 1.0e-5

void system::push_particles() {

  do_morton_order = true;

  real dt  = dt_global;
  real dth = 0.5*dt;
  const real cr  = 0.025f; //0.18;
  
  int idx = 0;
  while (idx < local_n) {
    if (!remove_particles_within_racc(idx)) idx++;
  }
  MPI_Allreduce(&local_n, &global_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (myid == 0) fprintf(stderr, "  global_n = %d\n" , global_n);
  
//   distribute_particles();

  
  // **************** predict @ half-step ***************

  do_first_order = true;
  do_first_order = false;
  
  do_predictor_step = true;
  do_predictor_step = false;

  int dens_p, ethm_p;

  /********* KICK1 ***********/

  const double tkick1 = get_time();
  derivs(false);
  fprintf(stderr, " ***** KICK1 ... done in %g sec\n ", get_time() - tkick1);

  dens_p = ethm_p = 0;
#pragma omp parallel for
  for (int i = 0; i < local_n; i++) {
    boundary_derivatives(i);
    
    ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
    const float cs2 = prim.ethm/prim.dens;
    
    pmhd[i]     += pmhd_dot[i] * dth;
    pmhd[i].psi *= exp(-0.5f*courant_no*cr);
    
    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i] -= pmhd_dot[i]*dth;
      fprintf(stderr, "i= %d  ngb= %d %d  few= %d\n",
	      i, gpu.nj_gather[i], gpu.nj_both[i], pvec[i].few);
      fprintf(stderr, "pos= %g %g %g  vel= %g %g %g\n",
	      pvec[i].pos.x.getu() - gravity_pos.x,
	      pvec[i].pos.y.getu() - gravity_pos.y,
	      pvec[i].pos.z.getu() - gravity_pos.z,
	      pvec[i].vel.x,
	      pvec[i].vel.y,
	      pvec[i].vel.z);
      fprintf(stderr, "  B= %g %g %g %g %g %g \n",
	      gpu.Bxx[i], gpu.Byy[i], gpu.Bzz[i],
	      gpu.Bxy[i], gpu.Bxz[i], gpu.Byz[i]);
      const ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
      fprintf(stderr, " B= %g  rho= %g  p= %g\n",
	      sqrt(sqr(prim.B.x) + 
		   sqr(prim.B.y) + 
		   sqr(prim.B.z)),
	      prim.dens, prim.ethm);
    }
    boundary_particles(i);
    
    prim = pmhd[i].to_primitive(pvec[i].wght);
    
    prim.dens = std::max(prim.dens, (real)DENSMIN);
    
    if (!eulerian_mode) 
      pvec[i].vel = (float3){prim.vel.x, prim.vel.y, prim.vel.z};
    
    if (prim.ethm <= 0.0f) {
      ethm_p++;
      prim.ethm = cs2*prim.dens;
    }
    
    pmhd[i] = prim.to_conservative(pvec[i].wght);
  }

  /******** DRIFT **********/
  const double tdrift = get_time();

#pragma omp parallel for  
  for (int i = 0; i < local_n; i++) {
    boundary_particles(i);
    
    if (kernel.ndim > 0) pvec[i].pos.x.add(pvec[i].vel.x * dt);
    if (kernel.ndim > 1) pvec[i].pos.y.add(pvec[i].vel.y * dt);
    if (kernel.ndim > 2) pvec[i].pos.z.add(pvec[i].vel.z * dt);
    
#if 1
#if 0
    const float f = gpu.dwdt[i]/pvec[i].wght * dt;
    pvec[i].h    *= (fabs(f) < 0.2) ? exp(f/kernel.ndim) : 1.0f;
    pvec[i].wght *= (fabs(f) < 0.2) ? exp(f)             : 1.0f;
#else
    const float f = gpu.dwdt[i]/pvec[i].wght * dt;
    pvec[i].h    *= exp(f/kernel.ndim);
    pvec[i].wght *= exp(f);
#endif
#endif
  }

  fprintf(stderr, " ***** DRIFT ... done in %g sec\n ", get_time() - tdrift);
  
  compute_dt();
  dt  = dt_global;
  dth = 0.5*dt;
  
  /******* KICK2 ********/

  const double tkick2 = get_time();
  derivs(false);
  fprintf(stderr, " ***** KICK2 ... done in %g sec\n ", get_time() - tkick2);

  dens_p = ethm_p = 0;
#pragma omp parallel for  
  for (int i = 0; i < local_n; i++) {
    boundary_derivatives(i);
    
    ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
    const float cs2 = prim.ethm/prim.dens;
    
    pmhd[i]     += pmhd_dot[i] * dth;
    pmhd[i].psi *= exp(-0.5f*courant_no*cr);
    
    if (pmhd[i].dens <= 0.0) {
      dens_p++;
      pmhd[i] -= pmhd_dot[i] * dth;
      fprintf(stderr, "i= %d  ngb= %d %d  few= %d\n",
	      i, gpu.nj_gather[i], gpu.nj_both[i], pvec[i].few);
      fprintf(stderr, "pos= %g %g %g  vel= %g %g %g\n",
	      pvec[i].pos.x.getu() - gravity_pos.x,
	      pvec[i].pos.y.getu() - gravity_pos.y,
	      pvec[i].pos.z.getu() - gravity_pos.z,
	      pvec[i].vel.x,
	      pvec[i].vel.y,
	      pvec[i].vel.z);
      fprintf(stderr, "  B= %g %g %g %g %g %g \n",
	      gpu.Bxx[i], gpu.Byy[i], gpu.Bzz[i],
	      gpu.Bxy[i], gpu.Bxz[i], gpu.Byz[i]);
      const ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
      fprintf(stderr, " B= %g  rho= %g  p= %g\n",
	      sqrt(sqr(prim.B.x) + 
		   sqr(prim.B.y) + 
		   sqr(prim.B.z)),
	      prim.dens, prim.ethm);
    }
    boundary_particles(i);
    
    prim = pmhd[i].to_primitive(pvec[i].wght);
    
    prim.dens = std::max(prim.dens, (real)DENSMIN);

    if (!eulerian_mode) 
      pvec[i].vel = (float3){prim.vel.x, prim.vel.y, prim.vel.z};
    
    if (prim.ethm <= 0.0f) {
      ethm_p++;
      prim.ethm = cs2*prim.dens;
    }
    
    pmhd[i] = prim.to_conservative(pvec[i].wght);

  }

  int dens_p_glob, ethm_p_glob;
  MPI_Allreduce(&dens_p,  &dens_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ethm_p,  &ethm_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  
  if (dens_p_glob > 0 && myid == 0)  {
    fprintf(stderr, " **proc= %d: correct,  dens_p= %d  (%g %c)\n", myid, dens_p_glob, 100.0*dens_p_glob/global_n, '%');
  }
  if (ethm_p_glob > 0 && myid == 0)  {
    fprintf(stderr, " **proc= %d: correct,  ethm_p= %d  (%g %c)\n", myid, ethm_p_glob, 100.0*ethm_p_glob/global_n, '%');
  }
//   assert(dens_p_glob == 0);
//   assert(ethm_p_glob == 0);

  
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
