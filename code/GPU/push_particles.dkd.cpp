#include "gn.h"

#define DENSMIN 1.0e-4
#define DENSDRIFT 1.0e-2

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
  
  distribute_particles();

  
  
  int dens_p, ethm_p;

  /********* DRIFT ***********/
  
#pragma omp parallel for  
  for (int i = 0; i < local_n; i++) {
    boundary_particles(i);
    const ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
    if (do_kepler_drift && prim.dens > DENSDRIFT) {
      assert(kernel.ndim == 3);
      
      double pos[3] = {pvec[i].pos.x.getu() - gravity_pos.x,
		       pvec[i].pos.y.getu() - gravity_pos.y,
		       pvec[i].pos.z.getu() - gravity_pos.z};
      double vel[3] = {pvec[i].vel.x,
		       pvec[i].vel.y,
		       pvec[i].vel.z};
//       pos[2] = vel[2] = 0.0;
      
      const double pos0[3] = {pos[0], pos[1], pos[2]};
      const double vel0[3] = {vel[0], vel[1], vel[2]};

//       assert(false);

      if (dth > 0.0) 
	kep_drift<double>::drift_shep(gravity_mass, pos, vel, dth);
      double dx[3] = {pos[0] - pos0[0], pos[1] - pos0[1], pos[2] - pos0[2]};
      double dv[3] = {vel[0] - vel0[0], vel[1] - vel0[1], vel[2] - vel0[2]};

//       assert(dx[2] == 0.0);
//       assert(dv[2] == 0.0);
#if 0
      pfloat3 ppos = pvec[i].pos;
      
      dx[0] = pvec[i].vel.x * dth*0.5;
      dx[1] = pvec[i].vel.y * dth*0.5;
      dx[2] = pvec[i].vel.z * dth*0.5;
      ppos.x.add(dx[0]);
      ppos.y.add(dx[1]);
      ppos.z.add(dx[2]);
      const float4 acc = body_forces(ppos);
      pvec[i].vel.x += acc.x*dth;
      pvec[i].vel.y += acc.y*dth;
      pvec[i].vel.z += acc.z*dth;
      dx[0] = pvec[i].vel.x * dth*0.5;
      dx[1] = pvec[i].vel.y * dth*0.5;
      dx[2] = pvec[i].vel.z * dth*0.5;
      ppos.x.add(dx[0]);
      ppos.y.add(dx[1]);
      ppos.z.add(dx[2]);

      pvec[i].pos = ppos;
#else

      pvec[i].pos.x.add(dx[0]);
      pvec[i].pos.y.add(dx[1]);
      pvec[i].pos.z.add(dx[2]);
      
      pvec[i].vel.x += dv[0];
      pvec[i].vel.y += dv[1];
      pvec[i].vel.z += dv[2];
#endif

    } else {
      if (kernel.ndim > 0) pvec[i].pos.x.add(pvec[i].vel.x * dth);
      if (kernel.ndim > 1) pvec[i].pos.y.add(pvec[i].vel.y * dth);
      if (kernel.ndim > 2) pvec[i].pos.z.add(pvec[i].vel.z * dth);
    }
    
#if 1
    if (dt == 0.0) continue;
#if 1
    const float f = gpu.dwdt[i]/pvec[i].wght * dth;
    pvec[i].h    *= (fabs(f) < 0.2) ? exp(f/kernel.ndim) : 1.0f;
    pvec[i].wght *= (fabs(f) < 0.2) ? exp(f)             : 1.0f;
#else
    const float f = gpu.dwdt[i]/pvec[i].wght * dth;
    pvec[i].h    *= exp(f/kernel.ndim);
    pvec[i].wght *= exp(f);
#endif
#endif
  }
  
  ////////

  do_predictor_step = false;
  do_first_order = false;

  const double tkick1 = get_time();
  derivs(true);
  fprintf(stderr, " ***** KICK1 ... done in %g sec\n ", get_time() - tkick1);


  for (int i = 0; i < local_n; i++) {
    ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
    const float cs2 = prim.ethm/prim.dens;
    if (cs2 <= 0.0f) {
      fprintf(stderr, "i= %d: d= %g  e= %g\n",
	      i, prim.dens, prim.ethm);
    }
    assert(cs2 > 0.0f);
  }

#if 1
  std::vector<ptcl_mhd> pmhd0 = pmhd;

  dens_p = ethm_p = 0;
#pragma omp parallel for
  for (int i = 0; i < local_n; i++) {
    boundary_derivatives(i);
    
    ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
    const float cs2 = prim.ethm/prim.dens;
    if (cs2 <= 0.0f) {
      fprintf(stderr, "i= %d: d= %g  e= %g\n",
	      i, prim.dens, prim.ethm);
    }
    assert(cs2 > 0.0f);
    
    pmhd[i]     += pmhd_dot[i] * dth;
    pmhd[i].psi *= exp(-0.5f*courant_no*cr);
    
    if (pmhd[i].mass <= 0.0f) {
      dens_p++;
      pmhd[i] = pmhd0[i];
    }
    boundary_particles(i);
    
    prim = pmhd[i].to_primitive(pvec[i].wght);
    
    prim.dens = std::max(prim.dens, (real)DENSMIN);
    
    if (!eulerian_mode) 
      pvec[i].vel = (float3){prim.vel.x, prim.vel.y, prim.vel.z};
    if (eulerian_mode) pvec[i].vel = (float3){0,0,0};
    
    if (prim.ethm <= 0.0f) {
      ethm_p++;
      prim.ethm = cs2*prim.dens;
    }
    
    pmhd[i] = prim.to_conservative(pvec[i].wght);
  }

  int dens_p_glob, ethm_p_glob;
  MPI_Allreduce(&dens_p,  &dens_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ethm_p,  &ethm_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  if (dens_p_glob > 0 && myid == 0)  
    fprintf(stderr, " **proc= %d: predict,  dens_p= %d  (%g %c)\n", myid, dens_p_glob, 100.0*dens_p_glob/global_n, '%');
  if (ethm_p_glob > 0 && myid == 0)  
    fprintf(stderr, " **proc= %d: predict,  ethm_p= %d  (%g %c)\n", myid, ethm_p_glob, 100.0*ethm_p_glob/global_n, '%');
  
  compute_dt();
  
  fprintf(stderr, " dt_old= %g  dt_new= %g      \n", dt, dt_global);
  if (dt_global/0.5 < dt) {
    fprintf(stderr, " *************************** \n");
    fprintf(stderr, " *************************** \n");
    fprintf(stderr, " dt_old= %g  dt_new= %g      \n", dt, dt_global);
    fprintf(stderr, " *************************** \n");
    fprintf(stderr, " *************************** \n");
    
  } else  {

    do_first_order = false;
    const double tkick2 = get_time();
    derivs(false);
    fprintf(stderr, " ***** KICK2 ... done in %g sec\n ", get_time() - tkick2);

    dens_p = ethm_p = 0;
#pragma omp parallel for
    for (int i = 0; i < local_n; i++) {
      boundary_derivatives(i);
    
      ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
      const float cs2 = prim.ethm/prim.dens;
      assert(cs2 > 0.0);
    
      pmhd[i]     = pmhd0[i] + pmhd_dot[i] * dt;
      pmhd[i].psi *= exp(-courant_no*cr);
    
      if (pmhd[i].mass <= 0.0) {
	dens_p++;
	pmhd[i] = pmhd0[i];
      }
      boundary_particles(i);
      
      prim = pmhd[i].to_primitive(pvec[i].wght);
    
      prim.dens = std::max(prim.dens, (real)DENSMIN);
    
      if (!eulerian_mode) 
	pvec[i].vel = (float3){prim.vel.x, prim.vel.y, prim.vel.z};
      if (eulerian_mode) pvec[i].vel = (float3){0,0,0};
    
      if (prim.ethm <= 0.0f) {
	ethm_p++;
	prim.ethm = cs2*prim.dens;
      }


      assert(prim.ethm > 0.0f);
      pmhd[i] = prim.to_conservative(pvec[i].wght);
    }

    MPI_Allreduce(&dens_p,  &dens_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ethm_p,  &ethm_p_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
    if (dens_p_glob > 0 && myid == 0)  {
      fprintf(stderr, " **proc= %d: correct,  dens_p= %d  (%g %c)\n", myid, dens_p_glob, 100.0*dens_p_glob/global_n, '%');
    }
    if (ethm_p_glob > 0 && myid == 0)  {
      fprintf(stderr, " **proc= %d: correct,  ethm_p= %d  (%g %c)\n", myid, ethm_p_glob, 100.0*ethm_p_glob/global_n, '%');
    }
  }

//   compute_dt();
#endif
  
  /******** DRIFT **********/

#pragma omp parallel for  
  for (int i = 0; i < local_n; i++) {
    boundary_particles(i);
    const ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
    if (do_kepler_drift && prim.dens > DENSDRIFT) {
      
      double pos[3] = {pvec[i].pos.x.getu() - gravity_pos.x,
		       pvec[i].pos.y.getu() - gravity_pos.y,
		       pvec[i].pos.z.getu() - gravity_pos.z};
      double vel[3] = {pvec[i].vel.x,
		       pvec[i].vel.y,
		       pvec[i].vel.z};
//       pos[2] = vel[2] = 0.0;

      const double pos0[3] = {pos[0], pos[1], pos[2]};
      const double vel0[3] = {vel[0], vel[1], vel[2]};
      
      if (dth > 0.0) 
	kep_drift<double>::drift_shep(gravity_mass, pos, vel, dth);
      double dx[3] = {pos[0] - pos0[0], pos[1] - pos0[1], pos[2] - pos0[2]};
      double dv[3] = {vel[0] - vel0[0], vel[1] - vel0[1], vel[2] - vel0[2]};

#if 0
      pfloat3 ppos = pvec[i].pos;
      
      dx[0] = pvec[i].vel.x * dth*0.5;
      dx[1] = pvec[i].vel.y * dth*0.5;
      dx[2] = pvec[i].vel.z * dth*0.5;
      ppos.x.add(dx[0]);
      ppos.y.add(dx[1]);
      ppos.z.add(dx[2]);
      const float4 acc = body_forces(ppos);
      pvec[i].vel.x += acc.x*dth;
      pvec[i].vel.y += acc.y*dth;
      pvec[i].vel.z += acc.z*dth;
      dx[0] = pvec[i].vel.x * dth*0.5;
      dx[1] = pvec[i].vel.y * dth*0.5;
      dx[2] = pvec[i].vel.z * dth*0.5;
      ppos.x.add(dx[0]);
      ppos.y.add(dx[1]);
      ppos.z.add(dx[2]);

      pvec[i].pos = ppos;
#else
      pvec[i].pos.x.add(dx[0]);
      pvec[i].pos.y.add(dx[1]);
      pvec[i].pos.z.add(dx[2]);
      
      pvec[i].vel.x += dv[0];
      pvec[i].vel.y += dv[1];
      pvec[i].vel.z += dv[2];
#endif
    } else {
      if (kernel.ndim > 0) pvec[i].pos.x.add(pvec[i].vel.x * dth);
      if (kernel.ndim > 1) pvec[i].pos.y.add(pvec[i].vel.y * dth);
      if (kernel.ndim > 2) pvec[i].pos.z.add(pvec[i].vel.z * dth);
    }

#if 1
    if (dt == 0.0) continue;
#if 1
    const float f = gpu.dwdt[i]/pvec[i].wght * dth;
    if (eulerian_mode) assert(f == 0.0f);
    pvec[i].h    *= (fabs(f) < 0.2) ? exp(f/kernel.ndim) : 1.0f;
    pvec[i].wght *= (fabs(f) < 0.2) ? exp(f)             : 1.0f;
#else
    const float f = gpu.dwdt[i]/pvec[i].wght * dth;
    pvec[i].h    *= exp(f/kernel.ndim);
    pvec[i].wght *= exp(f);
#endif
#endif
  }

  
  morton_order();

}
