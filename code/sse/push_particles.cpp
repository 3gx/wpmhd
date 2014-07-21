#include "gn.h"

void system::push_particles() {

  do_morton_order = false;

#if 1
  if (first_step_flag) {
    
    compute_dt();

    distribute_particles();
    
    derivs();
    
    int dens_p = 0;
    int ethm_p = 0;
    
    for (int i = 0; i < local_n; i++) {
      
      if (kernel.ndim > 0) pvec[i].pos.x.add(pvec[i].vel.x * dt_global);
      if (kernel.ndim > 1) pvec[i].pos.y.add(pvec[i].vel.y * dt_global);
      if (kernel.ndim > 2) pvec[i].pos.z.add(pvec[i].vel.z * dt_global);
      
      pmhd[i]     += pmhd_dot[i]*dt_global;
      pmhd_dot0[i] = pmhd_dot[i];
      pvel0[i]     = pvec[i].vel;

      if (pmhd[i].dens <= 0.0) {
	dens_p++;
	pmhd[i].dens -= pmhd_dot[i].dens*dt_global;
      }
      if (pmhd[i].ethm <= 0.0) {
	ethm_p++;
	pmhd[i].ethm -=  pmhd_dot[i].ethm*dt_global;
      }
     
      
    }

    distribute_particles();
    
    dt_global0 = dt_global;
    compute_dt();
    
    
    derivs();
    

    first_step_flag = false;
    
  } else {

    // predict

    const float lambda = dt_global/dt_global0;
    const float p1 = 0.5f*lambda * dt_global0;
    const float p2 = p1 + dt_global0;

    int dens_p = 0;
    int ethm_p = 0;
    for (int i = 0; i < local_n; i++) {
      const float3 dr = {pvec[i].vel.x*p2 - pvel0[i].x*p1,
			 pvec[i].vel.y*p2 - pvel0[i].y*p1,
			 pvec[i].vel.z*p2 - pvel0[i].z*p1};

      if (kernel.ndim > 0) pvec[i].pos.x.add(dr.x);
      if (kernel.ndim > 1) pvec[i].pos.y.add(dr.y);
      if (kernel.ndim > 2) pvec[i].pos.z.add(dr.z);

      pmhd[i] += pmhd_dot[i]*p2 - pmhd_dot0[i]*p1;

      if (pmhd[i].dens <= 0.0) {
	dens_p++;
	pmhd[i].dens -= pmhd_dot[i].dens*p2 - pmhd_dot0[i].dens*p1;
      }
      if (pmhd[i].ethm <= 0.0) {
	ethm_p++;
	pmhd[i].ethm -=  pmhd_dot[i].ethm*p2 - pmhd_dot0[i].ethm*p1;
      }
      
    }
    
    distribute_particles();
    std::vector<ptcl_mhd> pmhd_dot1 = pmhd_dot;
    std::vector<float3>   pvel1(local_n);
    for (int i = 0; i < local_n; i++) pvel1[i] = pvec[i].vel;
    
    derivs();

    // corrector

    const float c1 = 0.5f*p1 + p2/6.0f;
    const float c2 = c1/(1.0f + lambda);

    for (int i = 0; i < local_n; i++) {
//       const float3 dr = {(pvec[i].vel.x - pvel0[i].x) * c2 + (pvel0[i].x - pvel1[i].x) * c1,
// 			 (pvec[i].vel.y - pvel0[i].y) * c2 + (pvel0[i].y - pvel1[i].y) * c1,
// 			 (pvec[i].vel.z - pvel0[i].z) * c2 + (pvel0[i].z - pvel1[i].z) * c1};


//       if (kernel.ndim > 0) pvec[i].pos.x.add(dr.x);
//       if (kernel.ndim > 1) pvec[i].pos.y.add(dr.y);
//       if (kernel.ndim > 2) pvec[i].pos.z.add(dr.z);

//       pmhd[i] += (pmhd_dot[i] - pmhd_dot0[i]) * c2 + (pmhd_dot0[i] - pmhd_dot1[i]) * c1;
      pmhd_dot0[i] = pmhd_dot1[i];
      pvel0[i]     = pvel1[i];
    }

    dt_global0 = dt_global;
    compute_dt();

  }

#else

  const real dt = dt_global;
  const real dth = 0.5*dt_global;

  
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
//     if (kernel.ndim > 0) pvec[i].pos.x.add((pvec[i].vel.x - pvec0[i].vel.x) * dth);
//     if (kernel.ndim > 1) pvec[i].pos.y.add((pvec[i].vel.y - pvec0[i].vel.y) * dth);
//     if (kernel.ndim > 2) pvec[i].pos.z.add((pvec[i].vel.z - pvec0[i].vel.z) * dth);
    
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

  compute_dt();

#endif

}
