#include "gn.h"

void system::compute_dt() {

  double t0 = get_time();
#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)
  convert_to_primitives();
#endif

  float dt_local = +HUGE;
  for (int i = 0; i < local_n; i++) {
    const particle &pi = pvec[i];
    const ptcl_mhd &mi = pmhd[i];

    const float  B2 = sqr(mi.B.x) + sqr(mi.B.y) + sqr(mi.B.z);
    const float  pres = compute_pressure(mi.dens, mi.ethm);
    const float  cs = sqrt((gamma_gas*pres + B2)/mi.dens);
    const float3 dv = {mi.vel.x - pi.vel.x,
		       mi.vel.y - pi.vel.y,
		       mi.vel.z - pi.vel.z};
		       
    
    float dt = 0;
    switch(kernel.ndim) {
    case 1:
      dt = pi.wght/(cs + std::abs(dv.x));
//       dt = pi.h/(cs + std::abs(dv.x));
      break;
    case 2:
      dt = sqrt(pi.wght/M_PI)/(cs + sqrt(dv.x*dv.x + dv.y*dv.y));
//       dt = pi.h/(cs + sqrt(dv.x*dv.x + dv.y*dv.y));
      break;
    case 3:
      dt = powf(pi.wght*3.0f/4.0f/M_PI, 1.0f/3.0f)/(cs + sqrt(dv.x*dv.x + dv.y*dv.y + dv.z*dv.z));
//       dt = pi.h/(cs + sqrt(dv.x*dv.x + dv.y*dv.y + dv.z*dv.z));
      break;
    default:
      assert(kernel.ndim > 0 && kernel.ndim <= 3);
    };
    

    assert(dt > 0.0f);
    dt_local = std::min(dt_local, dt);
    
    
  }
  dt_local *= courant_no;
//   fprintf(stderr, "proc= %d: ******** dt_local= %g ********* \n", myid, dt_local);

  // communicate dt_local to compute dt_global
  t_compute += get_time() - t0;

  t0 = get_time();

  float dt_glob = 0;
  MPI_Allreduce(&dt_local, 
		&dt_glob, 
		1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  dt_global = dt_glob;
  
  t_communicate += get_time() - t0;

  t0 = get_time();
#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)
  restore_conservative();
#endif
  t_compute += get_time() - t0;

  int local_n_min, local_n_max, local_n_mean;
  int import_n_min, import_n_max, import_n_mean;
  MPI_Allreduce(&local_n, &local_n_min,  1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_n, &local_n_max,  1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_n, &local_n_mean, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MPI_Allreduce(&import_n, &import_n_min,  1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&import_n, &import_n_max,  1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&import_n, &import_n_mean, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  double local_n2 = sqr((double)local_n);
  double import_n2 = sqr((double)import_n);
  double l2, i2;

  MPI_Allreduce(&local_n2,  &l2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&import_n2, &i2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);



  double t_communicate_local = t_communicate;
  double t_compute_local     = t_compute;
  double tcomp_loc = t_compute;
  double t2comp_loc = sqr(t_compute);
  double tcomm_loc = t_communicate;
  double t2comm_loc = sqr(t_communicate);
  MPI_Allreduce(&t_communicate_local, &t_communicate, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&t_compute_local, &t_compute, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double tcomp, t2comp, tcomm, t2comm;
  MPI_Allreduce(&tcomp_loc, &tcomp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&t2comp_loc, &t2comp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&tcomm_loc, &tcomm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&t2comm_loc, &t2comm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


  if (myid == 0) {
    double l = local_n_mean/nproc;
    double i = import_n_mean/nproc;
    l2 = l2/nproc;
    i2 = i2/nproc;

    tcomp = tcomp/nproc;
    t2comp = t2comp/nproc;
    tcomm = tcomm/nproc;
    t2comm = t2comm/nproc;
    
    fprintf(stderr, "local_n= (%d %d; %g +/- %g);  import_n= (%d %d; %g +/- %g)\n",
	    local_n_min,  local_n_max,  (float)l, (float)sqrt(l2 - l*l),
	    import_n_min, import_n_max, (float)i, (float)sqrt(i2 - i*i));
    fprintf(stderr, "tcomp= %g +/- %g sec;  tcomm= %g +/- %g sec\n",
	    tcomp, (float)sqrt(t2comp - sqr(tcomp)),
	    tcomm, (float)sqrt(t2comm - sqr(tcomm)));
  }



  

  double t_interaction_local = t_interaction;
  double t_renorm_local      = t_renorm;
  double t_grad_local        = t_grad;
  double t_tree_local        = t_tree;
  double t_weight_local      = t_weight;
  MPI_Allreduce(&t_interaction_local, &t_interaction, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&t_renorm_local,      &t_renorm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&t_weight_local,      &t_weight, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&t_grad_local,        &t_grad,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&t_tree_local,        &t_tree,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


}
