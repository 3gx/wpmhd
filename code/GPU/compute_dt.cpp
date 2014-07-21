#include "gn.h"

#define HSCALE 1.0f
#define DTMUL  (1.5f)
#define DENSDRIFT 1.0e-2

void system::compute_dt() {

  for (int i = 0; i < local_n; i++) {
    boundary_particles(i);
  }

  double t0 = get_time();
  convert_to_primitives();

  float dt_local = +HUGE;
  float h_min = dt_local;
  float Lmin_loc = HUGE;

  int i_dtmin = 0;
  for (int i = 0; i < local_n; i++) {
    const particle &pi = pvec[i];
    const ptcl_mhd &mi = pmhd[i];

    const float  B2 = sqr(mi.B.x) + sqr(mi.B.y) + sqr(mi.B.z);
    if (!(mi.ethm > 0)) {
      fprintf(stderr, "i= %d :  mi.ethm= %g  mi.dens= %g mi.velx= %g %g %g  B= %g %g %g h= %g %g\n",
	      i, mi.ethm, mi.dens,
	      mi.vel.x, mi.vel.y, mi.vel.z,
	      mi.B.x, mi.B.y, mi.B.z,
	      pi.h, pi.wght);
//       assert(mi.ethm > 0);
    }
    const float  pres = compute_pressure(mi.dens, mi.ethm);
    float  cs = 1.0f*sqrt((gamma_gas*pres + B2)/mi.dens);
    float3 dv = {mi.vel.x - pi.vel.x, mi.vel.y - pi.vel.y, mi.vel.z - pi.vel.z};
    dv = (float3){0.0, 0.0, 0.0};

#if 0
    const float asq = gamma_gas * pres/mi.dens;
    
    const float b1 = mi.B.x;
    const float b2 = mi.B.y;
    const float b3 = mi.B.z;

    const float tsum = B2/mi.dens + asq;
    const float tdif = B2/mi.dens - asq;
#if 0
    const float cf1sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b2*b2+b3*b3)/mi.dens));
    const float cf2sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b1*b1+b3*b3)/mi.dens));
    const float cf3sq = 0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*(b1*b1+b2*b2)/mi.dens));

    cs =1.1f*std::sqrt(std::max(std::max(cf1sq, cf2sq), cf3sq));
#else
    cs = 1.1f* std::sqrt(0.5*(tsum + sqrt(tdif*tdif + 4.0*asq*B2/mi.dens)));
#endif
#endif    

    float dt = 0;
    float L  = 0;
    switch(kernel.ndim) {
    case 1:
      L = pi.wght;
      dt = L/(cs + std::abs(dv.x));
      break;
    case 2:
      L  = sqrt(pi.wght/M_PI);
      L = 0.5*pi.h;
      dt = L/(cs + sqrt(dv.x*dv.x + dv.y*dv.y));
      break;
    case 3:
      L = powf(pi.wght*3.0f/4.0f/M_PI, 1.0f/3.0f);
      L = 0.333*pi.h;
      dt = L/(cs + sqrt(dv.x*dv.x + dv.y*dv.y + dv.z*dv.z));
      break;
    default:
      assert(kernel.ndim > 0 && kernel.ndim <= 3);
    };
    Lmin_loc = std::min(Lmin_loc, L);
    if (pvec[i].few == 1) dt = 1e10f;

    if (!(dt > 0.0f)) {
      fprintf(stderr, "proc= %d dt= %g\n", myid, dt);
      assert(dt > 0.0f);
    }

#if 1    
    if (gravity_mass > 0.0f && mi.dens > DENSDRIFT) {
      float dt_acc1 = 1e10f;
      float dt_acc2 = 1e10f;
      if (true) {
	const float4 acc = body_forces(pi.pos);
	const float pv = sqrt(sqr(pi.vel.x) + sqr(pi.vel.y) + sqr(pi.vel.z));
	const float pa = sqrt(sqr(   acc.x) + sqr(   acc.y) + sqr(   acc.z));
	const float va = acc.x*pi.vel.x + acc.y*pi.vel.y + acc.z*pi.vel.z;
 	const float Rc_d = sqr(pv)*sqr(pa) - sqr(va);
// 	const float Rc_d = (float)(std::pow((double)pv*(double)pa, 2.0)- std::pow((double)va, 2.0));
// 	if (!(Rc_d > 0)) {
// 	  fprintf(stderr, "i= %d, Rc= %g: pos= %g %g %g, vel= %g %g %g, pv= %g pa= %g va =%g \n",
// 		  i, Rc_d,
// 		  pi.pos.x.getu() - gravity_pos.x,
// 		  pi.pos.y.getu() - gravity_pos.y,
// 		  pi.pos.z.getu() - gravity_pos.z,
// 		  pi.vel.x, pi.vel.y, pi.vel.z,
// 		  pv ,pa, va);
// 	}
// 	assert(Rc_d > 0);
	if (Rc_d > 0.0) {
	  const float Rcurv = (Rc_d == 0) ? HUGE : sqr(pv)*pv/sqrt(Rc_d);
	  const float Ldist = 2*M_PI/25.0 * Rcurv;
	  dt_acc1 = Ldist/(pv + TINY);
// 	} else {
// 	  const float vabs = sqrt(sqr(pi.vel.x) + sqr(pi.vel.y) + sqr(pi.vel.z));
// 	  dt_acc1 = L/vabs;
	}
      }
      if (0) {
	const float4 acc = body_forces(pi.pos);
	const float facc = sqrt(sqr(acc.x) + sqr(acc.y) + sqr(acc.z));
	float h = pi.h * HSCALE;
//  	h = 2.0*L;
	
 	float dt_acc = sqrt(2*h/facc);
	const float vabs = sqrt(sqr(pi.vel.x) + sqr(pi.vel.y) + sqr(pi.vel.z));
	
	dt_acc = 1e10f;
	dt_acc2 = std::min(dt_acc, h/vabs);
	dt_acc = 2.0*L/vabs;
      }
      
      const float dt_acc = std::min(dt_acc1, dt_acc2);
      dt = std::min(dt, dt_acc); ///courant_no);
      h_min = std::min(h_min, pi.h);
    }
#endif
    if (dt < dt_local) {
      i_dtmin = i;
    }
    dt_local = std::min(dt_local, dt);
    
    
  }
  dt_local *= courant_no;
//    if ((iteration > 98 && iteration < 110)) {
//      dt_local *= 0.1;
//    }

  
  {
    const int i = i_dtmin;
    const particle &pi = pvec[i];
    const ptcl_mhd &mi = pmhd[i];

    const float  B2 = sqr(mi.B.x) + sqr(mi.B.y) + sqr(mi.B.z);
    if (!(mi.ethm > 0)) {
      fprintf(stderr, "i= %d :  mi.ethm= %g  mi.dens= %g\n",
	      i, mi.ethm, mi.dens);
      assert(mi.ethm > 0);
    }
    const float  pres = compute_pressure(mi.dens, mi.ethm);
    const float  cs = 1.0f*sqrt((gamma_gas*pres + B2)/mi.dens);
    float3 dv = {mi.vel.x - pi.vel.x, mi.vel.y - pi.vel.y, mi.vel.z - pi.vel.z};
    dv = (float3){0.0, 0.0, 0.0};
//     const float3 dv = {mi.vel.x - 1*pi.vel.x,
// 		       mi.vel.y - 1*pi.vel.y,
// 		       mi.vel.z - 1*pi.vel.z};
    
    
    float dt = 0;
    float L  = 0;
    float wi = pi.wght;
    switch(kernel.ndim) {
    case 1:
      L = wi;
      dt = L/(cs + std::abs(dv.x));
      break;
    case 2:
      L  = sqrt(wi/M_PI);
      dt = L/(cs + sqrt(dv.x*dv.x + dv.y*dv.y));
      break;
    case 3:
      L = powf(wi*3.0f/4.0f/M_PI, 1.0f/3.0f);
      dt = L/(cs + sqrt(dv.x*dv.x + dv.y*dv.y + dv.z*dv.z));
      break;
    default:
      assert(kernel.ndim > 0 && kernel.ndim <= 3);
    };

    float dt_acc = 0;
    if (gravity_mass > 0.0f && mi.dens > DENSDRIFT) {
#if 1
      const float4 acc = body_forces(pi.pos);
      const float pv = sqrt(sqr(pi.vel.x) + sqr(pi.vel.y) + sqr(pi.vel.z));
      const float pa = sqrt(sqr(   acc.x) + sqr(   acc.y) + sqr(   acc.z));
      const float va = acc.x*pi.vel.x + acc.y*pi.vel.y + acc.z*pi.vel.z;
      const float Rc_d = sqr(pv)*sqr(pa) - sqr(va);
      assert(Rc_d > 0);
      const float Rcurv = (Rc_d == 0) ? HUGE : sqr(pv)*pv/sqrt(Rc_d);
      const float Ldist = 2*M_PI/25.0 * Rcurv;
      dt_acc = Ldist/(pv + TINY);
      
//       dt_local = std::min(dt_local, dt_acc);
//       h_min = std::min(h_min, pi.h);
#else
      const float4 acc = body_forces(pi.pos);
      const float facc = sqrt(sqr(acc.x) + sqr(acc.y) + sqr(acc.z));
      float h = pi.h * HSCALE;
//  	h = 2.0*L;

      dt_acc = sqrt(2.0f*h/facc);
      const float vabs = sqrt(sqr(pi.vel.x) + sqr(pi.vel.y) + sqr(pi.vel.z));
      
      dt_acc = 1e10f;
      dt_acc = std::min(dt_acc, 1.0f*h/vabs);

      h_min = std::min(h_min, pi.h);
#endif
    }

    fprintf(stderr, "i_dtmin= %d dt:%g dt_acc:%g [%g] L:%g h:%g w:%g pos(%g %g %g) vel(%g %g %g)  v(%g %g %g) B(%g %g %g) d:%g e:%g \n",
	    i, courant_no*dt, courant_no*dt_acc, courant_no,
	    L, pvec[i].h, pvec[i].wght,
	    pvec[i].pos.x.getu() - gravity_pos.x,
	    pvec[i].pos.y.getu() - gravity_pos.y,
	    pvec[i].pos.z.getu() - gravity_pos.z,
	    pvec[i].vel.x,
	    pvec[i].vel.y,
	    pvec[i].vel.z,
	    pmhd[i].vel.x,
	    pmhd[i].vel.y,
	    pmhd[i].vel.z,
	    pmhd[i].B.x,
	    pmhd[i].B.y,
	    pmhd[i].B.z,
	    pmhd[i].dens,
	    pmhd[i].ethm);
  }
	  
	  
//   fprintf(stderr, "proc= %d: ******** dt_acc_min= %g  h_min= %g ********* \n", myid, dt_acc_min, h_min);

  // communicate dt_local to compute dt_global
  t_compute += get_time() - t0;

  t0 = get_time();

  float dt_glob = 0;
  float Lmin_glob = 0;
  MPI_Allreduce(&dt_local, &dt_glob,   1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&Lmin_loc, &Lmin_glob, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  
  
  if (true || gravity_mass > 0.0) {
    
    if (dt_global > 0.0) {
      if (dt_glob < dt_global) {
	dt_global = dt_glob;
      } else if (DTMUL*dt_global < dt_glob) {
	dt_global *= DTMUL;
      }
    } else {
      dt_global = dt_glob;
    }
  } else {

    dt_global = dt_glob;

  }

  ch_glob = Lmin_glob/dt_global;
  
  t_communicate += get_time() - t0;

  t0 = get_time();

  restore_conservative();

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
