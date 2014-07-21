#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_qrng.h>

#define _CS0_ 1.0f

float system::compute_pressure(const float dens, const float ethm) {
  return ethm; //*(gamma_gas - 1.0f);
}

float4 system::body_forces(const int idx) {
  const pfloat3 pos = pvec[idx].pos;
  const float3 dr   = {pos.x - global_domain.centre.x,
		       pos.y - global_domain.centre.y,
		       pos.z - global_domain.centre.z};
  
  float4 acc;
  acc.x = +2*qOmega*sqr(Omega)*dr.x + 2* Omega*pmhd[idx].vel.z;
  acc.z = -2* Omega*pmhd[idx].vel.x;
  acc.w = 0.0f;

  return acc;
};

bool system::remove_particles_within_racc(const int idx) {return false;}

void system::boundary_particles(const int idx) { 
  const particle pi = pvec[idx];
  ptcl_mhd mi = (pmhd[idx] * (1.0f/pi.wght)).to_primitive();

  const float cs    = _CS0_;
  mi.ethm = sqr(cs) * mi.dens;

  pmhd[idx] = mi.to_conservative() * pi.wght;
  return;
}

void system::setup_particles(const bool init_data) {
  
  gsl_qrng *q = gsl_qrng_alloc (gsl_qrng_sobol, 3);
  kernel.set_dim(2);
  
  global_n = 0;
  local_n  = 0;
  
  NGBmin  = 18.9;
  NGBmean = 19.0;
  NGBmax  = 19.1;  
  
  int nx = 256;
  int ny = 256;

//   nx = ny = 128;
//   nx = ny = 64;

  const float3 rmin = (float3){0.0f, 0.0f, 0.0f};
  const float3 rmax = (float3){1.0f, 1.0f, 1.0f};
  
  pfloat<0>::set_range(rmin.x, rmax.x);
  pfloat<1>::set_range(rmin.y, rmax.y);
  pfloat<2>::set_range(rmin.z, rmax.z);
  
  global_domain.set_x(rmin.x, rmax.x);
  global_domain.set_y(rmin.y, rmax.y);
  global_domain.set_z(rmin.z, rmax.z);
  
  pvec.clear();
  pmhd.clear();
  
  pvec.reserve(128);
  pmhd.reserve(128);

  gamma_gas = 1.001f; //01;   // 5.0/3.0;
  gamma_gas = 1.0f;
  qOmega = 1.5f;
  Omega  = 1.0f;
  gravity_mass = 0;
  

  if (myid == 0) {
    
    const float3 L = global_domain.size();

    const int nlevels = 1;
    std::vector<boundary> level_boxes(nlevels);
    for (int lev = 0; lev < nlevels; lev++) {
      const float3 p = {0.5f, 0.5f, 0.0f};
      const float  h = L.x/(1 << (lev + 1));
      const boundary bnd = boundary(p, h);
      level_boxes[lev] = bnd;
    }

    
    // populate level boxes with particles

    int idx = 0;
    float xmin = 0;
    float ymin = 0;
    for (int lev = 0; lev < nlevels; lev++) {
      const float3 dr = {L.x/(1 << lev)/nx, L.y/(1 << lev)/nx, L.z/(1 << lev)/nx};
      for (int j = 0; j < ny; j++) {
	for (int i = 0; i < nx; i++) {
	  const float ff  = 0.00f;
	  float3 pos = (float3){
	    xmin + i*dr.x + dr.x*(0.5 - drand48()) * ff + 0.5*dr.x*(j & 1),
	    ymin + j*dr.y + dr.y*(0.5 - drand48()) * ff, 
	    0.0};
	  
	  pfloat3 ppos;
 	  ppos.x.set(pos.x);
 	  ppos.y.set(pos.y);
 	  ppos.z.set(pos.z);
	  
	  bool drop = false;
	  for (int l = lev+1; l < nlevels; l++) 
	    drop = drop || level_boxes[l].isinbox(ppos);
	  
	  if (drop) continue;
	  particle p;
	  
	  p.pos = ppos;
	  
	  p.h   = 1.5*dr.x;
	  p.vel = (float3){0.0f,0.0f,0.0f};
	  p.global_idx = idx;
	  p.local_idx  = idx++;
	  p.wght = 1.0;
	  
	  pvec.push_back(p);
	  local_n++;
	  
	}
      }
      xmin += L.x/(1 << (lev + 2));
      ymin += L.y/(1 << (lev + 2));
    }
    global_n = pvec.size();
    
    fprintf(stderr, " ****** global_n= %d  local_n= %d\n", global_n, local_n);
    
    const float d0 = 1.0;
    const float beta = 1348.0f; // * sqr(1.0/_CS0_);

#if 0
    const float ff = 2*M_PI*sqrt(16.0/15)*sqrt(2*gamma_gas/beta);
    const float cs = 2*Omega*global_domain.hsize.x.getu()/ff / 8.0;
#else
    const float cs = _CS0_;
#endif

    const float p0   = d0 * sqr(cs)/gamma_gas;
    const float vamp = 0.01*cs; ///sqrt(3.0);
    

    const float bx0 = 0.0;
#if 0
    const float by0 = 0.0;
#else
    const float by0 = std::sqrt(2.0*p0/beta);
#endif
    const float bz0 = 0.0;
    const float lambda = global_domain.hsize.x.getu() * 2.0f;


    const int n = pvec.size();
    for (int pc = 0; pc < n; pc++) {
      const pfloat3 pos = pvec[pc].pos;
      const float x = pos.x - global_domain.centre.x;
      
      float dens = d0;
      float pres = p0;
      
      double v[3];
      gsl_qrng_get(q, v);
//       float vx =  vamp * (2.0*v[0] - 1); // (2.0*drand48() - 1);
//       float vy =  vamp * (2.0*v[1] - 1); // (2.0*drand48() - 1);
      float vx =  vamp * (2.0*drand48() - 1);
      float vy =  vamp * (2.0*drand48() - 1);
      float vz =  -qOmega*Omega*x;// + vamp * (2*v[2] - 1);;
      
      float bx = bx0;
      float by = by0 * sin(2*M_PI*x/lambda);
      float bz = bz0;

      ptcl_mhd p;

      p.dens = dens;
      p.ethm = sqr(cs)*dens; //pres /(gamma_gas - 1.0);
      p.vel  = (real3){vx, vy, vz};
      p.B    = (real3){bx, by, bz};
      p.psi = 0;
      p.marker = 1;
      pmhd.push_back(p);


    }
  }
  
  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);

  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);

  
  MPI_Barrier(MPI_COMM_WORLD);

  
  const int3 nt = {2, 2, 1};
  box.set(nproc, nt, global_domain);
  
  distribute_particles();

  pmhd_dot.resize(local_n);
  divB_i.resize(local_n);
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;

  for (int i = 0; i < local_n; i++)
    pmhd_dot[i].set(0.0);

  
  fprintf(stderr, "proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

}
