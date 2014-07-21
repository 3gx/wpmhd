#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_qrng.h>

float system::compute_pressure(const float dens, const float ethm) {
  return ethm*(gamma_gas - 1.0f);
}
void system::boundary_particles(const int idx) {return;}
float4 system::body_forces(const pfloat3 pos) {return float4(0,0,0,0);}
bool system::remove_particles_within_racc(const int idx) {return false;}

#ifndef _PERIODIC_FLOAT_
#error "Please define _PERIODIC_FLOAT_ in pfloat.h"
#endif

void system::setup_particles(const bool init_data) {

//   gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_halton, 4);
  gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_sobol, 2);

  kernel.set_dim(2);
  int nx = 64;
  int ny = 64;
  int nz = 1;

  nx = ny = 128;
  nx = ny = 256;
//   nx = ny = 512;
  nx = ny = 1024;

  global_n = nx*ny*nz;
  local_n  = 0;
  
  NGBmin  = 18.999;
  NGBmean = 19.0;
  NGBmax  = 19.001;
  
  NGBmin  = 15.999;
  NGBmean = 16.0;
  NGBmax  = 16.001;

//   NGBmin  = 12.999;
//   NGBmean = 13.0;
//   NGBmax  = 13.001;


  float3 rmin = (float3){0,0,0};
  float3 rmax = (float3){1,1,1};
  
//   rmax.y = rmax.x/nx * ny;
//   rmax.z = rmax.x/nx * nz;

  
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

  if (myid == 0) {

    const float3 L = global_domain.size();

    // generate level boxes

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
 	  double v[16];
 	  gsl_qrng_get (q, v);
//   	  pos.x = v[0];
//   	  pos.y = v[1];
//   	  pos.x = drand48();
//   	  pos.y = drand48();
	  
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

	  p.h   = 2.5*dr.x;
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

    const real b0 = 1.0/sqrt(4.0*M_PI);
    const real d0 = 25.0/(36.0*M_PI);
    const real v0 = 1.0;
    const real p0 = 5.0/(12*M_PI);
    gamma_gas = 5.0/3;
    
    const int n = pvec.size();
    const real adv = 0*10;
    for (int pc = 0; pc < n; pc++) {
      pfloat3 pos = pvec[pc].pos;
      float x = pos.x.getu();
      float y = pos.y.getu();
      
      real vx = -v0 * sin(2.0*M_PI*y) + adv;
      real vy = +v0 * sin(2.0*M_PI*x) + adv;
      real vz =  adv;
      
      real bx = -b0*sin(2*M_PI*y);
      real by = +b0*sin(4*M_PI*x);
      real bz = 0;
      
      ptcl_mhd p;

      p.dens = d0;
      p.ethm = p0/(gamma_gas - 1.0); //* (1 + 0.5*sin(2*M_PI*x));  // * (sin(2.0*M_PI*x)*sin(2.0*M_PI*y) + 2);
      p.vel  = (real3){vx, vy, vz};
      p.B    = (real3){bx, by, bz};
      if (x < 0.5) p.psi  = 1.0;
      else  	   p.psi  = 2.0;
      p.psi = 0;
      
      pmhd.push_back(p);

    }
  }
  
  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);

  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);

  
  MPI_Barrier(MPI_COMM_WORLD);
  
  const int3 nt = {4, 4, 1};
  box.set(nproc, nt, global_domain);
  
  distribute_particles();
  
//   Bxx.resize(local_n);
//   Bxy.resize(local_n);
//   Bxz.resize(local_n);
//   Byy.resize(local_n);
//   Byz.resize(local_n);
//   Bzz.resize(local_n);
  
//   ptcl_aux.resize(local_n);
  
//   ptcl_mhd_grad[0].resize(local_n);
//   ptcl_mhd_grad[1].resize(local_n);
//   ptcl_mhd_grad[2].resize(local_n);

//   ptcl_mhd_min.resize(local_n);
//   ptcl_mhd_max.resize(local_n);

  pmhd_dot.resize(local_n);
  divB_i.resize(local_n);
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;

  for (int i = 0; i < local_n; i++)
    pmhd_dot[i].set(0.0);

  
  fprintf(stderr, "proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

}
