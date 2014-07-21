#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

float system::compute_pressure(const float dens, const float ethm) {
  return ethm*(gamma_gas - 1.0f);
}
void system::boundary_particles(const int idx) {return;}
float4 system::body_forces(const pfloat3 pos) {return float4(0,0,0,0);}
bool system::remove_particles_within_racc(const int idx) {return false;}
void system::boundary_derivatives(const int idx) {}

#ifndef _PERIODIC_FLOAT_
#error "Please define _PERIODIC_FLOAT_ in pfloat.h"
#endif

void system::setup_particles(const bool init_data) {
  
  gamma_gas = 5.0/3;


  kernel.set_dim(3);

  int nx = 128;
  int ny = 64;
  int nz = 32;
  
  global_n = nx*ny*nz;
  local_n  = 0;

  NGBmin  = 32.9;
  NGBmean = 33.0;
  NGBmax  = 33.1;
  
  float3 rmin = (float3){0,0,0};
  float3 rmax = (float3){2,1,0.5};
  
  rmax.y = rmax.x/nx * ny;
  rmax.z = rmax.x/nx * nz;
  
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

    float xmin = 0.0;
    float ymin = 0.0;
    float zmin = 0.0;

    int idx = 0;

    const float3 L = global_domain.size();
    const float3 dr = {L.x/nx, L.y/ny, L.z/nz};
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
	for (int i = 0; i < nx; i++) {
	  const float3 pos = (float3){
	    xmin + i*dr.x + 0.0f*dr.x*(j & 1),
	    ymin + j*dr.y,
	    zmin + k*dr.z};
	  
	  pfloat3 ppos;
 	  ppos.x.set(pos.x);
 	  ppos.y.set(pos.y);
 	  ppos.z.set(pos.z);

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
    }
    
    global_n = pvec.size();

    fprintf(stderr, " ****** global_n= %d  local_n= %d\n", global_n, local_n);

    
    const float xc = 1.0f;
    const float yc = 0.5f;
    const float r0 = 0.3;

    const int n = pvec.size();

    const float amp = 0.001;
    const real vel0 = 1;

    const real angle = 30 * M_PI/180.0;
    const real vel_x = 2.0; //vel0*cos(angle);
    const real vel_y = 1.0; //vel0*sin(angle);
    const real vel_z = 0.5;

    const real x0 = xc;
    const real y0 = yc;
    
    for (int pc = 0; pc < n; pc++) {
      const pfloat3 pos = pvec[pc].pos;
      const float x = pos.x.getu() - xc;
      const float y = pos.y.getu() - yc;
      const float r = sqrt(x*x + y*y);
      
      real vx = vel_x;
      real vy = vel_y;
      real vz = vel_z;

      real bx = 0;
      real by = 0;
      real bz = 0;
      
      float d0 = 1.0f;
      if (r < r0) d0 *= 2.0f;

#if 1
      d0 = 1;
      if (r < r0 &&  r > 0) {
	real azx = amp * (x)/r;
	real azy = amp * (y)/r;
	bx = azy;
	by = -azx;
	d0 = 2.0;
      }
#endif
//       bx= 1;
//       by = 0;


//       bx = by = bz = 0;
     
      
      ptcl_mhd p;

      p.dens = d0;
      p.ethm = 1.0/(gamma_gas - 1.0);
      p.vel  = (real3){vx, vy, vz};
      p.B    = (real3){bx, by, bz};
      p.psi  = 0;
      
      pmhd.push_back(p);

    }
  }
  
  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);

  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);

  
  MPI_Barrier(MPI_COMM_WORLD);
  
  const int3 nt = {4, 2, 1};
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
