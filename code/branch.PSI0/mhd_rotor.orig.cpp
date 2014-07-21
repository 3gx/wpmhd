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

#ifndef _PERIODIC_FLOAT_
#error "Please define _PERIODIC_FLOAT_ in pfloat.h"
#endif

void system::setup_particles(const bool init_data) {

  gamma_gas = 1.4;

  kernel.set_dim(2);
  int nx = 128;
  int ny = 128;
  int nz = 1;

//   nx = ny = 64;
  nx = ny = 256;
//   nx = ny = 512;

  global_n = nx*ny*nz;
  local_n  = 0;

  NGBmin  = 15.999;
  NGBmean = 16.0;
  NGBmax  = 16.001;

#if 0
  const int nn = +3;
  NGBmin  += nn;
  NGBmax  += nn;
  NGBmean += nn;
#endif

#if 1
  const int nn = 3;
  NGBmin  += nn;
  NGBmax  += nn;
  NGBmean += nn;
#endif

  float3 rmin = (float3){0,0,0};
  float3 rmax = (float3){1,0,1};
  
  rmax.y = rmax.x/nx * ny;
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

  const int3 nt = {4, 4, 1};
  box.set(nproc, nt, global_domain);

  if (!init_data) return;

  if (myid == 0) {
    
    const float3 L = global_domain.size();

    // generate level boxes

    const int nlevels = 1;
    std::vector<boundary> level_boxes(nlevels);
    for (int lev = 0; lev < nlevels; lev++) {
      float4 p;
      p.x = p.y = 0.5f;
      p.z = (rmin.z + rmax.z)*0.5f;
      p.w = L.x/(1 << (lev + 1));
      level_boxes[lev] = boundary(p);
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
	  const float3 pos = (float3){
	    xmin + i*dr.x + dr.x*(0.5 - drand48()) * ff + 0.0*dr.x*(j & 1),
	    ymin + j*dr.y + dr.y*(0.5 - drand48()) * ff , 
	    (rmin.z + rmax.z)*0.5f};
	  
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

    real x0 = 0.5;
    real y0 = 0.5;
    
    real r0 = 0.1;
    real r1 = 0.115;
    
    const int n = pvec.size();
//     const real adv = 0;
    for (int pc = 0; pc < n; pc++) {
      pfloat3 pos = pvec[pc].pos;
      float x = pos.x.getu() - x0;
      float y = pos.y.getu() - y0;
      real r = sqrt(sqr(x) + sqr(y));
      
      real vx = 0;
      real vy = 0;
      real vz = 0;
      real d0 = 1.0;
      real p0 = 1.0;
      real bx = 5.0/sqrt(4*M_PI);
      real by = 0.0;
      real bz = 0.0;
      
      real fr = 0;
      if (r <= r0) {
	fr = 1;
      } else if (r < r1) {
	fr = (r1 - r)/(r1 - r0);
      } else {
	fr = 0;
      }

      d0 = 1 + 9*fr;
      vx = -2*y/r*fr;
      vy = +2*x/r*fr;

      if (r <= r0) {
	vx = -2*y/r0*fr;
	vy = +2*x/r0*fr;
      }
      

//       if        (r <= r0*1.01) {
// 	d0 = 10;
// 	vx = -2*fr*y/r0;
// 	vy = +2*fr*x/r0;
//       } else if (r < r1) {
// 	d0 = 1 + 9*fr;
// 	vx = -2*fr*y/r;
// 	vy = +2*fr*x/r;
//       } else {
// 	d0 = 1;
// 	vx = vy = 0;
//       }

//       d0 = 1;
//       vx = vy = vz = 0;
      
      ptcl_mhd p;

      p.dens = d0;
      p.ethm = p0/(gamma_gas - 1.0);
      p.vel  = (real3){vx, vy, vz};
      p.B    = (real3){bx, by, bz};
      p.psi  = 0;
      
      pmhd.push_back(p);

    }
  }
  MPI_Bcast(&global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);
  
  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);

  
  MPI_Barrier(MPI_COMM_WORLD);
  
  
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

  for (int i = 0; i < local_n; i++)
    pmhd_dot[i].set(0.0);

  divB_i.resize(local_n);
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;
  
  fprintf(stderr, "proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

}
