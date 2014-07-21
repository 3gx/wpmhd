#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_qrng.h>

float system::compute_pressure(const float dens, const float ethm) {
  return ethm*(gamma_gas - 1.0f);
}
float4 system::body_forces(const pfloat3 pos) {return float4(0,0,0,0);}
bool system::remove_particles_within_racc(const int idx) {return false;}
void system::boundary_derivatives(const int idx) {}

void system::boundary_particles(const int idx) {
  ptcl_mhd prim = pmhd[idx].to_primitive(pvec[idx].wght);

//   prim.dens = 1.0;
//   prim.ethm = 1.0;

  static bool do_init = false;
  if (iteration == 50 && !do_init) {
    if (myid == 0 && idx == local_n-1){
      fprintf(stderr, "\n********************************************\n");
      fprintf(stderr, " ******** SETTING INITIAL CONDITIONS *********\n");
      fprintf(stderr, " *******************************************\n\n");
    }
    
    if (idx == local_n - 1) do_init = true;
  
    const float Dblob = 10.0f;
    const float Damb  = 1.0f;
    const float Pamb  = 1.0f;
    const float Vamb  = 2.7f * sqrt(gamma_gas * Pamb/Damb);
    const float Rblob = 0.1f;
    const float xc = 0.5f;
    const float yc = 0.5f;
    const float zc = 0.5f;


    const pfloat3 pos = pvec[idx].pos;
    const float x = pos.x.getu();
    const float y = pos.y.getu();
    const float z = pos.z.getu();
    
    float d, p, vx, vy, vz, bx, by, bz;
    
    const float r = sqrt(sqr(x-xc) + sqr(y-yc) + sqr(z-zc));
    if (r < Rblob) {
      d = Dblob;
      vx = vy = vz = 0;
    } else {
      d = Damb;
      vx = Vamb;
      vy = vz = 0;
    }
    p = Pamb;
//     vx = Vamb;
    bx = by = bz = 0;
    bx = 0.0f;
    
      
    prim.dens = d;
    prim.ethm = p/(gamma_gas - 1.0);
    prim.vel  = (real3){vx, vy, vz};
    prim.B    = (real3){bx, by, bz};
    prim.psi = 0;
      
    

  }

  pmhd[idx] = prim.to_conservative(pvec[idx].wght);

  return;
}

#ifndef _PERIODIC_FLOAT_
#error "Please define _PERIODIC_FLOAT_ in pfloat.h"
#endif

void system::setup_particles(const bool init_data) {
  gamma_gas = 5.0/3;
  gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_sobol, 3);


  kernel.set_dim(3);

  NGBmin  = 32.9;
  NGBmean = 33.0;
  NGBmax  = 33.1;
  
  float3 rmin = (float3){0, 0.0, 0.0};
  float3 rmax = (float3){3, 1.0, 1.0};
  
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

  global_n = 1e6;

  
  const int3 nt = {1, 1, 1};
  box.set(nproc, nt, global_domain);

  if (!init_data) return;

  if (myid == 0) {

    const float Rblob = 0.1f;
    const float xc = 0.5f;
    const float yc = 0.5f;
    const float zc = 0.5f;


#if 0
    const int n0 = 30;
    const float dx = rmax.x / rmax.x/n0;
    int idx = 0;
    for (int k = 0; k < n0; k++) {
      for (int j = 0; j < n0; j++) {
	for (int i = 0; i < (int)(rmax.x*n0); i++) {
	  
	  float3 pos = (float3){
	    rmin.x + i*dx, 
	    rmin.y + j*dx, 
	    rmin.z + k*dx};
	  
	  particle p;
	  
	  p.pos.x.set(pos.x);
	  p.pos.y.set(pos.y);
	  p.pos.z.set(pos.z);
	  
	  p.h   = 1.2 * dx;
	  p.vel = (float3){0.0f,0.0f,0.0f};
	  p.global_idx = idx;
	  p.local_idx  = idx++;
	  p.wght = 1.0;
	  
	  pvec.push_back(p);
	  local_n++;
	  
	}
      }
    }
#else
    int idx = 0;
    const int Namb = 2e5;     // ambient fluid
    const int Nblb = 1e5;     // blob stripe
    for (int i = 0; i < Namb; i++) {
      double v[16];
      gsl_qrng_get(q, v);
//       v[0] = drand48();
//       v[1] = drand48();
//       v[2] = drand48();
      float3 pos;
      
      particle p;

      //       pos.x = rmax.x * v[0];

      pos.y = rmax.y * v[1];
      pos.z = rmax.z * v[2];
      pos.x = v[0];
      
      p.pos.x.set(pos.x);
      p.pos.y.set(pos.y);
      p.pos.z.set(pos.z);
      
      p.h   = 1.2 * rmax.x/100.0;
      p.vel = (float3){0.0f,0.0f,0.0f};
      p.global_idx = idx;
      p.local_idx  = idx++;
      p.wght = 1.0;
      
      pvec.push_back(p);
      local_n++;

      /////////////

      pos.y = rmax.y * v[1];
      pos.z = rmax.z * v[2];
      pos.x += 1;

      
      p.pos.x.set(pos.x);
      p.pos.y.set(pos.y);
      p.pos.z.set(pos.z);
      
      p.h   = 1.2 * rmax.x/100.0;
      p.vel = (float3){0.0f,0.0f,0.0f};
      p.global_idx = idx;
      p.local_idx  = idx++;
      p.wght = 1.0;
      
      pvec.push_back(p);
      local_n++;

      /////////////

      pos.y = rmax.y * v[1];
      pos.z = rmax.z * v[2];
      pos.x += 1;


      p.pos.x.set(pos.x);
      p.pos.y.set(pos.y);
      p.pos.z.set(pos.z);
      
      p.h   = 1.2 * rmax.x/100.0;
      p.vel = (float3){0.0f,0.0f,0.0f};
      p.global_idx = idx;
      p.local_idx  = idx++;
      p.wght = 1.0;
      
      pvec.push_back(p);
      local_n++;
      
    }
#if 0
    for (int i = 0; i < Nblb; i++) {
      double v[16];
      gsl_qrng_get(q, v);
//       v[0] = drand48();
//       v[1] = drand48();
//       v[2] = drand48();
      float3 pos;

      const float f = 1.1f;
//       pos.x = xc + f*Rblob*(2*v[0] - 1);
      pos.x = rmax.x*v[0]; //xc + f*Rblob*(2*v[0] - 1);
      pos.y = yc + f*Rblob*(2*v[1] - 1);
      pos.z = zc + f*Rblob*(2*v[2] - 1);

      particle p;
      
      p.pos.x.set(pos.x);
      p.pos.y.set(pos.y);
      p.pos.z.set(pos.z);
      
      p.h   = 1.2 * rmax.x/100.0;
      p.vel = (float3){0.0f,0.0f,0.0f};
      p.global_idx = idx;
      p.local_idx  = idx++;
      p.wght = 1.0;
      
      pvec.push_back(p);
      local_n++;
      
    }
#elif 1 ////////////
    for (int i = 0; i < Nblb; i++) {
      double v[16];
      gsl_qrng_get(q, v);
      v[0] = drand48();
      v[1] = drand48();
      v[2] = drand48();
      float3 pos;

      const float f = 1.1f;
//       pos.x = xc + f*Rblob*(2*v[0] - 1);
      pos.x = f*Rblob*(2*v[0] - 1);
      pos.y = f*Rblob*(2*v[1] - 1);
      pos.z = f*Rblob*(2*v[2] - 1);

      const float r0 = sqrt(sqr(pos.x) + sqr(pos.y) + sqr(pos.z));
      if (r0 > Rblob) continue;

      pos.x += xc;
      pos.y += yc;
      pos.z += zc;
      

      particle p;
      
      p.pos.x.set(pos.x);
      p.pos.y.set(pos.y);
      p.pos.z.set(pos.z);
      
      p.h   = 1.2 * rmax.x/100.0;
      p.vel = (float3){0.0f,0.0f,0.0f};
      p.global_idx = idx;
      p.local_idx  = idx++;
      p.wght = 1.0;
      
      pvec.push_back(p);
      local_n++;
      
    }
#endif
#endif

    global_n = pvec.size();

    fprintf(stderr, " ****** global_n= %d  local_n= %d\n", global_n, local_n);

    const float Dblob = 10.0f;
    const float Damb  = 1.0f;
    const float Pamb  = 1.0f;
    const float Vamb  = 2.7f * sqrt(gamma_gas * Pamb/Damb);


    const int n = pvec.size();
    for (int pc = 0; pc < n; pc++) {
      const pfloat3 pos = pvec[pc].pos;
      const float x = pos.x.getu();
      const float y = pos.y.getu();
      const float z = pos.z.getu();

      float d, p, vx, vy, vz, bx, by, bz;
      
      const float r = sqrt(sqr(x-xc) + sqr(y-yc) + sqr(z-zc));
      if (r < Rblob) {
	d = Dblob;
	vx = vy = vz = 0;
      } else {
	d = Damb;
	vx = Vamb;
	vy = vz = 0;
      }
      p = Pamb;
      bx = by = bz = 0;

      ptcl_mhd mi;

      vx = vy = vz = 0;
      p = Pamb;
      d = Damb;

      mi.dens = d;
      mi.ethm = p/(gamma_gas - 1.0);
      mi.vel  = (real3){vx, vy, vz};
      mi.B    = (real3){bx, by, bz};
      mi.psi = 0;
      
      pmhd.push_back(mi);

    }
  }
  
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
  divB_i.resize(local_n);
  gpu.dwdt.cmalloc(local_n);
  for (int i = 0; i < local_n; i++) 
    gpu.dwdt[i] = 0.0f;
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;

  for (int i = 0; i < local_n; i++)
    pmhd_dot[i].set(0.0);

  
  fprintf(stderr, "proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

}
