#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_qrng.h>

float system::compute_pressure(const float dens, const float ethm) {return ethm*(gamma_gas - 1.0f);}
void system::boundary_particles(const int idx) {return;}
float4 system::body_forces(const pfloat3 pos) {return float4(0,0,0,0);}
bool system::remove_particles_within_racc(const int idx) {return false;}
void system::boundary_derivatives(const int idx) {}


void system::set_parallel_domain() {
  if (myid == 0)fprintf(stderr, " *** setting up parallel domain *** \n");

  const int3 nt = {2, 2, 1};
  box.set(nproc, nt, global_domain);
}

void system::set_geometry(const bool init_flag) {

  if (myid == 0) fprintf(stderr, " *** setting up the geometry *** \n");

  // # of dimensions in the problem
  kernel.set_dim(2);
  
  // # neighbours
  NGBmin  = 18.9;
  NGBmean = 19.0;
  NGBmax  = 19.1;

  // periodic box size
  const float3 box_rmin = (float3){0,  0,  0};
  const float3 box_rmax = (float3){1,  1,  1};
  
  pfloat<0>::set_range(box_rmin.x, box_rmax.x);
  pfloat<1>::set_range(box_rmin.y, box_rmax.y);
  pfloat<2>::set_range(box_rmin.z, box_rmax.z);
  
  global_domain.set_x(box_rmin.x, box_rmax.x);
  global_domain.set_y(box_rmin.y, box_rmax.y);
  global_domain.set_z(box_rmin.z, box_rmax.z);
  const float3 box_L = global_domain.size();
  
  // clean up geometry array
  pvec.clear();
  pvec.reserve(128);
  
  // setup geometry on proc0
  local_n = global_n = 0;
  if (myid == 0) {
    
    /* ********** NUMBER OF NESTED LEVELS *********** */
    const int nlev = 0; 

    int   idx   = 0;
    int   np    = 5e4;

    for (int i = 0; i < np; i++) {

      float3 L    = box_L;
      float3 rmin = box_rmin;
      float3 rmax = box_rmax;

      float3 pos;
      particle p;
      p.wght = 1.0f;
      p.vel = (float3){0.0f, 0.0f, 0.0f};
      
      pos.x = rmin.x +  drand48() * L.x;
      pos.y = rmin.y +  drand48() * L.y;
      pos.z = rmin.z +  drand48() * L.z;
      if (kernel.ndim == 2) pos.z = 0.0f;

      p.pos.x.set(pos.x);
      p.pos.y.set(pos.y);
      p.pos.z.set(pos.z);
      p.h = 1.2f*L.x/std::pow(1.0f*np, 1.0f/kernel.ndim);
      p.local_idx  = idx;
      p.global_idx = idx++;
      pvec.push_back(p);
      local_n++;
      
      for (int l = 0; l < nlev; l++) {
	L = (float3){0.5f*L.x, 0.5f*L.y, 0.5f*L.z};
	rmin.x += 0.5f*L.x;
	rmin.y += 0.5f*L.y;
	rmin.z += 0.5f*L.z;
	
	pos.x = rmin.x +  drand48() * L.x;
	pos.y = rmin.y +  drand48() * L.y;
	pos.z = rmin.z +  drand48() * L.z;
	if (kernel.ndim == 2) pos.z = 0.0f;
	
	p.pos.x.set(pos.x);
	p.pos.y.set(pos.y);
	p.pos.z.set(pos.z);
	p.h = 1.2f*L.x/std::pow(1.0f*np, 1.0f/kernel.ndim);
	p.local_idx  = idx;
	p.global_idx = idx++;
	pvec.push_back(p);
	local_n++;
      }
    }

  }

  global_n = pvec.size();
  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);
  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);
  
  pmhd.resize(global_n);
  pmhd_dot.resize(global_n);
  box.set(nproc, box.nt, global_domain);
  distribute_particles();
  
}


void system::set_problem(const bool init_data) {
  if (myid == 0) fprintf(stderr, "  *** orszag problem *** \n");

  gamma_gas = 5.0/3;

  if (!init_data) return;

  if (myid == 0) fprintf(stderr, "  *** setting up the problem *** \n");

  
  const real beta  = 0.1;
  const real Acnst = 0.1;
  const real d0 = 1.0;
  const real p0 = 0.5 * beta;
  gamma_gas = 5.0/3;
  const float x0 = 0.5;
  
  for (int i = 0; i < local_n; i++) {
    pfloat3 pos = pvec[i].pos;
    float x = pos.x.getu();
    float y = pos.y.getu();
    
    float d, p, vx, vy, vz, bx, by, bz;
    
    vx = 0;
    vy = 0;
    vz = 0;
    
    bx = 0;
    by = 0;
    bz = 0;
    
    vx = Acnst * sin(2*M_PI*y);
    if (fabs(x - x0) < 0.25) by = -1;
    else                     by = +1;
      

    d = d0;
    p = p0;
    
    ptcl_mhd m;
    
    m.dens = d;
    m.ethm = p/(gamma_gas - 1.0); 
    m.vel  = (real3){vx, vy, vz};
    m.B    = (real3){bx, by, bz};
    m.psi = 0;
    
    pmhd[i] = m.to_conservative(pvec[i].wght);
    
  }

  divB_i.resize(local_n);
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;
  
  gpu.dwdt.cmalloc(local_n);
  for (int i = 0; i < local_n; i++) 
    gpu.dwdt[i] = 0.0f;
  
  for (int i = 0; i < local_n; i++)
    pmhd_dot[i].set(0.0);
  
}




