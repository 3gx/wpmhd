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

  int nx, ny;
  
  nx = ny = 256;
  
  // # neighbours
  NGBmin  = 18.9;
  NGBmean = 19.0;
  NGBmax  = 19.1;

  const int nn = 0;
  NGBmin  += nn;
  NGBmean += nn;
  NGBmax  += nn;

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
    
#if 0
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
#else
    int idx = 0;
    const float3 L    = box_L;
    const float3 dr = {L.x/nx, L.y/ny, 0.0};
    const float3 rmin = box_rmin;
    const float3 rmax = box_rmax;
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
	const float ff  = 0.00f;
	float3 pos = (float3){
	  rmin.x + i*dr.x + 0.5*(j&1)*dr.x,
	  rmin.y + j*dr.y,
	  (rmin.z + rmax.z)*0.5};
	  
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
#endif

  }

  global_n = pvec.size();
  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);
  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);
  
  pmhd.resize(global_n);
  pmhd_dot.resize(global_n);
  divB_i.resize(global_n);
  box.set(nproc, box.nt, global_domain);
  distribute_particles();
  
  const int niter = 2;
  const float eps = 0.2f;
  adjust_positions(niter, eps);

}


void system::set_problem(const bool init_data) {
  if (myid == 0) fprintf(stderr, "  *** orszag problem *** \n");

  gamma_gas = 5.0/3;

  if (!init_data) return;

  if (myid == 0) fprintf(stderr, "  *** setting up the problem *** \n");

  
  const real b0 = 1.0/sqrt(4.0*M_PI);
  const real d0 = 25.0/(36.0*M_PI);
  const real v0 = 1.0;
  const real p0 = 5.0/(12*M_PI);
  gamma_gas = 5.0/3;
  
  const float adv = 0.0f;
  for (int i = 0; i < local_n; i++) {
    pfloat3 pos = pvec[i].pos;
    float x = pos.x.getu();
    float y = pos.y.getu();
    
    float d, p, vx, vy, vz, bx, by, bz;

    vx = -v0 * sin(2.0*M_PI*y) + adv;
    vy = +v0 * sin(2.0*M_PI*x) + adv;
    vz =  adv;
    
    bx = -b0*sin(2*M_PI*y);
    by = +b0*sin(4*M_PI*x);
    bz = 0;
    

    d = d0;
    p = p0;
    
    ptcl_mhd m;
    
    m.dens = d;
    m.ethm = p/(gamma_gas - 1.0); 
    m.vel  = (real3){vx, vy, vz};
    m.B    = (real3){bx, by, bz};
    m.psi = 0;
    
    pmhd[i] = m.to_conservative(pvec[i].wght);

    if (!eulerian_mode) pvec[i].vel = (float3){vx, vy, vz};
    else                pvec[i].vel = (float3){0,0,0};
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




