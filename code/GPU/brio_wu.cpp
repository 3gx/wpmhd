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
  const float3 box_rmin = (float3){0,  0,     0};
  const float3 box_rmax = (float3){4,  0.25,  1};
  
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
  divB_i.resize(global_n);
  box.set(nproc, box.nt, global_domain);
  distribute_particles();

  const int niter = 30;
  const float eps = 0.2f;
  adjust_positions(niter, eps);
  
}


void system::set_problem(const bool init_data) {
  if (myid == 0) fprintf(stderr, "  *** orszag problem *** \n");
 
  real fp = 1.0/sqrt(4*M_PI);
  real Bt = 5.0*fp;

  real dL  = 1.0;
  real pL  = 20.0;
  real vtL = 10.0;
  real vnL = 0;
  real BnL = 5*fp;

  real dR  = 1.0;
  real pR  = 1.0;
  real vtR = -10.0;
  real vnR = 0;
  real BnR = 5*fp;
  
  gamma_gas = 5.0/3;

#if 0
  // BRIO_WU

  Bt = 0.75;
  
  dL = 1.0;
  pL = 1.0;
  vtL = vnL = 0;
  BnL = 1.0;

  dR = 0.125;
  pR = 0.1;
  vtR = vnR = 0;
  BnR = -1;
  
  gamma_gas = 2;
#endif

  if (!init_data) return;

  if (myid == 0) fprintf(stderr, "  *** setting up the problem *** \n");

  

  real xc = global_domain.centre.x.getu();
  for (int i = 0; i < local_n; i++) {
    pfloat3 pos = pvec[i].pos;
    float x = pos.x.getu() - xc;
    
    float d, p, vx, vy, vz, bx, by, bz;
    vx = vy = vz = 0;
    bx = by = bz = 0;

#if 0
    const float eps = 0.02;
#define smooth(y1, y2,x) ((y1) + ((y2) - (y1)) * (1 + tanh((x)/eps))*0.5f)
    d  = smooth(dL, dR, x);
    p  = smooth(pL, pR, x);
    vx = smooth(vtL, vtR, x);
    vy = smooth(vnL, vnR, x);
    by = smooth(BnL, BnR, x);
    bx = Bt;
#else
    if (x < 0) {
      d = dL;
      p = pL;
      vx = vtL;
      vy = vnL;
      bx = Bt;
      by = BnL;
    } else {
      d = dR;
      p = pR;
      vx = vtR;
      vy = vnR;
      bx = Bt;
      by = BnR;
    }
#endif

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




