#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

void system::set_geometry() {

  if (myid == 0) fprintf(stderr, " *** setting up the geometry *** \n");

  // # of dimensions in the problem
  kernel.set_dim(2);
  
  // # neighbours
  NGBmin  = 18.9;
  NGBmean = 19.0;
  NGBmax  = 19.1;

//   NGBmin  = 15.9;
//   NGBmean = 16.0;
//   NGBmax  = 16.1;
  
  // periodic box size
  const float3 box_rmin = (float3){0,  0,    0};
  const float3 box_rmax = (float3){1,  1.5,  1};
  
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
    const int nlev = 2; 

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
  if (myid == 0) fprintf(stderr, "  *** blast2d problem *** \n");

  gamma_gas = 5.0/3;

  if (!init_data) return;

  if (myid == 0) fprintf(stderr, "  *** setting up the problem *** \n");
  
  float b0    = 0;
  b0 = 1.0;
  
  const float angle = M_PI/4;
  const float r0    = 0.1;
  const float d0    = 1.0;
  
  for (int i = 0; i < local_n; i++) {
    pfloat3 pos = pvec[i].pos;
    float x = pos.x.getu();
    float y = pos.y.getu();
    
    const float r = sqrt(sqr(x - 0.5) + sqr(y - 0.75));
    
    float d, p, vx, vy, vz, bx, by, bz;
    
    d = d0;
    p = 0.1;
    if (r < r0) p = 10;
    vx = vy = vz = 0;
    bx = by = bz = 0;
    bx = b0 * cos(angle);
    by = b0 * sin(angle);
    
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
