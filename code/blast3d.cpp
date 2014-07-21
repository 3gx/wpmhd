#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void system::setup_particles() {
  
  gamma_gas = 5.0/3;


  kernel.set_dim(3);

  int nx = 64;
  int ny = 64;
  int nz = 64;

  nx = ny = nz = 256;

  global_n = nx*ny*nz;
  local_n  = 0;

  NGBmin  = 32.9999;
  NGBmean = 33.0000;
  NGBmax  = 33.0001;
  
  float3 rmin = (float3){0,0,0};
  float3 rmax = (float3){1,0,0};
  
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
	    xmin + i*dr.x,
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

    
    const float xc = 0.5f;
    const float yc = 0.5f;
    const float zc = 0.5f;
    const float r0 = 0.1f;

    const float d0 = 1.0f;
    const float p1 = 0.1f;
    const float p2 = 10.0f;
    const float b0 = 1.0/sqrt(3.0);

    const int n = pvec.size();
    
    for (int pc = 0; pc < n; pc++) {
      const pfloat3 pos = pvec[pc].pos;
      const float x = pos.x.getu() - xc;
      const float y = pos.y.getu() - yc;
      const float z = pos.z.getu() - zc;
      const float r = sqrt(x*x + y*y + z*z);
      
      const real vx = 0;
      const real vy = 0;
      const real vz = 0;

      real bx = 4*b0;
      real by = 2*b0;
      real bz = 1*b0;

      float       p0 = p1;
      if (r < r0) p0 = p2;

      
      ptcl_mhd p;

      p.dens = d0;
      p.ethm = p0/(gamma_gas - 1.0);
      p.vel  = (real3){vx, vy, vz};
      p.B    = (real3){bx, by, bz};
      p.psi  = 0;
      
      pmhd.push_back(p);

    }
  }
  
  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);

  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);

  
  MPI_Barrier(MPI_COMM_WORLD);
  
  box.set_np(nproc, np);
  box.set_bnd_sampling(global_domain);
  
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
