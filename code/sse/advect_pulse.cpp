#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void system::setup_particles() {

  kernel.set_dim(2);
  int nx = 128;
  int ny = 128;
  int nz = 1;

//    nx = ny = 64;
//   nx = ny = 512;
//    nx = ny = 256;

  global_n = nx*ny*nz;
  local_n  = 0;

  NGBmin  = 15.99;
  NGBmean = 16.0;
  NGBmax  = 16.01;

#if 1
  const int nn = 3;
  NGBmin  += nn;
  NGBmax  += nn;
  NGBmean += nn;
#endif

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
    
    
    const float3 L = global_domain.size();

    // generate level boxes

    const int nlevels = 1;
    std::vector<boundary> level_boxes(nlevels);
    for (int lev = 0; lev < nlevels; lev++) {
      float4 p;
      p.x = p.y = 0.5;
      p.z = 0.0;
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
	    xmin + i*dr.x + dr.x*(0.5 - drand48()) * ff + 0.5*dr.x*(j&1),
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
    
    const float xc = 0.5f;
    const float yc = 0.5f;
    const float r0 = 0.3/2;

    const int n = pvec.size();

    const float amp = 0.001;
    const real vel0 = 1;

    const real angle = 30 * M_PI/180.0;
    const real vel_x = vel0*cos(angle);
    const real vel_y = vel0*sin(angle);

    const real x0 = xc;
    const real y0 = yc;
    for (int pc = 0; pc < n; pc++) {
      const pfloat3 pos = pvec[pc].pos;
      const float x = pos.x.getu() - xc;
      const float y = pos.y.getu() - yc;
      const float r = sqrt(x*x + y*y);
      
      const real vx = vel_x;
      const real vy = vel_y;
      const real vz = 1;

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
	d0 = 1.0;
      }
#endif


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

//   ptcl_mhd_dot.resize(local_n);

//   ptcl_mhd_min.resize(local_n);
//   ptcl_mhd_max.resize(local_n);
  
  divB_i.resize(local_n);
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;

  fprintf(stderr, "proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

}
