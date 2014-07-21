#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

#if 1


#if 0
#include "orszag_tang.cpp"
#elif 0
#include "orszag_tang3d.cpp"
#elif 0
#include "cloud2d.cpp"
#elif 1
#include "disk3d.cpp"
#elif 1
#include "mri3d.cpp"
#elif 0
#include "mri2d.cpp"
#elif 1
#include "mhd_rotor.cpp"
#elif 1
#include "advect_pulse.cpp"
#elif 0
#include "khi.cpp"
#elif 0
#include "advect3d.cpp"
#elif 1
#include "blast3d.cpp"
#elif 1
#include "khi3d.cpp"
#elif 1
#include "current_sheet.cpp"
#else
#error "choose something pls in setup_particles.cpp"
#endif


#else
void system::setup_particles() {

  kernel.set_dim(3);
  int nx = 128;
  int ny = 64;
  int nz = 32;
  nx = ny = nz = 64; 

//   nx = ny = 128; nz = 1;
//   int nxyz = 128;
//    nx = ny = nz = nxyz;
  nx = ny = 256; nz = 1; kernel.set_dim(2);
  global_n = nx*ny*nz;
  local_n  = 0;
    

  NGBmin  = 12.99999;
  NGBmean = 13.0;
  NGBmax  = 13.00001;

//   int nn = 8;
//   NGBmin  += nn;
//   NGBmean += nn;
//   NGBmax  += nn;
  
  float3 rmin = (float3){0,0,0};
  float3 rmax = (float3){1,0,0};
  
  rmax.y = rmax.x/nx * ny;
  rmax.z = rmax.x/nx * nz;

  
#ifdef _PERIODIC_FLOAT_
  pfloat<0>::set_range(rmin.x, rmax.x);
  pfloat<1>::set_range(rmin.y, rmax.y);
  pfloat<2>::set_range(rmin.z, rmax.z);
#endif

  global_domain.set_x(rmin.x, rmax.x);
  global_domain.set_y(rmin.y, rmax.y);
  global_domain.set_z(rmin.z, rmax.z);

  pvec.clear();

  if (myid == 0) {
    
    float3 dr = global_domain.size();
    dr.x *= 1.0/nx;
    dr.y *= 1.0/ny;
    dr.z *= 1.0/nz;
    real   h = dr.x *1.2;
    
//     h = 0.00794126;
    int idx = 0;
    for (int k = 0; k < nz; k++) 
      for (int j = 0; j < ny; j++) 
	for (int i = 0; i < nx; i++) {
	  particle p;
	  real ff = 0.15;
// 	  p.pos = (pfloat3){
// 	    i*dr.x + dr.x*(0.5 - drand48()) * ff,
// 	    j*dr.y + dr.y*(0.5 - drand48()) * ff, 
// 	    k*dr.z + dr.z*(0.5 - drand48()) * ff};
	  float3 pos = (float3){
	    i*dr.x + dr.x*(0.5 - drand48()) * ff,
	    j*dr.y + dr.y*(0.5 - drand48()) * ff, 
	    k*dr.z + dr.z*(0.5 - drand48()) * ff};
//  	  p.pos.x = pos.x;
#if 0
 	  pos.x = drand48();
	  pos.y = drand48();
 	  pos.z = drand48();
#endif
 	  p.pos.x.set(pos.x);
 	  p.pos.y.set(pos.y);
 	  p.pos.z.set(pos.z);

// 	  p.pos.x += dr.x*(0.5 - drand48()) * ff;
// 	  p.pos.y += dr.y*(0.5 - drand48()) * ff;
// 	  p.pos.z += dr.z*(0.5 - drand48()) * ff;
//       	  p.pos = (pfloat3){drand48(), drand48(), drand48()};

// 	  float myh;
// 	  std::cin >> myh;
	  p.h   = h;
	  p.vel = (real3){0,0,0};
	  p.global_idx = idx;
	  p.local_idx  = idx++;

	  if (kernel.ndim < 3) p.pos.z.set(0.0);
	  if (kernel.ndim < 2) p.pos.y.set(0.0);

	  pvec.push_back(p);
	  local_n++;

// 	  fprintf(stderr, "i= %d\n", (int)pvec.size() - 1);
// 	  pfloat3 x = {p.pos.x, p.pos.y, p.pos.z};
	}
    
  }
  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);

  ptcl_mhd.resize(pvec.size());
  ptcl_aux.resize(pvec.size());
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  

  box.set_np(nproc, np);
  box.set_bnd_sampling(global_domain);

//   fprintf(stderr ,"proc= %d: dist particles on proc\n", myid);
  distribute_particles();

  
  Bxx.resize(local_n);
  Bxy.resize(local_n);
  Bxz.resize(local_n);
  Byy.resize(local_n);
  Byz.resize(local_n);
  Bzz.resize(local_n);
  
  ptcl_aux.resize(local_n);
  
  ptcl_mhd_grad[0].resize(local_n);
  ptcl_mhd_grad[1].resize(local_n);
  ptcl_mhd_grad[2].resize(local_n);
//   if (myid == 2) {
//     for (int i = 0; i < local_n; i++) {
//       particle p = pvec[i];
//       fprintf(stderr, "local_n= %d  %g %g %g\n",
// 	      i, p.pos.x, p.pos.y, p.pos.z);
//     }
//   }


  fprintf(stderr, "proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

#if 0
  char fn[256];
  sprintf(fn, "proc_%.2d.dump", myid);
  FILE *fout = fopen(fn, "w");

  for (int i = 0; i < local_n; i++) {
      particle p = pvec[i];
//       fprintf(stderr, "local_n= %d  %g %g %g\n",
// 	      i, p.pos.x, p.pos.y, p.pos.z);

      fprintf(fout, "%d %d  %g %g %g %g\n", 
	      p.global_idx,
	      p.local_idx,
	      (float)p.pos.x,
	      (float)p.pos.y,
	      (float)p.pos.z,
	      (float)p.h);
  }
  fclose(fout);
#endif
}
#endif

