#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void system::boundary_particles(const int idx) {
  return;
}

void system::setup_particles() {


  kernel.set_dim(2);

  global_n = 0;
  local_n  = 0;
  
  NGBmin  = 18.99;
  NGBmean = 19.00;
  NGBmax  = 19.01;  

//   int nn = 7;
//   NGBmin  += nn;
//   NGBmean += nn;
//   NGBmax  += nn;

  const float3 rmin = (float3){ 0.0f,  0.0f,  0.0f};
  const float3 rmax = (float3){20.0f, 20.0f, +1.0f};
  
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

  gravity_mass = 10.0f;
  gravity_eps2 = 0.1f;
//   gravity_rin  = 0.01;
//   gravity_rout = 8.0f;
  gravity_pos  = (float3){(rmin.x + rmax.x)*0.5f, (rmin.y + rmax.y)*0.5f, (rmin.z + rmax.z)*0.5f};
  gamma_gas = 5.0f/3.0f;
  gamma_gas = 1.01;

  if (myid == 0) {

    const float Rcl = 0.4f;
    const float Mcl = 0.1f;
    const int   nr  = 64;
    const float dR  = Rcl/nr;
    const float Xc = 1.5*Rcl*2;
    const float Yc = 3.0*Rcl*2;
    const float nr_high = 1.0/sqr(1*dR);
    const float nr_low  = 1.0/sqr(4*dR);
    const float sigma   = 0.2;

    int idx = 0;
    float rp = dR*0.5;
    while(rp < 1.5*Rcl) {
      const float nr = nr_low + 0.5f*(nr_high - nr_low)*(1 - tanh((rp - 1.1*Rcl)/sigma));
//       const float nr = nr_high + (nr_low - nr_high)*tanh((rp - 1.1*Rcl)/sigma);
      const float dr = 1.0/sqrt(nr);
      
      const float dtheta1 = dr/rp;
      int   ntheta  = (int)(2.0f*M_PI/dtheta1);
      float dxi     = 2.0f*M_PI - ntheta*dtheta1;
      float dtheta  = dtheta1 + dxi/ntheta;
      if (ntheta == 0) {
	ntheta = 3;
	dtheta = 2*M_PI/ntheta;
      }
      double theta = 0.5f*dtheta;
      for (int itheta = 0; itheta < ntheta; itheta++) {
	
	const float xp = Xc + gravity_pos.x + rp * cos(theta);
	const float yp = Yc + gravity_pos.y + rp * sin(theta);

	if (xp < rmin.x || xp > rmax.x ||
	    yp < rmin.y || yp > rmax.y) continue;

	theta += dtheta ;
	particle p;

	p.h   = 1.5*dr;
	p.vel = (float3){0.0f,0.0f,0.0f};
	p.wght = 1.0;
	
	p.pos.x.set(xp);
	p.pos.y.set(yp);
	p.pos.z.set(gravity_pos.z);

	p.global_idx = idx;
	if (rp > 1.1*Rcl) p.global_idx = -idx;
	p.local_idx  = idx++;
	pvec.push_back(p);
	local_n++;

      }
      rp += dr;
    }
    

    
    global_n = pvec.size();
    
    fprintf(stderr, " ****** global_n= %d  local_n= %d\n", global_n, local_n);

    
    const float beta = 1;
    const float dens0 = Mcl/sqr(Rcl)/M_PI;
    const float Mach = 100;
    fprintf(stderr, "dens0= %g\n", dens0);
    const int n = pvec.size();
    for (int pc = 0; pc < n; pc++) {
      const pfloat3 pos = pvec[pc].pos;
      const float x = pos.x.getu() - gravity_pos.x;
      const float y = pos.y.getu() - gravity_pos.y;
      const float z = pos.z.getu() - gravity_pos.z;
      const float rsoft = sqrt(x*x + y*y + z*z + gravity_eps2);

      const float xc = x - Xc + dR*0.5;
      const float yc = y - Yc + dR*0.5;
      const float zc = z;
      float rc = sqrt(xc*xc + yc*yc + zc*zc);
      
      const real vphi = sqrt(gravity_mass/sqrt(Xc*Xc + Yc*Yc + gravity_eps2));
      const float cs0 = vphi/Mach;

      const float dens_min = log(dens0*1.0e-3f);
      const float dens_max = log(dens0);
      float d0 = dens_min + 0.5f*(dens_max - dens_min)*(1 - tanh((rc - 1.0*Rcl)/sigma));
      d0 = exp(d0);

//       fprintf(stdout, "rc= %g  d0= %g rp= %g\n", rc, d0, rp);
//       float d0 = dens0;
      float p0 = sqr(cs0) * d0/gamma_gas;
      const float b0 = 0*sqrt(2.0f*p0/beta);

      real vx = -vphi*0.3f;
      real vy = -vphi*0.15f;
      real vz = 0;
      
      real bx =  0.0;
      real by =  0.0;
      real bz = +b0;

      bx = by = b0;
      bz = 0;

      real azx = b0 * (xc)/rc;
      real azy = b0 * (yc)/rc;
      bx = azy;
      by = -azx;
      
//       d0 = dens0;
      if (rc > Rcl) {
//   	d0 = dens0*1.0e-3;
 	p0 = sqr(cs0) * d0/gamma_gas;
	bx = by = bz = 0;
      }
//       bx = by = bz = 0;
      
      ptcl_mhd p;

      p.dens = d0;
      p.ethm = p0/(gamma_gas - 1.0);
      p.vel  = (real3){vx, vy, vz};
      p.B    = (real3){bx, by, bz};
      p.psi = 0;
      p.marker = 1;
      pmhd.push_back(p);


    }
  }
  
  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);

  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);

  
  MPI_Barrier(MPI_COMM_WORLD);
  
  box.set_np(nproc, np);
  box.set_bnd_sampling(global_domain);
  
  distribute_particles();
  
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
