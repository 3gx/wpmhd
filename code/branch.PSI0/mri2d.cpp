#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

float system::compute_pressure(const float dens, const float ethm) {
  const float cs0 = 0.1;
  return sqr(cs0) * dens;
//   return ethm*(gamma_gas - 1.0f);
}


float4 system::body_forces(const pfloat3 pos) {
  const float3 fpos = {pos.x.getu(), pos.y.getu(), pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2   = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + gravity_eps2;
  const float ids   = (ds2 > 0.0f) ? 1.0f/sqrt(ds2) : 0.0f;
  const float ids2  = ids*ids;
  const float mids  = ids  * gravity_mass;
  const float mids3 = ids2 * mids;

  return float4(-mids3*dr.x, -mids3*dr.y, -mids3*dr.z, -mids);
//   return float4(0,0,0,0);
};


bool system::remove_particles_within_racc(const int idx) {
  if (gravity_rin == 0.0f && gravity_rout == 0.0f) return false;
  const particle &pi = pvec[idx];
  const float3 fpos = {pi.pos.x.getu(), pi.pos.y.getu(), pi.pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2   = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  
  if (ds2 <= sqr(gravity_rin) || ds2 >= sqr(gravity_rout)) {
    local_n--;
    for (int j = idx; j < local_n; j++) {
      pvec[j] = pvec[j+1];
      pmhd[j] = pmhd[j+1];
    }
      
//     std::swap(pvec    [idx], pvec    [local_n-1]);
//     std::swap(pmhd    [idx], pmhd    [local_n-1]);
//     std::swap(pmhd_dot[idx], pmhd_dot[local_n-1]);
//     pvec[idx].local_idx = idx;
//     local_n--;
  }

  return (ds2 <= 0.0f);
  
}

void system::boundary_particles(const int idx) {
  pmhd[idx].ethm = compute_pressure(pmhd[idx].dens, 1.0);
  return;
  particle &pi = pvec[idx];
  ptcl_mhd &mi = pmhd[idx];
  const float Vi = weights[idx];

  const float3 fpos = {pi.pos.x.getu(), pi.pos.y.getu(), pi.pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2   = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

  const float R1    = 1.3;
  const float R2    = 3.7;
  
  const float cs0    = 0.1f;
  const float dens0  = 1.0f;
  
  mi.marker = 1;
  if (ds2 < sqr(R1) || ds2 > sqr(R2)) {
    const float rsoft = sqrt(ds2 + gravity_eps2);
    const float ids   = 1.0f/sqrt(ds2);
    real vphi = sqrt(gravity_mass/rsoft);
    
    real vx = -vphi * dr.y*ids;
    real vy = +vphi * dr.x*ids;
    
    pi.vel.x = vx;
    pi.vel.y = vy;
    pi.vel.z = 0.0;
    
    mi.marker  = -1;

#if 1

#ifdef _SEMICONSERVATIVE_
    mi.B = (real3){0.0, 0.0, 0.0};
    mi.ethm = sqr(cs0) * dens0 /(gamma_gas - 1.0)/gamma_gas;
    mi.etot = mi.ethm*Vi  + 0.5*(sqr(mi.B.x  ) + sqr(mi.B.y  ) + sqr(mi.B.z  ))*Vi;

    mi.mass = dens0*Vi;
    mi.mom.x = pi.vel.x * mi.mass;
    mi.mom.y = pi.vel.y * mi.mass;
    mi.mom.z = pi.vel.z * mi.mass;
#else
    mi.B = (real3){0.0, 0.0, 0.0};
    mi.vel.x = pi.vel.x;
    mi.vel.y = pi.vel.y;
    mi.vel.z = pi.vel.z;
//     if (ds2 < sqr(R1*0.8)) {
//       mi.dens = dens0;
//       mi.ethm = sqr(cs0) * mi.dens /(gamma_gas - 1.0)/gamma_gas;
//     }

#endif

#endif
    
    
  }
  
  
}

void system::setup_particles() {


  kernel.set_dim(2);

  global_n = 0;
  local_n  = 0;
  
  NGBmin  = 18.9999;
  NGBmean = 19.0000;
  NGBmax  = 19.0001;  

//   int nn = 1;
//   NGBmin  += nn;
//   NGBmax  += nn;
//   NGBmean += nn;

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

  const float Rin    = 1.0f;
  const float Rout   = 4.0f;
  const float cs0    = 0.1f;
  const float dens0  = 1.0f;
  const int   nmode  = 4;
  const float b0     = 0.05513f/nmode;
  const float B_Rin  = 2.0f;
  const float B_Rout = 3.0f;

  gravity_mass = 1.0f;
  gravity_eps2 = sqr(0.0f);
  gravity_rin  = Rin *0.8;
  gravity_rout = Rout*1.2;
  gravity_pos  = (float3){(rmin.x + rmax.x)*0.5f, (rmin.y + rmax.y)*0.5f, (rmin.z + rmax.z)*0.5f};
  gamma_gas = 5.0f/3.0f;
//   gamma_gas = 1.0001;

  if (myid == 0) {
    
    int idx = 0;
    const int nr_max =  32;
    const int nr_min  = 16;
    const float R1    = 1.8;
    const float R2    = 3.2;
    const float dR    = R2 - R1;
    const float sigma = 0.2f;
    const float nr_high = sqr(nr_max/dR);
    const float nr_low = sqr(nr_min/dR);
    
    float rp = Rin;
    while(rp < Rout) {
      const float nr = nr_low + 0.5f*(nr_high - nr_low)*(tanh((rp - R1)/sigma) - tanh((rp - R2)/sigma));
      const float dr = 1.0/sqrt(nr);
      const float dtheta1 = dr/rp;
      const int   ntheta  = (int)(2.0f*M_PI/dtheta1);
      const float dxi     = 2.0f*M_PI - ntheta*dtheta1;
      const float dtheta  = dtheta1 + dxi/ntheta;
      double theta = 0.5f*dtheta;
      for (int itheta = 0; itheta < ntheta; itheta++) {
	
	const float xp = gravity_pos.x + rp * cos(theta);
	const float yp = gravity_pos.y + rp * sin(theta);

	theta += dtheta ;
	particle p;

	p.h   = 1.5*dr;
	p.vel = (float3){0.0f,0.0f,0.0f};
	p.wght = 1.0;
	
	p.pos.x.set(xp);
	p.pos.y.set(yp);
	p.pos.z.set(gravity_pos.z);
	
	p.global_idx = idx;
	p.local_idx  = idx++;
	pvec.push_back(p);
	local_n++;

      }
      rp += dr;
    }
    
    
    global_n = pvec.size();
    
    fprintf(stderr, " ****** global_n= %d  local_n= %d\n", global_n, local_n);
    
    const int n = pvec.size();
    for (int pc = 0; pc < n; pc++) {
      pfloat3 pos = pvec[pc].pos;
      float x = pos.x.getu() - gravity_pos.x;
      float y = pos.y.getu() - gravity_pos.y;
      float z = pos.z.getu() - gravity_pos.z;
      float rsoft = sqrt(x*x + y*y + z*z + gravity_eps2);
      float r = sqrt(x*x + y*y + z*z);
      
      real vphi = sqrt(gravity_mass/rsoft);
      
      real vx = -vphi * y/r;
      real vy = +vphi * x/r;
      real vz = 0;
      
      const real dv = vphi * 1.0e-4;
      vx += drand48() * dv;
      vy += drand48() * dv;
      
      real bx =  0.0;
      real by =  0.0;
      real bz = +b0;
      
      if (r < B_Rin || r > B_Rout) bz = 0.0f;

      ptcl_mhd p;

      p.dens = dens0;
      p.ethm = sqr(cs0) * dens0 /(gamma_gas - 1.0)/gamma_gas;
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
