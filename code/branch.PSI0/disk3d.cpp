#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_qrng.h>

#define DENS_MIN 1.0e-7
#define HoverR   0.1f
#define _ISO_

float system::compute_pressure(const float dens, const float ethm) {
  return ethm; //*(gamma_gas - 1.0f);
}


float4 system::body_forces(const pfloat3 pos) {
  const float3 fpos = {pos.x.getu(), pos.y.getu(), pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2   = dr.x*dr.x + dr.y*dr.y + 0*dr.z*dr.z + gravity_eps2;
  const float ids   = (ds2 > 0.0f) ? 1.0f/sqrt(ds2) : 0.0f;
  const float ids2  = ids*ids;
  const float mids  = ids  * gravity_mass;
  const float mids3 = ids2 * mids;
  
  return float4(-mids3*dr.x, -mids3*dr.y, -mids3*dr.z, -mids);
};


bool system::remove_particles_within_racc(const int idx) {
  if (gravity_rin == 0.0f && gravity_rout == 0.0f) return false;
  const particle &pi = pvec[idx];
  const float3 fpos = {pi.pos.x.getu(), pi.pos.y.getu(), pi.pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  
  if (ds2 <= sqr(gravity_rin)) {
    local_n--;
    for (int j = idx; j < local_n; j++) {
      pvec[j] = pvec[j+1];
      pmhd[j] = pmhd[j+1];
    }
  }
  
  if (ds2 >= sqr(gravity_rout)) {
    local_n--;
    for (int j = idx; j < local_n; j++) {
      pvec[j] = pvec[j+1];
      pmhd[j] = pmhd[j+1];
    }
  }

      
//     std::swap(pvec    [idx], pvec    [local_n-1]);
//     std::swap(pmhd    [idx], pmhd    [local_n-1]);
//     std::swap(pmhd_dot[idx], pmhd_dot[local_n-1]);
//     pvec[idx].local_idx = idx;
//     local_n--;
//   }

  return (ds2 <= 0.0f);
  
}

void system::boundary_particles(const int idx) {
  const particle pi = pvec[idx];
  ptcl_mhd mi = (pmhd[idx] * (1.0f/pi.wght)).to_primitive();

  const float x = pi.pos.x.getu() - gravity_pos.x;
  const float y = pi.pos.y.getu() - gravity_pos.y;
  const float R = sqrt(x*x + y*y);

//   const float z = pi.pos.z.getu() - gravity_pos.z;
//   const float r = sqrt(x*x + y*y + z*z);
  
#ifdef _ISO_
  const float vcirc = sqrt(gravity_mass/R);
#if 1
  const float cs    = vcirc * HoverR;
#else
  const float cs    = HoverR; 
#endif
  mi.ethm = sqr(cs) * mi.dens;
#endif
  
  static bool do_magnetic = false;
  //  do_magnetic = true;
  if (iteration == 200 && !do_magnetic) {
    if (myid == 0 && idx == local_n-1){
      fprintf(stderr, "\n********************************************\n");
      fprintf(stderr, " ******** ENABLING MAGNETIC FIELDS *********\n");
      fprintf(stderr, " *******************************************\n\n");
    }
    
    if (idx == local_n - 1) do_magnetic = true;
// #define _TOROIDAL_
#ifdef  _TOROIDAL_
    const float Rtube = 2.5f;
    const float Dtube = 0.5f;
    const float beta  = 400.0f;
    
    const float Rt  = Rtube - R;
    const float dR  = sqrt(Rt*Rt + z*z);
    const float pres = compute_pressure(mi.dens, mi.ethm);
    const float Bt   = sqrt(2.0 * pres/beta);

    float bx = 0, by = 0, bz = 0;
    if (dR < Dtube) {
      bx = -y/R * Bt;
      by = +x/R * Bt;
    }

#else

    const float beta = 13480.0f;
    const float dens_mid = 0.5; //exp(-sqr(2.0)/2.0);
    const float pres = cs * dens_mid;
    const float B0   = sqrt(2.0 * pres/beta);

    float bx =  0.0f;
    float by =  0.0f;
    float bz =  0.0f;
    
    const float B_Rin  = 2.5f;
    const float B_Rout = 3.5f;
    const float dB     = (B_Rout - B_Rin); //*0.5f;
#if 1
    if (R >= B_Rin && R <= B_Rout) {
      bz = B0 * sin(2*M_PI * (R - B_Rin) / dB);
//       bz = B0;
      //       if (R < (B_Rin  + B_Rout)*0.5f) bz = -b0;
    }
#else
    const int n = R/dB;
    bz = b0 * 2.0*(2.0*(n & 1) - 1);
#endif

#endif
    mi.B = (real3){bx, by, bz};
    mi.psi = 0;
  }
  
#if 0
  const float Rin  = 1.00;
  const float Rout = 4.250;
  if (R < Rin || R > Rout) {
    mi.B = (real3){0.0f, 0.0f, 0.0f};
  }
#endif
  

//   pfloat3 prmin = global_domain.rmin();
//   pfloat3 prmax = global_domain.rmin();
//   float3 rmin = {prmin.x.getu(), prmin.y.getu(), prmin.z.getu()};
//   float3 rmax = {prmax.x.getu(), prmax.y.getu(), prmax.z.getu()};
//   const float3 L = {rmax.x - rmin.x,
// 		    rmax.y - rmin.y,
// 		    rmax.z - rmin.z};

//   const float3 h = {0.25, 0.25, 0.25};
  
//   rmin.x += h.x;
//   rmax.x -= h.x;
  
//   rmin.y += h.y;
//   rmax.y -= h.y;

//   rmin.z += h.z;
//   rmax.z -= h.z;

//   boundary bnd;
//   bnd.set_x(rmin.x, rmax.x);
//   bnd.set_y(rmin.y, rmax.y);
//   bnd.set_z(rmin.z, rmax.z);

//   assert(global_domain.isinbox(bnd));
//   assert(!bnd.isinbox(global_domain));

//   const float dens_min = DENS_MIN;
//   if (!bnd.isinbox(pi.pos)) {
//     mi.dens = dens_min;
//     mi.ethm = sqr(cs) * mi.dens;
// #ifndef _ISO_
// #error "Only _ISO_ is supported ..."
// #endif    
//   }
  
  




  pmhd[idx] = mi.to_conservative() * pi.wght;
  return;

  
}

void system::setup_particles(const bool init_data) {
//   gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_halton, 3);
  gsl_qrng *q = gsl_qrng_alloc (gsl_qrng_sobol, 3);


  kernel.set_dim(3);

  global_n = 0;
  local_n  = 0;
  
  NGBmin  = 32.99;
  NGBmean = 33.00;
  NGBmax  = 33.01;  

  int nn = 0;
  NGBmin  += nn;
  NGBmean += nn;
  NGBmax  += nn;
  

  const float3 rmin = (float3){ 0.0f,  0.0f,  0.0f};
  const float3 rmax = (float3){20.0f, 20.0f, 20.0f};
  
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

  ///////////// generate disc
  
  global_n = 300000;
  const int N_per_ann = 1000;
  
  const float ecc   = 0.0f;
  const float Mbh   = 1.0f;
//   const float Mdisk = 0.1f;
  const float sig_slope = 0.0f;
  gamma_gas = 5.0f/3;
  gamma_gas = 1.0f;

//   const float cs0 = _CS0_;
  
  const float Rin  = 0.5f;
  const float Rout = 4.25f;
  
  const int   nmode  = 4;
  const float b0     = 0*0.05513f/nmode;
  const float B_Rin  = 2.0f;
  const float B_Rout = 3.0f;
  
  const int  Nann = global_n/N_per_ann;
  global_n = Nann * N_per_ann;
  
  const double epsilon = std::pow(Rout/Rin, 1.0f/(float)Nann);
  if (myid == 0) {
    fprintf(stderr, " Nann= %d , annulus scaling factor = %lg\n", Nann, epsilon);
  }
  
  
  
  
  gravity_mass = Mbh;
  gravity_eps2 = sqr(0.0f);
  gravity_rin  = Rin *0.5;
  gravity_rout = Rout*2.0;
  gravity_pos  = (float3){(rmin.x + rmax.x)*0.5f, 
			  (rmin.y + rmax.y)*0.5f, 
			  (rmin.z + rmax.z)*0.5f};
  
  
  int3 nt = {16,16,4};
   nt = (int3){8,8,8};
//   nt = (int3){2,2,2};
  box.set(nproc, nt, global_domain);

  if (!init_data) return;

  if (myid == 0) {

    int idx = 0;

    for (int j = 0; j < Nann; j++) {
      const float Rj  = Rin * std::pow(epsilon, (double)(j+0));
      const float Rjp = Rin * std::pow(epsilon, (double)(j+1));
      const float Rrange = Rjp - Rj;

      const float sig_slope = -1.0f;
#if 1
      const float mass_ratio = 
	(std::pow(Rjp,  2.0f - sig_slope) - std::pow(Rj,  2.0f - sig_slope))/
	(std::pow(Rout, 2.0f - sig_slope) - std::pow(Rin, 2.0f - sig_slope));
      const int Nj = (int)(mass_ratio * global_n); 
#else
      const int Nj = N_per_ann; 
#endif
      
      for (int i = 0; i < Nj; i++) {
	double v[3];
	gsl_qrng_get(q, v);
	bool flag = true;
	float r, phi, aaa;
	while (flag) {

	  // SEMI-MAJOR AXIS
	  
	  aaa = Rrange * drand48() + Rj;
 	  phi = drand48() * 2*M_PI;
	  aaa = Rrange * v[0] + Rj;
	  phi = 2*M_PI * v[1];
	  
	  // CONVERT SEMI-MAJOR AXIS TO RADIUS
	  r = aaa * ((1.0-sqr(ecc))/(1.0 + (ecc*cos(phi))));
	  
	  const float area_scale = r / (aaa*(1.0+ecc));
	  const float scale = area_scale;
	  const float fv = drand48();

	  if (fv <= scale) flag = false;
	  flag = false;
	}
	
	const float vcirc = sqrt(Mbh/r);
#if 1
	const float cs    = vcirc * HoverR;
#else
	const float cs = HoverR;
#endif
	const float H0    = cs/vcirc * r;
	const float H     = cs/vcirc * r * 2;// std::pow(r/Rin, 0.7);
	
	// 1st scale height

	flag  = true;
	float height;
	while (flag) {
	  height = (2.0*drand48() - 1.0) *H;
	  height = 2.0*v[2] - 1.0;
	  const float p = 1.5;
	  height = sign(height)*std::pow(std::abs(height), 1.0/p) * H;
// 	  gsl_qrng_get(q, v);
	  const float fp = 0.5f;
	  const float fz = drand48() * std::pow(exp(1.0*sqr(H)/2/sqr(H0)), fp);
	  const double scale = std::pow(exp( + 1.0*sqr(height)/2.0/sqr(H0)), fp);
	  if (fz <= scale) flag = false;
 	  flag = false;
	}

// 	height = (2.0*drand48() - 1.0) * 1*H;
	
	particle p;

	const float h  = std::pow(M_PI*(sqr(Rjp) - sqr(Rj))*2*H/Nj, 1.0/3);
	p.h = 3.0*h;
	p.vel = (float3){0.0f,0.0f,0.0f};
	p.wght = 1.0f;
	
	const float xp = gravity_pos.x + r * cos(phi);
	const float yp = gravity_pos.y + r * sin(phi);
	const float zp = gravity_pos.z + height;
	
	p.pos.x.set(xp);
	p.pos.y.set(yp);
	p.pos.z.set(zp);
	
	p.global_idx = idx;
	p.local_idx  = idx++;
	pvec.push_back(p);
	local_n++;

	
      }
     
    }
    
    global_n = pvec.size();
    
    fprintf(stderr, " ****** global_n= %d  local_n= %d\n", global_n, local_n);
    
    const int n = pvec.size();
    for (int pc = 0; pc < n; pc++) {
      pfloat3 pos = pvec[pc].pos;
      float x = pos.x.getu() - gravity_pos.x;
      float y = pos.y.getu() - gravity_pos.y;
      float z = pos.z.getu() - gravity_pos.z;
      float rsoft = sqrt(x*x + y*y + 0*z*z + gravity_eps2);
      float r = sqrt(x*x + y*y + 0*z*z);
      
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

      const float vcirc = sqrt(Mbh/r);
#if 1
      const float cs = vcirc * HoverR;
      const float H  = cs/vcirc * r;
      p.dens = std::max(exp(-sqr(z)/2/sqr(H)), DENS_MIN); //powf(r, -sig_slope);
//       float4 acc  = body_forces(pos);
//       pfloat3 pos0;
//       pos0.x = pos.x;
//       pos0.y = pos.y;
//       pos0.z.set(gravity_pos.z);

//       float4 acc0 = body_forces(pos0);
//       p.dens = std::max(exp(-(acc.w - acc0.w)/sqr(cs)), DENS_MIN);
#else
      const float cs = HoverR;
      float4 acc  = body_forces(pos);
      pfloat3 pos0;
      pos0.x.set(gravity_pos.x + Rin);
      pos0.y.set(gravity_pos.y);
      pos0.z.set(gravity_pos.z);
      float4 acc0 = body_forces(pos0);
      p.dens = std::max(exp(-(acc.w - acc0.w)/sqr(cs)), DENS_MIN);
#endif
      if (gamma_gas > 1.0f)
	p.ethm = sqr(cs) * p.dens /(gamma_gas - 1.0)/gamma_gas;
      else
	p.ethm = sqr(cs) * p.dens;
      
      p.vel  = (real3){vx, vy, vz};
      p.B    = (real3){bx, by, bz};
      p.psi = 0;
      p.marker = 1;
      pmhd.push_back(p);


    }
  }

  MPI_Bcast(&global_n,  1, MPI_INT, 0, MPI_COMM_WORLD);

  fprintf(stderr, "proc= %d  local_n= %d  global_n= %d \n", myid, local_n, global_n);

  
  MPI_Barrier(MPI_COMM_WORLD);
  
  pmhd_dot.resize(local_n);
  distribute_particles();

  divB_i.resize(local_n);
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;

  for (int i = 0; i < local_n; i++)
    pmhd_dot[i].set(0.0);

  
  fprintf(stderr, "proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

}

