#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_qrng.h>

#define DENS_MIN 1.0e-7
#define HoverR   0.1f
#define HoverR_I 0.1f

#define  Rin  1.0f
#define  Rout 4.0f
#define  Mbh  1.0f

#define _CYLYNDER_

float system::compute_pressure(const float dens, const float ethm) {return ethm; } //*(gamma_gas - 1.0f);}



  float4 system::body_forces(const pfloat3 pos) {
  const float3 fpos = {pos.x.getu(), pos.y.getu(), pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
#ifndef _CYLYNDER_
  const float ds2   = sqr(dr.x)  + sqr(dr.y) + sqr(dr.z) + gravity_eps2;
#else
  const float ds2   = sqr(dr.x)  + sqr(dr.y) + 0.0f*sqr(dr.z) + gravity_eps2;
#endif
  const float ids   = (ds2 > 0.0f) ? 1.0f/sqrt(ds2) : 0.0f;
  const float ids2  = ids*ids;
  const float mids  = ids  * gravity_mass;
  const float mids3 = ids2 * mids;
  
  return float4(-mids3*dr.x, -mids3*dr.y, -mids3*dr.z, -mids);
};


bool system::remove_particles_within_racc(const int idx) {
//   if (gravity_rin == 0.0f && gravity_rout == 0.0f) return false;
  const particle &pi = pvec[idx];
  const float3 fpos = {pi.pos.x.getu(), pi.pos.y.getu(), pi.pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

  
  bool flag = false;
  flag = flag | (ds2 <= sqr(gravity_rin));
  flag = flag | (ds2 >= sqr(gravity_rout));

  if (1) {
    const pfloat3 prmin = global_domain.rmin();
    const pfloat3 prmax = global_domain.rmax();
    float3 rmin = {prmin.x.getu(), prmin.y.getu(), prmin.z.getu()};
    float3 rmax = {prmax.x.getu(), prmax.y.getu(), prmax.z.getu()};
    const float3 L = {rmax.x - rmin.x,
		      rmax.y - rmin.y,
		      rmax.z - rmin.z};
    
    const float3 h = {L.x*0.1, L.y*0.1, L.z*0.1};
    
    rmin.x += h.x/2;
    rmax.x -= h.x/2;
    
    rmin.y += h.y/2;
    rmax.y -= h.y/2;
    
    rmin.z += h.z/2;
    rmax.z -= h.z/2;

    boundary bnd;
    bnd.set_x(rmin.x, rmax.x);
    bnd.set_y(rmin.y, rmax.y);
    bnd.set_z(rmin.z, rmax.z);
    
    assert(global_domain.isinbox(bnd));
    assert(!bnd.isinbox(global_domain));
    const bool inbox = bnd.isinbox(pvec[idx].pos);
    flag = flag | !inbox;

    if (flag) {
       
      const float x = pi.pos.x.getu() - gravity_pos.x;
      const float y = pi.pos.y.getu() - gravity_pos.y;
      const float z = pi.pos.z.getu() - gravity_pos.z; 
      const float R = sqrt(sqr(x) + sqr(y));
      const float vcirc = sqrt(gravity_mass/R);
      const float cs = vcirc * HoverR;
      const float H  = cs/vcirc * R;
      const float dens = std::max(exp(-sqr(z)/2/sqr(H)), DENS_MIN);

      fprintf(stderr, " ********************** ");
      fprintf(stderr, "idx= %d  : mass= %g  pos= %g %g %g    wght= %g  dens= %g  H= %g %g\n",
	      idx, pmhd[idx].mass,
	      pvec[idx].pos.x.getu() - gravity_pos.x,	
	      pvec[idx].pos.y.getu() - gravity_pos.y,
	      pvec[idx].pos.z.getu() - gravity_pos.z,
	      pvec[idx].wght,
	      pmhd[idx].mass/pvec[idx].wght, H, dens
      );
      if (inbox) fprintf(stderr, "  ********* T ********* \n");
      else       fprintf(stderr, "  ********* F ********* \n");
      fprintf(stderr, "bnd: "); bnd.dump(stderr, true);
    }
//     fprintf(stderr, "domain: "); global_domain.dump(stderr, true);
//     fprintf(stderr, "bnd: ");    bnd.dump(stderr, true);
//     if (bnd.isinbox(pvec[idx].pos)) 
//       fprintf(stderr, " flag= T\n");
//     else
//       fprintf(stderr, " flag= F\n");
    

  }


#if 1
  const ptcl_mhd prim = pmhd[idx].to_primitive(pi.wght);
  const float pB   = 0.5f*(sqr(prim.B.x) + sqr(prim.B.y) + sqr(prim.B.z));
  const float pgas = compute_pressure(prim.dens, prim.ethm);
  const float magn = pB/pgas;
  flag = flag | (magn > 1000.0f);
#endif
  
  float4 acc  = body_forces(pi.pos);
  const float vesc = sqrt(fabs(acc.w));
  const float vp = sqrt(sqr(pi.vel.x) + sqr(pi.vel.y) + sqr(pi.vel.z));
  flag = flag | (vp > 10*vesc);
						
  if (flag) {
    local_n--;
    std::swap(pvec    [idx], pvec    [local_n]);
    std::swap(pmhd    [idx], pmhd    [local_n]);
    pvec[idx].local_idx = idx;
  }
      

  return flag;
  
}


void system::boundary_derivatives(const int idx) {
//   return;
  const float bRin  = 1.2f;
  const float dRin = 0.02f;

  const float bRout = 6.5f;
  const float dRout = 0.5f;

  const particle pi = pvec[idx];
  const float x = pi.pos.x.getu() - gravity_pos.x;
  const float y = pi.pos.y.getu() - gravity_pos.y;
  const float R = sqrt(x*x + y*y);

  const float fin  = + tanh((R - bRin )/dRin );
  const float fout = - tanh((R - bRout)/dRout);

   const float fp = (fin + fout)*0.5f;
   
//   pmhd_dot[idx].wB.x *= fp;
//   pmhd_dot[idx].wB.y *= fp;
//   pmhd_dot[idx].wB.z *= fp;
//   pmhd_dot[idx].psi  *= fp;

  const float c = 0.1f * (dt_global > 0) ? 1.0f/dt_global : 0.0f;
  const float fm = c * (1 - fp);
  pmhd_dot[idx].wB.x += -fm * pmhd[idx].wB.x;
  pmhd_dot[idx].wB.y += -fm * pmhd[idx].wB.y;
  pmhd_dot[idx].wB.z += -fm * pmhd[idx].wB.z;
//   pmhd_dot[idx].psi  += -fm * pmhd[idx].psi;

}

void system::boundary_particles(const int idx) {
  const particle pi = pvec[idx];
  ptcl_mhd mi = pmhd[idx].to_primitive(pi.wght);
  
  const float x = pi.pos.x.getu() - gravity_pos.x;
  const float y = pi.pos.y.getu() - gravity_pos.y;
  const float z = pi.pos.z.getu() - gravity_pos.z; 

#if 0
  const float R = sqrt(sqr(x) + sqr(y) + sqr(z));
#else
  const float R = sqrt(x*x + y*y);
#endif
  
  const float vcirc = sqrt(gravity_mass/R);
  const float cs    = vcirc * HoverR;
  mi.ethm = sqr(cs) * mi.dens;

  /*****************************/
  /*****************************/
  /*****************************/

  static bool do_magnetic = false;
  //  do_magnetic = true;
  if (iteration == 1400000000 && !do_magnetic) {
    if (myid == 0 && idx == local_n-1){
      fprintf(stderr, "\n********************************************\n");
      fprintf(stderr, " ******** ENABLING MAGNETIC FIELDS *********\n");
      fprintf(stderr, " *******************************************\n\n");
    }
    
    if (idx == local_n - 1) do_magnetic = true;


//     const float cs = vcirc * HoverR;
//     const float H  = cs/vcirc * R;
//     mi.dens = std::max(exp(-sqr(z)/2/sqr(H)), DENS_MIN);
//     if (gamma_gas > 1.0f)  mi.ethm = sqr(cs) * mi.dens /(gamma_gas - 1.0)/gamma_gas;
//     else                   mi.ethm = sqr(cs) * mi.dens;

    float bx, by, bz;
    bx = by = bz = 0;

// #define _TOROIDAL_
#ifdef _TOROIDAL_  
    const float Rtube = 2.5f;
    const float Dtube = 0.5f;
    const float beta  = 100.0f;
    
    const float Rt  = Rtube - R;
    const float dR  = sqrt(Rt*Rt + z*z);
    const float pres = compute_pressure(mi.dens, mi.ethm);
    const float Bt   = sqrt(2.0 * pres/beta);

    if (dR < Dtube) {
      bx = -y/R * Bt;
      by = +x/R * Bt;
    }
#else
    const float B_Rin  = 2.0f;
    const float B_Rout = 3.0f;
    const float dB     = (B_Rout - B_Rin); //*0.5f;


#if 0
    const float beta = 13480.0f;
    const float dens_mid = 1.0; //exp(-sqr(2.0)/2.0);
    const float pres = cs * dens_mid;
    const float B0   = sqrt(2.0 * pres/beta);
#else
    const float B0   = 0.05513/4/1;
#endif
#if 1
    if (R >= B_Rin && R <= B_Rout) {
      bz = B0 * sin(2*M_PI * (R - B_Rin) / dB);
//       bz = B0;
    }
#else
    const int n = R/dB;
    bz = B0 * sin(2*M_PI * R / dB);
//     bz = b0 * 2.0*(2.0*(n & 1) - 1);
#endif
    
    if (R < B_Rin || R > B_Rout) bz = 0.0f;

#endif

    mi.B = (real3){bx, by, bz};
  
//     const float Brin  = 1.0f;
//     const float Brout = 4.0f;
//     if (R < Brin || R > Brout) {
//       mi.B = (real3){0.0f, 0.0f, 0.0f};
//     }
  
  }
  /*****************************/
  /*****************************/
  /*****************************/

  pmhd[idx] = mi.to_conservative(pi.wght);
  return;

  
}

void system::set_problem(const bool init_data) {
  fprintf(stderr, " ****** disk ****** \n");
  const pfloat3 centre = global_domain.centre;

  gravity_mass = Mbh;
  gravity_eps2 = 0*sqr(0.25f);
  gravity_rin  = Rin  * 0.25f/2;
  gravity_rout = Rout * 2.00f;
//   gravity_rin  = 0;   //Rin  * 0.25f;
  gravity_rout = 1e6; //Rout * 12.00f;
  gravity_pos  = (float3){centre.x.getu(), centre.y.getu(), centre.z.getu()};
  gamma_gas    = 1.0f;
  
  if (!init_data) return;
  fprintf(stderr, " ****** setup ****** \n");
  
  const float B_Rin  = 2.0f;
  const float B_Rout = 3.0f;
  const float dB     = (B_Rout - B_Rin); //*0.5f;

  for (int i = 0; i < local_n; i++) {
    const pfloat3 pos = pvec[i].pos;
    const float x = pos.x.getu() - gravity_pos.x;
    const float y = pos.y.getu() - gravity_pos.y;
    const float z = pos.z.getu() - gravity_pos.z;
    const float R = sqrt(x*x + y*y + gravity_eps2);
    const float r = sqrt(x*x + y*y + z*z + gravity_eps2);

#ifndef _CYLYNDER_
    const float vphi = sqrt(gravity_mass/r);
#else
    const float vphi = sqrt(gravity_mass/R);
#endif
    float d, e, vx, vy, vz, bx, by, bz;
    bx = by = bz = 0;
    
    vx = -vphi * y/R;
    vy = +vphi * x/R;
    vz = 0;
    
    const float dv = vphi * 1.0e-4;
    vx += drand48() * dv;
    vy += drand48() * dv;

    const float vcirc = sqrt(gravity_mass/R);
    const float cs = vcirc * HoverR;
    const float H  = cs/vcirc * R;


#if 0
    const float beta = 13480.0f;
    const float dens_mid = 1.0; //exp(-sqr(2.0)/2.0);
    const float pres = cs * dens_mid;
    const float B0   = sqrt(2.0 * pres/beta);
#else
    const float B0   = 0.05513/4/1;
#endif
#if 1
    if (R >= B_Rin && R <= B_Rout) {
      bz = B0 * sin(2*M_PI * (R - B_Rin) / dB);
//       bz = B0;
    }
#else
    const int n = R/dB;
    bz = B0 * sin(2*M_PI * R / dB);
//     bz = b0 * 2.0*(2.0*(n & 1) - 1);
#endif
    
    if (R < B_Rin || R > B_Rout) bz = 0.0f;
    

    
    d = std::max(exp(-sqr(z)/2/sqr(H)), DENS_MIN);
    if (gamma_gas > 1.0f)  e = sqr(cs) * d /(gamma_gas - 1.0)/gamma_gas;
    else                   e = sqr(cs) * d;

    bx = by = bz = 0;
    
    
    ptcl_mhd m;
    m.dens = d;
    m.ethm = e;
    m.vel  = (real3){vx, vy, vz};
    m.B    = (real3){bx, by, bz};
    m.psi = 0;

    pmhd[i] = m.to_conservative(pvec[i].wght);

//     if (i%1000 == 0)
//       fprintf(stderr, "i= %d: pos= %g %g %g   hw= %g %g  mhd=  %g %g  %g %g %g  %g %g %g \n",
// 	      i, x,y, z, pvec[i].h, pvec[i].wght,
// 	      pmhd[i].mass,
// 	      pmhd[i].etot,
// 	      pmhd[i].mom.x,
// 	      pmhd[i].mom.y,
// 	      pmhd[i].mom.z,
// 	      pmhd[i].wB.x,
// 	      pmhd[i].wB.y,
// 	      pmhd[i].wB.z);

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
void system::set_geometry(const bool init_data) {

  if (myid == 0) fprintf(stderr, " *** setting up the geometry *** \n");

  // # of dimensions in the problem
  kernel.set_dim(3);
  
  // # neighbours
  NGBmin  = 32.0f;
  NGBmean = 33.0f;
  NGBmax  = 34.0f;
  
  const int nn = 0;
  NGBmin  += nn;
  NGBmean += nn;
  NGBmax  += nn;

  // periodic box size
//   const float3 box_rmin = (float3){ 0.0f,  0.0f,  0.0f};
//   const float3 box_rmax = (float3){16.0f, 16.0f,  6.0f};
  const float3 box_rmin = (float3){ 0.0f,  0.0f,  0.0f};
  const float3 box_rmax = (float3){20.0f, 20.0f, 20.0f};
  
  pfloat<0>::set_range(box_rmin.x, box_rmax.x);
  pfloat<1>::set_range(box_rmin.y, box_rmax.y);
  pfloat<2>::set_range(box_rmin.z, box_rmax.z);
  
  global_domain.set_x(box_rmin.x, box_rmax.x);
  global_domain.set_y(box_rmin.y, box_rmax.y);
  global_domain.set_z(box_rmin.z, box_rmax.z);
  
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
  
  if (!init_data) return;

  // setup geometry on proc0
  local_n = global_n = 0;
  if (myid == 0) {
    int idx = 0;

    const int np    = 1e5;
    
    const int N_per_ann = 1000;
    const int Nann      =  np/N_per_ann;
    const double epsilon = std::pow(Rout/Rin, 1.0f/(float)Nann);

    const float3 bh_pos  = {(box_rmin.x + box_rmax.x)*0.5f, 
			    (box_rmin.y + box_rmax.y)*0.5f, 
			    (box_rmin.z + box_rmax.z)*0.5f};

    fprintf(stderr, "np= %d : Nann= %d\n", np, Nann);
    for (int j = 0; j < Nann; j++) {
      const float Rj  = Rin * std::pow(epsilon, (double)(j+0));
      const float Rjp = Rin * std::pow(epsilon, (double)(j+1));
      const float Rrange = Rjp - Rj;

      const float sig_slope = -1.0f;
      const float mass_ratio = 
	(std::pow(Rjp,  2.0f - sig_slope) - std::pow(Rj,  2.0f - sig_slope))/
	(std::pow(Rout, 2.0f - sig_slope) - std::pow(Rin, 2.0f - sig_slope));
      const int Nj = (int)(mass_ratio * np); 
      
      for (int i = 0; i < Nj; i++) {
	bool flag = true;
	float aaa, phi, r;
	const float ecc = 0.0f;
	while (flag) {

	  // SEMI-MAJOR AXIS
	  
	  aaa = Rrange * drand48() + Rj;
 	  phi = drand48() * 2*M_PI;
	  
	  // CONVERT SEMI-MAJOR AXIS TO RADIUS
	  r = aaa * ((1.0-sqr(ecc))/(1.0 + (ecc*cos(phi))));
	  
	  const float area_scale = r / (aaa*(1.0+ecc));
	  const float scale = area_scale;
	  const float fv = drand48();

	  if (fv <= scale) flag = false;
	  flag = false;
	}
	
	const float vcirc = sqrt(Mbh/r);
	const float cs    = vcirc * HoverR_I;
	const float H     = cs/vcirc * r;
	
	// 1st scale height

	float height;
	flag = true;
	while (flag) {
	  height = (2.0*drand48() - 1.0) * r;
	  const float fz = drand48();
	  const double scale = exp( - 1.0*sqr(height)/2.0/sqr(H));
	  if (fz <= scale) flag = false;
	}
	
	particle p;

	const float h  = std::pow(M_PI*(sqr(Rjp) - sqr(Rj))*2*H/Nj, 1.0/3);
	p.h = 3.0*h;
	p.vel = (float3){0.0f,0.0f,0.0f};
	p.wght = 1.0f;
	
	const float xp = bh_pos.x + r * cos(phi);
	const float yp = bh_pos.y + r * sin(phi);
	const float zp = bh_pos.z + height;
	
	p.pos.x.set(xp);
	p.pos.y.set(yp);
	p.pos.z.set(zp);
	
	p.global_idx = idx;
	p.local_idx  = idx++;
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
  
}

void system::set_parallel_domain() {
  if (myid == 0)fprintf(stderr, " *** setting up parallel domain *** \n");

  const int3 nt = {2, 2, 1};
  box.set(nproc, nt, global_domain);
}
