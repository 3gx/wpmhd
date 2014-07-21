#include "gn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_qrng.h>

#define RCLOUD 0.5f
#define TCLOUD 1000.0f
#define DENSDRIFT 1.0e-3f

float system::compute_pressure(const float dens, const float ethm) {
  return (gamma_gas > 1.0f) ? ethm*(gamma_gas - 1.0f) : ethm;
}


float4 system::body_forces(const pfloat3 pos) {
  const float3 fpos = {pos.x.getu(), pos.y.getu(), pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2   = sqr(dr.x)  + sqr(dr.y) + sqr(dr.z) + gravity_eps2;
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

  
  bool flag = false;

  if (0) {
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
  }


  flag = flag | (ds2 <= sqr(gravity_rin));
  flag = flag | (ds2 >= sqr(gravity_rout));

#if 1
  const ptcl_mhd prim = pmhd[idx].to_primitive(pi.wght);
  const float pB   = 0.5f*(sqr(prim.B.x) + sqr(prim.B.y) + sqr(prim.B.z));
  const float pgas = compute_pressure(prim.dens, prim.ethm);
  const float magn = pB/pgas;
  if (prim.dens > 1.0e-2)
    flag = flag | (magn > 100.0f);
#endif
  
  float4 acc  = body_forces(pi.pos);
  const float vesc = sqrt(fabs(acc.w));
  const float vp = sqrt(sqr(pi.vel.x) + sqr(pi.vel.y) + sqr(pi.vel.z));
  if (prim.dens > 1.0e-2)
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
  return;
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

#if 1

  const float vunit = 65.58f; // km/s
  const float tunit = 1.491f; // Myr
  const float Tunit = 60.14;
  
  float cs2 = TCLOUD/Tunit/sqr(vunit);
  if (mi.dens < DENSDRIFT) cs2 *= 100;

  if (gamma_gas > 1.0f)  mi.ethm = cs2 * mi.dens /(gamma_gas - 1.0)/gamma_gas;
    else                 mi.ethm = cs2 * mi.dens;

#endif

  pmhd[idx] = mi.to_conservative(pi.wght);
  return;

  
}

void system::set_geometry(const bool init_data) {

  if (myid == 0) fprintf(stderr, " *** setting up the geometry *** \n");

  // # of dimensions in the problem
  kernel.set_dim(3);
  
  // # neighbours
  NGBmin  = 32.9f;
  NGBmean = 33.0f;
  NGBmax  = 33.1f;

  const int nn = 20;
  NGBmin  += nn;
  NGBmean += nn;
  NGBmax  += nn;

  // setup periodic box of size 2*RCLOUD = DCLOUD

  const float Dcl  = 2.0f*RCLOUD;
  const float Rbox = 10.0f;

  const float3 box_rmin = (float3){0.0f, 0.0f, 0.0f};
  const float3 box_rmax = (float3){Rbox, Rbox, Rbox};
  
  pfloat<0>::set_range(box_rmin.x, box_rmax.x);
  pfloat<1>::set_range(box_rmin.y, box_rmax.y);
  pfloat<2>::set_range(box_rmin.z, box_rmax.z);
  
  global_domain.set_x(box_rmin.x, box_rmax.x);
  global_domain.set_y(box_rmin.y, box_rmax.y);
  global_domain.set_z(box_rmin.z, box_rmax.z);

  const float3 box_centre = {global_domain.centre.x.getu(),
			     global_domain.centre.y.getu(),
			     global_domain.centre.z.getu()};
  const float3  box_L = global_domain.size();
  
  // clean up geometry array
  pvec.clear();
  pvec.reserve(128);

  if (!init_data) return;
  if (!init_data) {
    const float box1_size = 100.0f;
    
    const float3 box1_rmin = (float3){0.0f, 0.0f, 0.0f};
    const float3 box1_rmax = (float3){box1_size, box1_size, box1_size};
    
    pfloat<0>::set_range(box1_rmin.x, box1_rmax.x);
    pfloat<1>::set_range(box1_rmin.y, box1_rmax.y);
    pfloat<2>::set_range(box1_rmin.z, box1_rmax.z);
    
    global_domain.set_x(box1_rmin.x, box1_rmax.x);
    global_domain.set_y(box1_rmin.y, box1_rmax.y);
    global_domain.set_z(box1_rmin.z, box1_rmax.z);
    
    return;
  }

  // setup geometry on proc0
  local_n = global_n = 0;
  if (myid == 0) {
    int idx = 0;
    
    /* ********** NUMBER OF NESTED LEVELS *********** */

    const int nlev = 5; 

    int   np    = 1e5;

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
  
  box.set(nproc, box.nt, global_domain);
  distribute_particles();
  
  const int niter = 30;
  const float eps = 0.05f;
  adjust_positions(niter, eps);

  pmhd.resize(local_n);
  pmhd_dot.resize(local_n);
  divB_i.resize(local_n); 
  gpu.dwdt.cmalloc(local_n);
 
}

void system::set_parallel_domain() {
  if (myid == 0)fprintf(stderr, " *** setting up parallel domain *** \n");

  const int3 nt = {2, 2, 2};
  box.set(nproc, nt, global_domain);
}

void system::set_problem(const bool init_data) {
  fprintf(stderr, " ******** cloud infall ****** \n");
  
  const pfloat3 pcentre = global_domain.centre;
  const  float3  centre = {pcentre.x.getu(), pcentre.y.getu(), pcentre.z.getu()};
  
  const float Mbh = 1.0f;

  gravity_mass = Mbh;
  gravity_eps2 = 0.0f;
  gravity_rin  = 0.05;
  gravity_rout = 1e6;
  gravity_pos  = centre;

  do_kepler_drift = true;

  gamma_gas    = 5.0/3; //1.0f;
  gamma_gas = 1.1f;
  
  if (!init_data) return;
  
  fprintf(stderr, " ****** setup ****** \n");

  const float vunit = 65.58f; // km/s
  const float tunit = 1.491f; // Myr
  const float Tunit = 60.14;
  
  const float rc = 3.0f;
  const float phi = M_PI*5.0/4.0f;

  const float cosph = cos(phi);
  const float sinph = sin(phi);

  const float xcl = rc * cosph;
  const float ycl = rc * sinph;
  
  const float vr   =  -40.0f / vunit;
  const float vt   =  20.0f / vunit;
  const float vorb = sqrt(vr*vr + vt*vt);
  const float tinfall = rc/vorb;

  const float dcl = 1.0e-00;
  const float cs2 = TCLOUD/Tunit/sqr(vunit);
  
  fprintf(stderr, "Ro= %g pc,  tinfall= %g Myr [%g]\n",
	  rc, tinfall * tunit, tinfall);

  const float vx_cl = vr*cosph - vt*sinph;
  const float vy_cl = vr*sinph + vt*cosph;

  for (int i = 0; i < local_n; i++) {
    pfloat3 &pos = pvec[i].pos;

    const float x = pos.x.getu() - centre.x;
    const float y = pos.y.getu() - centre.y;
    const float z = pos.z.getu() - centre.z;
    const float r = std::sqrt(x*x + y*y + z*z);
    
    pos.x.add(xcl);
    pos.y.add(ycl);

    float d, e, vx, vy, vz, bx, by, bz;
    bx = by = bz = 0;
    
    vx = vx_cl;
    vy = vy_cl;
    vz = 0.0;
    
    //       const float f = 1.1;
    //       const float p1 = 10.0f;
    //       const float p2 = 10.0f;
    d = dcl; // * std::pow(1 - std::pow(r/(f*RCLOUD), p1), p2);
    
    if (gamma_gas > 1.0f)  e = cs2 * d /(gamma_gas - 1.0)/gamma_gas;
    else                   e = cs2 * d;
    
#if 1
    const float inv_beta = 0.1f;
    
    const float d0 = dcl * 0.5;
    const float e0 = cs2 * d0;
    
    const float beta = 1.0f/inv_beta;
    const float B0 = sqrt(2.0f*e0/beta);
    
    //     constB.x = 0*B0;
    //     constB.y = 0*B0; //B0;
    //     constB.z = B0;
    
    bx = 0;
    by = 0;
    bz = B0;
#endif

    if (r > RCLOUD) {
      d = 1.0e-3;
      if (gamma_gas > 1.0f)  e = cs2 * d /(gamma_gas - 1.0)/gamma_gas;
      else                   e = cs2 * d;
      vx = vy = vz = 0;
    }
  
    ptcl_mhd m;
    m.dens = d;
    m.ethm = e;
    m.vel  = (real3){vx, vy, vz};
    m.B    = (real3){bx, by, bz};
    m.psi = 0;
    
    pmhd[i] = m.to_conservative(pvec[i].wght);
    pvec[i].vel = (float3){vx, vy, vz};
    
  }
  
  for (int i = 0; i < local_n; i++) {
    divB_i[i] = 0;
    gpu.dwdt[i] = 0.0f;
    pmhd_dot[i].set(0.0);
  }
}
