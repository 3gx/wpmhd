#ifndef _PRIMITIVES_H_
#define _PRIMITIVES_H_

#ifndef NMAXPROC
#define NMAXPROC 256
#endif

#ifndef NMAXIMPORT
#define NMAXIMPORT 256
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include "pfloat.h"

#ifndef HUGE
#define HUGE 1e+10
#endif

#ifndef TINY
#define TINY 1e-10
#endif

// #define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
typedef double real;
#else
typedef  float real;
#endif


struct bool3 {
  bool x, y, z;
};

struct float3 {
  float x, y, z;
};

#if 0
struct float4 {
  real x, y, z, w;
  float3 xyz() {return (float3){x,y,z};}
};
#else
struct float4{
  typedef float  v4sf __attribute__ ((vector_size(16)));
  typedef double v2df __attribute__ ((vector_size(16)));
  static v4sf v4sf_abs(v4sf x){
    typedef int v4si __attribute__ ((vector_size(16)));
    v4si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    return __builtin_ia32_andps(x, (v4sf)mask);
  }
  union{
    v4sf v;
    struct{
      float x, y, z, w;
    };
  };
  float4() : v((v4sf){0.f, 0.f, 0.f, 0.f}) {}
  float4(float x, float y, float z, float w) : v((v4sf){x, y, z, w}) {}
  float4(v4sf _v) : v(_v) {}
  float4 abs(){
    typedef int v4si __attribute__ ((vector_size(16)));
    v4si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    return float4(__builtin_ia32_andps(v, (v4sf)mask));
  }
  void dump(){
    std::cerr << x << " "
	      << y << " "
	      << z << " "
	      << w << std::endl;
  }
  float3 xyz() {return (float3){x,y,z};}
  v4sf operator=(const float4 &rhs){
    v = rhs.v;
    return v;
  }
  float4(const float4 &rhs){
    v = rhs.v;
  }
};
#endif

struct real4 {
  real x, y, z, w;
};

struct real3 {
  real x, y, z;
};

struct int3 {
  int x, y, z;
};


/////////////

template<typename T>
bool safe_resize(std::vector<T> &vec, size_t size) {
  T* ptr_old = &vec[0];
  vec.resize(size);
  return (ptr_old == &vec[0]);
}

/////////////////

inline float  sqr(float  x) {return x*x;}
inline float  cub(float  x) {return sqr(x)*x;}
inline float sign(float x) {return (x < 0.0f) ? -1.0f : 1.0f;};

inline double sqr(double x) {return x*x;}
inline double cub(double x) {return sqr(x)*x;}
inline double sign(double x) {return (x < 0.0) ? -1.0 : 1.0;};

/////////////////////

struct particle {  // 8 words
  pfloat3 pos;
  float3  vel;
  float   h, wght;
  int     global_idx, local_idx;
};

//////////////////

struct ptcl_mhd {  // 10 64bit words, 8 mhd + 2 scalar 
  union {
    real dens;
    real mass;
  };
  union {
    real ethm;
    real etot;
  };
  union {
    real3 vel;
    real3 mom;
  };
  union {
    real3  B;
    real3 wB;
  };
  union {
    real  psi;
    real mpsi;
  };
  union {
    real  scal;
    real mscal;
  };
  int marker;

  ptcl_mhd to_primitive() const {
    ptcl_mhd prim;
    const ptcl_mhd &cons = *this;
    prim.dens  = cons.mass;
    prim.vel.x = cons.mom.x/cons.mass;
    prim.vel.y = cons.mom.y/cons.mass;
    prim.vel.z = cons.mom.z/cons.mass;
    prim.B.x   = cons.wB.x;
    prim.B.y   = cons.wB.y;
    prim.B.z   = cons.wB.z;
    
    prim.scal  = cons.mscal /cons.mass;
    prim.marker = cons.marker;

    prim.psi   = cons.mpsi;//  /cons.mass;
    prim.psi   = cons.mpsi /cons.mass;

#ifdef _CONSERVATIVE_
    prim.ethm  = cons.etot
      - 0.5*(sqr(prim.vel.x) + sqr(prim.vel.y) + sqr(prim.vel.z))*prim.dens
      - 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#elif defined _SEMICONSERVATIVE_
    prim.ethm  = cons.etot
      - 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#else
    prim.ethm = cons.etot;
#endif

    return prim;
  }

  ptcl_mhd to_conservative() const  {
    ptcl_mhd cons;
    const ptcl_mhd &prim = *this;
    cons.mass  = prim.dens;
    cons.mom.x = prim.vel.x * cons.mass;
    cons.mom.y = prim.vel.y * cons.mass;
    cons.mom.z = prim.vel.z * cons.mass;
    cons.wB.x  = prim.B.x;
    cons.wB.y  = prim.B.y;
    cons.wB.z  = prim.B.z;
    cons.mscal  = prim.scal  * cons.mass;
    cons.marker = prim.marker;
    cons.mpsi  = prim.psi; //   * cons.mass;
    cons.mpsi  = prim.psi * cons.mass;
#ifdef _CONSERVATIVE_
    cons.etot = prim.ethm 
      + 0.5*(sqr(prim.vel.x) + sqr(prim.vel.y) + sqr(prim.vel.z))*prim.dens
      + 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#elif defined _SEMICONSERVATIVE_
    cons.etot = prim.ethm 
      + 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#else
    cons.etot = prim.ethm;
#endif

    return cons;
  }

  

  void set(const real x) {
    dens = ethm = psi = scal = x;
    vel = B = (real3){x,x,x};
  }
  ptcl_mhd& operator-=(const ptcl_mhd &x) {
    dens  -= x.dens;
    ethm  -= x.ethm;
    vel.x -= x.vel.x;
    vel.y -= x.vel.y;
    vel.z -= x.vel.z;
    B.x   -= x.B.x;
    B.y   -= x.B.y;
    B.z   -= x.B.z;
    psi   -= x.psi;
    scal  -= x.scal;
    return *this;
  }
  ptcl_mhd operator-(const ptcl_mhd &y) const {
    ptcl_mhd x = *this;
    x -= y;
    return x;
  }
  ptcl_mhd& operator+=(const ptcl_mhd &x) {
    dens  += x.dens;
    ethm  += x.ethm;
    vel.x += x.vel.x;
    vel.y += x.vel.y;
    vel.z += x.vel.z;
    B.x   += x.B.x;
    B.y   += x.B.y;
    B.z   += x.B.z;
    psi   += x.psi;
    scal  += x.scal;
    return *this;
  }
  ptcl_mhd operator+(const ptcl_mhd &y) const {
    ptcl_mhd x = *this;
    x += y;
    return x;
  }
  ptcl_mhd& operator*=(const ptcl_mhd &x) {
    dens  *= x.dens;
    ethm  *= x.ethm;
    vel.x *= x.vel.x;
    vel.y *= x.vel.y;
    vel.z *= x.vel.z;
    B.x   *= x.B.x;
    B.y   *= x.B.y;
    B.z   *= x.B.z;
    psi   *= x.psi;
    scal  *= x.scal;
    return *this;
  }
  ptcl_mhd operator*(const ptcl_mhd &y) const {
    ptcl_mhd x = *this;
    x *= y;
    return x;
  }
  
  ptcl_mhd operator*(const real y) const {
    ptcl_mhd x = *this;
    x.dens  *= y;
    x.ethm  *= y;
    x.vel.x *= y;
    x.vel.y *= y;
    x.vel.z *= y;
    x.B.x   *= y;
    x.B.y   *= y;
    x.B.z   *= y;
    x.psi   *= y;
    x.scal  *= y;
    return x;
  }
  friend ptcl_mhd operator*(const real x,
				   const ptcl_mhd &y) {
    return y*x;
  }
#if 0
  friend ptcl_mhd min(const ptcl_mhd &x,
			     const ptcl_mhd &y) {
    ptcl_mhd z = x;
    z.dens  = std::min(x.dens,  y.dens);
    z.ethm  = std::min(x.ethm,  y.ethm);
    z.vel.x = std::min(x.vel.x, y.vel.x);
    z.vel.y = std::min(x.vel.y, y.vel.y);
    z.vel.z = std::min(x.vel.z, y.vel.z);
    z.B.x   = std::min(x.B.x,   y.B.x);
    z.B.y   = std::min(x.B.y,   y.B.y);
    z.B.z   = std::min(x.B.z,   y.B.z);
    z.psi   = std::min(x.psi,   y.psi);
    z.scal  = std::min(x.scal,  y.scal);
    return z;
  }
  friend ptcl_mhd max(const ptcl_mhd &x,
			     const ptcl_mhd &y) {
    ptcl_mhd z = x;
    z.dens  = std::max(x.dens,  y.dens);
    z.ethm  = std::max(x.ethm,  y.ethm);
    z.vel.x = std::max(x.vel.x, y.vel.x);
    z.vel.y = std::max(x.vel.y, y.vel.y);
    z.vel.z = std::max(x.vel.z, y.vel.z);
    z.B.x   = std::max(x.B.x,   y.B.x);
    z.B.y   = std::max(x.B.y,   y.B.y);
    z.B.z   = std::max(x.B.z,   y.B.z);
    z.psi   = std::max(x.psi,   y.psi);
    z.scal  = std::max(x.scal,  y.scal);
    return z;
  }

  friend ptcl_mhd abs(const ptcl_mhd &x) {
    ptcl_mhd z = x;
    z.dens  = std::abs(x.dens);
    z.ethm  = std::abs(x.ethm);
    z.vel.x = std::abs(x.vel.x);
    z.vel.y = std::abs(x.vel.y);
    z.vel.z = std::abs(x.vel.z);
    z.B.x   = std::abs(x.B.x);
    z.B.y   = std::abs(x.B.y);
    z.B.z   = std::abs(x.B.z);
    z.psi   = std::abs(x.psi);
    z.scal  = std::abs(x.scal);
    return z;
  }
#else
  ptcl_mhd& min(const ptcl_mhd &x) {
    dens  = std::min(x.dens,  dens);
    ethm  = std::min(x.ethm,  ethm);
    vel.x = std::min(x.vel.x, vel.x);
    vel.y = std::min(x.vel.y, vel.y);
    vel.z = std::min(x.vel.z, vel.z);
    B.x   = std::min(x.B.x,   B.x);
    B.y   = std::min(x.B.y,   B.y);
    B.z   = std::min(x.B.z,   B.z);
    psi   = std::min(x.psi,   psi);
    scal  = std::min(x.scal,  scal);
    return *this;
  }
  ptcl_mhd& max(const ptcl_mhd &x) {
    dens  = std::max(x.dens,  dens);
    ethm  = std::max(x.ethm,  ethm);
    vel.x = std::max(x.vel.x, vel.x);
    vel.y = std::max(x.vel.y, vel.y);
    vel.z = std::max(x.vel.z, vel.z);
    B.x   = std::max(x.B.x,   B.x);
    B.y   = std::max(x.B.y,   B.y);
    B.z   = std::max(x.B.z,   B.z);
    psi   = std::max(x.psi,   psi);
    scal  = std::max(x.scal,  scal);
    return *this;
  }

  ptcl_mhd& abs() {
    dens  = std::abs(dens);
    ethm  = std::abs(ethm);
    vel.x = std::abs(vel.x);
    vel.y = std::abs(vel.y);
    vel.z = std::abs(vel.z);
    B.x   = std::abs(B.x);
    B.y   = std::abs(B.y);
    B.z   = std::abs(B.z);
    psi   = std::abs(psi);
    scal  = std::abs(scal);
    return *this;
  }
#endif
			     
};

#endif // _PRIMITIVES_H_
