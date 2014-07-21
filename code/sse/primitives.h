#ifndef _PRIMITIVES_H_
#define _PRIMITIVES_H_

#include "v4sf.h"

#ifndef NMAXPROC
#define NMAXPROC 256
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
#if 1
  real dens;     // mass
  real ethm;     // etot
  real3 vel;     // mom
  real3 B;       // wB
  real  psi;     // mpsi
  real  scal;    // mscal
#else
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
#endif

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

/////////

struct v4sf3 {
  v4sf x, y, z;
};

struct ptcl_mhd4 {  // 10 64bit words, 8 mhd + 2 scalar 
  v4sf dens;     // mass
  v4sf ethm;     // etot
  v4sf3 vel;     // mom
  v4sf3 B;       // wB
  v4sf  psi;     // mpsi
  v4sf  scal;    // mscal

  ptcl_mhd4(const ptcl_mhd v1,
	    const ptcl_mhd v2,	
	    const ptcl_mhd v3,
	    const ptcl_mhd v4) {
    set(v1, v2, v3, v4);
  }
  
  ptcl_mhd4& set(const ptcl_mhd v1,
		 const ptcl_mhd v2,	
		 const ptcl_mhd v3,
		 const ptcl_mhd v4) {
    dens  = v4sf(v1.dens,  v2.dens,  v3.dens,   v4.dens);
    ethm  = v4sf(v1.ethm,  v2.ethm,  v3.ethm,   v4.ethm);
    vel.x = v4sf(v1.vel.x, v2.vel.x, v3.vel.x,  v4.vel.x);
    vel.y = v4sf(v1.vel.y, v2.vel.y, v3.vel.y,  v4.vel.y);
    vel.z = v4sf(v1.vel.z, v2.vel.z, v3.vel.z,  v4.vel.z);
    B.x = v4sf(v1.B.x, v2.B.x, v3.B.x,  v4.B.x);
    B.y = v4sf(v1.B.y, v2.B.y, v3.B.y,  v4.B.y);
    B.z = v4sf(v1.B.z, v2.B.z, v3.B.z,  v4.B.z);
    psi = v4sf(v1.psi, v2.psi, v3.psi, v4.psi);
    scal = v4sf(v1.scal, v2.scal, v3.scal, v4.scal);
    return *this;
  }

  ptcl_mhd4() {};
  ~ptcl_mhd4() {};
  
  ptcl_mhd reduce() {
    ptcl_mhd v;
    v.dens  = dens [0] + dens [1] + dens [2] + dens [3];
    v.ethm  = ethm [0] + ethm [1] + ethm [2] + ethm [3];
    v.vel.x = vel.x[0] + vel.x[1] + vel.x[2] + vel.x[3];
    v.vel.y = vel.y[0] + vel.y[1] + vel.y[2] + vel.y[3];
    v.vel.z = vel.z[0] + vel.z[1] + vel.z[2] + vel.z[3];
    v.B.x   = B.x  [0] + B.x  [1] + B.x  [2] + B.x  [3];
    v.B.y   = B.y  [0] + B.y  [1] + B.y  [2] + B.y  [3];
    v.B.z   = B.z  [0] + B.z  [1] + B.z  [2] + B.z  [3];
    v.psi   = psi  [0] + psi  [1] + psi  [2] + psi  [3];
    v.scal  = scal [0] + scal [1] + scal [2] + scal [3];
    return v;
  }

  void set(const float x) {
    dens = ethm = psi = scal = v4sf(x);
    vel = B = (v4sf3){dens,dens,dens};
  }
  ptcl_mhd4& operator-=(const ptcl_mhd4 &x) {
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
  ptcl_mhd4 operator-(const ptcl_mhd4 &y) const {
    ptcl_mhd4 x = *this;
    x -= y;
    return x;
  }


 
  ptcl_mhd4& operator-=(const ptcl_mhd &x) {
    dens  -= v4sf(x.dens);
    ethm  -= v4sf(x.ethm);
    vel.x -= v4sf(x.vel.x);
    vel.y -= v4sf(x.vel.y);
    vel.z -= v4sf(x.vel.z);
    B.x   -= v4sf(x.B.x);
    B.y   -= v4sf(x.B.y);
    B.z   -= v4sf(x.B.z);
    psi   -= v4sf(x.psi);
    scal  -= v4sf(x.scal);
    return *this;
  }
  ptcl_mhd4 operator-(const ptcl_mhd &y) const {
    ptcl_mhd4 x = *this;
    x -= y;
    return x;
  }
  
  ptcl_mhd4& operator+=(const ptcl_mhd4 &x) {
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
  ptcl_mhd4 operator+(const ptcl_mhd4 &y) const {
    ptcl_mhd4 x = *this;
    x += y;
    return x;
  }
  ptcl_mhd4& operator*=(const ptcl_mhd4 &x) {
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
  ptcl_mhd4 operator*(const ptcl_mhd4 &y) const {
    ptcl_mhd4 x = *this;
    x *= y;
    return x;
  }
  
  ptcl_mhd4 operator*(const v4sf y) const {
    ptcl_mhd4 x = *this;
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
  friend ptcl_mhd4 operator*(const v4sf x,
			     const ptcl_mhd4 &y) {
    return y*x;
  }

  ptcl_mhd4& min(const ptcl_mhd4 &x) {
    dens  = v4sf::min(x.dens,  dens);
    ethm  = v4sf::min(x.ethm,  ethm);
    vel.x = v4sf::min(x.vel.x, vel.x);
    vel.y = v4sf::min(x.vel.y, vel.y);
    vel.z = v4sf::min(x.vel.z, vel.z);
    B.x   = v4sf::min(x.B.x,   B.x);
    B.y   = v4sf::min(x.B.y,   B.y);
    B.z   = v4sf::min(x.B.z,   B.z);
    psi   = v4sf::min(x.psi,   psi);
    scal  = v4sf::min(x.scal,  scal);
    return *this;
  }
  ptcl_mhd4& max(const ptcl_mhd4 &x) {
    dens  = v4sf::max(x.dens,  dens);
    ethm  = v4sf::max(x.ethm,  ethm);
    vel.x = v4sf::max(x.vel.x, vel.x);
    vel.y = v4sf::max(x.vel.y, vel.y);
    vel.z = v4sf::max(x.vel.z, vel.z);
    B.x   = v4sf::max(x.B.x,   B.x);
    B.y   = v4sf::max(x.B.y,   B.y);
    B.z   = v4sf::max(x.B.z,   B.z);
    psi   = v4sf::max(x.psi,   psi);
    scal  = v4sf::max(x.scal,  scal);
    return *this;
  }

  ptcl_mhd4& abs() {
    dens  = v4sf::abs(dens);
    ethm  = v4sf::abs(ethm);
    vel.x = v4sf::abs(vel.x);
    vel.y = v4sf::abs(vel.y);
    vel.z = v4sf::abs(vel.z);
    B.x   = v4sf::abs(B.x);
    B.y   = v4sf::abs(B.y);
    B.z   = v4sf::abs(B.z);
    psi   = v4sf::abs(psi);
    scal  = v4sf::abs(scal);
    return *this;
  }

  //////////

  ptcl_mhd4& min(const ptcl_mhd &x) {
    dens  = v4sf::min(v4sf(x.dens),  dens);
    ethm  = v4sf::min(v4sf(x.ethm),  ethm);
    vel.x = v4sf::min(v4sf(x.vel.x), vel.x);
    vel.y = v4sf::min(v4sf(x.vel.y), vel.y);
    vel.z = v4sf::min(v4sf(x.vel.z), vel.z);
    B.x   = v4sf::min(v4sf(x.B.x),   B.x);
    B.y   = v4sf::min(v4sf(x.B.y),   B.y);
    B.z   = v4sf::min(v4sf(x.B.z),   B.z);
    psi   = v4sf::min(v4sf(x.psi),   psi);
    scal  = v4sf::min(v4sf(x.scal),  scal);
    return *this;
  }
  ptcl_mhd4& max(const ptcl_mhd &x) {
    dens  = v4sf::max(v4sf(x.dens),  dens);
    ethm  = v4sf::max(v4sf(x.ethm),  ethm);
    vel.x = v4sf::max(v4sf(x.vel.x), vel.x);
    vel.y = v4sf::max(v4sf(x.vel.y), vel.y);
    vel.z = v4sf::max(v4sf(x.vel.z), vel.z);
    B.x   = v4sf::max(v4sf(x.B.x),   B.x);
    B.y   = v4sf::max(v4sf(x.B.y),   B.y);
    B.z   = v4sf::max(v4sf(x.B.z),   B.z);
    psi   = v4sf::max(v4sf(x.psi),   psi);
    scal  = v4sf::max(v4sf(x.scal),  scal);
    return *this;
  }


//   static ptcl_mhd min(ptcl_mhd x, ptcl_mhd4 y) {
//     y.dens  = v4sf::min(y.dens,  v4sf(x.dens));
//     y.ethm  = v4sf::min(y.ethm,  v4sf(x.ethm));
//     y.vel.x = v4sf::min(y.vel.x, v4sf(x.vel.x));
//     y.vel.y = v4sf::min(y.vel.y, v4sf(x.vel.y));
//     y.vel.z = v4sf::min(y.vel.z, v4sf(x.vel.z));
//     y.B.x   = v4sf::min(y.B.x,   v4sf(x.B.x));
//     y.B.y   = v4sf::min(y.B.y,   v4sf(x.B.y));
//     y.B.z   = v4sf::min(y.B.z,   v4sf(x.B.z));
//     y.psi   = v4sf::min(y.psi,   v4sf(x.psi));
//     y.scal  = v4sf::min(y.scal,  v4sf(x.scal));

//     x.dens  = std::min(std::min(y.dens [0], y.dens [1]), std::min(y.dens [2], y.dens [3]));
//     x.ethm  = std::min(std::min(y.ethm [0], y.ethm [1]), std::min(y.ethm [2], y.ethm [3]));
//     x.vel.x = std::min(std::min(y.vel.x[0], y.vel.x[1]), std::min(y.vel.x[2], y.vel.x[3]));
//     x.vel.y = std::min(std::min(y.vel.y[0], y.vel.y[1]), std::min(y.vel.y[2], y.vel.y[3]));
//     x.vel.z = std::min(std::min(y.vel.z[0], y.vel.z[1]), std::min(y.vel.z[2], y.vel.z[3]));
//     x.B.x = std::min(std::min(y.B.x[0], y.B.x[1]), std::min(y.B.x[2], y.B.x[3]));
//     x.B.y = std::min(std::min(y.B.y[0], y.B.y[1]), std::min(y.B.y[2], y.B.y[3]));
//     x.B.z = std::min(std::min(y.B.z[0], y.B.z[1]), std::min(y.B.z[2], y.B.z[3]));
//     x.scal  = std::min(std::min(y.scal [0], y.scal [1]), std::min(y.scal [2], y.scal [3]));
//     x.psi  = std::min(std::min(y.psi [0], y.psi [1]), std::min(y.psi [2], y.psi [3]));
//     return x;
//   }

//   static ptcl_mhd max(ptcl_mhd x, ptcl_mhd4 y) {
//     y.dens  = v4sf::max(y.dens,  v4sf(x.dens));
//     y.ethm  = v4sf::max(y.ethm,  v4sf(x.ethm));
//     y.vel.x = v4sf::max(y.vel.x, v4sf(x.vel.x));
//     y.vel.y = v4sf::max(y.vel.y, v4sf(x.vel.y));
//     y.vel.z = v4sf::max(y.vel.z, v4sf(x.vel.z));
//     y.B.x   = v4sf::max(y.B.x,   v4sf(x.B.x));
//     y.B.y   = v4sf::max(y.B.y,   v4sf(x.B.y));
//     y.B.z   = v4sf::max(y.B.z,   v4sf(x.B.z));
//     y.psi   = v4sf::max(y.psi,   v4sf(x.psi));
//     y.scal  = v4sf::max(y.scal,  v4sf(x.scal));

//     x.dens  = std::max(std::max(y.dens [0], y.dens [1]), std::max(y.dens [2], y.dens [3]));
//     x.ethm  = std::max(std::max(y.ethm [0], y.ethm [1]), std::max(y.ethm [2], y.ethm [3]));
//     x.vel.x = std::max(std::max(y.vel.x[0], y.vel.x[1]), std::max(y.vel.x[2], y.vel.x[3]));
//     x.vel.y = std::max(std::max(y.vel.y[0], y.vel.y[1]), std::max(y.vel.y[2], y.vel.y[3]));
//     x.vel.z = std::max(std::max(y.vel.z[0], y.vel.z[1]), std::max(y.vel.z[2], y.vel.z[3]));
//     x.B.x = std::max(std::max(y.B.x[0], y.B.x[1]), std::max(y.B.x[2], y.B.x[3]));
//     x.B.y = std::max(std::max(y.B.y[0], y.B.y[1]), std::max(y.B.y[2], y.B.y[3]));
//     x.B.z = std::max(std::max(y.B.z[0], y.B.z[1]), std::max(y.B.z[2], y.B.z[3]));
//     x.scal  = std::max(std::max(y.scal [0], y.scal [1]), std::max(y.scal [2], y.scal [3]));
//     x.psi  = std::max(std::max(y.psi [0], y.psi [1]), std::max(y.psi [2], y.psi [3]));
//     return x;
//   }
			     
};

#endif // _PRIMITIVES_H_
