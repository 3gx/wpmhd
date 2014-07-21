

#ifndef _PFLOAT_H_
#define _PFLOAT_H_

#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdlib>

#if 0

#include "nfloat.h"

#define pfloat nfloat
#define pfloat3 nfloat3
#define pfloat_merge nfloat_merge

#else


#define LARGEU 0xffffffffu

#define _PERIODIC_FLOAT_

template <int ch> 
struct pfloat{
  unsigned int uval;
  static double xmin,  xmax;
  static double xscale_i2f,  xscale_f2i;
  
  static void set_range(const double _xmin, const double _xmax){
    xmin = _xmin;
    xmax = _xmax;
    xscale_i2f = (xmax - xmin) / double(LARGEU);
    xscale_f2i = 1.0 / xscale_i2f;
  }
  pfloat& set(double f){
    const float forig = f;
    int niter = 100;
    while (f < xmin && niter-- > 0) f += (xmax - xmin);
    if (!(niter > 2)) {
      fprintf(stderr, "forig= %lg  (%g; %g)\n", forig, xmin, xmax);
      assert(niter > 2);
    }
    niter = 100;
    while (f > xmax && niter-- > 0) f -= (xmax - xmin);
    if (!(niter > 2)) {
      fprintf(stderr, "forig= %lg  (%g; %g)\n", forig, xmin, xmax);
      assert(niter > 2);
    }
    const double tmp = (f - xmin) * xscale_f2i;
    uval = static_cast<unsigned int>(tmp) & 0xfffffffeu;
    return *this;
  }
  pfloat& setp(const pfloat rhs) {uval = rhs.uval; return *this;}
  pfloat& setu(const unsigned int u) {uval = u; return *this;}

  pfloat& operator=(const pfloat rhs) {uval = rhs.uval; return *this;};
//   pfloat& operator=(const unsigned int u){uval = u; return *this;};
  
  double getu() const{return xmin + xscale_i2f * static_cast<double>(     uval);}
  double gets() const{return xmin + xscale_i2f * static_cast<double>((int)uval);}
  
  pfloat(){ uval = 0;}
//   pfloat(const unsigned int u) {set(u);}
    
  double operator-(const pfloat rhs) const{
    const unsigned int s = (uval - rhs.uval);
    return xscale_i2f * static_cast<double>((int)s);
  }
  
  pfloat half() {
    const unsigned int lhs = uval >> 1; 
    uval = (lhs <= uval) ? lhs : 0;
    return *this;
  };
  pfloat twice() {
    const unsigned int lhs = uval << 1;
    uval = (lhs >= uval) ? lhs : 0xffffffffu;
    return *this;
  };
  pfloat& add(const pfloat rhs) {
    const unsigned int lhs = uval + rhs.uval;
    uval = (lhs >= uval) ? lhs : 0xffffffffu;
    return *this;
  }
  pfloat& sub(const pfloat rhs) {
    const unsigned int lhs = uval - rhs.uval;
    uval = (lhs <= uval) ? lhs : 0;
    return *this;
  }
  pfloat& padd(const pfloat rhs) {
    uval += rhs.uval;
    return *this;
  }
  pfloat& psub(const pfloat rhs) {
    uval -= rhs.uval;
    return *this;
  }
  bool aleq(const pfloat rhs) const {
    const int           s = uval;
    const unsigned int  p = abs(s); // & -2; ///? may not work here
    return p <= rhs.uval;
  }
  bool ale(const pfloat rhs) const {
    const int           s = uval;
    const unsigned int  p = abs(s); // & -2; ///? may not work here
    return p < rhs.uval;
  }
  bool leq(const pfloat rhs) const {
    return uval <= rhs.uval;
  }
  
  pfloat& min(const pfloat rhs) {
    uval = std::min(uval, rhs.uval);
    return *this;
  }
  pfloat& max(const pfloat rhs) {
    uval = std::max(uval, rhs.uval);
    return *this;
  }
  pfloat& div(const unsigned int rhs) {
    const unsigned int lhs = uval/rhs;
    uval = (lhs <= uval) ? lhs : 0;
    return *this;
  }

  pfloat& add(const double x) {
    double xh = x / 2.0;
    while (xh < xmin) xh += (xmax - xmin);
    while (xh > xmax) xh -= (xmax - xmin);
    const double tmp = (xh - xmin) * xscale_f2i;
    const unsigned int itmp = static_cast<unsigned int>(floor(tmp));
    uval += (itmp << 1);
    return *this;
  }

  friend void pfloat_merge(const pfloat x1, const pfloat h1, 
			   const pfloat x2, const pfloat h2,
			   pfloat &xc, pfloat &hs) {
    unsigned long long_shift = 0x100000000lu;
//     long_shift += 0xu;
//     long_shift = 0;

    const unsigned long ux1 = x1.uval + long_shift;
    const unsigned long uh1 = h1.uval;
    const unsigned long ux2 = x2.uval + long_shift;
    const unsigned long uh2 = h2.uval;

//     fprintf(stderr, "long_shift= 0x%.16lx\n", long_shift);

    const unsigned long min1 = ux1 - uh1;
    const unsigned long max1 = ux1 + uh1;
    const unsigned long min2 = ux2 - uh2;
    const unsigned long max2 = ux2 + uh2;

//     fprintf(stderr, "min1= 0x%.16lx\n", min1);
//     fprintf(stderr, "max1= 0x%.16lx\n", max1);
//     fprintf(stderr, "min2= 0x%.16lx\n", min2);
//     fprintf(stderr, "max2= 0x%.16lx\n", max2);
    
    unsigned long minc = std::min(min1, min2);
    unsigned long maxc = std::max(max1, max2);

//     fprintf(stderr, "minc= 0x%.16lx\n", minc);
//     fprintf(stderr, "maxc= 0x%.16lx\n", maxc);

    
    minc = minc >> 1;
    maxc = maxc >> 1;

    const unsigned long uxc = maxc + minc - long_shift;
    const unsigned long uhs = maxc - minc;
      
      
    xc.uval = uxc;
    hs.uval = uhs;
    
//     xc.uval = x2.uval;
//     hs.uval = h2.uval;
      
    
  }
  
};

template <int ch> double pfloat<ch>::xmin       = 0.0;
template <int ch> double pfloat<ch>::xmax       = 1.0;
template <int ch> double pfloat<ch>::xscale_i2f = 1.0 / double(LARGEU);
template <int ch> double pfloat<ch>::xscale_f2i = 1.0 * double(LARGEU);

struct pfloat3 {
  pfloat<0> x;
  pfloat<1> y;
  pfloat<2> z;
};

#endif

#endif // _PERIODIC_H_
    
