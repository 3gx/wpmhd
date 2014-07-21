#ifndef _NFLOAT_H_
#define _NFLOAT_H_

#include <cassert>
#include <iostream>
#include <cmath>

template <int ch> 
struct nfloat{
  float uval;
  
  static void set_range(const double _xmin, const double _xmax){}
  
  nfloat& set(double f){
    uval = f; 
    return *this;
  }
  nfloat& setp(const nfloat rhs) {
    uval = rhs.uval; 
    return *this;
  }

  nfloat& operator=(const nfloat rhs) {
    uval = rhs.uval;
    return *this;
  };
  
  double getu() const{return uval;}
  
  nfloat(){ uval = 0.0f;}
    
  double operator-(const nfloat rhs) const{return uval - rhs.uval;}
  
  nfloat half() {
    uval *= 0.5f;
    return *this;
  };
  nfloat twice() {
    uval *= 2.0f;
    return *this;
  };
  nfloat& add(const nfloat rhs) {
    uval += rhs.uval;
    return *this;
  }
  nfloat& sub(const nfloat rhs) {
    uval -= rhs.uval;
    return *this;
  }
  nfloat& padd(const nfloat rhs) {
    uval += rhs.uval;
    return *this;
  }
  nfloat& psub(const nfloat rhs) {
    uval -= rhs.uval;
    return *this;
  }
  bool aleq(const nfloat rhs) const {
//     fprintf(stderr, "  uval= %g\n", uval);
//     fprintf(stderr, "  rhs.uval= %g\n", rhs.uval);
//     bool b = abs(uval) <= rhs.uval;
//     fprintf(stderr, "  b= %d\n", b);
    return std::abs(uval) <= rhs.uval;
  }
  
  nfloat& min(const nfloat rhs) {
    uval = std::min(uval, rhs.uval);
    return *this;
  }
  nfloat& max(const nfloat rhs) {
    uval = std::max(uval, rhs.uval);
    return *this;
  }
  nfloat& div(const unsigned int rhs) {
    uval *= 1.0f/rhs;
    return *this;
  }
  
  friend void nfloat_merge(const nfloat x1, const nfloat h1, 
			   const nfloat x2, const nfloat h2,
			   nfloat &xc, nfloat &hs) {
    const float ux1 = x1.uval;
    const float uh1 = h1.uval;
    const float ux2 = x2.uval;
    const float uh2 = h2.uval;


    const float min1 = ux1 - uh1;
    const float max1 = ux1 + uh1;
    const float min2 = ux2 - uh2;
    const float max2 = ux2 + uh2;
    
    const float minc = std::min(min1, min2);
    const float maxc = std::max(max1, max2);

    const float uxc = (maxc + minc)*0.5f;
    const float uhs = (maxc - minc)*0.5f;

    xc.uval = uxc;
    hs.uval = uhs;

//     if (uhs > 0.2)
//       fprintf(stderr, "uxc= %g  uhs= %g\n", uxc, uhs);
    
  }
  
};

struct nfloat3 {
  nfloat<0> x;
  nfloat<1> y;
  nfloat<2> z;
};


#endif // _NFLOAT_H_
    
