#ifndef _PFLOAT_SHIFT_H_
#define _PFLOAT_SHIFT_H_

#include <cassert>
#include <iostream>
#include <cmath>

#ifndef pfloat

#define _PERIODIC_FLOAT_
#define _PERIODIC_WITH_SHIFT_

template <int ch> 
struct pfloat{
  unsigned int uval;
  static float xmin,  xmax;
  static float xscale_i2f,  xscale_f2i;
  static float xpmin,  xpmax;
  static float xpscale_i2f, xpscale_f2i;
  
  static void set_range(const float _xpmin, const float _xpmax){
    const float L = _xpmax - _xpmin;
    xpmin = _xpmin;
    xpmax = _xpmax;
    xpscale_i2f = L   / float(0xffffffffu);
    xpscale_f2i = 1.f / xpscale_i2f;

    xmin = _xpmin - 0.5f*L;
    xmax = _xpmax + 0.5f*L;
    xscale_i2f = (xmax - xmin) / float(0xffffffffu);
    xscale_f2i = 1.f / xscale_i2f;
  }
  
  void set(float f){
    while (f < xmin) f += (xmax - xmin);
    while (f > xmax) f -= (xmax - xmin);
    if (!(xmin <= f && f<=xmax)) {
      fprintf(stderr, "ch= %d:  xmin= %g <  f= %g  < xmax= %g\n",
	      ch, xmin, f, xmax);
      assert(xmin <= f && f<= xmax);
    }
    const float tmp = (f - xmin) * xscale_f2i;
    uval = static_cast<unsigned int>(tmp);
  }
  float getu() const{return xmin + xscale_i2f * static_cast<float>(     uval);}
  float gets() const{return xmin + xscale_i2f * static_cast<float>((int)uval);}
  operator float() const{return getu();}

  unsigned int p() const {
    float f = getu();
    while (f < xpmin) f += (xpmax - xpmin);
    while (f > xpmax) f -= (xpmax - xpmin);
    const float tmp = (f - xpmin) * xpscale_f2i;
    return static_cast<unsigned int>(tmp);
  }
  
  pfloat(){ uval = 0;}
  pfloat(const float  f){set(f);}
  pfloat(const double f){set(f);}
  pfloat(const unsigned int u) {uval = u;}
    
  float operator-(const pfloat rhs) const{
    const unsigned int s = (p() - rhs.p());
    return xpscale_i2f * static_cast<float>((int)s);
    //     const unsigned int s = (uval - rhs.uval);
    //     const float ff = xscale_i2f * static_cast<float>((int)s);
    //     return xscale_i2f * static_cast<float>((int)s);
  }
  
//   pfloat operator-(const float rhs) const{
//     return pfloat(uval - pfloat(rhs).uval);
//   }
//   pfloat operator+(const float rhs) const{
//     return pfloat(uval + pfloat(rhs).uval);
//   }
//   bool operator<(const pfloat rhs) const{
//     const int s = (uval - rhs.uval);
//     return (s < 0);
//   }
//   bool operator>(const pfloat rhs) const{
//     const int s = (uval - rhs.uval);
//     return (s > 0);
//   }

//   bool operator<=(const pfloat rhs) const{
//     const int s = (uval - rhs.uval);
//     return (s <= 0);
//   }
//   bool operator>=(const pfloat rhs) const{
//     const int s = (uval - rhs.uval);
//     return (s >= 0);
//   }
  

//   pfloat operator+=(const pfloat rhs) {
//     uval += rhs.uval;
//     return *this;
//   }

//   pfloat operator-=(const pfloat rhs) {
//     uval -= rhs.uval;
//     return *this;
//   }

//   friend pfloat max(const pfloat lhs, const pfloat rhs) {
//     return (lhs > rhs) ? lhs : rhs;
//   }
//   friend pfloat min(const pfloat lhs, const pfloat rhs) {
//     return (lhs < rhs) ? lhs : rhs;
//   }

};

template <int ch> float pfloat<ch>::xmin   = -0.5f;
template <int ch> float pfloat<ch>::xmax   = +1.5f;
template <int ch> float pfloat<ch>::xpmin  = 0.f;
template <int ch> float pfloat<ch>::xpmax  = 1.f;
template <int ch> float pfloat<ch>::xscale_i2f = 2.f / float(0xffffffffu);
template <int ch> float pfloat<ch>::xscale_f2i = 0.5 * float(0xffffffffu);
template <int ch> float pfloat<ch>::xpscale_i2f = 1.f / float(0xffffffffu);
template <int ch> float pfloat<ch>::xpscale_f2i = float(0xffffffffu);



struct pfloat3 {
  pfloat<0> x;
  pfloat<1> y;
  pfloat<2> z;
};

struct pfloat4 {
  pfloat<0> x;
  pfloat<1> y;
  pfloat<2> z;
  float     w;
  pfloat3 xyz() {return (pfloat3){x,y,z};}
};

#endif


#endif // _PERIODIC_H_

#ifdef _TOOLBOX_PERIODIC_
using namespace std;

int main(int argc, char *argv[]) {
  pfloat<99> x, y;

  pfloat<99>::set_range(0, 1);


  x = pfloat<99>(atof(argv[1]));
  y = pfloat<99>(atof(argv[2]));


//   cout << x.uval << endl;
//   cout << y.uval << endl;

//   cout << pfloat<99>::xscale_f2i << endl;
//   cout << pfloat<99>::xscale_i2f << endl;



  cout << (float)x << endl;
  cout << (float)y << endl;
  cout << y - x << endl;

  
  return 0;
}
#endif
