#ifndef _PREAL_H_
#define _PREAL_H_

template<class T>
struct preal {
  T value;
  
  preal() {value = 0;}
  preal(const T v) {
    value  = v;
    while (value >=  0.5) value -= 1.0;
    while (value <  -0.5) value += 1.0;
  };
  preal(const T v, const T xmin, const T xmax) {
    value = preal((v - xmin)/(xmax - xmin)).value;
  }
  ~preal() {};
  

  preal operator-(const preal v) const {
    return preal(value - v.value);
  }
  
  bool operator<(const preal v) const {
    return (*this - v).value < 0.0;
  }

  bool operator>(const preal v) const {
    return (*this - v).value > 0.0;
  }
  


};

#endif // _PREAL_H_

#ifdef _TOOLBOX_
#include <cstdio>
#include <iostream>

int main(int argc, char *argv[]) {
  preal<float> a(atof(argv[1]));
  preal<float> b(atof(argv[2]));
  preal<float> c = a - b;

  fprintf(stderr, " %g < %g = %d\n", a.value, b.value, a< b);
  
  return 0;
}

#endif
