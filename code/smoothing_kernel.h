#ifndef _SMOOTHING_KERNEL_H_
#define _SMOOTHING_KERNEL_H_

inline float pow3(float x) {return x*sqr(x);}
inline float pow4(float x) {return sqr(x)*sqr(x);}
inline float pow5(float x) {return x*sqr(x)*sqr(x);}



#if 1  // cubic spline

struct smoothing_kernel {
  int   ndim ;
  float KERNEL_COEFF_1;
  float KERNEL_COEFF_2;
  float KERNEL_COEFF_3;
  float KERNEL_COEFF_4;
  float KERNEL_COEFF_5;
  float KERNEL_COEFF_6;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
      KERNEL_COEFF_1 = (5.0/7*2.546479089470);
      KERNEL_COEFF_2 = (5.0/7*15.278874536822);
      KERNEL_COEFF_3 = (5.0/7*45.836623610466);
      KERNEL_COEFF_4 = (5.0/7*30.557749073644);
      KERNEL_COEFF_5 = (5.0/7*5.092958178941);
      KERNEL_COEFF_6 = (5.0/7*(-15.278874536822));
      NORM_COEFF     =  M_PI;
      break;
    default:
      KERNEL_COEFF_1 = 2.546479089470;
      KERNEL_COEFF_2 = 15.278874536822;
      KERNEL_COEFF_3 = 45.836623610466;
      KERNEL_COEFF_4 = 30.557749073644;
      KERNEL_COEFF_5 = 5.092958178941;
      KERNEL_COEFF_6 = (-15.278874536822);
      NORM_COEFF     = 4.0/3*M_PI;
    }
  }
  
  float w(float u) {
    if      (u < 0.5) return (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u - 1) * u * u);
    else if (u < 1.0) return  KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u) * (1.0 - u);
    else              return 0.0;
  }
  float w4(float u) {return w(u);}
  
  float dw(float u) {
    if      (u < 0.5) return (0*KERNEL_COEFF_1 + KERNEL_COEFF_2 * (3 * u*u - 2 * u));
    else if (u < 1.0) return -3*KERNEL_COEFF_5 * (1.0 - u) * (1.0 - u);
    else              return 0.0;
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }

};

#elif 0       // quintic spline


struct smoothing_kernel {
  int   ndim ;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
      NORM_COEFF     =  M_PI/23.8362;
      break;
    default:
      NORM_COEFF     =  4.0*M_PI/3/13.9626;
    }
  }
  
  float w(float u) {
    float wk = 0;
    u *= 3;
    if (u < 1) {
      wk = pow5(3 - u) - 6*pow5(2-u) + 15*pow5(1-u);
    } else if (u < 2) {
      wk = pow5(3 - u) - 6*pow5(2-u);
    } else if (u < 3) {
      wk = pow5(3 - u);
    }
    return wk;
  }
  float w4(float u) {return w(u);}

  float dw(float u) {
    return -1;
    const float sigma = 3.0f;
    return -2*u*sqr(sigma)*w(u);
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }

};

#elif 0  // gaussian

struct smoothing_kernel {
  int   ndim ;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
      NORM_COEFF     =  M_PI/0.349023;
      break;
    default:
      NORM_COEFF     =  4.0*M_PI/3/0.206144;
    }
  }
  
  float w(float u) {
    const float sigma = 3.0f;
    if (u < 1.0) return exp(-sqr(sigma*u));
    else         return 0.0;
  }
  
  float dw(float u) {
    const float sigma = 3.0f;
    return -2*u*sqr(sigma)*w(u);
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }

};

#elif 1      // quadratic function

struct smoothing_kernel {
  int   ndim ;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
#if 1
      NORM_COEFF     =  M_PI/0.785398;
#else
      NORM_COEFF     =  M_PI/1.5708;
#endif
      break;
    default:
      NORM_COEFF     =  4.0*M_PI/3/0.628319;
    }
  }

#if 1
  float w(float u) {
    u *= 2;
    if (u < 2.0) return 0.375*u*u - 1.5*u + 1.5;
    else         return 0.0;
  }
#else
  float w(float u) {
    if (u < 1) return (1 - u*u);
    else       return 0.0;
  }
#endif
  float w4(float u){return w(u);}
  
  float dw(float u) {
    u *= 2;
    if (u < 2.0) return 0.75*u - 1.5;
    else         return 0.0;
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }

};

#elif 1  // super-quadratic

struct smoothing_kernel {
  int   ndim ;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
      NORM_COEFF     =  M_PI/0.785398;
//       NORM_COEFF     =  M_PI/0.628319;
      break;
    default:
      NORM_COEFF     =  4.0*M_PI/3/0.628319;
    }
  }

  float w(float u) {
    return w4(u);
    if (u < 1.0) return (1.5*u*u - 3.0*u + 1.5)*(1 - u*u);
    else         return 0.0;
  }

  float w4(float u) {
    if (u < 1.0) return 1.5*u*u - 3.0*u + 1.5;
    else         return 0.0;
  }
  
  float dw(float u) {
    u *= 2;
    if (u < 2.0) return 0.75*u - 1.5;
    else         return 0.0;
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }
};

#elif 1 ///////////////////////////////

struct smoothing_kernel {
  int   ndim ;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
      NORM_COEFF     =  M_PI/0.628319;
      break;
    default:
      NORM_COEFF     =  4.0*M_PI/3/0.448799;
    }
  }

  float w4(float u) {
    if (u < 1.0) return (1.5*u*u - 3.0*u + 1.5)*(1 - u*u);
    else         return 0.0;
  }

  float w(float u) {
    if (u < 1.0) return (1.5*u*u - 3.0*u + 1.5);
    else         return 0.0;
  }
  
  float dw(float u) {
    if (u < 1.0) return 3.0*u - 3;
    else         return 0.0;
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }
};

#elif 0

struct smoothing_kernel {
  int   ndim ;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
      NORM_COEFF     =  M_PI/0.628319;
      break;
    default:
      NORM_COEFF     =  4.0*M_PI/3/0.478719;
    }
  }

  float w(float u) {
    if (u < 1.0) return 1.0 - 6.0*u*u + 8*u*u*u - 3.0*u*u*u*u;
    else         return 0.0;
  }
  
  float dw(float u) {
    if (u < 1.0) return -12*u + 24*u*u - 12*u*u*u;
    else         return 0.0;
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }

};

#elif 1

struct smoothing_kernel {
  int   ndim ;
  float NORM_COEFF;
  
  smoothing_kernel() {};
  smoothing_kernel(int ndim) {set_dim(ndim);}
  ~smoothing_kernel() {};

  void set_dim(int NDIM) {
    ndim = NDIM;
    assert(NDIM >= 1 && NDIM <= 3);
    switch(NDIM) {
    case 1:
      break;
    case 2:
      NORM_COEFF     =  M_PI/0.516119;
      break;
    default:
      NORM_COEFF     =  4.0*M_PI/3/0.478719;
    }
  }

  float w(float u) {
    if (u < 1.0) return (1.0 - 6.0*u*u + 8*u*u*u - 3.0*u*u*u*u)*(1 - u*u);
    else         return 0.0;
  }
  
  float dw(float u) {
    if (u < 1.0) return -12*u + 24*u*u - 12*u*u*u;
    else         return 0.0;
  }

  float pow(float x) {
    switch(ndim) {
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    default: assert(ndim >= 1 && ndim <= 3); return -1;
    }
  }

};


#else
#error "choose something pls in smoothing_kernel.h"
#endif
#endif // _KERNEL_H_
