#ifndef _PERIODIC_H_
#define _PERIODIC_H_

#if 1
#define pfloat float
#define pfloat4 float4
#define pfloat3 float3
#endif


#if 1
#include "pfloat.h"
#else
#include "pfloat_shift.h"
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
