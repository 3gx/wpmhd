#include "pfloat.h"
using namespace std;

int main(int argc, char *argv[]) {
#if 1
  pfloat<0> x, y, z;
  
  x.uval = 0xffffff00u;
  y.uval = 0x0000000fu;
  z = x; z.add(y);
  fprintf(stderr, "x= 0x%.8x  y= 0x%.8x x+y= 0x%.8x \n", x.uval, y.uval, z.uval);
//   fprintf(stderr, "x <= y = %d  x + y <= y = %d \n", x.aleq(uval, y.uval, z.uval);

  x.uval = 0xffffff00u;
  y.uval = 0x0000010fu;
  z = x; z.add(y);
  fprintf(stderr, "x= 0x%.8x  y= 0x%.8x x+y= 0x%.8x\n", x.uval, y.uval, z.uval);


  x.uval = 0xffffff00u;
  y.uval = 0x0000010fu;
  z = x; z.sub(y);
  fprintf(stderr, "x= 0x%.8x  y= 0x%.8x x-y= 0x%.8x\n", x.uval, y.uval, z.uval);
  z = y; z.sub(x);
  fprintf(stderr, "x= 0x%.8x  y= 0x%.8x y-x= 0x%.8x\n", x.uval, y.uval, z.uval);

  
  x.uval = 0xfffffff0u;
  y = x;
  fprintf(stderr, "x= 0x%.8x  x/2= 0x%.8x \n", x.uval, y.half().uval);

  x.uval = 0xffffffffu;
  y = x;
  fprintf(stderr, "x= 0x%.8x  x*2= 0x%.8x \n", x.uval, y.twice().uval);

  x.uval = 0x0000000fu;
  y = x;
  fprintf(stderr, "x= 0x%.8x  x/2= 0x%.8x \n", x.uval, y.div(36).uval);
  fprintf(stderr, " comparison \n");
  
  x.uval = 0x000000f0u;
  y.uval = 0xffffffffu;
  fprintf(stderr, "x= 0x%.8x  y= 0x%.8x\n", x.uval, y.uval);
  fprintf(stderr, "x<=y = %d\n", x.aleq(y));
  fprintf(stderr, "y<=x = %d\n", y.aleq(x));
  fprintf(stderr, " -x<=y--\n");
  int s = x.uval;
  cerr << "s=" << s << endl;

  fprintf(stderr, " **** overlap **** \n");
  

  pfloat<0> xc1, hs1, xc2, hs2;
  xc1.set(0.7);
  hs1.set(0.1);

  xc2.set(0.9);
  hs2.set(0.2);

  fprintf(stderr, "x1= %g  h1= %g \n", xc1.getu(), hs1.getu());
  fprintf(stderr, "x2= %g  h2= %g \n", xc2.getu(), hs2.getu());
  
  pfloat<0> xc3, hs3;
  pfloat_merge(xc1, hs1, xc2, hs2, xc3, hs3);

  fprintf(stderr, "x3= %g  h3= %g \n", xc3.getu(), hs3.getu());

  fprintf(stderr, " ---------------- \n");

  xc1.set(0.2);
  hs1.set(0.12);
  
  xc2.set(0.1);
  hs2.set(0.2);
  
  fprintf(stderr, "x1= %g  h1= %g \n", xc1.getu(), hs1.getu());
  fprintf(stderr, "x2= %g  h2= %g \n", xc2.getu(), hs2.getu());
  
  pfloat_merge(xc1, hs1, xc2, hs2, xc3, hs3);

  fprintf(stderr, "x3= %g  h3= %g \n", xc3.getu(), hs3.getu());

  
#else
  unsigned int  ux= 0xffffff00u;
  unsigned int  uy= 0x0000000fu;

  unsigned long lux= ux; //0xffffff00u;
  unsigned long luy= uy; //0x0000000fu;


  ux += uy;
  fprintf(stderr, "ux = 0x%.16x  ux = 0x%.16x\n", ux, uy);
  lux += luy;
  fprintf(stderr, "lux= 0x%.16lx  lux= 0x%.16lx\n", lux, luy);
  
#endif

  return 0;
}
