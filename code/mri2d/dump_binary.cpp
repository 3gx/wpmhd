#include "gn.h"

void system::dump_binary(const char *filename, bool flag) {

  if (flag) convert_to_primitives();

  FILE *fout; 
  if (!(fout = fopen(filename, "w"))) {
    std::cerr << "Cannot open file " << filename << std::endl;
    exit(-1);
  }

  int ival;
  float fval;

#define fdump(x) { fval = x; fwrite(&fval, sizeof(float), 1, fout); }
#define idump(x) { ival = x; fwrite(&ival, sizeof(int),   1, fout); }

  idump(20*4);
  idump(myid);
  idump(nproc);
  idump(nproc);
  idump(1);
  idump(1);

  idump(global_n);
  idump(local_n);
  idump(kernel.ndim);
  fdump(t_global);
  fdump(dt_global);
  idump(iteration);
  fdump(courant_no);
  fdump(gamma_gas);
  
#ifdef _PERIODIC_FLOAT_
  const int periodic_on = 1;
#else 
  const int periodic_on = 0;
#endif
  idump(periodic_on);

  const pfloat3 rmin = global_domain.rmin();
  const pfloat3 rmax = global_domain.rmax();
  fdump(rmin.x.getu());
  fdump(rmin.y.getu());
  fdump(rmin.z.getu());
  fdump(rmax.x.getu());
  fdump(rmax.y.getu());
  fdump(rmax.z.getu());
  idump(20*4);

  
  for (int i = 0; i < local_n; i++) {
    particle &p = pvec[i];
    ptcl_mhd &m = pmhd[i];
    
    float L = 0.0f;
    switch(kernel.ndim) {
    case 1:
      L = p.wght;
      break;
    case 2:
      L = sqrt(p.wght/M_PI);
      break;
    case 3:
      L = powf(p.wght*3.0f/4.0f/M_PI, 1.0f/3.0f);
      break;
    default:
      assert(kernel.ndim > 0 && kernel.ndim <= 3);
    };
    
    idump(25*4);
    fdump(p.pos.x.getu());
    fdump(p.pos.y.getu());
    fdump(p.pos.z.getu());
    fdump(p.vel.x);
    fdump(p.vel.y);
    fdump(p.vel.z);
    fdump(m.dens);
    fdump(m.ethm);
    fdump(compute_pressure(m.dens, m.ethm));
    fdump(    (sqr(m.B.x  ) + sqr(m.B.y  ) + sqr(m.B.z  ))*0.5f);
    fdump(sqrt(sqr(m.vel.x) + sqr(m.vel.y) + sqr(m.vel.z))); 
    fdump(m.vel.x);
    fdump(m.vel.y);
    fdump(m.vel.z);
    fdump(m.B.x);
    fdump(m.B.y);
    fdump(m.B.z);
    fdump(p.h);
    fdump(p.wght);
    fdump(m.psi);
    fdump(L*divB_i[i]);
    fdump(-1.0f);
    fdump(-1.0f);
    fdump(-1.0f);
    fdump(-1.0f);
    idump(25*4);
    
  }

  fclose(fout);
  
  
  if (flag) restore_conservative();

}
