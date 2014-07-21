#include "gn.h"

void system::dump_snapshot(const char *filename) {

  convert_to_primitives();

  FILE *fout = fopen(filename, "w");

  fprintf(fout, "# x(1) y(2) z(3)  | ax(4) ay(5) az(6) | dens(7) pres(8) pB(9) v(10) | ");
  fprintf(fout, " vx(11) vy(12) vz(13) | bx(14) vy(15) bz(16) | h(17) w(18) | psi(19) divB(20)\n");
  for (int i = 0; i < local_n; i++) {
    particle &p = pvec[i];
    ptcl_mhd &m = pmhd[i];

    float L = 0;
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

    fprintf(fout, "%g %g %g  %g %g %g  %g %g %g %g  %g %g %g  %g %g %g  %g %g  %g %g\n",
	    p.pos.x.getu(), p.pos.y.getu(), p.pos.z.getu(),       // 3
	    p.vel.x, p.vel.y, p.vel.z,                            // 6
	    m.dens, compute_pressure(m.dens, m.ethm),             // 8
	    (sqr(m.B.x  ) + sqr(m.B.y  ) + sqr(m.B.z  ))*0.5f,    // 9
	    sqrt(sqr(m.vel.x) + sqr(m.vel.y) + sqr(m.vel.z)),     // 10
	    m.vel.x, m.vel.y, m.vel.z,                            // 13
	    m.B.x,   m.B.y,   m.B.z,                              // 16
	    p.h, p.wght,                                          // 18
	    m.psi, L*divB_i[i]);                                  // 20
  }
  fclose(fout);


  restore_conservative();

}
