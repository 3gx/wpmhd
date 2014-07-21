#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

struct particle {
  float x, y,z;
  float vx, vy, vz;
  float h, wght;
  int global_idx;
};

struct ptcl_mhd {
  float dens, ethm, pres;
  float velx, vely, velz;
  float Bx, By, Bz;
};


class analysis {
public:
  std::vector<particle> pvec;
  std::vector<ptcl_mhd> pmhd;
  float gamma_gas, t_global;

  analysis() {};
  ~analysis() {};

  void read_snap(const char *filename) {
    FILE *fin; 
    if (!(fin = fopen(filename, "r"))) {
      std::cerr << "Cannot open file " << filename << std::endl;
      exit(-1);
    }
    
    std::cerr  << " read snapshot: " << filename << std::endl;
    
    int ival;
    float fval;
    
#define fload(x) { fread(&fval, sizeof(float), 1, fin); x = fval; }
#define iload(x) { fread(&ival, sizeof(int),   1, fin); x = ival;}
    
    float ftmp;
    int itmp, np0, npx, npy, npz;
    iload(itmp); // 20*4
    iload(itmp); // myid
    iload(np0);
    iload(npx);
    iload(npy);
    iload(npz);
    
    int nglob, nloc, ndim;
    iload(nglob);
    iload(nloc);
    iload(ndim);  

    float dt_global;
    fload(t_global);
    fload(dt_global);
    fprintf(stderr, "t_global= %g\n", t_global);

    int iteration;
    iload(iteration);
    fprintf(stderr, "iteration= %d\n", iteration);
    
    float courant_no;
    fload(courant_no);
    fload(gamma_gas);
  
    int periodic_on;
    iload(periodic_on);

    float rmin_x, rmin_y, rmin_z;
    float rmax_x, rmax_y, rmax_z;
    fload(rmin_x);
    fload(rmin_y);
    fload(rmin_z);
    fload(rmax_x);
    fload(rmax_y);
    fload(rmax_z);

    iload(itmp);     // 20*4
    assert(itmp == 80);

    pvec.clear();
    pmhd.clear();
    pvec.reserve(1024);
    pmhd.reserve(1024);
    
    fprintf(stderr, "np =%d   nglob= %d \n", np0, nloc);
    for (int p = 0; p < np0; p++) {
      fprintf(stderr, " p= %d out of %d; nloc= %d\n", p, np0, nloc);
      for (int i = 0; i < nloc; i++) {
	particle p;
	ptcl_mhd m;
	iload(ival);    // 25*4
	assert(ival == 100);
	fload(p.x);
	fload(p.y); 
	fload(p.z);
	fload(p.vx);
	fload(p.vy);
	fload(p.vz);
	fload(m.dens);
	fload(m.ethm);
	fload(m.pres);
	fload(ftmp); //dump(    (sqr(m.B.x  ) + sqr(m.B.y  ) + sqr(m.B.z  ))*0.5f);
	fload(ftmp); //dump(sqrt(sqr(m.vel.x) + sqr(m.vel.y) + sqr(m.vel.z))); 
	fload(m.velx);
	fload(m.vely);
	fload(m.velz);
	fload(m.Bx);
	fload(m.By);
	fload(m.Bz);
	fload(p.h);
	fload(p.wght);
	fload(ftmp); //fload(m.psi);
	fload(ftmp); //L*divB_i[i]);
	fload(ftmp); // -1
	fload(ftmp); //-1.0f);
	fload(ftmp); //-1.0f);
	fload(ftmp); //-1.0f);
	iload(ival); //25*4);
	assert(ival == 100);
	
	
	pvec.push_back(p);
	pmhd.push_back(m);
      }

      if (!(p < np0-1)) break;
      iload(itmp); // 20*4
      iload(itmp); // myid
      iload(np0);
      iload(npx);
      iload(npy);
      iload(npz);
      
      int nglob1;
      iload(nglob1);
      if (nglob != nglob1) {
	fprintf(stderr, "np; npx, npy, npz = %d; %d %d %d \n", 
		np0, npx, npy, npz);
	fprintf(stderr, "nglob= %d  nglob1= %d\n", nglob, nglob1);
      }
      assert(nglob == nglob1);
      iload(nloc);
      iload(ndim);
      fload(t_global);
      fload(dt_global);
      iload(iteration);
      fload(courant_no);
      fload(gamma_gas);
  
      iload(periodic_on);

      fload(rmin_x);
      fload(rmin_y);
      fload(rmin_z);
      fload(rmax_x);
      fload(rmax_y);
      fload(rmax_z);
      iload(itmp);     // 20*4
    }

    fprintf(stderr, "read= %d particles\n", (int)pvec.size());
  }

  float get_mass() {
    float mass = 0;
    const float Dblob = 10.0f;
    const float Tamb  = sqrt(gamma_gas);
    const int ni = pvec.size();
    fprintf(stderr, "ni= %d\n", ni);
    for (int i = 0; i < ni; i++) {
      const particle &p = pvec[i];
      const ptcl_mhd &m = pmhd[i];
      
      const float Tel = sqrt(gamma_gas*m.pres/m.dens);
      const float Del = m.dens;
      
      if (Del > 0.64*Dblob && Tel < Tamb)  {
	mass += m.dens * p.wght;
      }
//       mass += m.dens * p.wght;
//       if (i%1000 == 0) {
// 	fprintf(stderr, "dens= %g [%g] Temp= %g [%g]  %g\n",
// 		Del, Dblob, Tel, Tamb, m.pres);
//       } 
    }
    
    return mass;
  }
  
};


int main(int argc, char *argv[]) {
  
  if (argc < 2) fprintf(stderr, " filename needed ...\n");
  assert(argc > 1);

  analysis s;
  
  s.read_snap(argv[1]);

  const float Mcl = s.get_mass();

  fprintf(stdout,  " t_global= %g  Mcl= %g\n", s.t_global, Mcl);

  std::cerr << "end-of-code\n";
}
