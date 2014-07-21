#ifndef __MACOSX_
#define __LINUX__
#endif

#ifdef __MACOSX__
#include <Accelerate/Accelerate.h>
#include <xmmintrin.h>
inline void fpe_catch() {
  _mm_setcsr( _MM_MASK_MASK &~
              (_MM_MASK_OVERFLOW|_MM_MASK_INVALID|_MM_MASK_DIV_ZERO) );
}
#elif defined __LINUX__
#include <fenv.h>
void fpe_catch(void) {
  /* Enable some exceptions. At startup all exceptions are masked. */
  feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
}
#else
void fpe_catch(void) {}
#endif


#include "gn.h"

int gwait = 0;

int main(int argc, char *argv[]) {
//   fpe_catch();
  MPI_Init(&argc, &argv);

  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);


  if (myid == 0) {
//     gwait = 1;
    while (gwait == 1) {
      std::cerr << " waiting ... zzzz \n";
      sleep(2);
    };
  }

  int3 np = {1,1,1};
  switch(nproc) {
  case 1: break;
  case 2: np.x = 2; break;
  case 4: np.x = 2; np.y = 2; break;
  case 8: np.x = 4; np.y = 2; break;
  case 16: np.x = 4; np.y = 4; break;
  case 32: np.x = 8; np.y = 4; break;
  case 64: np.x = 8; np.y = 8; break;
  case 128: np.x = 16; np.y = 8; break;
  case 256: np.x = 16; np.y = 16; break;
  default: 
    std::cerr << "please choose correct nproc ... \n";
    exit(-1);
  }

  bool do_ppm = false;
//   do_ppm = true;

  class system s(np, nproc, myid, do_ppm);

  s.setup_particles();
//   s.dump_snapshot("data/init0.dat");
  fprintf(stderr, "proc= %d: initialising ...\n", myid);
  double t0 = get_time();
  s.init_conservative();
  system("uname -n");
  fprintf(stderr, "proc= %d: done initialising ... in %g sec\n", myid, get_time() - t0);
  

  s.eulerian_mode   = false;
//   s.eulerian_mode   = true;
  s.do_morton_order = false;
  
//   s.dump_snapshot("data/init0.dat");
//   exit(-1);

  s.courant_no = 0.8;

  s.iterate();
  
//   s.dump_snapshot("data/init1.dat");
  
  double Volume0, Mtot0, Etot0, Ekin0, Emag0, Ethm0;
  s.compute_conserved(Mtot0, Etot0, Ethm0, Ekin0, Emag0, Volume0);

  float  t_out = 0.0f;
  float  t_bin = 0.0f;
  float dt_out = 0.1f;
  float dt_bin = 0.1f;
  int    i_out = 0;
  int    i_bin = 0;

//   t_out = 1e10f;
  
//  dt_out = 0.01f;
//   dt_bin = 0.01f;
  

  float t_end = 1.1;
  
  t_end = 10;
//   dt_out = 1.0f;


  const int   n_end = 1000000;
  fprintf(stderr, " *** t_end= %g  n_end= %d \n", t_end, n_end);
  for (int i = 0; i < n_end; i++) {
    double t0  = get_time();
    s.iterate();
    double dt_done = get_time() - t0;
    if (s.t_global >= t_end) exit(-1);
//     if (i%1 == 0) {
    if (s.t_global >= t_out) {
      char fn[256];
      sprintf(fn, "%s/iter_p%.3d_%.5d.ascii", "data", s.myid, i_out++);
      fprintf(stderr, "dumping output to %s @ time= %g \n", fn, s.t_global);
      s.dump_snapshot(fn);
      t_out += dt_out;
    }
    if (s.t_global >= t_bin) {
      char fn[256];
      sprintf(fn, "%s/iter_p%.3d_%.5d.bin", "data", s.myid, i_bin++);
      fprintf(stderr, "dumping output to %s @ time= %g \n", fn, s.t_global);
      s.dump_binary(fn);
      t_bin += dt_bin;
    }


    double Volume, Mtot, Etot, Ekin, Emag, Ethm;
    s.compute_conserved(Mtot, Etot, Ethm, Ekin, Emag, Volume);
    
    if (s.myid == 0) {
//       if (s.iteration%2 == 0) {
      fprintf(stderr, " ----------------------------------------------------------- \n");
	fprintf(stderr, "proc= %d: iter= %d: t= %g dt = %g cons= (%g %g %g) diff= (%g %g %g)  done in %g sec.\n",
		s.myid, s.iteration, s.t_global, s.dt_global,
		Mtot, Etot, Volume,
		(Mtot - Mtot0)/Mtot0, (Etot - Etot0)/Etot0, (Volume - Volume0)/Volume0, dt_done);
//       } else {
	fprintf(stderr, "proc= %d: ratio= %g;  t_comm= %g: %g %g  %g %g %g  %g %g \n",
		myid, s.t_communicate/s.t_compute, s.t_communicate,
		s.t_distribute, s.t_import_bnd, s.t_pmhd, s.t_pvec, s.t_pvel, s.t_Bmatrix, s.t_grad_comm);
	fprintf(stderr, "                            t_comp= %g: %g %g  %g %g  %g\n",
		s.t_compute, s.t_tree, s.t_weight, s.t_renorm, s.t_grad, s.t_interaction);
//       }

#if 0      
      fprintf(stderr, " Ethm= %g (%g; %g) Ekin= %g (%g; %g) Emagn= %g (%g; %g )\n",
	      Ethm, Ethm0, (Ethm - Ethm0)/(Ethm0 + TINY),
	      Ekin, Ekin0, (Ekin - Ekin0)/(Ekin0 + TINY),
	      Emag, Emag0, (Emag - Emag0)/(Emag0 + TINY)); 
#endif
    }
	    

  }

  MPI_Finalize();
  fprintf(stderr, "end-of-program\n");

#if 0
  

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = get_time();

//    fprintf(stderr, "setup_particles\n");
  s.setup_particles();
  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = get_time();



  fprintf(stderr, "build_tree\n");
  s.build_tree();
  MPI_Barrier(MPI_COMM_WORLD);
  double t2 = get_time();


  fprintf(stderr, "import_boundary particles\n");
    s.import_boundary_particles();
  MPI_Barrier(MPI_COMM_WORLD);
  double t3 = get_time();

   int ncnt = 1;
//    for (int i = 0; i < ncnt; i++)
//      s.ngb_search();
   MPI_Barrier(MPI_COMM_WORLD);
   double t4 = get_time();

  for (int i = 0; i < ncnt; i++)
    s.compute_weights();
  for (int i = 0; i < ncnt; i++)
    s.compute_weights();

  s.import_boundary_pmhd();
  s.import_boundary_pmhd();
  MPI_Barrier(MPI_COMM_WORLD);
  double t5 = get_time();

  MPI_Barrier(MPI_COMM_WORLD);
  double t6 = get_time();

  fprintf(stderr, "proc= %d: vain\n", myid);
  s.vain();
  
  double t7 = get_time();
  
#if 0
  
  fprintf(stderr, "build_tree\n");
  s.build_tree();

  fprintf(stderr, "import_boundary particles\n");
  s.import_boundary_particles();

  fprintf(stderr, "proc= %d: compute_weights\n", myid);
  s.compute_weights();

  fprintf(stderr, "proc= %d: import_boundary_pmhd\n", myid);
  s.import_boundary_pmhd();

  fprintf(stderr, "proc= %d: renorm\n", myid);
  s.renorm();

  MPI_Barrier(MPI_COMM_WORLD);
  fprintf(stderr, "proc= %d: gradient\n", myid);
  s.gradient();

  MPI_Barrier(MPI_COMM_WORLD);
  fprintf(stderr, "proc= %d: mhd_interaction \n", myid);

  s.mhd_interaction();
#else
  fprintf(stderr, "proc= %d: derivs\n", myid);
  s.derivs();
#endif

  
  
  MPI_Barrier(MPI_COMM_WORLD);
  double t8 = get_time();
  fprintf(stderr, "proc= %d: dump\n", myid);

  s.dump_snapshot("snap.dat");
  MPI_Finalize();
  //   fprintf(stderr, "end-of-program\n");
  
  fprintf(stderr, "proc= %d, total= %g: setup= %g / tree= %g / export= %g / ngb= %g / weights= %g // w= %g sec vain= %g\n",
	  myid, t5 - t1, t1 - t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6);
  fprintf(stderr, "derivs= %g sec\n", t8 - t7);
  
  //   fprintf(stderr, "end-of-program\n");
#endif  
  return 0;
}
