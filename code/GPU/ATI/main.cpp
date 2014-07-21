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
  MPI_Init(&argc, &argv);

  int nproc, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);


  if (myid == 14) {
    gwait = 0;
    while (gwait == 1) {
      std::cerr << " waiting ... zzzz \n";
      sleep(2);
    };
  }

  char path[256] = "ZZZZZZ";
  if (argc > 1) {
    sprintf(path, "%s", argv[1]);
  } else {
    if (myid == 0) {
      fprintf(stderr, " ./main data_path \n");
      exit(-1);
    }
  }

  bool do_ppm = false;
//   do_ppm = true;

  class system s(nproc, myid, do_ppm);
  s.gpu.init();
  fpe_catch();
  
  bool load_data = false;
  if (myid == 0) 
    fprintf(stderr, "  >>>>>>>> output path= %s <<<<<<<<<< \n", path);
		   

  if (argc > 2) {
    if (myid == 0) 
      fprintf(stderr, "  reading snapshot ... \n");
    load_data = true;
    s.setup_particles(false);
    s.read_binary((const char*)argv[2], 1);
    if (argc > 3) {
      s.t_global  = 0.0;
      s.iteration = 0;
      if (myid == 0) 
	fprintf(stderr, "reset time and iteration count .. \n");
    }
  } else { //if (argv < 3) {
    if (myid == 0) 
      fprintf(stderr, "  setting up data ... \n");
    s.setup_particles(true);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
#if 0
  {
    assert(myid == 0);

    const int n = 10000;
    std::vector<pfloat3> tpos[n];
    for (int i = 0; i < n; i++) tpos[i].reserve(128);
    int ntile = s.box.ntile;
    assert(ntile <= n);
    for (int i = 0; i < s.global_n; i++) {
      const pfloat3 pos = s.pvec[i].pos;
      int tile = s.box.which_tile(pos);
      tpos[tile].push_back(pos);
    }

    char fn[256];
    for (int tile = 0; tile < ntile; tile++) {
      sprintf(fn, "tiles/%.5d", tile);
      fprintf(stderr, "tile= %d out of %d to %s \n", tile, ntile, fn);
      FILE *fout = fopen(fn, "w");
      for (size_t i = 0; i < tpos[tile].size(); i++) {
	fprintf(fout, "%g %g %g\n",
		tpos[tile][i].x.getu(),
		tpos[tile][i].y.getu(),
		tpos[tile][i].z.getu());
      }
      fclose(fout);
    }
    
    exit(-1);
  }
#endif

  char fnn[256];
  sprintf(fnn, "%s/init_p%.3d.bin", path, s.myid);
  fprintf(stderr, " dump init to %s \n",  fnn);
  s.dump_binary(fnn, false);

  fprintf(stderr, "proc= %d: initialising ...\n", myid);
  double t0 = get_time();
  s.init_conservative();
  system("uname -n");
  fprintf(stderr, "proc= %d: done initialising ... in %g sec\n", myid, get_time() - t0);


  s.eulerian_mode   = false;
//   s.eulerian_mode   = true;
  s.do_morton_order = false;
  
  s.courant_no = 0.8;

  s.iterate();
  
  
  double Volume0, Mtot0, Etot0, Ekin0, Emag0, Ethm0;
  s.compute_conserved(Mtot0, Etot0, Ethm0, Ekin0, Emag0, Volume0);

  float  t_out = 0.0f;
  float  t_bin = 0.0f;
  float dt_out = 0.1f;
  float dt_bin = 0.1f;
  int    i_out = 0;
  int    i_bin = 0;

  t_out = 1e10f;
  
//    dt_out = 0.01f;
//   dt_bin = 0.01f;
//   dt_bin = 1.0e-4;
//   dt_bin = 1.0f;
  dt_bin = 0.1f;
  float t_end = 10.01;

//   dt_bin = 0.1f;
  t_end = 200.0f;
//   dt_bin = 0.1f;


  float t_restart = t_bin;
  float dt_restart = 0.1f;
//   t_restart = 1e10f;

//   dt_bin    = 0.1f;
//   dt_restart = 0.01f;








  if (load_data) {
    i_bin = (int)(s.t_global/dt_bin) + 1;
    t_bin = i_bin*dt_bin;
    const int i_restart = (int)(s.t_global/dt_restart) + 1;
    t_restart = i_restart * dt_restart;
  }

  t_restart = 1e10f;

  if (myid == 0) {
    fprintf(stderr, " t_bin= %g  i_bin = %d  t_restart= %g\n", t_bin, i_bin, t_restart);
  }
  
  assert(t_end > s.t_global);

  const int   n_end = 1000000;
  fprintf(stderr, " *** t_end= %g  n_end= %d \n", t_end, n_end);
  for (int i = 0; i < n_end; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0  = get_time(); 
    s.iterate();
    MPI_Barrier(MPI_COMM_WORLD);
    double dt_done = get_time() - t0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (s.t_global >= t_end) exit(-1);
    if (s.t_global >= t_out) {
//       char fn[256];
//       sprintf(fn, "%s/iter_p%.3d_%.5d.ascii", "data", s.myid, i_out++);
//       fprintf(stderr, "dumping output to %s @ time= %g \n", fn, s.t_global);
//       s.dump_snapshot(fn);
//       t_out += dt_out;
    }
    if (s.t_global >= t_bin) {
//     if (true) {
      char fn[256];
      sprintf(fn, "%s/iter_p%.3d_%.5d.bin", path, s.myid, i_bin++);
      fprintf(stderr, "dumping output to %s @ time= %g \n", fn, s.t_global);
      s.dump_binary(fn);
      t_bin += dt_bin;
    }
    if (s.t_global >= t_restart) {
      char fn[256];
      sprintf(fn, "%s/iter_p%.3d.restart", path, s.myid);
      fprintf(stderr, "dumping restart file to %s @ time= %g \n", fn, s.t_global);
      s.dump_binary(fn);
      t_restart += dt_restart;
    }
    
    
    double Volume, Mtot, Etot, Ekin, Emag, Ethm;
    s.compute_conserved(Mtot, Etot, Ethm, Ekin, Emag, Volume);
    
    if (s.myid == 0) {
//       if (s.iteration%2 == 0) {
      fprintf(stderr, " ----------------------------------------------------------- \n");
	fprintf(stderr, "proc= %d: iter= %d: t= %g dt = %g cons= (%g %g %g) diff= (%g %g %g) Ethm= %g Ekin= %g Emag= %g  done in %g sec.\n",
		s.myid, s.iteration, s.t_global, s.dt_global,
		Mtot, Etot, Volume, 
		(Mtot - Mtot0)/Mtot0, (Etot - Etot0)/Etot0, (Volume - Volume0)/Volume0, Ethm, Ekin, Emag, dt_done);
//       } else {
	fprintf(stderr, "proc= %d: ratio= %g;  t_comm= %g: %g %g  %g %g %g  %g %g \n",
		myid, s.t_communicate/s.t_compute, s.t_communicate,
		s.t_distribute_particles, s.t_import_bnd, s.t_pmhd, s.t_pvec, s.t_pvel, s.t_Bmatrix, s.t_grad_comm);
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

//   s.dump_snapshot("snap.dat");
  MPI_Finalize();
  //   fprintf(stderr, "end-of-program\n");
  
  fprintf(stderr, "proc= %d, total= %g: setup= %g / tree= %g / export= %g / ngb= %g / weights= %g // w= %g sec vain= %g\n",
	  myid, t5 - t1, t1 - t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6);
  fprintf(stderr, "derivs= %g sec\n", t8 - t7);
  
  //   fprintf(stderr, "end-of-program\n");
#endif  
  return 0;
}
