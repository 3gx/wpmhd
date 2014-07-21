#include "gn.h"

void system::read_binary(const char *filename, const int n_files) {
  assert(n_files == 1);
  
  float3 rmin, rmax;
  if (myid == 0) {
    
    FILE *fin; 
    if (!(fin = fopen(filename, "r"))) {
      std::cerr << "Cannot open file " << filename << std::endl;
      exit(-1);
    }

    std::cerr << "proc= " << myid << " read snapshot: " << filename << std::endl;
    
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
    
    float courant_No;
    int nglob, nloc, ndim;
    iload(nglob);
    iload(nloc);
    iload(ndim);  kernel.set_dim(ndim);
    fload(t_global);
    fload(dt_global);
    iload(iteration);
    fload(courant_No);
    fload(gamma_gas);
  
    int periodic_on;
    iload(periodic_on);
#ifdef _PERIODIC_FLOAT_
    if (!periodic_on) {
      fprintf(stderr,  "\n ************************** \n");
      fprintf(stderr,  " WARNING, SNAPSHOT *DOES NOT* USE _PERIODIC_FLOAT_, which is enabled in the code \n");
      fprintf(stderr,  "\n ************************** \n");
    }
#else
    if (periodic_on) {
      fprintf(stderr,  "\n ************************** \n");
      fprintf(stderr,  " WARNING, SNAPSHOT USES _PERIODIC_FLOAT_, which is not eanbled in the code \n");
      fprintf(stderr,  "\n ************************** \n");
    }
#endif
    
    fload(rmin.x);
    fload(rmin.y);
    fload(rmin.z);
    fload(rmax.x);
    fload(rmax.y);
    fload(rmax.z);
    iload(itmp);     // 20*4
    
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
	float3 pos;
	iload(ival);    // 25*4
	fload(pos.x); p.pos.x.set(pos.x);
	fload(pos.y); p.pos.y.set(pos.y);
	fload(pos.z); p.pos.z.set(pos.z);
	fload(p.vel.x);
	fload(p.vel.y);
	fload(p.vel.z);
	fload(m.dens);
	fload(m.ethm);
	fload(ftmp); // compute_pressure(m.dens, m.ethm));
	fload(ftmp); //dump(    (sqr(m.B.x  ) + sqr(m.B.y  ) + sqr(m.B.z  ))*0.5f);
	fload(ftmp); //dump(sqrt(sqr(m.vel.x) + sqr(m.vel.y) + sqr(m.vel.z))); 
	fload(m.vel.x);
	fload(m.vel.y);
	fload(m.vel.z);
	fload(m.B.x);
	fload(m.B.y);
	fload(m.B.z);
	fload(p.h);
	fload(p.wght);
	fload(m.psi);
	fload(ftmp); //L*divB_i[i]);
	fload(ftmp); // -1
	fload(ftmp); //-1.0f);
	fload(ftmp); //-1.0f);
	fload(ftmp); //-1.0f);
	iload(ival); //25*4);

	m.B.x -= constB.x;
	m.B.y -= constB.y;
	m.B.z -= constB.z;

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
      iload(ndim); assert(ndim == kernel.ndim);
      fload(t_global);
      fload(dt_global);
      iload(iteration);
      fload(courant_No);
      fload(gamma_gas);
  
      iload(periodic_on);

      fload(rmin.x);
      fload(rmin.y);
      fload(rmin.z);
      fload(rmax.x);
      fload(rmax.y);
      fload(rmax.z);
      iload(itmp);     // 20*4
    }
    assert(nglob == (int)pmhd.size());
    fclose(fin);

    
    pfloat<0>::set_range(rmin.x, rmax.x);
    pfloat<1>::set_range(rmin.y, rmax.y);
    pfloat<2>::set_range(rmin.z, rmax.z);
    global_domain.set_x(rmin.x, rmax.x);
    global_domain.set_y(rmin.y, rmax.y);
    global_domain.set_z(rmin.z, rmax.z);

    local_n = pmhd.size();
  }

  MPI_Bcast(&rmin.x,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rmin.y,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rmin.z,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rmax.x,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rmax.y,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rmax.z,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  pfloat<0>::set_range(rmin.x, rmax.x);
  pfloat<1>::set_range(rmin.y, rmax.y);
  pfloat<2>::set_range(rmin.z, rmax.z);
  
  global_domain.set_x(rmin.x, rmax.x);
  global_domain.set_y(rmin.y, rmax.y);
  global_domain.set_z(rmin.z, rmax.z);

  global_n = pvec.size();		      
  local_n = global_n;

  MPI_Bcast(&global_n,   1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&iteration,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(& t_global,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dt_global,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  dt_global = 0.0f;
  
  pmhd_dot.resize(local_n);

  box.set(nproc, box.nt, global_domain);

  distribute_particles();

  divB_i.resize(local_n);
  for (int i = 0; i < local_n; i++) 
    divB_i[i] = 0;

  for (int i = 0; i < local_n; i++)
    pmhd_dot[i].set(0.0);

  MPI_Barrier(MPI_COMM_WORLD);

  fprintf(stderr, " >>> proc= %d  local_n= %d  global_n=  %d\n",
 	  myid, local_n, global_n);

  MPI_Barrier(MPI_COMM_WORLD);

  build_tree();
  import_boundary_pvec_into_a_tree();
  compute_weights_cl();
  
  for (int i = 0; i < local_n; i++) {
    pmhd[i] = pmhd[i].to_conservative(pvec[i].wght);
  }

  fprintf(stderr, " ....... done ....... \n");

}
