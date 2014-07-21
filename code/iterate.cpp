#include "gn.h"

void system::iterate() {
  push_particles();
  
  t_global += dt_global;
  iteration++;

  for (int i = 0; i < local_n; i++) {
    boundary_particles(i);
  }
  compute_dt();
}

void system::compute_conserved(double &Mtot, double &Etot, 
			       double &Ethm, double &Ekin, double &Emag,
			       double &Volume) {
  
  double Mtot_loc = 0.0f;
  double Etot_loc = 0.0f;
  double Ethm_loc = 0.0f;
  double Ekin_loc = 0.0f;
  double Emag_loc = 0.0f;
  double Epot_loc = 0.0f;
  double Volume_loc = 0.0f;

  for (int i = 0; i < local_n; i++) {
    const particle &pi = pvec[i];
    const ptcl_mhd &mi = pmhd[i];

    const double ekin = (sqr(mi.mom.x) + sqr(mi.mom.y) + sqr(mi.mom.z))/mi.mass * 0.5f;
    const double emag = (sqr(mi.B.x)   + sqr(mi.B.y)   + sqr(mi.B.z)  )/pi.wght * 0.5f;
    const double ener = mi.ethm;
    
    const float4 acc = body_forces(pi.pos);
    Epot_loc += acc.w*mi.mass;

    Volume_loc += pi.wght;
    Mtot_loc += mi.mass;
    Ekin_loc += ekin;
    Emag_loc += emag;

#ifdef _CONSERVATIVE_
    Etot_loc += ener;
    Ethm_loc += ener - ekin - emag;
#elif defined _SEMICONSERVATIVE_
    Etot_loc += ener + ekin;
    Ethm_loc += ener - emag;
#else
    Etot_loc += ener + ekin + emag;
    Ethm_loc += ener;
#endif

  }

  MPI_Allreduce(&Mtot_loc, &Mtot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Etot_loc, &Etot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Ethm_loc, &Ethm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Ekin_loc, &Ekin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Emag_loc, &Emag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Allreduce(&Volume_loc, &Volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double Epot;
  MPI_Allreduce(&Epot_loc, &Epot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  Etot += Epot;
   
  
}
