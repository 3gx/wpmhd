#include "gn.h"

void system::iterate() {
  push_particles();
  t_global += dt_global;
  iteration++;
}

void system::compute_conserved(double &Mtot, double &Etot, 
			       double &Ethm, double &Ekin, double &Emag,
			       double &Volume) {
  
  double Mtot_loc = 0.0f;
  double Etot_loc = 0.0f;
  double Ethm_loc = 0.0f;
  double Ekin_loc = 0.0f;
  double Emag_loc = 0.0f;
  double Volume_loc = 0.0f;

  for (int i = 0; i < local_n; i++) {
//     const particle &pi = pvec[i];
    const ptcl_mhd &mi = pmhd[i];
    const float     Vi = weights[i];

#ifdef _CONSERVATIVE_
    Etot_loc += mi.ethm;
#elif defined _SEMICONSERVATIVE_
    Mtot_loc += mi.dens;
    const double ekin = (sqr(mi.vel.x) + sqr(mi.vel.y) + sqr(mi.vel.z))/mi.dens * 0.5f;
    Etot_loc += mi.ethm + ekin;;
#else
    Mtot_loc += mi.dens*Vi;
    const double ethm = mi.ethm*Vi;
    const double ekin = mi.dens*Vi*(sqr(mi.vel.x) + sqr(mi.vel.y) + sqr(mi.vel.z)) * 0.5f;
    const double emag =         Vi*(sqr(mi.B.x)   + sqr(mi.B.y)   + sqr(mi.B.z)  ) * 0.5f;

    Ethm_loc += ethm;
    Ekin_loc += ekin;
    Emag_loc += emag;
    Etot_loc += ethm + ekin + emag;
#endif

    Volume_loc += Vi;
    
  }

  MPI_Allreduce(&Mtot_loc, &Mtot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Etot_loc, &Etot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Ethm_loc, &Ethm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Ekin_loc, &Ekin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Emag_loc, &Emag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Allreduce(&Volume_loc, &Volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
  
}
