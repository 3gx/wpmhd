#include "gn.h"

////////////


void system::init_conservative() {
#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)

  distribute_particles();
  
  // build tree
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "build_tree ...\n");
#endif
  build_tree();
  
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "import_boundary_particles ...\n");
#endif
  import_boundary_particles();
  
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "compute_weights ...\n");
#endif
  compute_weights();

  build_tree();

  import_boundary_pvec();

  build_ngb_leaf_lists();
  
  improve_weights();

  for (size_t i = 0; i < pmhd.size(); i++) {
    const particle &pi   = pvec[i];
    const ptcl_mhd prim = pmhd[i];
    ptcl_mhd cons;
    const float Vi = weights[pi.local_idx];
    
    cons.dens  = prim.dens  * Vi;
    cons.vel.x = prim.vel.x * cons.dens;
    cons.vel.y = prim.vel.y * cons.dens;
    cons.vel.z = prim.vel.z * cons.dens;
    cons.B.x  = prim.B.x   * Vi;
    cons.B.y  = prim.B.y   * Vi;
    cons.B.z  = prim.B.z   * Vi;
    cons.scal = prim.scal  * cons.dens;

    cons.psi  = prim.psi   * cons.dens;
//     cons.psi  = prim.psi ;
#ifdef _CONSERVATIVE_
    cons.ethm = prim.ethm 
      + 0.5*(sqr(prim.vel.x) + sqr(prim.vel.y) + sqr(prim.vel.z))*prim.dens
      + 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#else
    cons.ethm = prim.ethm 
      + 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#endif
    cons.ethm *= Vi;
    
    pmhd[i] = cons;
  }
#endif
}

////////////////

void system::convert_to_primitives() {
  double t0  = get_time();
#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)

  conservative.resize(pmhd.size());
  for (size_t i = 0; i < pmhd.size(); i++) {
    const particle &pi  = pvec[i];
    const ptcl_mhd cons = pmhd[i];
    ptcl_mhd prim;

    const float Vi = weights[pi.local_idx];

    prim.dens  = cons.dens/Vi;
    prim.vel.x = cons.vel.x/cons.dens;
    prim.vel.y = cons.vel.y/cons.dens;
    prim.vel.z = cons.vel.z/cons.dens;
    prim.B.x   = cons.B.x  /Vi;
    prim.B.y   = cons.B.y  /Vi;
    prim.B.z   = cons.B.z  /Vi;
    prim.scal  = cons.scal /cons.dens;

    prim.psi   = cons.psi  /cons.dens;
//     prim.psi   = cons.psi;
#ifdef _CONSERVATIVE_
    prim.ethm  = cons.ethm/Vi
      - 0.5*(sqr(prim.vel.x) + sqr(prim.vel.y) + sqr(prim.vel.z))*prim.dens
      - 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#else
    prim.ethm  = cons.ethm/Vi
      - 0.5*(sqr(prim.B.x  ) + sqr(prim.B.y  ) + sqr(prim.B.z  ));
#endif
    
    conservative[i] = cons;
    pmhd      [i] = prim;
  }
#endif
  t_compute += get_time() - t0;
}

void system::restore_conservative() {
  double t0 = get_time();
#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)
  for (size_t i = 0; i < pmhd.size(); i++) {
    pmhd[i] = conservative[i];
  }
#endif
  t_compute += get_time() - t0;
}


