#include "gn.h"

////////////


void system::init_conservative() {
  distribute_particles();
  
  build_tree();

  import_boundary_pvec_into_a_tree();
  compute_weights();
  import_boundary_wght();
  import_boundary_pvec_scatter_into_a_tree();
  import_boundary_pmhd();
  build_ngb_leaf_lists();

  for (size_t i = 0; i < pmhd.size(); i++) 
    pmhd[i] = pmhd[i].to_conservative() * pvec[i].wght;
}

////////////////

void system::convert_to_primitives() {
  double t0  = get_time();

  conservative.resize(pmhd.size());
  for (size_t i = 0; i < pmhd.size(); i++) {
    const particle &pi  = pvec[i];
    conservative[i] = pmhd[i];
    pmhd        [i] = (pmhd[i] * (1.0f/pi.wght)).to_primitive();
  }
  t_compute += get_time() - t0;
}

void system::restore_conservative() {
  double t0 = get_time();
  for (size_t i = 0; i < pmhd.size(); i++) {
    pmhd[i] = conservative[i];
  }
  t_compute += get_time() - t0;
}


