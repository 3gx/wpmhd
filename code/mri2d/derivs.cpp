#include "gn.h"

void system::derivs() {
  t_compute = t_communicate = 0;
  
  build_tree();

  import_boundary_pvec_into_a_tree();

  compute_weights();
  import_boundary_wght();
  import_boundary_pvec_scatter_into_a_tree();
  import_boundary_pmhd();
  build_ngb_leaf_lists();
  
  convert_to_primitives();
  
  if (do_ppm) {
    if (do_first_order) {
      gradient();
    } else {
      renorm();
      gradient_ppm();
    }
  } else {
    gradient();
  }
  
  mhd_interaction();
  
  restore_conservative();

  
}


