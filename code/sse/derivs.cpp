#include "gn.h"

void system::derivs() {
  
  t_compute = t_communicate = 0;
  t_weight = t_renorm = t_grad = t_interaction = t_tree = 0;

  double t0 = get_time();
  build_tree();
  t_tree += get_time() - t0;

  import_boundary_particles();
  
  t0 = get_time();
  compute_weights();
  t_weight += get_time() - t0;
  
  if (do_morton_order) {
    morton_order();
    do_morton_order = false;
  }

  t0 = get_time();
  build_tree();
  t_tree += get_time() - t0;

  //////////////

  import_boundary_pvec();
  import_boundary_pmhd();
  
  build_ngb_leaf_lists();

  improve_weights();
  
#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)
  convert_to_primitives();
#endif
  
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "renorm ...\n");
#endif
//   t0 = get_time();
//   renorm();
//   t_renorm += get_time() - t0;

  // gradients
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "gradient ...\n");
#endif
  if (do_ppm) {
    renorm();
    gradient_ppm();
  } else {
    gradient();
//     gradient_v4sf();
  }
  
  // compute_defect
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "compute_defect ...\n");
#endif
  compute_defect();

  // mhd_interaction
#ifdef _DEBUG_PRINT_
  fprintf(stderr, "mhd_interaction ...\n");
#endif
  t0 = get_time();
  mhd_interaction();
  t_interaction = get_time() - t0;

#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)
  restore_conservative();
#endif

  
}


