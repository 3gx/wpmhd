#include "gn.h"

void system::derivs() {
  t_compute = t_communicate = 0;
  
  fprintf(stderr,  "Build tree ... \n");
  build_tree();
  fprintf(stderr,  "Build_tree ... done in %g sec\n", t_build_tree);
  
  import_boundary_pvec_into_a_tree();
  
  fprintf(stderr,  "Weights ... \n");
  compute_weights_cl();
  fprintf(stderr,  "Weights ... done in %g sec\n", t_compute_weights);

//   fprintf(stderr,  "Weights ... \n");
//   compute_weights_cl();
//   fprintf(stderr,  "Weights ... done in %g sec\n", t_compute_weights);
//   exit(-1);
  import_boundary_wght();
  import_boundary_pvec_scatter_into_a_tree();
  import_boundary_pmhd();

  fprintf(stderr, "Build NGB list ... \n");
  const double tbuildngb = get_time();
  build_ngb_leaf_lists();
  fprintf(stderr, "Build NGB list ... done in %g sec \n", get_time() - tbuildngb);
  
  convert_to_primitives();

  fprintf(stderr,  "Copy2gpu ... \n"); 
  const double tcopy2gpu = get_time();
  copy2gpu();
  fprintf(stderr,  "Copy2gpu ... done in %g sec \n", get_time() - tcopy2gpu);

#if 1

  fprintf(stderr,  "Nngb... \n");
  nngb_cl();
  fprintf(stderr,  "Nngb ... done in %g sec \n", t_nngb);
  
  fprintf(stderr,  "Renorm ... \n");
  renorm_cl();
  fprintf(stderr,  "Renorm ... done in %g sec \n", t_renorm);

//   fprintf(stderr,  "Gradient ... \n");
//   gradient_cl();
//   fprintf(stderr,  "Gradient ... done in %g sec\n", t_gradient);

//   fprintf(stderr,  "Interaction ... \n");
//   mhd_interaction();
//   fprintf(stderr,  "Interaction ... done in %g sec\n", t_mhd_interaction);
  
  fprintf(stderr,  "InteractionCL ... \n");
  mhd_interaction_cl();
  fprintf(stderr,  "InteractionCL ... done in %g sec\n", t_mhd_interaction);
//   exit(-1);
#else
  fprintf(stderr,  "Renorm ... \n");
  renorm_cl();
  fprintf(stderr,  "Renorm ... done in %g sec \n", t_renorm);

  fprintf(stderr,  "Gradient ... \n");
  gradient_cl();
  fprintf(stderr,  "Gradient ... done in %g sec\n", t_gradient);

  fprintf(stderr,  "Interaction ... \n");
  mhd_interaction();
  fprintf(stderr,  "Interaction ... done in %g sec\n", t_mhd_interaction);
#endif

// //   exit(-1);
//   if (do_ppm) {
//     if (do_first_order) {
//       gradient();
//     } else {
//       renorm();
//       gradient_ppm();
//     }
//   } else {
//     gradient();
//   }

#if 0
  fprintf(stderr,  "Interaction ... \n");
  mhd_interaction();
  fprintf(stderr,  "Interaction ... done in %g sec\n", t_mhd_interaction);
#endif
  
  restore_conservative();
  
}


