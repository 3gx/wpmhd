#include "gn.h"

void system::derivs(const bool first) {
  t_compute = t_communicate = 0;

  if (first) {
    gpu.domain_hsize = float4(global_domain.hsize.x.getu(),
			      global_domain.hsize.y.getu(),
			      global_domain.hsize.z.getu(), 0.0f);
    
    fprintf(stderr,  "Build tree ... \n");
    build_tree();
    fprintf(stderr,  "Build_tree ... done in %g sec\n", t_build_tree);
    
    import_boundary_pvec_into_a_tree();
    
    std::vector<float> wght0(local_n);
    for (int i = 0; i < local_n; i++) {
      wght0[i] = pvec[i].wght;
    }
    
    if (!do_predictor_step) {
      fprintf(stderr,  "Weights ... \n");
#if 1
      compute_weights_cl();
#else
      compute_weights();
#endif
      fprintf(stderr,  "Weights ... done in %g sec\n", t_compute_weights);
    } else {
      //     local_tree.root.calculate_inner_boundary();
      //     local_tree.root.calculate_outer_boundary();
    }

#if 0
#pragma omp parallel for 
    for (int i = 0; i < local_n; i++) {
      ptcl_mhd prim = pmhd[i].to_primitive(pvec[i].wght);
      prim.vel.x = pvec[i].vel.x;
      prim.vel.y = pvec[i].vel.y;
      prim.vel.z = pvec[i].vel.z;
      pmhd[i] = prim.to_conservative(pvec[i].wght);
    }
#endif

    import_boundary_wght();
    import_boundary_pvec_scatter_into_a_tree();
  }

  import_boundary_pmhd();

  if (first) {
    fprintf(stderr, "Build NGB list ... \n");
    const double tbuildngb = get_time();
    build_ngb_leaf_lists();
    fprintf(stderr, "Build NGB list ... done in %g sec \n", get_time() - tbuildngb);
  }
  
  // copy mesh-data to GPU
  
  convert_to_primitives();
  
  fprintf(stderr,  "Copy2gpu ... \n"); 
  const double tcopy2gpu = get_time();
  copy2gpu();
  fprintf(stderr,  "Copy2gpu ... done in %g sec \n", get_time() - tcopy2gpu);

  // call MHD solver

  fprintf(stderr,  "Renorm ... \n");
  renorm_cl();
  fprintf(stderr,  "Renorm ... done in %g sec \n", t_renorm);

  fprintf(stderr,  "InteractionCL ... \n");
  mhd_interaction_cl();
  fprintf(stderr,  "InteractionCL ... done in %g sec\n", t_mhd_interaction);
  
  //

  restore_conservative();
  
}


