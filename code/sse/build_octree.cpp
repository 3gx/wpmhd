#include "gn.h"
#include "myMPI.h"

void system::build_tree() {

  double t0 = get_time();
  // reset the tree

  octp.clear();
  root_node.clear();
  octn.reset();

  // allocate nodes

  octp.resize(local_n);
  const size_t expected_node_count = (size_t)(local_n*100.0f/octnode::nleaf);
  octn.allocate(expected_node_count);
  
  root_node.assign_root(global_domain);
  
  // inserd local bodies
  
  for (int i = 0; i < local_n; i++) {
    octp[i] = octbody(pvec[i]);
  }
  
  for (int i = 0; i < local_n; i++) {
    octnode::insert_octbody(root_node, octp[i], octn);
  }

#ifdef _DEBUG_  
#ifndef _PERIODIC_FLOAT_
  root_node.sanity_check();
#endif
#endif
  
  const float scaling_factor = 1.1f;
  root_node.calculate_inner_boundary();
  root_node.calculate_outer_boundary(scaling_factor);
  
  t_compute += get_time() - t0;

  ///////////////// communicate boundary

  t0 = get_time();

  boundary bnd = root_node.outer;
  myMPI_allgather<boundary>(bnd, box.outer, myid, nproc);

  t_communicate += get_time() - t0;
  
}
