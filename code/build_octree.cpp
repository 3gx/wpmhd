#include "gn.h"
#include "myMPI.h"

/////////     Build local tree

void system::build_tree() {
  
  double t0 = get_time();

  // reset the tree

  const int expected_node_count = (int)(local_n*10.0f/TREE_NLEAF);

  local_tree.clear();
  local_tree.resize(local_n, expected_node_count);
  local_tree.set_domain(global_domain);
  local_tree.insert(&pvec[0], local_n);
  local_tree.root.calculate_inner_boundary();
  local_tree.root.calculate_outer_boundary();
  local_tree.get_leaves();

  /////////
  //// build mini-tree
  ////////

  const int expected_domain_node_count = (int)(local_n*10.0f/DOMAIN_TREE_NLEAF);

  domain_tree.clear();
  domain_tree.resize(local_n, expected_domain_node_count);
  domain_tree.set_domain(global_domain);
  domain_tree.insert(&pvec[0], local_n);
  domain_tree.get_leaves();
  
  ///////////////// communicate boundary
  
  const float scale_factor = 1.1f; 
  calculate_outer_domain_boundaries(scale_factor);


  //////////////////
  //////////////////
  //////////////////

  t_compute += get_time() - t0;
  
  import_tree.clear();
  import_tree.set_domain(global_domain);

  import_n = 0;
  Nimport  = 0;
  

  t_build_tree = get_time() - t0;
 }
