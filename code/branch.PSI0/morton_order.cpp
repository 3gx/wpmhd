#include "gn.h"

void system::morton_order() {

  std::vector<particle> pvec_morton;
  std::vector<ptcl_mhd> pdot_morton = pmhd_dot;
  std::vector<ptcl_mhd> pmhd_morton = pmhd;

  // dump particles in morton order
  pvec_morton.reserve(256);
  local_tree.root.dump_particles_in_Zorder(pvec_morton);
  assert(local_n = (int)pvec_morton.size());
  
  pvec = pvec_morton;
  for (int i = 0; i < local_n; i++) {
    pmhd    [i] = pmhd_morton[pvec_morton[i].local_idx];
    pmhd_dot[i] = pdot_morton[pvec_morton[i].local_idx];

    pvec[i].local_idx = i;
  }
  
  

}
