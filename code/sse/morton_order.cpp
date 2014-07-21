#include "gn.h"

void system::morton_order() {
  return;

#ifdef _DEBUG_PRINT_
  fprintf(stderr, "proc= %d: morton order\n", myid);
#endif
  // allocate arrays for morton order
  std::vector<particle> pvec_morton;
  std::vector<ptcl_mhd> pdot_morton = pmhd_dot;
  std::vector<ptcl_mhd> pmhd_morton = pmhd;

  // dump particles in morton order
  pvec_morton.reserve(256);
  root_node.dump_particles_in_Zorder(pvec_morton);
//   pvec_morton = pvec;
  assert(local_n = (int)pvec_morton.size());
//   return;
  
  // move particles in memory
  pvec = pvec_morton;
  for (int i = 0; i < local_n; i++) {
    pmhd    [i] = pmhd_morton[pvec_morton[i].local_idx];
    pmhd_dot[i] = pdot_morton[pvec_morton[i].local_idx];
    
//     assert(pvec[i].local_idx == i);
    pvec[i].local_idx = i;
  }
  
  

}
