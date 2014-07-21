#include "gn.h"
#include "myMPI.h"

void system::import_boundary_particles() {
  double t0 = get_time();

  std::vector<particle> pvec_send[NMAXPROC];
  std::vector<particle> pvec_recv[NMAXPROC];
  
  for (int p = 0; p < nproc; p++) {
    pvec_send[p].reserve(128);
    pidx_send[p].reserve(128);
    if (p == myid) continue;
    root_node.walk_boundary(box.outer[p], pvec_send[p]);
    const int np = pvec_send[p].size();
    pidx_send[p].resize(np);
    for (int i = 0; i < np; i++) {
      pidx_send[p][i] = pvec_send[p][i].local_idx;
    }
  }
  
  
  //////
  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<particle>(pvec_send, pvec_recv, myid, nproc, debug_flag);
  
  // copy imported particles into the system

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvec_recv[p].size(); i++) 
      cntr++;
  
  assert(safe_resize(pvec, local_n + cntr));
  
  cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvec_recv[p].size(); i++) 
      pvec[local_n + cntr++] = pvec_recv[p][i];

  import_n = pvec.size() - local_n;

#ifdef _DEBUG_PRINT_
  fprintf(stderr, "proc= %d: import_n= %d\n", myid, import_n);
#endif
  
  // create octbodies of import particles
  
  assert(safe_resize(octp, local_n + import_n));
  
  for (int i = local_n; i < local_n + import_n; i++) {
    octp[i] = octbody(pvec[i], true);
  }
  
  // insert octobdies into the tree
  
  for (int i = local_n; i < local_n + import_n; i++) {
    octnode::insert_octbody(root_node, octp[i], octn);
  }
    
  root_node.calculate_inner_boundary();
  root_node.calculate_outer_boundary();
  
  // update local index
  for (int i = local_n; i < local_n + import_n; i++) {
    pvec[i].local_idx = i;
  }
  
  leaf_list.clear();
  group_list.clear();
  leaf_list.reserve(128);
  group_list.reserve(128);

  root_node.extract_leaves(leaf_list);
  root_node.extract_leaves(group_list);

  ///////////
  t_communicate += get_time() - t0;
  t_import_bnd = get_time() - t0;
}

//////////

void system::import_boundary_pmhd() {
  double t0 = get_time();

  std::vector<ptcl_mhd> pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd> pmhd_recv[NMAXPROC];
  
  int send = 0;
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pidx_send[p].size();
    pmhd_send[p].resize(np);
    for (int i = 0; i < np; i++) {
      send++;
      int idx = pidx_send[p][i];
      assert(idx < local_n);
      pmhd_send[p][i] = pmhd[idx];				   
    }
  }
#ifdef _DEBUG_PRINT_
  fprintf(stderr, " proc= %d:  send= %d\n", myid, send);
#endif

  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);

  
  if (debug_flag) {
    std::vector<int> pgdx_recv[NMAXPROC];
    std::vector<int> pgdx_send[NMAXPROC];
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      const int np = pidx_send[p].size();
      pgdx_send[p].resize(np);
      for (int i = 0; i < np; i++) {
	const int idx = pidx_send[p][i];
 	pgdx_send[p][i] = pvec[idx].global_idx;
      }
    }

    myMPI_all2all<int>(pgdx_send, pgdx_recv, myid, nproc, true);

    int cntr = 0;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pgdx_recv[p].size(); i++) {
	assert(pvec[local_n + cntr].global_idx == pgdx_recv[p][i]);
	cntr++;
      }
  }

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pmhd_recv[p].size(); i++) 
      cntr++;
  
  assert(safe_resize(pmhd, local_n + cntr));

  cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pmhd_recv[p].size(); i++) 
      pmhd[local_n + cntr++] = pmhd_recv[p][i];
  
  assert((size_t)cntr == pvec.size() - local_n);

#if _DEBUG_PRINT_
  fprintf(stderr, "proc= %d: pmhd.size= %d   pvec.size= %d  local_n= %d  import_n= %d\n",
	  myid, (int)pmhd.size(), (int)pvec.size(), local_n, import_n);
#endif
  
  assert(pmhd.size() == pvec.size());

  t_communicate += get_time() - t0;
  t_pmhd = get_time() - t0;
}

/////////

void system::import_boundary_Bmatrix() {
  double t0 = get_time();

  std::vector<float> B_send[NMAXPROC];
  std::vector<float> B_recv[NMAXPROC];
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pidx_send[p].size();
    B_send[p].resize(6*np);
    for (int i = 0; i < np; i++) {
      int idx = pidx_send[p][i];
      assert(idx < local_n);
      int i6 = i*6;
      B_send[p][i6  ] = Bxx[idx];
      B_send[p][i6+1] = Bxy[idx];
      B_send[p][i6+2] = Bxz[idx];
      B_send[p][i6+3] = Byy[idx];
      B_send[p][i6+4] = Byz[idx];
      B_send[p][i6+5] = Bzz[idx];
    }
  }
  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<float>(B_send, B_recv, myid, nproc, debug_flag);

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < B_recv[p].size(); i += 6) 
      cntr++;
  
  Bxx.resize(local_n + cntr);
  Bxy.resize(local_n + cntr);
  Bxz.resize(local_n + cntr);
  Byy.resize(local_n + cntr);
  Byz.resize(local_n + cntr);
  Bzz.resize(local_n + cntr);

  cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < B_recv[p].size(); i += 6) {
      Bxx[local_n + cntr] = B_recv[p][i  ];
      Bxy[local_n + cntr] = B_recv[p][i+1];
      Bxz[local_n + cntr] = B_recv[p][i+2];
      Byy[local_n + cntr] = B_recv[p][i+3];
      Byz[local_n + cntr] = B_recv[p][i+4];
      Bzz[local_n + cntr] = B_recv[p][i+5];
      cntr++;
    }

  assert((size_t)cntr == pvec.size() - local_n);

  assert(Bxx.size() == pvec.size());
  assert(Bxy.size() == pvec.size());
  assert(Bxz.size() == pvec.size());
  assert(Byy.size() == pvec.size());
  assert(Byz.size() == pvec.size());
  assert(Bzz.size() == pvec.size());
  
  t_communicate += get_time() - t0;
  t_Bmatrix = get_time() - t0;
}

/////////

void system::import_boundary_pvel() {
  double t0 = get_time();

  std::vector<float>  pvel_send[NMAXPROC];
  std::vector<float>  pvel_recv[NMAXPROC];
  
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pidx_send[p].size();
    pvel_send[p].resize(3*np);
    for (int i = 0; i < np; i++) {
      int idx = pidx_send[p][i];
      assert(idx < local_n);
      pvel_send[p][3*i  ] = pvec[idx].vel.x;
      pvel_send[p][3*i+1] = pvec[idx].vel.y;
      pvel_send[p][3*i+2] = pvec[idx].vel.z;
    }
  }
  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<float>(pvel_send, pvel_recv, myid, nproc, debug_flag);

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvel_recv[p].size(); i += 3) 
      cntr++;
  
  cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvel_recv[p].size(); i += 3) {
      pvec[local_n + cntr].vel.x = pvel_recv[p][i  ];
      pvec[local_n + cntr].vel.y = pvel_recv[p][i+1];
      pvec[local_n + cntr].vel.z = pvel_recv[p][i+2];
      cntr++;
    }

  assert((size_t)cntr == pvec.size() - local_n);
  t_communicate += get_time() - t0;
  t_pvel= get_time() - t0;
}

/////////

void system::import_boundary_pmhd_grad() {
  double t0 = get_time();

  std::vector<ptcl_mhd>  pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd>  pmhd_recv[NMAXPROC];
  
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pidx_send[p].size();
    pmhd_send[p].resize(3*np);
    for (int i = 0; i < np; i++) {
      int idx = pidx_send[p][i];
      assert(idx < local_n);
      pmhd_send[p][3*i  ] = pmhd_grad[0][idx];
      pmhd_send[p][3*i+1] = pmhd_grad[1][idx];
      pmhd_send[p][3*i+2] = pmhd_grad[2][idx];
    }
  }
  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pmhd_recv[p].size(); i += 3) 
      cntr++;
  
  pmhd_grad[0].resize(local_n + cntr);
  pmhd_grad[1].resize(local_n + cntr);
  pmhd_grad[2].resize(local_n + cntr);

  cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pmhd_recv[p].size(); i += 3) {
      pmhd_grad[0][local_n + cntr] = pmhd_recv[p][i  ];
      pmhd_grad[1][local_n + cntr] = pmhd_recv[p][i+1];
      pmhd_grad[2][local_n + cntr] = pmhd_recv[p][i+2];
      cntr++;
    }

  assert((size_t)cntr == pvec.size() - local_n);

  assert(pmhd_grad[0].size() == pvec.size());
  assert(pmhd_grad[1].size() == pvec.size());
  assert(pmhd_grad[2].size() == pvec.size());
  t_communicate += get_time() - t0;
  t_grad_comm = get_time() - t0;
}

////////////////

// void system::import_scatter_particles() {

//   std::vector<particle> pvec_send[NMAXPROC];
//   std::vector<particle> pvec_recv[NMAXPROC];

//   for (int p = 0; p < nproc; p++) 
//     pvec_send[p].reserve(128);

//   for (int p = 0; p < nproc; p++) {
//     if (p == myid) continue;
//     root_node.walk_outer_boundary(box.get_bnd(p), box.get_outer(p), pvec_send[p]);
//     const int np = pvec_send[p].size();
//     for (int i = 0; i < np; i++) {
//       for (size_t j = 0; j < pidx_send[p].size(); j++) {
// 	assert(pvec_send[p][i].local_idx < local_n);
// 	assert(pidx_send[p][j] != pvec_send[p][i].local_idx);
//       }
//     }
//     for (int i = 0; i < np; i++) 
//       pidx_send[p].push_back(pvec_send[p][i].local_idx);
//   }
  
  
// #if _DEBUG_
//   const bool debug_flag = true;
// #else
//   const bool debug_flag = false;
// #endif
//   myMPI_all2all<particle>(pvec_send, pvec_recv, myid, nproc, debug_flag);
  
//   const int import_n_old = import_n;
  
//   int cntr = 0;
//   for (int p = 0; p < nproc; p++)
//     for (size_t i = 0; i < pvec_recv[p].size(); i++)
//       cntr++;
  
//   fprintf(stderr, "proc= %d// imported= %d  total= %d\n", myid, cntr, import_n + cntr);
//   MPI_Barrier(MPI_COMM_WORLD);
  
//   assert(safe_resize(pvec, local_n + import_n_old + cntr));
  
//   cntr = 0;
//   for (int p = 0; p < nproc; p++)
//     for (size_t i = 0; i < pvec_recv[p].size(); i++)      
//       pvec[local_n + import_n_old + cntr++] = pvec_recv[p][i];


// #if 0

//   import_n = pvec.size() - local_n;
//   assert(safe_resize(octp, local_n + import_n));

//   for (int i = local_n + import_n_old; i < local_n + import_n; i++) {
//     octp[i] = octbody(pvec[i], true);
//   }
  
//   for (int i = 0;       i < local_n;            i++) octp[i].update();
//   for (int i = local_n; i < local_n + import_n; i++) octp[i].update(true);
  
//   for (int i = local_n + import_n_old; i < local_n + import_n; i++) {
//     octnode::insert_octbody(root_node, octp[i], octn);
//   }

//   for (int i = local_n + import_n_old; i < local_n + import_n; i++) {
//     pvec[i].local_idx = i;
//   }

//   leaf_list.clear();
//   group_list.clear();

//   root_node.extract_leaves(leaf_list);
//   root_node.extract_leaves(group_list);

//   root_node.calculate_inner_boundary();
//   root_node.calculate_outer_boundary();
  
//   leaf_list.clear();
//   group_list.clear();
//   leaf_list.reserve(128);
//   group_list.reserve(128);

//   root_node.extract_leaves(leaf_list);
//   root_node.extract_leaves(group_list);
// #endif
//   ///////////



//   /////////

//   if (true) {
//     std::vector<int> pgdx_recv[NMAXPROC];
//     std::vector<int> pgdx_send[NMAXPROC];
//     for (int p = 0; p < nproc; p++) {
//       if (p == myid) continue;
//       const int np = pidx_send[p].size();
//       pgdx_send[p].resize(np);
//       for (int i = 0; i < np; i++) {
// 	const int idx = pidx_send[p][i];
//  	pgdx_send[p][i] = myid*1000000 + i; //pvec[idx].global_idx;
//  	pgdx_send[p][i] = pvec[idx].global_idx;
//       }
//     }
    
//     myMPI_all2all<int>(pgdx_send, pgdx_recv, myid, nproc, true);

//     if (myid == 0) {
//     int cntr = 0;
//     for (int p = 0; p < nproc; p++)
//       for (size_t i = 0; i < pgdx_recv[p].size(); i++) {
// 	if (!(pvec[local_n + cntr].global_idx == pgdx_recv[p][i])) {
// 	  fprintf(stderr, "proc= %d: p= %d i= %d gidx= %d  gidx_recv= %d\n",
// 		  myid, p, i, pvec[local_n + cntr].global_idx,
// 		  pgdx_recv[p][i]);
	  
// 	}
// //  	assert(pvec[local_n + cntr].global_idx == pgdx_recv[p][i]);
// 	cntr++;
//       }
// 	fprintf(stderr, "proc= %d: cntr= %d  n= %d\n",
// 		myid, cntr, import_n);

//     }
//   }

//   ////////////

//   fprintf(stderr, "*** proc= %d done ***** \n", myid);
//   MPI_Barrier(MPI_COMM_WORLD);
//   sleep(100);
//   exit(-1);

// }

// //////////////

void system::import_boundary_pvec() {
  double t0 = get_time();

  std::vector<particle> pvec_send[NMAXPROC];
  std::vector<particle> pvec_recv[NMAXPROC];

  for (int p = 0; p < nproc; p++) {
    pidx_send[p].clear();
    pidx_send[p].reserve(128);
    pvec_send[p].reserve(128);
  }
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    root_node.walk_outer_boundary(box.get_bnd(p), box.get_outer(p), pvec_send[p]);
    const int np = pvec_send[p].size();
    for (int i = 0; i < np; i++) 
      pidx_send[p].push_back(pvec_send[p][i].local_idx);
  }
  
  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<particle>(pvec_send, pvec_recv, myid, nproc, debug_flag);
  
  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvec_recv[p].size(); i++)
      cntr++;
  
  assert(safe_resize(pvec, local_n + cntr));
  
  cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvec_recv[p].size(); i++)      
      pvec[local_n + cntr++] = pvec_recv[p][i];

  import_n = pvec.size() - local_n;
  
  // create octbodies of import particles
  
  assert(safe_resize(octp, local_n + import_n));
  
  for (int i = local_n; i < local_n + import_n; i++) {
    octp[i] = octbody(pvec[i], true);
  }
  
  // insert octobdies into the tree
  
  for (int i = local_n; i < local_n + import_n; i++) {
    octnode::insert_octbody(root_node, octp[i], octn);
  }
    
  root_node.calculate_inner_boundary();
  root_node.calculate_outer_boundary();
  
  // update local index
  for (int i = local_n; i < local_n + import_n; i++) {
    pvec[i].local_idx = i;
  }
  
  leaf_list.clear();
  group_list.clear();
  leaf_list.reserve(128);
  group_list.reserve(128);

  root_node.extract_leaves(leaf_list);
  root_node.extract_leaves(group_list);

  t_communicate += get_time() - t0;
  t_pvec = get_time() - t0;
}



//////////

void system::import_boundary_pmhd_cross() {
  double t0 = get_time();

  std::vector<ptcl_mhd>  pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd>  pmhd_recv[NMAXPROC];
  
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pidx_send[p].size();
    pmhd_send[p].resize(6*np);
    for (int i = 0; i < np; i++) {
      int idx = pidx_send[p][i];
      assert(idx < local_n);
      const int i6 = i*6;
      pmhd_send[p][i6  ] = pmhd_cross[0][idx];
      pmhd_send[p][i6+1] = pmhd_cross[1][idx];
      pmhd_send[p][i6+2] = pmhd_cross[2][idx];
      pmhd_send[p][i6+3] = pmhd_cross[3][idx];
      pmhd_send[p][i6+4] = pmhd_cross[4][idx];
      pmhd_send[p][i6+5] = pmhd_cross[5][idx];
    }
  }
 
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pmhd_recv[p].size(); i += 6) 
      cntr++;
  
  for (int k = 0; k < 6; k++)
    pmhd_cross[k].resize(local_n + cntr);

  cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pmhd_recv[p].size(); i += 6) {
      pmhd_cross[0][local_n + cntr] = pmhd_recv[p][i  ];
      pmhd_cross[1][local_n + cntr] = pmhd_recv[p][i+1];
      pmhd_cross[2][local_n + cntr] = pmhd_recv[p][i+2];
      pmhd_cross[3][local_n + cntr] = pmhd_recv[p][i+3];
      pmhd_cross[4][local_n + cntr] = pmhd_recv[p][i+4];
      pmhd_cross[5][local_n + cntr] = pmhd_recv[p][i+5];
      cntr++;
    }

  assert((size_t)cntr == pvec.size() - local_n);
  
  assert(pmhd_cross[0].size() == pvec.size());
  assert(pmhd_cross[1].size() == pvec.size());
  assert(pmhd_cross[2].size() == pvec.size());
  assert(pmhd_cross[3].size() == pvec.size());
  assert(pmhd_cross[4].size() == pvec.size());
  assert(pmhd_cross[5].size() == pvec.size());

  t_communicate += get_time() - t0;
  t_grad_comm += get_time() - t0;
}
