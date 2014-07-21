#include "gn.h"
#include "myMPI.h"

void system::import_boundary_pvec_scatter_into_a_tree() {
  const double t0 = get_time();
  
  //****** allocate communicaiton buffers
  
  std::vector<particle> pvec_send[NMAXPROC];
  
  //****** prepare particles for communication

  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    pvec_send[p].reserve(128);
    const std::vector<int> &tiles = box.get_tiles(p);
    for (size_t tile = 0; tile < tiles.size(); tile++) 
      local_tree.root.walk_scatter(box.inner(tiles[tile]), pvec_send[p]);
  }
  import_pvec_buffers(pvec_send);

  t_import_boundary_pvec_scatter_into_a_tree = get_time() - t0;
}

void system::import_boundary_pvec_into_a_tree() {
  const double t0 = get_time();
  
  //****** allocate communicaiton buffers
  
  std::vector<particle> pvec_send[NMAXPROC];
  
  //****** prepare particles for communication
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    pvec_send[p].reserve(128);
    for (size_t tile = 0; tile < outer_tiles[p].size(); tile++) {
      local_tree.root.walk_boundary(outer_tiles[p][tile], pvec_send[p]);
    }
  }
  
  import_pvec_buffers(pvec_send);
  
  t_import_boundary_pvec_into_a_tree = get_time() - t0;
}

void system::import_pvec_buffers(std::vector<particle> pvec_send_in[NMAXPROC]) {
  assert(Nimport < NMAXIMPORT);

  const double t0 = get_time();
  std::vector<particle> pvec_send[NMAXPROC];

  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
     const int np = pvec_send_in[p].size();
     pvec_send[p].clear();
     pvec_send[p].reserve(16);
     for (int i = 0; i < np; i++) {
       if (pidx_sent[p].insert(pvec_send_in[p][i].local_idx))
	 pvec_send[p].push_back(pvec_send_in[p][i]);
     }
  }

#if 0
  for (int p = 0; p < nproc; p++) 
    for (size_t i = 0; i < pvec_send[p].size(); i++) 
      for (int ci = 0 ; ci < Nimport; ci++) 
	for (size_t j = 0; j < pidx_send[ci][p].size(); j++) 
	  assert(pvec_send[p][i].local_idx != pidx_send[ci][p][j]);
#endif

  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pvec_send[p].size();
    pidx_send[Nimport][p].resize(np);
    for (int i = 0; i < np; i++) {
      pidx_send[Nimport][p][i] = pvec_send[p][i].local_idx;
    }
  }
  Nimport++;


  std::vector<particle> pvec_recv[NMAXPROC];
  
  //****** communicate particles
  
  myMPI_all2all<particle>(pvec_send, pvec_recv, myid, nproc, debug_flag);
  
  //***** add imported  particles to the system
  
  const int import_n_old = import_n;
  
  int cntr = import_n_old;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvec_recv[p].size(); i++) 
      cntr++;

  ////////  Stored particle's index from octbody vector

  std::vector<int> pvec_idx     (local_n);
  std::vector<int> pvec_mini_idx(local_n);
  for (int i = 0; i < local_n; i++) {
    pvec_idx     [i] = local_tree.body_list [i].pp->local_idx;
    pvec_mini_idx[i] = domain_tree.body_list[i].pp->local_idx;
  }
  std::vector<int> pvec_import_idx(import_n_old);
  for (int i = 0; i < import_n_old; i++) 
    pvec_import_idx[i] = import_tree.body_list[i].pp->local_idx;
  
  //////// Safely resize pvec vector, and correct octbody-pointers if necessary
  
  if (!safe_resize(pvec, local_n + cntr)) {
    for (int i = 0; i < local_n; i++) {
      local_tree.body_list [i].pp = &pvec[pvec_idx     [i]];
      domain_tree.body_list[i].pp = &pvec[pvec_mini_idx[i]];
    }
    for (int i = 0; i < import_n_old; i++) {
      import_tree.body_list[i].pp = &pvec[pvec_import_idx[i]];
    }
  };
  
  /////// import particles into pvec array
  
  cntr = import_n_old;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pvec_recv[p].size(); i++) 
      pvec[local_n + cntr++] = pvec_recv[p][i];
  
  import_n = pvec.size() - local_n;
  
  /////// safely resize octBody_import vector, and correct pointers if necessary


  const int ni = std::min(nproc, 4)*std::max(import_n, local_n);
  import_tree.resize_nodes(ni);
  import_tree.resize_bodies(ni);
  import_tree.insert(&pvec[local_n + import_n_old], import_n - import_n_old);

  import_tree.root.calculate_inner_boundary();
  
  //***** update local index
  
  for (int i = local_n; i < local_n + import_n; i++) {
    pvec[i].local_idx = i;
  }
  
  t_communicate = get_time() - t0;
}

///////////////////

void system::import_boundary_pmhd() {

  const double t0 = get_time();

  pmhd.resize(local_n);

  std::vector<ptcl_mhd> pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd> pmhd_recv[NMAXPROC];

  int imported = 0;
  for (int cimport = 0; cimport < Nimport; cimport++) {
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      const int np = pidx_send[cimport][p].size();
      pmhd_send[p].resize(np);
      for (int i = 0; i < np; i++) {
	int idx = pidx_send[cimport][p][i];
	assert(idx < local_n);
	pmhd_send[p][i] = pmhd[idx];				   
      }
    }
    myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);
    
    int cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pmhd_recv[p].size(); i++) 
	cntr++;

    pmhd.resize(local_n + cntr);
    
    cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pmhd_recv[p].size(); i++) 
	pmhd[local_n + cntr++] = pmhd_recv[p][i];

    imported = cntr;
  }
  
  assert(imported    == import_n);
  assert(pmhd.size() == pvec.size());

  t_import_boundary_pmhd = get_time() - t0;
  
  int cntr = 0;
  if (debug_flag) {
    std::vector<int> pgdx_recv[NMAXPROC];
    std::vector<int> pgdx_send[NMAXPROC];
    
    for (int cimport = 0; cimport < Nimport; cimport++) {
      for (int p = 0; p < nproc; p++) {
	if (p == myid) continue;
	const int np = pidx_send[cimport][p].size();
	pgdx_send[p].resize(np);
	for (int i = 0; i < np; i++) {
	  const int idx = pidx_send[cimport][p][i];
	  pgdx_send[p][i] = pvec[idx].global_idx;
	}
      }
      
      myMPI_all2all<int>(pgdx_send, pgdx_recv, myid, nproc, true);
      
      for (int p = 0; p < nproc; p++)
	for (size_t i = 0; i < pgdx_recv[p].size(); i++) {
	  assert(pvec[local_n + cntr].global_idx == pgdx_recv[p][i]);
	  cntr++;
	}
    }
  }
  
  t_communicate += get_time() - t0;
}

/////////

void system::import_boundary_Bmatrix() {
  const double t0 = get_time();

  Bxx.resize(local_n);
  Bxy.resize(local_n);
  Bxz.resize(local_n);
  Byy.resize(local_n);
  Byz.resize(local_n);
  Bzz.resize(local_n);
 
  std::vector<float> B_send[NMAXPROC];
  std::vector<float> B_recv[NMAXPROC];
  
  int imported = 0;
  for (int cimport = 0; cimport < Nimport; cimport++) {
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      const int np = pidx_send[cimport][p].size();
      B_send[p].resize(6*np);
      for (int i = 0; i < np; i++) {
	int idx = pidx_send[cimport][p][i];
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
    myMPI_all2all<float>(B_send, B_recv, myid, nproc, debug_flag);

    int cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < B_recv[p].size(); i += 6) 
	cntr++;
    
    Bxx.resize(local_n + cntr);
    Bxy.resize(local_n + cntr);
    Bxz.resize(local_n + cntr);
    Byy.resize(local_n + cntr);
    Byz.resize(local_n + cntr);
    Bzz.resize(local_n + cntr);
    
    cntr = imported;
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

    imported = cntr;
  }
    
  assert(imported   == import_n);
  assert(Bxx.size() == pvec.size());
  assert(Bxy.size() == pvec.size());
  assert(Bxz.size() == pvec.size());
  assert(Byy.size() == pvec.size());
  assert(Byz.size() == pvec.size());
  assert(Bzz.size() == pvec.size());
  
  t_communicate += get_time() - t0;

  t_import_boundary_Bmatrix = get_time() - t0;
}

///////////

void system::import_boundary_pmhd_grad() {
  const double t0 = get_time();
  
  pmhd_grad[0].resize(local_n);
  pmhd_grad[1].resize(local_n);
  pmhd_grad[2].resize(local_n);

  std::vector<ptcl_mhd> pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd> pmhd_recv[NMAXPROC];

  int imported = 0;
  for (int cimport = 0; cimport < Nimport; cimport++) {
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      const int np = pidx_send[cimport][p].size();
      pmhd_send[p].resize(3*np);
      for (int i = 0; i < np; i++) {
	int idx = pidx_send[cimport][p][i];
	assert(idx < local_n);
	pmhd_send[p][3*i  ] = pmhd_grad[0][idx];
	pmhd_send[p][3*i+1] = pmhd_grad[1][idx];
	pmhd_send[p][3*i+2] = pmhd_grad[2][idx];
      }
    }
    myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);

    int cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pmhd_recv[p].size(); i += 3) 
	cntr++;
    
    pmhd_grad[0].resize(local_n + cntr);
    pmhd_grad[1].resize(local_n + cntr);
    pmhd_grad[2].resize(local_n + cntr);

    cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pmhd_recv[p].size(); i += 3) {
	pmhd_grad[0][local_n + cntr] = pmhd_recv[p][i  ];
	pmhd_grad[1][local_n + cntr] = pmhd_recv[p][i+1];
	pmhd_grad[2][local_n + cntr] = pmhd_recv[p][i+2];
	cntr++;
      }

    imported = cntr;
  }
    
  assert(imported            == import_n);
  assert(pmhd_grad[0].size() == pvec.size());
  assert(pmhd_grad[1].size() == pvec.size());
  assert(pmhd_grad[2].size() == pvec.size());
  
  t_communicate += get_time() - t0;

  t_import_boundary_pmhd_grad = get_time() - t0;
}

/////////

void system::import_boundary_pvel() {
  const double t0 = get_time();

  std::vector<float>  pvel_send[NMAXPROC];
  std::vector<float>  pvel_recv[NMAXPROC];
  
  
  int imported = 0;
  for (int cimport = 0; cimport < Nimport; cimport++) {
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      const int np = pidx_send[cimport][p].size();
      pvel_send[p].resize(3*np);
      for (int i = 0; i < np; i++) {
	int idx = pidx_send[cimport][p][i];
	assert(idx < local_n);
	pvel_send[p][3*i  ] = pvec[idx].vel.x;
	pvel_send[p][3*i+1] = pvec[idx].vel.y;
	pvel_send[p][3*i+2] = pvec[idx].vel.z;
      }
    }
    myMPI_all2all<float>(pvel_send, pvel_recv, myid, nproc, debug_flag);
    
    int cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pvel_recv[p].size(); i += 3) 
	cntr++;
    
    cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pvel_recv[p].size(); i += 3) {
	pvec[local_n + cntr].vel.x = pvel_recv[p][i  ];
	pvec[local_n + cntr].vel.y = pvel_recv[p][i+1];
	pvec[local_n + cntr].vel.z = pvel_recv[p][i+2];
	cntr++;
      }
    
    imported = cntr;
  }
  
  assert(imported == import_n);

  t_communicate += get_time() - t0;

  t_import_boundary_pvel = get_time() - t0;
}

/////////

void system::import_boundary_wght() {
  const double t0 = get_time();

  std::vector<float> pwght_send[NMAXPROC];
  std::vector<float> pwght_recv[NMAXPROC];
  
  int imported = 0;
  for (int cimport = 0; cimport < Nimport; cimport++) {
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      const int np = pidx_send[cimport][p].size();
      pwght_send[p].resize(2*np);
      for (int i = 0; i < np; i++) {
	int idx = pidx_send[cimport][p][i];
	assert(idx < local_n);
	pwght_send[p][2*i  ] = pvec[idx].wght;
	pwght_send[p][2*i+1] = pvec[idx].h;
      }
    }
    myMPI_all2all<float>(pwght_send, pwght_recv, myid, nproc, debug_flag);
    
    int cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pwght_recv[p].size(); i += 2) 
	cntr++;
    
    cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pwght_recv[p].size(); i += 2) {
	pvec[local_n + cntr].wght = pwght_recv[p][i  ];
	pvec[local_n + cntr].h    = pwght_recv[p][i+1];
	cntr++;
      }
    
    imported = cntr;
  }
  
  assert(imported == import_n);

  t_import_boundary_wght = get_time() - t0;
}


//////////

void system::import_boundary_pmhd_cross() {
  const double t0 = get_time();
  

  for (int i = 0; i < 6; i++) pmhd_cross[i].resize(local_n);

  std::vector<ptcl_mhd>  pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd>  pmhd_recv[NMAXPROC];
  
  int imported = 0;
  for (int cimport = 0; cimport < Nimport; cimport++) {
    for (int p = 0; p < nproc; p++) {
      if (p == myid) continue;
      const int np = pidx_send[cimport][p].size();
      pmhd_send[p].resize(6*np);
      for (int i = 0; i < np; i++) {
	int idx = pidx_send[cimport][p][i];
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
    myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);

    int cntr = imported;
    for (int p = 0; p < nproc; p++)
      for (size_t i = 0; i < pmhd_recv[p].size(); i += 6) 
	cntr++;
    
    for (int k = 0; k < 6; k++)
      pmhd_cross[k].resize(local_n + cntr);
    
    cntr = imported;
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
    
    imported = cntr;
  }
  
  assert(imported == import_n);
  assert(pmhd_cross[0].size() == pvec.size());
  assert(pmhd_cross[1].size() == pvec.size());
  assert(pmhd_cross[2].size() == pvec.size());
  assert(pmhd_cross[3].size() == pvec.size());
  assert(pmhd_cross[4].size() == pvec.size());
  assert(pmhd_cross[5].size() == pvec.size());

  t_communicate += get_time() - t0;

  t_import_boundary_pmhd_cross = get_time() - t0;
}
