#include "gn.h"
#include "myMPI.h"

void system::distribute_particles() {

  double t0 = get_time();

  recalculate_domain_boundaries = false; 
#if 1
  recalculate_domain_boundaries = true; 
#endif
  if (recalculate_domain_boundaries) {
    std::vector<pfloat3> sample_pos;
    determine_sample_freq();
    //   fprintf(stderr, "proc= %d  sample_freq= %d\n", myid, sample_freq);
    collect_sample_coords(sample_pos);
    if (myid == 0) {
      box.determine_division(sample_pos);
    }
    MPI_Bcast(box.ntile_per_proc, nproc, MPI_INT, 0, MPI_COMM_WORLD);
    myMPI_Bcast<boundary>(box.inner_tiles, 0, nproc);
    for (int proc = 0; proc < nproc; proc++) 
      myMPI_Bcast<int>(box.procs_tiles[proc], 0, nproc);
  }
  
  
  pvec.resize(local_n);
  pmhd.resize(local_n);
  pmhd_dot.resize(local_n);

  int iloc = 0;
  for (int i = 0; i < local_n; i++)  
    if (box.isinproc(pvec[i].pos, myid)) {
      std::swap(pvec    [i], pvec    [iloc  ]);
      std::swap(pmhd    [i], pmhd    [iloc  ]);
      std::swap(pmhd_dot[i], pmhd_dot[iloc++]);
    }
  
#if _DEBUG_
  for (int i = 0;    i < iloc;    i++) assert( box.isinproc(pvec[i].pos, myid));  
  for (int i = iloc; i < local_n; i++) assert(!box.isinproc(pvec[i].pos, myid));
#endif
  
  std::vector<particle> pvec_send[NMAXPROC];
  std::vector<particle> pvec_recv[NMAXPROC];
  std::vector<ptcl_mhd> pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd> pmhd_recv[NMAXPROC];
  
  for (int p= 0; p < nproc; p++) {
    pvec_send[p].reserve(128);
    pmhd_send[p].reserve(128);
  }
  
  for (int i = iloc; i < local_n; i++) {
    const int proc = box.which_proc(pvec[i].pos);
    assert(proc >= 0);
    
    pvec_send[proc].push_back(pvec    [i]);
    pmhd_send[proc].push_back(pmhd    [i]);
    pmhd_send[proc].push_back(pmhd_dot[i]);
  } 

  myMPI_all2all<particle>(pvec_send, pvec_recv, myid, nproc, debug_flag);
  myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);
  
  int nrecv = 0;
  for (int p = 0; p < nproc; p++) 
    nrecv += pvec_recv[p].size();
  
  pvec.resize    (iloc + nrecv);
  pmhd.resize    (iloc + nrecv);
  pmhd_dot.resize(iloc + nrecv);
  
  for (int p = 0; p < nproc; p++) 
    for (size_t i = 0; i < pvec_recv[p].size(); i++) {
      pvec    [iloc  ] = pvec_recv[p][i    ];
      pmhd    [iloc  ] = pmhd_recv[p][2*i  ];
      pmhd_dot[iloc++] = pmhd_recv[p][2*i+1];
    }
  
  local_n  = iloc;
  import_n = 0;

  assert(iloc == (int)pvec.size());
  assert(iloc == (int)pmhd.size());
  assert(iloc == (int)pmhd_dot.size());
  assert(pmhd.size() == pvec.size());
  
  for (int i = 0; i < iloc; i++) 
    assert(box.isinproc(pvec[i].pos, myid));
  
  /////////

  int nglob, nloc = local_n;
  MPI_Allreduce(&nloc, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  assert(nglob == global_n);
  
  //*******  update local index
  
  for (int i = 0;  i < local_n; i++) {
    pvec[i].local_idx = i;
  }
  
  import_tree.clear();
  import_tree.resize(local_n, local_n);
  import_tree.set_domain(global_domain);
  
  //////////////

  for (int proc = 0; proc < nproc; proc++) {
    pidx_sent[proc].clear();
  }
  
#if 0
  MPI_Barrier(MPI_COMM_WORLD);
  exit(-1);
#endif
  
  import_n = 0;
  Nimport = 0;

  t_distribute_particles = get_time() - t0;
};
