#include "gn.h"
#include "myMPI.h"

void system::distribute_particles() {

  recalculate_domain_boundaries = true; 
//   recalculate_domain_boundaries = false; 
  if (recalculate_domain_boundaries) {
    std::vector<pfloat3> sample_pos;
    determine_sample_freq();
    //   fprintf(stderr, "proc= %d  sample_freq= %d\n", myid, sample_freq);
    collect_sample_coords(sample_pos);
    if (myid == 0) {
      box.determine_division(sample_pos);
    }
    myMPI_Bcast<boundary>(box.bnd_vec, 0, nproc);
  }
  double t0 = get_time();

  pvec.resize(local_n);
  pmhd.resize(local_n);
  pmhd_dot.resize(local_n);
  pmhd_dot0.resize(local_n);
  pvel0.resize(local_n);

#ifdef _DEBUG_PRINT_
  fprintf(stderr, "proc= %d  local_n= %d\n", myid, local_n);
#endif
  const boundary bnd = box.get_bnd(myid);
  
  int iloc = 0;
  for (int i = 0; i < local_n; i++)  
    if (bnd.isinbox(pvec [i].pos)) {
      std::swap(pvec     [i], pvec     [iloc  ]);
      std::swap(pmhd     [i], pmhd     [iloc  ]);
      std::swap(pvel0    [i], pvel0    [iloc  ]);
      std::swap(pmhd_dot [i], pmhd_dot [iloc  ]);
      std::swap(pmhd_dot0[i], pmhd_dot0[iloc++]);
    }
  
#if _DEBUG_
  for (int i = 0;    i < iloc;    i++) assert( bnd.isinbox(pvec[i].pos));  
  for (int i = iloc; i < local_n; i++) assert(!bnd.isinbox(pvec[i].pos));
#endif
  
  std::vector<particle> pvec_send[NMAXPROC];
  std::vector<particle> pvec_recv[NMAXPROC];
  std::vector<ptcl_mhd> pmhd_send[NMAXPROC];
  std::vector<ptcl_mhd> pmhd_recv[NMAXPROC];
  std::vector<ptcl_mhd> pdot_send[NMAXPROC];
  std::vector<ptcl_mhd> pdot_recv[NMAXPROC];
  std::vector<ptcl_mhd> pdot0_send[NMAXPROC];
  std::vector<ptcl_mhd> pdot0_recv[NMAXPROC]; 
  std::vector<float3> pvel0_send[NMAXPROC];
  std::vector<float3> pvel0_recv[NMAXPROC];
 
  for (int p= 0; p < nproc; p++) {
    pvec_send[p].reserve(128);
    pmhd_send[p].reserve(128);
    pdot_send[p].reserve(128);
    pdot0_send[p].reserve(128);
    pvel0_send[p].reserve(128);
  }
  
  for (int i = iloc; i < local_n; i++) {
    const int ibox = box.which_box(pvec[i].pos);
    assert(ibox >= 0);
    
    pvec_send[ibox].push_back(pvec    [i]);
    pmhd_send[ibox].push_back(pmhd    [i]);
    pdot_send[ibox].push_back(pmhd_dot[i]);
    pdot0_send[ibox].push_back(pmhd_dot0[i]);
    pvel0_send[ibox].push_back(pvel0[i]);
  } 

#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<particle>(pvec_send, pvec_recv, myid, nproc, debug_flag);
  myMPI_all2all<ptcl_mhd>(pmhd_send, pmhd_recv, myid, nproc, debug_flag);
  myMPI_all2all<ptcl_mhd>(pdot_send, pdot_recv, myid, nproc, debug_flag);
  myMPI_all2all<ptcl_mhd>(pdot0_send, pdot0_recv, myid, nproc, debug_flag);
  myMPI_all2all<float3>  (pvel0_send, pvel0_recv, myid, nproc, debug_flag);
  
  int nrecv = 0;
  for (int p = 0; p < nproc; p++) 
    nrecv += pvec_recv[p].size();
  
  pvec.resize    (iloc + nrecv);
  pmhd.resize    (iloc + nrecv);
  pmhd_dot.resize(iloc + nrecv);
  pmhd_dot0.resize(iloc + nrecv);
  pvel0.resize(iloc + nrecv);

  for (int p = 0; p < nproc; p++) 
    for (size_t i = 0; i < pvec_recv[p].size(); i++) {
      pvec     [iloc  ] = pvec_recv [p][i];
      pmhd     [iloc  ] = pmhd_recv [p][i];
      pvel0    [iloc  ] = pvel0_recv[p][i];
      pmhd_dot [iloc  ] = pdot_recv [p][i];
      pmhd_dot0[iloc++] = pdot0_recv[p][i];
    }
  
  local_n  = iloc;
  import_n = 0;

  assert(iloc == (int)pvec.size());
  assert(iloc == (int)pmhd.size());
  assert(iloc == (int)pmhd_dot.size());
  assert(iloc == (int)pmhd_dot0.size());
  assert(iloc == (int)pvel0.size());
  assert(pmhd.size() == pvec.size());
  
  for (int i = 0; i < iloc; i++) 
    assert(bnd.isinbox(pvec[i].pos));
  
  /////////

  int nglob, nloc = local_n;
  MPI_Allreduce(&nloc, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  assert(nglob == global_n);
  
  //  update local index
  
  for (int i = 0;  i < local_n; i++) {
    pvec[i].local_idx = i;
  }
  
  pvec.reserve(2*local_n);
  pmhd.reserve(2*local_n);
  octp.reserve(2*local_n);
  
  for (int p = 0; p < nproc; p++)
    pidx_send[p].clear();

  t_distribute = get_time() - t0;
};
