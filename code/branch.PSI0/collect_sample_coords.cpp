#include "gn.h"
#include "myMPI.h"

#define NMAXSAMPLE 30000

void system::determine_sample_freq() {
  const int maxsample = (int)(NMAXSAMPLE*0.8f); // 0.8 is safety factor 
  sample_freq = (global_n + maxsample - 1)/maxsample;
}

void system::collect_sample_coords(std::vector<pfloat3> &sample_pos) {
  sample_pos.clear();
  sample_pos.reserve(128);
  const int n = local_n;
  for (int i = 0; i < n; i += sample_freq) {
    sample_pos.push_back(pvec[i].pos);
  }
  
  MPI_Status status;
  int nsample = sample_pos.size();
  
  // exchange samples

  if (myid != 0) {
    MPI_Send(&nsample, 1, MPI_INT, 0, myid*2 , MPI_COMM_WORLD);
    MPI_Send((float*)&sample_pos[0], nsample*3, MPI_FLOAT, 0, myid*2+1,  MPI_COMM_WORLD);
  } else {
    for (int p = 1; p < nproc; p++) {
      int nreceive;
      MPI_Recv(&nreceive, 1, MPI_INT, p, p*2, MPI_COMM_WORLD, &status);
      sample_pos.resize(nsample + nreceive);
      MPI_Recv((float*)&sample_pos[nsample], 3*nreceive, MPI_FLOAT, p, p*2+1, MPI_COMM_WORLD, &status);
      nsample += nreceive;
    }
  }
}

//////////////

void system::calculate_outer_domain_boundaries(const float scale_factor) {
  const double t0 = get_time();
  
  for (int i = 0; i < local_n; i++) domain_tree.body_list[i].update();
  domain_tree.root.calculate_outer_boundary(scale_factor);
  outer_tiles[myid].resize(domain_tree.leaf_list.size());
  
  for (size_t i = 0; i < domain_tree.leaf_list.size(); i++) {
    outer_tiles[myid][i] = domain_tree.leaf_list[i]->outer;
  }
  for (int proc = 0; proc < nproc; proc++) {
    myMPI_Bcast<boundary>(outer_tiles[proc], proc, nproc);
  }

  t_communicate = get_time() - t0;
}
