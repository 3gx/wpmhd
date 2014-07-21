#include "gn.h"

#define NMAXSAMPLE 10000

void system::determine_sample_freq() {
  const int maxsample = (int)(NMAXSAMPLE*0.8f); // 0.8 is safety factor 
  sample_freq = (global_n + maxsample - 1)/maxsample;
}

void system::collect_sample_coords(std::vector<pfloat3> &sample_pos) {
  sample_pos.clear();
  sample_pos.reserve(128);
  const int n = pvec.size();
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
//       assert(status == MPI_SUCCESS);
    }
  }

  //

//   pfloat3 tmp = 0;
//   for(i = 0;i<nbody; i++){
//     vector3 r= pb[i].get_pos();
//     for(int k=0;k<3;k++)if(fabs(r[k])>tmp) tmp=fabs(r[k]);
//   }
//   rmax = MP_doublemax(tmp);
  
  

}
