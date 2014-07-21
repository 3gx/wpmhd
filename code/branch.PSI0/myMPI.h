#ifndef _myMPI_H_
#define _myMPI_H_

#include "primitives.h"
#include "boundary.h"

template<class T>
MPI_Datatype datatype();

template<class T>
void  myMPI_all2all(std::vector<T> sbuf[], std::vector<T> rbuf[], 
		    const int myid, 
		    const int nproc,
		    bool DEBUG = false) {

#if 1
  for (int dist = 1; dist < nproc; dist++) {
    int src = (nproc + myid - dist) % nproc;
    int dst = (nproc + myid + dist) % nproc;
    int scount = sbuf[dst].size();
    int rcount = 0;
    MPI_Status stat;
    MPI_Sendrecv(&scount, 1, MPI_INT, dst, 0,
		 &rcount, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &stat);
    rbuf[src].resize(rcount);
    MPI_Sendrecv(&sbuf[dst][0], scount, datatype<T>(), dst, 1,
		 &rbuf[src][0], rcount, datatype<T>(), src, 1, MPI_COMM_WORLD, &stat);

  }
#elif 0

  int scount = sbuf[dst].size();
  int rcount = 0;
  MPI_Alltoall(
#else
  MPI_Status  stat[NMAXPROC  ];
  MPI_Request  req[NMAXPROC*2];
  int        nrecv[NMAXPROC  ];

  int nreq = 0;
  
  fprintf(stderr, "Isend 1\n");
  for (int p = 0; p < nproc; p++) {
    int nsend = sbuf[p].size();
    MPI_Isend(&nsend, 1, MPI_INT, myid, 0, MPI_COMM_WORLD, &req[nreq++]);
  }

  fprintf(stderr, "Irecv 1\n");
  for (int p = 0; p < nproc; p++) {
    MPI_Irecv(&nrecv[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &req[nreq++]);
  }
  fprintf(stderr, "Waitall 1\n");
  MPI_Waitall(nreq, req, stat);

  nreq = 0;
  fprintf(stderr, "Isend 2\n");
  for (int p = 0; p < nproc; p++) {
    int nsend = sbuf[p].size();
    MPI_Isend(&sbuf[p][0], nsend, datatype<T>(), p, 0, MPI_COMM_WORLD, &req[nreq++]);
  }
  fprintf(stderr, "Irecv 2\n");
  for (int p = 0; p < nproc; p++) {
    rbuf[p].resize(nrecv[p]);
    MPI_Irecv(&rbuf[p][0], nrecv[p], datatype<T>(), p, 0, MPI_COMM_WORLD, &req[nreq++]);
  }
  fprintf(stderr, "Waitall 2\n");
  MPI_Waitall(nreq, req, stat);
#endif

  if (DEBUG) {
    int nsend_recv_loc[2] = {0,0};
    for (int p = 0; p < nproc; p++) {
      nsend_recv_loc[0] += sbuf[p].size();
      nsend_recv_loc[1] += rbuf[p].size();
    }
    
    int nsend_recv_glob[2];
    MPI_Allreduce(&nsend_recv_loc, &nsend_recv_glob, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    assert(nsend_recv_glob[0] == nsend_recv_glob[1]);
    
  }

}

template<class T>
void myMPI_allgather(T &sbuf, std::vector<T> &rbuf, 
		     const int myid, 
		     const int nproc) {
  
  rbuf.resize(nproc);
  MPI_Allgather(&sbuf,    1, datatype<T>(), 
		&rbuf[0], 1, datatype<T>(), 
		MPI_COMM_WORLD);
  
}

template<class T>
  void myMPI_Bcast(std::vector<T> &buf,
		   const int myid, 
		   const int nproc) {
  
  int n = buf.size();
  MPI_Bcast(&n, 1, MPI_INT, myid, MPI_COMM_WORLD);
  buf.resize(n);
  MPI_Bcast(&buf[0], n, datatype<T>(), myid, MPI_COMM_WORLD);
}




#endif // _myMPI_H_
