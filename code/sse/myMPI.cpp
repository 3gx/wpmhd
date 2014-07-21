#include "myMPI.h"

static MPI_Datatype MPI_FLOAT3   = 0;
static MPI_Datatype MPI_PARTICLE = 0;
static MPI_Datatype MPI_BOUNDARY = 0;
static MPI_Datatype MPI_PTCL_MHD = 0;

template <> MPI_Datatype datatype<int>      () {
  return MPI_INT;
}

template <> MPI_Datatype datatype<float>      () {
  return MPI_FLOAT;
}

template <> MPI_Datatype datatype<double>      () {
  return MPI_DOUBLE;
}

template <> MPI_Datatype datatype<float3>      () {
  if (MPI_FLOAT3) return MPI_FLOAT3;
  else {
    int ss = sizeof(float3) / sizeof(float);
    assert(0 == sizeof(float3) % sizeof(float));
    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_FLOAT3);
    MPI_Type_commit(&MPI_FLOAT3);
    return MPI_FLOAT3;
  }
}

template <> MPI_Datatype datatype<boundary>      () {
  if (MPI_BOUNDARY) return MPI_BOUNDARY;
  else {
    int ss = sizeof(boundary) / sizeof(float);
    assert(0 == sizeof(boundary) % sizeof(float));
    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_BOUNDARY);
    MPI_Type_commit(&MPI_BOUNDARY);
    return MPI_BOUNDARY;
  }
}

template <> MPI_Datatype datatype<particle> () {
  
  if (MPI_PARTICLE) return MPI_PARTICLE;
  else {
    int ss = sizeof(particle) / sizeof(real);
    assert(0 == sizeof(particle) % sizeof(real));
#ifdef DOUBLE_PRECISION
    MPI_Type_contiguous(ss, MPI_DOUBLE, &MPI_PARTICLE);
#else
    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_PARTICLE);
#endif
    MPI_Type_commit(&MPI_PARTICLE);
    return MPI_PARTICLE;
  }
  
}

template <> MPI_Datatype datatype<ptcl_mhd> () {
  
  if (MPI_PTCL_MHD) return MPI_PTCL_MHD;
  else {
    int ss = sizeof(ptcl_mhd) / sizeof(real);
    assert(0 == sizeof(ptcl_mhd) % sizeof(real));
#ifdef DOUBLE_PRECISION
    MPI_Type_contiguous(ss, MPI_DOUBLE, &MPI_PTCL_MHD);
#else
    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_PTCL_MHD);
#endif
    MPI_Type_commit(&MPI_PTCL_MHD);
    return MPI_PTCL_MHD;
  }
  
}

