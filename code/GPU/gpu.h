#ifndef _GPU_H_
#define _GPU_H_

#define NGPUBLOCKS 120
#define NBLOCKDIM  64

#include "my_ocl.h"

struct gpu_struct {
  my_dev::context context;

  my_dev::kernel compute_wngb;
  my_dev::kernel compute_wngb_dr;
  my_dev::kernel compute_nngb;
  my_dev::kernel extract_gather_list;
  my_dev::kernel compute_Bmatrix;
  my_dev::kernel compute_gradient;
  my_dev::kernel limit_gradient;
  my_dev::kernel limit_gradient2;

  my_dev::kernel extract_ijlist;
  my_dev::kernel compute_dwij;
  my_dev::kernel compute_states;
  my_dev::kernel compute_fluxes;
  my_dev::kernel reduce_fluxes;

  

  float4 domain_hsize;
  
  // tree-data
  my_dev::dev_mem<int>  leaf_list;    // list of leaves to process, size = #NGPUBLOCKS
  my_dev::dev_mem<int2> body_list;    // ptcl info in leaves, (ifirst, ni), size = #leaves
  my_dev::dev_mem<int>  ilist;

  my_dev::dev_mem<int4> in_group_bodies;
  my_dev::dev_mem<int2> in_bodyIds;

  // neighbour info
  my_dev::dev_mem<int> jlist, joffset;
  my_dev::dev_mem<int2> ijlist;
  my_dev::dev_mem<int> ijlist_offset, leaf_ngb_max;

  my_dev::dev_mem<int>  in_ilist, in_jlist, jidx, nj;
  my_dev::dev_mem<int2> in_joffset;

  /////////////

  // position (.w - h),  velocity (.w - wght) / local & imported
  my_dev::dev_mem<float4> ppos, pvel, drmean;

  //////////

  // Weights
  my_dev::dev_mem<float2> in_h_out_wngb;

  // Renormalisation matrix / local & imported
  my_dev::dev_mem<float> Bxx, Bxy, Bxz, Byy, Byz, Bzz;
  my_dev::dev_mem<float> dwdt;
  
  // #neighbour info / local
  my_dev::dev_mem<int2> nngb;   // both = gather & scater
  my_dev::dev_mem<int>  nj_gather, nj_both;
  
  ///////// MHD DATA
  
  my_dev::dev_mem<float4> mhd1, mhd2, mhd3;
  // mhd1 = (velx, vely, velz, dens)
  // mhd2 = (Bx,   By,   Bz,   pres)
  // mhd3 = (psi,  s1, s2, s3)     , i.e. div clean data and 3 extra scalars

  // gradient data
  my_dev::dev_mem<float4> mhd1_grad_x, mhd1_grad_y, mhd1_grad_z;
  my_dev::dev_mem<float4> mhd2_grad_x, mhd2_grad_y, mhd2_grad_z;
  my_dev::dev_mem<float4> mhd3_grad_x, mhd3_grad_y, mhd3_grad_z;
  my_dev::dev_mem<float4> mhd1_psi, mhd2_psi, mhd3_psi;

  // interaction data
  my_dev::dev_mem<float4> dwij, drij;
  my_dev::dev_mem<float2> dndt_ij;
  my_dev::dev_mem<float4> mhd1_statesL, mhd1_statesR;
  my_dev::dev_mem<float4> mhd2_statesL, mhd2_statesR;
  my_dev::dev_mem<float4> mhd3_statesL, mhd3_statesR;
  my_dev::dev_mem<float4> fluxes1, fluxes2, fluxes3, divBij;
  my_dev::dev_mem<float4> dqdt1, dqdt2, dqdt3, divB, dndt;

  void init();
  void finialize();
  void load_kernels();

};
#endif // _GPU_H_
