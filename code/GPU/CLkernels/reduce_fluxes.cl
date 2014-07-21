#ifndef _REDUCE_FLUXES_CL_
#define _REDUCE_FLUXES_CL_


__kernel void reduce_fluxes(__global float4      *out_fluxes1,	 //  0
			    __global float4      *out_fluxes2,   //  1
			    __global float4      *out_fluxes3,   //  2
			    __global float4      *out_divB,      //  3
			    __global int         *leaf_list,     //  4
			    __global int2        *body_list,     //  5
			    __global int         *ilist,         //  6
			    __global int         *ijlist_offset, //  7
			    __global int         *leaf_ngb_max,  //  8
			    __global float4      *in_fluxes1,    //  9
			    __global float4      *in_fluxes2,    // 10
			    __global float4      *in_fluxes3,    // 11
			    __global float4      *in_divB) {     // 12

  // Get threadIdx, blockIdx & blockDim
  const int threadIdx = get_local_id(0);
  const int blockIdx  = get_group_id(0);
  const int blockDim  = get_local_size(0);

  // Get bodyId
  const int leaf_id  = leaf_list[blockIdx];
  const int2 ifirst  = body_list[leaf_id];
  const int  ni      = ifirst.y;
  const int  bodyIdx = ifirst.x + min(threadIdx, ni-1); 
  const int  bodyId  = ilist[bodyIdx];
  
  // Compute offset where to write neighbours idx
  const int ijoffset    = ijlist_offset[leaf_id] + min(threadIdx, ni-1);
  const int nj          = leaf_ngb_max [leaf_id];

  const int ngb_max = ni * nj;
  
  float4 fluxes1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 fluxes2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 fluxes3 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 divB    = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int i = 0; i < ngb_max; i += ni) {
    const int iaddr = ijoffset + i;
    fluxes1 += in_fluxes1[iaddr];
    fluxes2 += in_fluxes2[iaddr];
    fluxes3 += in_fluxes3[iaddr];
    divB    += in_divB   [iaddr];
  }
  
  if (threadIdx < ni) {
    out_fluxes1[bodyId] = fluxes1;
    out_fluxes2[bodyId] = fluxes2;
    out_fluxes3[bodyId] = fluxes3;
    out_divB   [bodyId] = divB;
  }
}

#endif
