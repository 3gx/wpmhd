#include "gn.h"
#include "myMPI.h"

void system::compute_defect() {
#if 0
  pdefect.resize(pvec.size());
  for (size_t i = 0; i < pvec.size(); i++) 
    pdefect[i] = (float3){0.0f, 0.0f, 0.0f};
  return;
  
  const int n_groups = group_list.size();
  for (int group = 0; group < n_groups; group++) {
    const octnode &inode              =   *group_list[group];
#if 0
    const int n_leaves                = ngb_leaf_list[group].size();
    std::vector<octnode*> &ileaf_list = ngb_leaf_list[group];
#else
    const int n_leaves                = ngb_leaf_list_outer[group].size();
    std::vector<octnode*> &ileaf_list = ngb_leaf_list_outer[group];
#endif
    
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (ibp->isexternal()) continue;
      const particle &pi = *ibp->pp;
      
      const boundary  bi = boundary(pi.pos, pi.h);
      const pfloat3 ipos = pi.pos;
      
      const float   hi      = pi.h;
      const float   wi      = pi.wght;
      
      const float   hi2     = hi*hi;
      const float inv_hi    = 1.0f/hi;
      const float inv_hidim = kernel.pow(inv_hi);
   
      float3 dw_ii = {0.0f, 0.0f, 0.0f};
      int    ngb_i = 0;

      for (int leaf = 0; leaf < n_leaves; leaf++) {
 	const octnode &jnode = *ileaf_list[leaf];
	
 	if (!boundary(bi).overlap(jnode.inner)) continue;
	
	for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	  const particle &pj = *jbp->pp;

	  const pfloat3 jpos = pj.pos; 
	  const float3  dr   = {jpos.x - ipos.x,
				jpos.y - ipos.y,
				jpos.z - ipos.z};
	  const float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	  
 	  const float hj  = pj.h;
 	  const float hj2 = hj*hj;
//   	  if (s2 <= hi2 && s2 <= hj2)  {
 	  if (s2 <= hi2 || s2 <= hj2)  {

	    const int li = pi.local_idx;
	    const int lj = pj.local_idx;
	    
	    const float inv_hj    = 1.0f/hj;
	    const float inv_hjdim = kernel.pow(inv_hj);
	    
	    const real wj   = pj.wght;
	    
	    const float s   = sqrt(s2);
	    const float wki = kernel.w(s*inv_hi)*inv_hidim * wi;
	    const float wkj = kernel.w(s*inv_hj)*inv_hjdim * wj;
	    
	    const float Vi = weights[li];
	    const float Vj = weights[lj];

	    const real dwi[3] = {wki * (Bxx[li]*dr.x + Bxy[li]*dr.y + Bxz[li]*dr.z),
				 wki * (Bxy[li]*dr.x + Byy[li]*dr.y + Byz[li]*dr.z),
				 wki * (Bxz[li]*dr.x + Byz[li]*dr.y + Bzz[li]*dr.z)};
	    
	    const real dwj[3] = {wkj * (Bxx[lj]*dr.x + Bxy[lj]*dr.y + Bxz[lj]*dr.z),
				 wkj * (Bxy[lj]*dr.x + Byy[lj]*dr.y + Byz[lj]*dr.z),
				 wkj * (Bxz[lj]*dr.x + Byz[lj]*dr.y + Bzz[lj]*dr.z)};

	    const real dwij[3] = {Vi*dwi[0] + Vj*dwj[0],
				  Vi*dwi[1] + Vj*dwj[1],
				  Vi*dwi[2] + Vj*dwj[2]};
	    
	    dw_ii.x += dwij[0];
	    dw_ii.y += dwij[1];
	    dw_ii.z += dwij[2];
	    ngb_i   += 1;
	  }
	}
      }

#if 0
      fprintf(stdout, "proc= %d:  i= %d ngb_i= %d\n",
	      myid, pi.local_idx, ngb_i);
#endif

      ngb_i = 1;
      pdefect[pi.local_idx] = (float3){dw_ii.x/ngb_i, dw_ii.y/ngb_i, dw_ii.z/ngb_i};
//       if (fabs(dw_ii.x) + fabs(dw_ii.y) + fabs(dw_ii.z) > 1.0e-2) {
// 	fprintf(stdout, "i= %d  : %g %g %g  %d \n",
// 		pi.local_idx, dw_ii.x, dw_ii.y, dw_ii.z, ngb_i);
//       }
    }
  }

//   exit(-1);
//   for (int i = 0; i < local_n; i++) 
//     pdefect[i] = (float3){0,0,0};

  //////////// exchange defects
  
  std::vector<float> pdefect_send[NMAXPROC];
  std::vector<float> pdefect_recv[NMAXPROC];
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pidx_send[p].size();
    pdefect_send[p].resize(3*np);
    for (int i = 0; i < np; i++) {
      int idx = pidx_send[p][i];
      assert(idx < local_n);
      pdefect_send[p][3*i  ] = pdefect[idx].x;
      pdefect_send[p][3*i+1] = pdefect[idx].y;
      pdefect_send[p][3*i+2] = pdefect[idx].z;
    }
  }
  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<float>(pdefect_send, pdefect_recv, myid, nproc, debug_flag);

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < pdefect_recv[p].size(); i += 3) {
      pdefect[local_n + cntr].x = pdefect_recv[p][i  ];
      pdefect[local_n + cntr].y = pdefect_recv[p][i+1];
      pdefect[local_n + cntr].z = pdefect_recv[p][i+2];
      cntr++;
    }
  
  assert((size_t)cntr == pvec.size() - local_n);
#endif  
}
