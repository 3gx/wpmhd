#include "gn.h"
#include "myMPI.h"

void system::improve_weights() {

  weights.resize(pvec.size());
  for (size_t i = 0; i < pvec.size(); i++)
    weights[i] = pvec[i].wght;
  return;
  
  const int n_groups = group_list.size();
  for (int group = 0; group < n_groups; group++) {
    const octnode &inode              =   *group_list[group];
    const int n_leaves                = ngb_leaf_list_outer[group].size();
    std::vector<octnode*> &ileaf_list = ngb_leaf_list_outer[group];
    
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
   
      float wght_i = 0.0f;

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
 	  if (s2 <= hi2 || s2 <= hj2)  {

	    const float inv_hj    = 1.0f/hj;
	    const float inv_hjdim = kernel.pow(inv_hj);
	    
	    const real wj   = pj.wght;
	    
	    const float s   = sqrt(s2);
	    const float wki = kernel.w(s*inv_hi)*inv_hidim;
	    const float wkj = kernel.w(s*inv_hj)*inv_hjdim;
	    
// 	    wght_i += 0.25f*(sqr(wi) + sqr(wj))*(wki + wkj);
//  	    wght_i += 0.5f*(sqr(wi)*wki + sqr(wj)*wkj);
	    wght_i += sqr(wj)*wkj;
	    
	  }
	}
      }

      weights[pi.local_idx] = wght_i;
      weights[pi.local_idx] = pi.wght;
    }
  }

  //////////// exchange weights
  
  std::vector<float> weights_send[NMAXPROC];
  std::vector<float> weights_recv[NMAXPROC];
  
  for (int p = 0; p < nproc; p++) {
    if (p == myid) continue;
    const int np = pidx_send[p].size();
    weights_send[p].resize(np);
    for (int i = 0; i < np; i++) {
      int idx = pidx_send[p][i];
      assert(idx < local_n);
      weights_send[p][i] = weights[idx];
    }
  }
  
#if _DEBUG_
  const bool debug_flag = true;
#else
  const bool debug_flag = false;
#endif
  myMPI_all2all<float>(weights_send, weights_recv, myid, nproc, debug_flag);

  int cntr = 0;
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < weights_recv[p].size(); i++) {
      weights[local_n + cntr++]= weights_recv[p][i];
    }
  
  assert((size_t)cntr == pvec.size() - local_n);
  
}
