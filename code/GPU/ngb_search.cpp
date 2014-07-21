#include "gn.h"

void system::ngb_search() {


  std::vector<int> ngb_list(local_n);

//   MPI_Barrier(MPI_COMM_WORLD);
//   double t0 = get_time();
  
  
  std::vector<std::pair<octnode*, octnode*> > interaction_list;
  interaction_list.reserve(128);
  
  leaf_leaf_interaction(root_node_local, root_node_local, interaction_list);

//   MPI_Barrier(MPI_COMM_WORLD);
//   double t1 = get_time();

  int ninter = 0;
  for (int i = 0; i < local_n; i++) {
    ngb_list[i] = 0;
  };

  int isize = interaction_list.size();
  for (int i = 0; i < isize; i++) {
    octnode &inode = *interaction_list[i].first;
    octnode &jnode = *interaction_list[i].second;
  
#if 0  
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) 
      for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) 
	ninter++;
#endif 

    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next) {
      if (ibp->isexternal()) continue;
      boundary ibnd = boundary(ibp->xcache,ibp->hcache);
      if (!ibnd.overlap(jnode.inner)) continue;

      real ih2 = sqr(ibp->pp->h); //sqr(ibp->xhcache.w);
      pfloat3 ipos = ibp->pp->pos;
      real  local_idx = ibp->pp->local_idx;
      
      for (octbody *jbp = jnode.pfirst; jbp != NULL; jbp = jbp->next) {
	pfloat3 jpos = jbp->pp->pos;
	float3 dr = {jpos.x - ipos.x,
		     jpos.y - ipos.y,
		     jpos.z - ipos.z};
	float s2 = sqr(dr.x) + sqr(dr.y) + sqr(dr.z);
	if (s2 <= ih2) ngb_list[local_idx] += 1;
	
      } 
      
    }
    
  }
  
//   MPI_Barrier(MPI_COMM_WORLD);
//   double t2 = get_time();

  fprintf(stderr, "nbgb: proc= %d: niter= %d _pp= %g\n", myid, ninter, 1.0*ninter/local_n);
  
#if 0
  fprintf(stderr, "dump ngb\n");
  
  char fn[256];
  sprintf(fn, "proc_%.2d_ngb.dump", myid);
  FILE *fout = fopen(fn, "w");
  
  for (int i = 0; i < local_n; i++) {
    particle &ptcl = pvec[i];
    
    fprintf(fout, "%d %d %g %g %g %g  ngb= %d\n", 
	    (int)ptcl.global_idx,
	    (int)ptcl.local_idx,
	    (float)ptcl.pos.x.getu(),
	    (float)ptcl.pos.y.getu(),
	    (float)ptcl.pos.z.getu(),
	    ptcl.h,
	    ngb_list[ptcl.local_idx]);
  }
  fclose(fout);
  
#endif


  
}


void system::leaf_leaf_interaction(octnode &inode,
				   octnode &jnode, 
				   std::vector< std::pair<octnode*,octnode*> > &interaction_list) {
  if (!inode.outer.overlap(jnode.inner)) return;

  bool itravel = false;
  bool jtravel = false;
  
  if (inode.isleaf()) {
    
    if (jnode.isleaf()) {
      // store interacting leaves
      interaction_list.push_back( std::pair<octnode*, octnode*>(&inode, &jnode) );
      return;
    } else {
      jtravel = true;
    }
    
  } else {

    if (jnode.isleaf()) {
      itravel = true;
    } else {
      if (inode.centre.w > jnode.centre.w) itravel = true;
      else                                 jtravel = true;
    }

  }

  if (itravel) {
    for (int ic = 0; ic < 8; ic++)
      if (!inode.child[ic].isempty())
	leaf_leaf_interaction(inode.child[ic], jnode, interaction_list);
    return;
  }

  if (jtravel) {
    for (int jc = 0; jc < 8; jc++)
      if (!jnode.child[jc].isempty())
	leaf_leaf_interaction(inode, jnode.child[jc], interaction_list);
    return;
  }
  
}
