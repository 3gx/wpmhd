#include "gn.h"

void system::copy2gpu() {

  const int n_leaves = local_tree.leaf_list.size();
  import_tree.get_leaves();
  const int n_import_leaves = import_tree.leaf_list.size();

  gpu.body_list.cmalloc(n_leaves + n_import_leaves);
  gpu.ilist.cmalloc(pvec.size());
  
  int ip = 0;
  for (int leaf = 0; leaf < n_leaves; leaf++) {
    const octnode<TREE_NLEAF> &inode = *local_tree.leaf_list[leaf];
    gpu.body_list[leaf] = (int2){ip, inode.nparticle};
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next)  
      gpu.ilist[ip++] = ibp->pp->local_idx;
  }
  for (int leaf = 0; leaf < n_import_leaves; leaf++) {
    const octnode<TREE_NLEAF> &inode = *import_tree.leaf_list[leaf];
    gpu.body_list[n_leaves + leaf] = (int2){ip, inode.nparticle};
    for (octbody *ibp = inode.pfirst; ibp != NULL; ibp = ibp->next)  
      gpu.ilist[ip++] = ibp->pp->local_idx;
  }
  assert(ip == (int)pvec.size());
  
  gpu.body_list.h2d();
  gpu.ilist.h2d();

  const int n_data = pvec.size();
  gpu.ppos.cmalloc(n_data);
  gpu.pvel.cmalloc(n_data);
  for (int i = 0; i < n_data; i++) {
    particle &p = pvec[i];
    const ptcl_mhd m  = pmhd[i]; //
    p.vel.x = m.vel.x;
    p.vel.y = m.vel.y;
    p.vel.z = m.vel.z;
    if (eulerian_mode) p.vel = (float3){0,0,0};
    gpu.ppos[i] = float4(p.pos.x.getu(), p.pos.y.getu(), p.pos.z.getu(), p.h   );
    gpu.pvel[i] = float4(p.vel.x,        p.vel.y,        p.vel.z,        p.wght);
  }
  gpu.ppos.h2d();
  gpu.pvel.h2d();
  
  gpu.joffset.cmalloc(NGPUBLOCKS + 1);
  gpu.leaf_list.cmalloc(NGPUBLOCKS);

  gpu.domain_hsize = float4(global_domain.hsize.x.getu(),
			    global_domain.hsize.y.getu(),
			    global_domain.hsize.z.getu(), 0.0f);

  gpu.mhd1.cmalloc(n_data);
  gpu.mhd2.cmalloc(n_data);
  gpu.mhd3.cmalloc(n_data);
  for (int i = 0; i < n_data; i++) {
    const ptcl_mhd &m = pmhd[i];
    gpu.mhd1[i] = float4(m.vel.x, m.vel.y, m.vel.z, m.dens);
    gpu.mhd2[i] = float4(m.B.x,   m.B.y,   m.B.z,   m.ethm);
    gpu.mhd3[i] = float4(m.psi,   0.0f,    0.0f,    0.0f);
    gpu.mhd2[i].x += constB.x;
    gpu.mhd2[i].y += constB.y;
    gpu.mhd2[i].z += constB.z;
  }  
  gpu.mhd1.h2d();
  gpu.mhd2.h2d();
  gpu.mhd3.h2d();
  
}
