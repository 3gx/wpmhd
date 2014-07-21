#include "gn.h"
#include "myMPI.h"

void system::export_boundary_particles() {

//   MPI_Barrier(MPI_COMM_WORLD);
//   fprintf(stderr, "proc= %d: root inner= ", myid); root_node.inner.dump(stderr, true);
//   MPI_Barrier(MPI_COMM_WORLD);
//   fprintf(stderr, "proc= %d: root outer= ", myid); root_node.outer.dump(stderr, true);

//   MPI_Barrier(MPI_COMM_WORLD);
//   fprintf(stderr, " ****\n");

  std::vector<particle> psend[NMAXPROC];
  std::vector<particle> precv[NMAXPROC];

  for (int p = 0; p < nproc; p++) {
    psend[p].reserve(128);
    if (p == myid) continue;
    root_node.walk_boundary(box.outer[p], psend[p]);
//     fprintf(stderr, "proc= %d: p= %d, send= %d\n", myid, p, (int)psend[p].size());
//     fprintf(stderr, "proc= %d: box.outer= ", myid); box.outer[p].dump(stderr);
//     fprintf(stderr, "\n");
    
  }

#if 0
  
  for (int p = 0; p < nproc; p++) {
    char fn[256];
    sprintf(fn, "proc_%.2d_export_%.2d.dump", myid, p);
    FILE *fout = fopen(fn, "w");
    fprintf (fout, "# n= %d \n", (int)psend[p].size());
    fprintf (stderr, "proc= %d  export= %d  n= %d \n", myid, p,(int)psend[p].size());

    for (size_t i = 0; i < psend[p].size(); i++) {
      particle &ptcl = psend[p][i];
      
      fprintf(fout, "%d %d %g %g %g %g\n", 
	      (int)ptcl.global_idx,
	      (int)ptcl.local_idx,
	      ptcl.pos.x,
	      ptcl.pos.y,
	      ptcl.pos.z,
	      ptcl.h);
    }
    fclose(fout);
  }

#endif


  //////
  
  myMPI_all2all<particle>(psend, precv, myid, nproc, true);
  
  
#if 1
  
  // sanity check
  
  for (int p = 0; p < nproc; p++) {
    for (size_t i = 0; i < precv[p].size(); i++) {
      assert(root_node.outer.isinbox(precv[p][i].pos));
    }
  }
  
#endif

  
  // copy imported particles into the system
  
  p_import.reserve(local_n);
  p_import.clear();
  
  particle *first = &p_import[0];
  for (int p = 0; p < nproc; p++)
    for (size_t i = 0; i < precv[p].size(); i++) 
      p_import.push_back(precv[p][i]);
  assert(first == &p_import[0]);
  
  import_n = p_import.size();
  
  // create octbodies of import particles

  octp_import.reserve(local_n);
  assert(safe_resize(octp_import, import_n));
  
  for (int i = 0; i < import_n; i++) {
    octp_import[i] = octbody(p_import[i], true);
  }
  
  // insert octobdies into the tree
  
  for (int i = 0; i < import_n; i++) {
    octnode::insert_octbody(root_node, octp_import[i], octn);
  }
    
  root_node.calculate_boundary();
  
  // update local index
  for (int i = 0; i < import_n; i++)
    p_import[i].local_idx = i;
  
  leaf_list.clear();
  group_list.clear();
  leaf_list.reserve(128);
  group_list.reserve(128);

  root_node.extract_leaves(leaf_list);
  root_node.extract_leaves(group_list);
//   fprintf(stderr, "leaf_list_size= %d\n", (int)leaf_list.size());
//   root_node.extract_groups(group_list, octnode::ngroup);
  

}
