#ifndef _GN_H_
#define _GN_H_


#define  TREE_NLEAF 32
#define  DOMAIN_TREE_NLEAF 32


#include "distribute.h"
#include "binary_tree.h"
#include "smoothing_kernel.h"
#include "octree.h"

#include <sys/time.h>
inline double get_time() {
  struct timeval Tvalue;
  struct timezone dummy;
  
  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
}


enum bc_type {BC_PERIODIC, BC_REFLECTING, BC_OUTFLOW};


class system {

private:
public:
  //*********** timing

  double t_distribute_particles;
  double t_build_tree;
  double t_import_boundary_pvec_into_a_tree;
  double t_import_boundary_pvec_scatter_into_a_tree;
  double t_import_boundary_pmhd;
  double t_import_boundary_Bmatrix;
  double t_import_boundary_pmhd_grad;
  double t_import_boundary_pvel;
  double t_import_boundary_wght;
  double t_import_boundary_pmhd_cross;
  double t_compute_weights;

  //********** domain decomposition data

  int sample_freq;
  bool recalculate_domain_boundaries;
  double t_compute, t_communicate;
  double t_weight, t_renorm, t_grad, t_interaction, t_tree;
  double t_import_bnd;
  double t_pmhd, t_pvec, t_pvel, t_grad_comm, t_Bmatrix;

  float gamma_gas, courant_no, ch_glob;
  real t_global, dt_global;
  int  iteration;
  float3 gravity_pos;
  float  gravity_mass, gravity_eps2, gravity_rin, gravity_rout;


  bool eulerian_mode, do_morton_order, do_ppm;

  int local_n, global_n;           // local & global number of particles
  int import_n, import_n_max;      // number of imported particles
  boundary global_domain;

  bc_type bc_xmin, bc_xmax;    
  bc_type bc_ymin, bc_ymax;    
  bc_type bc_zmin, bc_zmax;    
  
  distribute box;
  int     Nimport;

  //// parallel tree for domain decomposotion
  octree<DOMAIN_TREE_NLEAF>   domain_tree;
  std::vector<boundary>       outer_tiles[NMAXPROC];
  bool debug_flag;

  /////////
  

  std::vector<particle> pvec; 
  std::vector<ptcl_mhd> pmhd;  
  std::vector<int> pidx_send[NMAXIMPORT][NMAXPROC];
  binary_tree<int> pidx_sent[NMAXPROC];
  
  std::vector<float >   dwdt_i, divB_i, psi_tau;
  
  std::vector<ptcl_mhd> conservative;
  std::vector<ptcl_mhd> pmhd_dot, pmhd_grad[3];    // derived mhd data
  std::vector<ptcl_mhd> pmhd_cross[6];

  std::vector<real> Bxx, Bxy, Bxz, Byy, Byz, Bzz;         // derived particles data
  
  ////////////
 
  int myid, nproc;

  real NGBmin, NGBmax, NGBmean;
  
  octree<TREE_NLEAF> local_tree;
  octree<TREE_NLEAF> import_tree;
  std::vector< std::vector<octnode<TREE_NLEAF>*> > ngb_leaf_list;           // list of neighbouring leaves
  std::vector< std::vector<octnode<TREE_NLEAF>*> > ngb_leaf_list_outer;     // list of neighbouring leaves
  
  
  smoothing_kernel kernel;


  ///////

public:
  
  system(int nproc, int myid, bool ppm = false, int ndim = 3)  {
#if _DEBUG_  
    debug_flag = true;
#else   
    debug_flag = false;   
#endif
    this->nproc = nproc;
    this->myid  = myid;
    
    assert(nproc <= NMAXPROC);

    bc_xmin = bc_xmax = BC_PERIODIC;
    bc_ymin = bc_ymax = BC_PERIODIC;
    bc_zmin = bc_zmax = BC_PERIODIC;

    kernel.set_dim(ndim);
    do_morton_order = eulerian_mode = false;

    t_global = dt_global = 0.0;
    iteration = 0;
    gamma_gas = 5.0/3;
    courant_no = 0.5;
    do_ppm = ppm;

    gravity_mass = 0.0f;
    gravity_eps2 = 1.0f;
    gravity_rin  = 0.0f;
    gravity_rout = 0.0f;
    gravity_pos = (float3){0.0f, 0.0f, 0.0f};

    import_n_max = import_n = 0;
    ch_glob = 1.0f;
  };
  ~system() {};


  void setup_particles(const bool init_data = false);
  void boundary_particles(const int idx);
  void distribute_particles();
  void determine_sample_freq();
  void collect_sample_coords(std::vector<pfloat3>&);

  /////////// neighbour search and weight calculation
  
  void calculate_outer_domain_boundaries(const float);
  void import_pvec_buffers(std::vector<particle> pvec_send[NMAXPROC]);

  void build_ngb_leaf_lists();  
  void import_boundary_pvec_into_a_tree();
  void import_boundary_pvec_scatter_into_a_tree();
  void import_boundary_particles();
  void import_scatter_particles();
  void import_boundary_pvec();
  void import_boundary_pmhd();
  void import_boundary_pmhd_grad();
  void import_boundary_pmhd_cross();
  void import_boundary_pvel();
  void import_boundary_wght();
  void import_boundary_Bmatrix();
  void build_tree();
  void ngb_search();
  void compute_weights();
  void compute_weights    (std::vector<octnode<TREE_NLEAF>*> &, const float);  
  void compute_weights_gpu(std::vector<boundary> &, const float);  

  ///////////////// mhd 

  void renorm();
  void gradient();
  void gradient_ppm();
  
  void init_conservative();
  void convert_to_primitives();
  void restore_conservative();
  
  void limit_gradients();
  void calculate_velocity_field();
  void update_weights();
  void calculate_interaction();

  void calculate_derivatives();
  void export_interactions();

  void improve_weights();
  void compute_defect();
  void mhd_interaction();
  float4 body_forces(const pfloat3 pos);
  bool remove_particles_within_racc(const int);
  
  ////// integrator

  void push_particles();
  void derivs();
  bool do_first_order;
  void iterate();
  void vain();
  void compute_dt();
  void compute_conserved(double &Mtot, double &Etot, 
			 double &Ethm, double &Ekin, double &Emag,
			 double &Volume);

  void morton_order();
  ptcl_mhd solve_riemann(const float3 &Vij, const float3 &Bij, 
			 const float3 &vij,
			 const ptcl_mhd &Qi, 
			 const ptcl_mhd &Qj, 
			 const float3 &ei,
			 float&, float&);
  float compute_pressure(const float, const float);

  void dump_snapshot(const char*);
  void dump_binary(const char*, bool flag = true);
  void read_binary(const char*, const int);

//   float get_time() const {return  t_global;}
//   float get_dt()   const {return dt_global;}
//   int   get_iter() const {return iteration;}
  
};


#endif // _GN_H_
