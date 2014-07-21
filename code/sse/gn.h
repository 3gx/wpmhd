#ifndef _GN_H_
#define _GN_H_

#include "distribute.h"
#include "node.h"
#include "smoothing_kernel.h"

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
  int sample_freq;
  bool recalculate_domain_boundaries;
  double t_compute, t_communicate;
  double t_weight, t_renorm, t_grad, t_interaction, t_tree;
  double t_distribute, t_import_bnd;
  double t_pmhd, t_pvec, t_pvel, t_grad_comm, t_Bmatrix;

  real gamma_gas, courant_no;
  real t_global, dt_global;
  int  iteration;

  bool eulerian_mode, do_morton_order, do_ppm;

  int3 np;
  int local_n, global_n;           // local & global number of particles
  int import_n;                    // number of imported particles
  boundary global_domain;

  bc_type bc_xmin, bc_xmax;    
  bc_type bc_ymin, bc_ymax;    
  bc_type bc_zmin, bc_zmax;    
  
  distribute box;
  octnode    root_node;

  /////////
  
  std::vector<particle> pvec;             // local    particles data
//   std::vector<particle> pvec_import;      // imported particles data

  std::vector<int> pidx_send[NMAXPROC];
  
  std::vector<ptcl_mhd> pmhd;         // local    mhd data
  std::vector<float3>   pdefect;
  std::vector<float >   weights, dwdt_i, divB_i;
  
#if (defined _CONSERVATIVE_) && (defined _SEMICONSERVATIVE_)
#error "Please difine only _CONSERVATIVE_ or _SEMICONSERVATIVE_ but not both in Makefile"
#endif

#if (defined _CONSERVATIVE_) || (defined _SEMICONSERVATIVE_)
  std::vector<ptcl_mhd> conservative;
#endif
  std::vector<ptcl_mhd> pmhd_dot, pmhd_dot0, pmhd_grad[3];    // derived mhd data
  std::vector<ptcl_mhd> pmhd_cross[6];
  std::vector<float3>   pvel0;
  bool first_step_flag;
  double dt_global0;

  std::vector<real> Bxx, Bxy, Bxz, Byy, Byz, Bzz;         // derived particles data
  
  std::vector< std::vector<octnode*> > ngb_leaf_list;           // list of neighbouring leaves
  std::vector< std::vector<octnode*> > ngb_leaf_list_outer;     // list of neighbouring leaves
  
  ////////////
 
  int myid, nproc;

  real NGBmin, NGBmax, NGBmean;
  
  std::vector<octbody> octp; //, octp_import, octp_bc;
  memory_pool<octnode> octn;
  std::vector<octnode*> leaf_list, group_list;
  
  smoothing_kernel kernel;
  
public:
  
  system(const int3 np, int nproc, int myid, bool ppm = false, int ndim = 3)  {
    this->nproc = nproc;
    this->myid  = myid;
    
    this->np = np;
    assert(nproc == np.x*np.y*np.z);
    
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
    first_step_flag = true;
  };
  ~system() {};


  void setup_particles();
  void distribute_particles();
  void determine_sample_freq();
  void collect_sample_coords(std::vector<pfloat3>&);
  /////////// neighbour search and weight calculation
  
  void build_ngb_leaf_lists();
  void import_boundary_particles();
  void import_scatter_particles();
  void import_boundary_pvec();
  void import_boundary_pmhd();
  void import_boundary_pmhd_grad();
  void import_boundary_pmhd_cross();
  void import_boundary_pvel();
  void import_boundary_Bmatrix();
  void build_tree();
  void leaf_leaf_interaction(octnode&, octnode&, std::vector< std::pair<octnode*, octnode*> >&);
  void ngb_search();
  void compute_weights();
  void compute_weights    (std::vector<boundary> &, const float);  
  void compute_weights_gpu(std::vector<boundary> &, const float);  

  ///////////////// mhd 

  void renorm();
  void gradient();
  void gradient_v4sf();
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
  
  ////// integrator

  void push_particles();
  void derivs();
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
			 const float3 &ei);
  float compute_pressure(const float, const float);

  void dump_snapshot(const char*);
  void dump_binary(const char*);

//   float get_time() const {return  t_global;}
//   float get_dt()   const {return dt_global;}
//   int   get_iter() const {return iteration;}
  
};


#endif // _GN_H_
