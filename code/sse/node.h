#ifndef _NODE_H_
#define _NODE_H_

#include "boundary.h"
#include "memory_pool.h"


struct octbody {
  particle *pp;
  octbody*  next;
  pfloat3 xcache;
  float   hcache;
  
  octbody() {};
  octbody(particle &p, bool isexternal = false) {
    pp      = &p;
    next    = NULL;
    xcache  = p.pos;
    hcache  = p.h;
    if (isexternal) hcache *= -1.0f;
  };
  bool islocal()    {return hcache >= 0.0f;}
  bool isexternal() {return !islocal();}
  void update(const bool isexternal = false)     {
    xcache = pp->pos;
    hcache = pp->h;
    if (isexternal) hcache *= -1.0f;
  }
  
  
  ~octbody() {};
  
};


////////


struct octnode {
  float4 centre;     // x,y,z, half-length      // 4
  boundary inner, outer;                        // 6
  int nparticle;                                // 7
  int touch;                                    // 8
  
  union {
    octnode *child;                                
    octbody *pfirst;                                    
  };                                            // 10
  
  enum {nleaf = 32};
  bool isleaf() const {return nparticle <= nleaf;}
  bool isempty() const {return nparticle == 0;}
  bool istouched() const {return touch;}
  
  octnode() {clear();}
  void clear()   {nparticle = 0; touch = 0; child = NULL;}
  octnode(const octnode &parent, const int ic) {
    centre.w = parent.centre.w*0.5f;
    centre.x = parent.centre.x + centre.w * ((ic & 1) ? 1.0f : -1.0f);
    centre.y = parent.centre.y + centre.w * ((ic & 2) ? 1.0f : -1.0f);
    centre.z = parent.centre.z + centre.w * ((ic & 4) ? 1.0f : -1.0f);

    child     = NULL;
    nparticle = 0;
    touch     = 0;
  };
  ~octnode() {};

  ////

  void assign_root(const boundary &bnd) {
    pfloat3 pc = bnd.centre;
    pfloat3 ps = bnd.hsize;
    float3 c   = {pc.x.getu(), pc.y.getu(), pc.z.getu()};
    float3 s   = {ps.x.getu(), ps.y.getu(), ps.z.getu()};
#if 0
    centre     = (float4){c.x, c.y, c.z,  fmax(s.x, fmax(s.y,s.z))};
#else
    centre.x = c.x;
    centre.y = c.y;
    centre.z = c.z;
    centre.w = fmax(s.x, fmax(s.y,s.z));
#endif
    
    float hsize = 1.0f;
    while(hsize > centre.w) hsize *= 0.5f;
    while(hsize < centre.w) hsize *= 2.0f;
    
    centre.w = hsize;
    child = NULL;

  };
  
  void push_octbody(octbody &p) {
    p.next = pfirst;
    pfirst = &p;
  }

  ///////

#if 0
  static inline int octant_index(const pfloat3 &lhs, const pfloat3 &rhs) {
    return (((lhs.x.leq(rhs.x)) ? 1 : 0) +  
	    ((lhs.y.leq(rhs.y)) ? 2 : 0) + 
	    ((lhs.z.leq(rhs.z)) ? 4 : 0));
  }
#else
  static inline int octant_index(float4 &center, float4 &pos){
    int i = __builtin_ia32_movmskps(
				    (float4::v4sf)__builtin_ia32_cmpltps(center.v, pos.v));
    return 7 & i;
  }
#endif

  //////
  
  void divide_treenode(memory_pool<octnode> &pool) {
    static octbody *bodyarray[nleaf + 1];
    assert(nparticle <= nleaf + 1);
    octbody *p = pfirst;
    
    for (int i = 0; i < nparticle; i++) {
      bodyarray[i] = p;
      p = p->next;
    }
    pfirst = NULL;

    assert(child == NULL);
    child = pool.get(8);
    
    for (int ic = 0; ic < 8; ic++) {
      child[ic] = octnode(*this, ic);
    }
    
    for (int i = 0; i < nparticle; i++) {
      octbody &p = *bodyarray[i];
      
#if 0      
      pfloat3 c;
      c.x.set(centre.x);
      c.y.set(centre.y);
      c.z.set(centre.z);
      int ic = octant_index(c, p.xcache);
#else
      float4 x4;
      x4.x = p.xcache.x.getu();
      x4.y = p.xcache.y.getu();
      x4.z = p.xcache.z.getu();
      x4.w = 0.0f;
      int ic = octant_index(centre, x4);
#endif

      child[ic].push_octbody(p);
      child[ic].nparticle++;
      child[ic].touch = 1;
    }
    for (int ic = 0; ic < 8; ic++) {
      if (!child[ic].isleaf()) {
	child[ic].divide_treenode(pool);
      }
    }
  }

  //////


  static void insert_octbody(octnode &node, octbody &p, memory_pool<octnode> &pool) {
    octnode *current = &node;
    while(true) {
      
      if (current->isleaf()) {
	current->push_octbody(p);
	current->nparticle++;
	current->touch = 1;
 	if (!current->isleaf()) 
 	  current->divide_treenode(pool);
	return;
      } else {
	current->nparticle++;
	current->touch = 1;

#if 0	
	pfloat3 c;
	c.x.set(current->centre.x);
	c.y.set(current->centre.y);
	c.z.set(current->centre.z);
	int ic = octant_index(c, p.xcache);
#else
      float4 x4;
      x4.x = p.xcache.x.getu();
      x4.y = p.xcache.y.getu();
      x4.z = p.xcache.z.getu();
      x4.w = 0.0f;
      int ic = octant_index(current->centre, x4);
#endif
	
	current = &current->child[ic];
	continue;
      }
      
    }
    
  }

  ////

  void sanity_check() {
    if (isleaf()) {
      
      boundary bnd = boundary(centre);
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next)  {
	if (!bnd.isinbox(bp->pp->pos)) {
	  pfloat3 pos = bp->pp->pos;
	  fprintf(stderr, "pos= %20.16lg %20.16lg %20.16lg\n",
		  pos.x.getu(),
		  pos.y.getu(),
		  pos.z.getu());
	  fprintf(stderr, "bnd= "); bnd.dump(stderr, true);
	  assert(bnd.isinbox(bp->pp->pos));
	}
      }
      
    } else {

      for (int ic = 0; ic < 8; ic++)
	child[ic].sanity_check();

    }

  }
  

  ////////

  boundary& calculate_inner_boundary() {
    inner = boundary();
    
    if (!touch   ) return inner;
    if (isempty()) return inner;
    
    if (isleaf()) {
      
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
	inner.merge(boundary(bp->xcache));
      
    } else {
      
      for (int ic = 0; ic < 8; ic++) 
	inner.merge(child[ic].calculate_inner_boundary());
      
    }
    return inner;
  }

  boundary& calculate_outer_boundary(const float scale_factor = 1.0f) {
    outer = boundary();
    
    if (isempty()) return outer;
    
    if (isleaf()) {
      
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
	if (bp->islocal())
	  outer.merge(boundary(bp->xcache, bp->hcache*scale_factor));
      
    } else {
      
      for (int ic = 0; ic < 8; ic++) 
	outer.merge(child[ic].calculate_outer_boundary(scale_factor));
      
    }
    
    return outer;
  }

  ///////////

//   void walk_outer_boundary(const boundary &bnd_in, 
// 			   const boundary &bnd_out,
// 			   std::vector<particle> &export_list) const {

//     if (isempty()) return;
    
//     if (!bnd_in.overlap(outer)) return;
    
//     if (isleaf()) {
      
//       for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
//  	if (!bnd_out.isinbox(bp->xcache) && boundary(bp->xcache, bp->hcache).overlap(bnd_in) && bp->islocal()) 
// 	  export_list.push_back(*bp->pp);
      
//     } else {
      
//       for (int ic = 0; ic < 8; ic++) 
// 	child[ic].walk_outer_boundary(bnd_in, bnd_out, export_list);
      
//     }
//   }

  void walk_outer_boundary(const boundary &bnd_in,
			   const boundary &bnd_out, 
			   std::vector<particle> &export_list) const {

    if (isempty()) return;
    
    if (!(bnd_in.overlap(outer) || bnd_out.overlap(inner))) return;
    
    if (isleaf()) {
      
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
 	if ((bnd_out.isinbox(bp->xcache) || boundary(bp->xcache, bp->hcache).overlap(bnd_in)) &&
	    bp->islocal()) 
	  export_list.push_back(*bp->pp);
      
    } else {
      
      for (int ic = 0; ic < 8; ic++) 
	child[ic].walk_outer_boundary(bnd_in, bnd_out, export_list);
      
    }
  }

  //////////////  
  
  void walk_boundary(const boundary &bnd, 
		     std::vector<particle> &export_list) const {

    if (isempty()) return;
    
    if (!bnd.overlap(inner)) return;
    
    if (isleaf()) {
      
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
	if (bnd.isinbox(bp->xcache) && bp->islocal())
	  export_list.push_back(*bp->pp);
      
    } else {
      
      for (int ic = 0; ic < 8; ic++) 
	child[ic].walk_boundary(bnd, export_list);
      
    }
  }

  void walk_boundary(const boundary &bnd, 
		     std::vector<particle*> &export_list) const {

    if (isempty()) return;
    
    if (!bnd.overlap(inner)) return;
    
    if (isleaf()) {
      
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
	if (bnd.isinbox(bp->xcache) && bp->islocal())
	  export_list.push_back(bp->pp);
      
    } else {
      
      for (int ic = 0; ic < 8; ic++) 
	child[ic].walk_boundary(bnd, export_list);
      
    }
  }


  ///////////

  void walk_boundary(const boundary &bnd_inner, 
		     const boundary &bnd_outer,
		     std::vector<particle> &export_list) const {
    
    if (isempty()) return;
    
    if (!bnd_outer.overlap(inner)) return;
    
    if (isleaf()) {
      
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
	if (!bnd_inner.isinbox(bp->xcache) && bnd_outer.isinbox(bp->xcache) && bp->islocal()) 
	  export_list.push_back(*bp->pp);
      
    } else {
      
      for (int ic = 0; ic < 8; ic++) 
	child[ic].walk_boundary(bnd_inner, bnd_outer, export_list);
      
    }
  }

  ///////////

  void extract_leaves(std::vector<octnode*> &leaf_list) {
    if (isempty()) return;

    if (isleaf()) 
      leaf_list.push_back(this);
    else
      for (int ic = 0; ic < 8; ic++)
	child[ic].extract_leaves(leaf_list);
  }

  ////////
  
  void find_leaves_inner(const boundary &bnd, std::vector<octnode*> &leaf_list) {


    if (isempty()) return;
    

    if (!bnd.overlap(inner)) return;
    
    if (isleaf()) {
      
      if (bnd.overlap(inner))
	leaf_list.push_back(this);
      
    } else {

      for (int ic = 0; ic < 8; ic++)
	child[ic].find_leaves_inner(bnd, leaf_list);

    }
  }

  void find_leaves_outer(const boundary &bnd_in,	       
			 const boundary &bnd_out,
			 std::vector<octnode*> &leaf_list) {

    if (isempty()) return;
    

    if (!(bnd_in.overlap(outer) || bnd_out.overlap(inner))) return;
    
    if (isleaf()) {
      
      if (bnd_in.overlap(outer) || bnd_out.overlap(inner))
	leaf_list.push_back(this);
      
    } else {

      for (int ic = 0; ic < 8; ic++)
	child[ic].find_leaves_outer(bnd_in, bnd_out, leaf_list);

    }
  }

  void dump_particles_in_Zorder(std::vector<particle*> &particle_list) {
    if (isempty()) return;

    if (isleaf()) {
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
	if (bp->islocal())
	  particle_list.push_back(bp->pp);
    } else {
      for (int ic = 0; ic < 8; ic++)
	child[ic].dump_particles_in_Zorder(particle_list);
      
    }
  }

  void dump_particles_in_Zorder(std::vector<particle> &particle_list) {
    if (isempty()) return;

    if (isleaf()) {
      for (octbody *bp = pfirst; bp != NULL; bp = bp->next) 
	if (bp->islocal())
	  particle_list.push_back(*bp->pp);
    } else {
      for (int ic = 0; ic < 8; ic++)
	child[ic].dump_particles_in_Zorder(particle_list);
      
    }
  }

  ////

};

#endif // _NODE_H_
