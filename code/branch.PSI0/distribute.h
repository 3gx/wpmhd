#ifndef _DISTRIBUTE_H_
#define _DISTRIBUTE_H_

#include "boundary.h"
#include "peano.h"
#include <algorithm>

struct cmp_peanokey_index{
  bool operator() (const peano_struct &lhs, const peano_struct &rhs){
    return lhs.key < rhs.key;
  }
};


class distribute {

private:
public:

  boundary global_domain;


  int                ntile_per_proc[NMAXPROC];
  std::vector<int>      procs_tiles[NMAXPROC];
  std::vector<boundary> inner_tiles;
  std::vector<peanokey> tile_keys;

  float3 rmin;
  int    domain_fac;

  int  nproc, ntile;
  int3 nt;

public:

  distribute()  {
    nproc = -1;
    ntile = -1;
    inner_tiles.clear();
  };
  ~distribute() {};
  
  void set(const int nproc, const int3 nt, const boundary &global_domain) {
    this->global_domain = global_domain;
    this->nproc = nproc;
    this->nt    = nt;
    ntile       = nt.x * nt.y * nt.z;
    
    for (int p = 0; p < nproc; p++) {
      procs_tiles[p].clear();
      ntile_per_proc[p] = -1;
    }
    
    inner_tiles.resize(ntile);
  }

  int idx(const int3 idx3) const  {
    return (idx3.z*nt.y + idx3.y)*nt.x + idx3.x;
  };
  
  ///////////
  
  template<int ch>
  void sort_coord_array(pfloat3 *r, int lo, int up) const {
    int i, j;
    pfloat3 tempr;
    while ( up>lo ) {
      i = lo;
      j = up;
      tempr = r[lo];
      
      /*** Split file in two ***/
      while ( i<j ) {
	switch(ch) {
	case 0:
	  for ( ; r[j].x.getu() > tempr.x.getu(); j-- );
	  for ( r[i]=r[j]; i<j && r[i].x.getu() <= tempr.x.getu(); i++ );
	  break;
	case 1:
	  for ( ; r[j].y.getu() > tempr.y.getu(); j-- );
	  for ( r[i]=r[j]; i<j && r[i].y.getu() <= tempr.y.getu(); i++ );
	  break;
	case 2:
	  for ( ; r[j].z.getu() > tempr.z.getu(); j-- );
	  for ( r[i]=r[j]; i<j && r[i].z.getu() <= tempr.z.getu(); i++ );
	  break;
	default: assert(ch >=0 && ch <= 2);
	}
	r[j] = r[i];
      }
      r[i] = tempr;
      /*** Sort recursively, the smallest first ***/
      if ( i-lo < up-i ) { sort_coord_array<ch>(r,lo,i-1);  lo = i+1; }
      else               { sort_coord_array<ch>(r,i+1,up);  up = i-1; }
    }
  }
  
  
  template<int ch>
  void calculate_boxdim(const int np, pfloat3 pos[], const int istart, const int iend, 
			pfloat3 &rlow, pfloat3 &rhigh) const {
    switch(ch) {
    case 0: rlow.x.set(pfloat<ch>::xmin); rhigh.x.set(pfloat<ch>::xmax); break;
    case 1: rlow.y.set(pfloat<ch>::xmin); rhigh.y.set(pfloat<ch>::xmax); break;
    case 2: rlow.z.set(pfloat<ch>::xmin); rhigh.z.set(pfloat<ch>::xmax); break;
    default: assert(ch >=0 && ch <= 2);
    }
    if(istart > 0) {
      pfloat3 r1 = pos[istart  ];
      pfloat3 r2 = pos[istart-1];
      switch(ch) {
      case 0: r1.x.half(); r2.x.half(); r1.x.add(r2.x); rlow.x = r1.x;break;
      case 1: r1.y.half(); r2.y.half(); r1.y.add(r2.y); rlow.y = r1.y;break;
      case 2: r1.z.half(); r2.z.half(); r1.z.add(r2.z); rlow.z = r1.z;break;
      default: assert(ch >=0 && ch <= 2);
      }
    }

    if (iend < np-1) {
      pfloat3 r1 = pos[iend  ];
      pfloat3 r2 = pos[iend+1];
      switch(ch) {
      case 0: r1.x.half(); r2.x.half(); r1.x.add(r2.x); rhigh.x = r1.x; break;
      case 1: r1.y.half(); r2.y.half(); r1.y.add(r2.y); rhigh.y = r1.y; break;
      case 2: r1.z.half(); r2.z.half(); r1.z.add(r2.z); rhigh.z = r1.z; break;
      default: assert(ch >= 0 && ch <= 2);
      }
    }
  }
  
  ////////////
  
  inline peanokey peano_key(const pfloat3 r) {
    const int x = (int)((r.x.getu() - rmin.x) * domain_fac);
    const int y = (int)((r.y.getu() - rmin.y) * domain_fac);
    const int z = (int)((r.z.getu() - rmin.z) * domain_fac);
    return peano_hilbert_key(x, y, z, BITS_PER_DIMENSION);
  }

  void determine_division(std::vector<pfloat3> &pos) {
    
    std::vector<int> istart(ntile);
    std::vector<int> iend  (ntile);
    
    const int n = pos.size();
    assert(ntile < n);
    
    sort_coord_array<0>(&pos[0], 0, n-1);
    
    for (int i = 0; i < ntile; i++) {
      istart[i] = (i * n)/ntile;
      if (i > 0) iend[i-1] = istart[i] - 1;
    }
    iend[ntile - 1] = n - 1;
    
    std::vector<pfloat3> rlow(ntile), rhigh(ntile);

    pfloat3 r0, r1;
    for (int ix = 0; ix < nt.x; ix++) {
      const int ix0 =  ix   *nt.y*nt.z;
      const int ix1 = (ix+1)*nt.y*nt.z;
      calculate_boxdim<0>(n, &pos[0], istart[ix0], iend[ix1-1], r0, r1);
      for (int i = ix0; i < ix1; i++){
	rlow [i].x=  r0.x;
	rhigh[i].x = r1.x;
      }
    }
    
    for(int ix = 0; ix < nt.x; ix++) {
      const int ix0 =  ix   *nt.y*nt.z;
      const int ix1 = (ix+1)*nt.y*nt.z;
      const int npy = iend[ix1-1] - istart[ix0] + 1;
      sort_coord_array<1>(&pos[0], istart[ix0], iend[ix1-1]);
      for (int iy = 0; iy < nt.y; iy++) {
	const int iy0 = ix0 +  iy   *nt.z;
	const int iy1 = ix0 + (iy+1)*nt.z;
	calculate_boxdim<1>(npy, &pos[0]+istart[ix0], istart[iy0]-istart[ix0], iend[iy1-1]-istart[ix0], r0,r1);
	for (int i = iy0; i < iy1; i++){
	  rlow [i].y = r0.y;
	  rhigh[i].y = r1.y;
	}
      }
    }
    
    for (int ix = 0; ix < nt.x; ix++) {
      const int ix0 = ix*nt.y*nt.z;
      for (int iy = 0; iy < nt.y; iy++) {
	const int iy0 = ix0 +  iy   *nt.z;
	const int iy1 = ix0 + (iy+1)*nt.z;
	const int npz = iend[iy1-1] - istart[iy0] + 1;
	sort_coord_array<2>(&pos[0], istart[iy0], iend[iy1-1]);
	for (int iz = 0; iz < nt.z; iz++) {
	  const int iz0 = iy0 + iz;
	  calculate_boxdim<2>(npz, &pos[0]+istart[iy0], istart[iz0]-istart[iy0], iend[iz0]-istart[iy0], r0, r1);
	  rlow [iz0].z = r0.z;
	  rhigh[iz0].z = r1.z;
	}
      }
    }

    inner_tiles.resize(ntile);
    for (int tile = 0; tile < ntile; tile++) {
      inner_tiles[tile].set(rlow[tile], rhigh[tile]);
    }

    /// sort inner tiles into peano-hilber order
    
    pfloat3 prmin = global_domain.rmin();
    rmin = (float3){prmin.x.getu(), prmin.y.getu(), prmin.z.getu()};
    const float size = 2.0f*std::max(global_domain.hsize.x.getu(), 
				     std::max(global_domain.hsize.y.getu(), global_domain.hsize.z.getu()));
    
    domain_fac = 1.0f / size * (((peanokey)1) << (BITS_PER_DIMENSION));
    std::vector<peano_struct> keys(ntile);

    for (int tile = 0; tile < ntile; tile++) {
      keys[tile].idx = tile;
      keys[tile].key = peano_key(inner_tiles[tile].centre);
    }
    
    std::sort(keys.begin(), keys.end(), cmp_peanokey_index());
    std::vector<boundary> inner_tiles_old = inner_tiles;
    tile_keys.resize(ntile);
    for (int tile = 0; tile < ntile; tile++) {
      inner_tiles[tile] = inner_tiles_old[keys[tile].idx];
      tile_keys  [tile] = keys[tile].key;
    }
    
    for (int proc = 0; proc < nproc; proc++)
      ntile_per_proc[proc] = ntile/nproc;
    
    const int residue = ntile - nproc * (ntile/nproc);
    for (int proc = 0; proc < residue; proc++)
      ntile_per_proc[proc] += 1;
        
    int tile = 0;
    for (int proc = 0; proc < nproc; proc++) {
      const int n = ntile_per_proc[proc];
      procs_tiles[proc].resize(n);
      for (int i = 0; i < n; i++)
	procs_tiles[proc][i] = tile++;
    }
    assert(tile == ntile);
    
    
  }


  /////////

  std::vector<int>& get_tiles(const int proc) {return procs_tiles[proc];}
  boundary& inner(const int tile) {return inner_tiles[tile];}
  
  ///////

#if 0 /// not functional, but a good beginning
  int locate_tile(const peanokey key) const {
    int l = 0;
    int r = ntile - 1;
    while (r - l > 0) {
      const int c = (l + r) >> 1;
      if      (key < tile_keys[c]) r = c;
      else if (key > tile_keys[c]) l = c;
      else return c;
    }
    return l;
  }

  int which_tile(const pfloat3 pos) {
    return locate_tile(peano_key(pos));
  }

  bool isinproc(const pfloat3 pos, const int proc) {
    const int tile = which_tile(pos);
    const int n0 = procs_tiles[proc][0];
    const int dn = procs_tiles[proc].size();
    if (n0 <= tile && tile < n0 + dn) return true;
    return false;
  }
    
#else

  int which_tile(const pfloat3 pos) {
    for (int tile = 0; tile < ntile; tile++) {
      if (inner_tiles[tile].isinbox(pos)) return tile;
    }
    return -1;
  }
  
  bool isinproc(const pfloat3 &pos, const int proc) {
    const int n = procs_tiles[proc].size();
    for (int i = 0; i < n; i++) 
      if (inner_tiles[procs_tiles[proc][i]].isinbox(pos)) return true;
    return false;
  }
#endif

  bool isinproc(const boundary &bnd, const int proc) {
    const int n = procs_tiles[proc].size();
    for (int i = 0; i < n; i++) 
      if (inner_tiles[procs_tiles[proc][i]].isinbox(bnd)) return true;
    return false;
  }


  int which_proc(const boundary &bnd) {
    for (int proc = 0; proc < nproc; proc++) 
      if (isinproc(bnd, proc)) return proc;
    return -1;
  }

  
};

#endif // _DISTRIBUTE_H_

