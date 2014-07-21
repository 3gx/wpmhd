#ifndef _DISTRIBUTE_H_
#define _DISTRIBUTE_H_

#include "boundary.h"

class distribute {

private:
public:

  boundary global_domain;
  std::vector<boundary> bnd_vec;
  std::vector<boundary> outer;
  
  int  nproc;
  int3 np;         // 3-dimensional grid of processors

public:

  distribute()  {};
  ~distribute() {};
  
  void set_np(int nproc, int3 np) {
    this->nproc = nproc;
    this->np    = np;
    assert(np.x*np.y*np.z == nproc);
    
    bnd_vec.resize(nproc);
    outer.resize(nproc);
  }
  int3 get_np() {return np;}

  int idx(int3 idx3) {
    return (idx3.z*np.y + idx3.y)*np.x + idx3.x;
  };
  
#if 0 // wont' compile anymore, so had to disable
  void set_bnd_equal_volume(boundary global_domain) {
    this->global_domain = global_domain;
    
    boundary bnd_p;
    float3 size = globa_domain.size();
    
    for (int k = 0; k < np.z; k++) {
      bnd_p.rmin.z =  k      * size.z + global_domain.rmin.z;
      bnd_p.rmax.z = (k + 1) * size.z + global_domain.rmin.z;
      
      for (int j = 0; j < np.y; j++) {
	bnd_p.rmin.y =  j      * size.y + global_domain.rmin.y;
	bnd_p.rmax.y = (j + 1) * size.y + global_domain.rmin.y;
	
	for (int i = 0; i < np.x; i++) {
	  bnd_p.rmin.x =  i      * size.x + global_domain.rmin.x;
	  bnd_p.rmax.x = (i + 1) * size.x + global_domain.rmin.x;
	  
	  bnd_vec[idx((int3){i,j,k})] = bnd_p;
	}

      }
      
    }

  }
#endif

  void set_bnd_sampling(boundary global_domain) {
    this->global_domain = global_domain;
    
    // equal partitioning in X-direction

    
    pfloat3 len = global_domain.hsize;
    len.x.div(np.x);
    len.y.div(np.y);
    len.z.div(np.z);
    len.x.twice();
    len.y.twice();
    len.z.twice();
    
    const pfloat3 rmin = global_domain.rmin();
    const pfloat3 rmax = global_domain.rmax();

    std::vector< pfloat<0> > xoffset(np.x + 1);
    xoffset[0] = rmin.x;
    for (int i = 1; i < np.x + 1; i++) {
      xoffset[i] = xoffset[i-1];
      xoffset[i].add(len.x);
    }
    xoffset[np.x] = rmax.x;
    
    for (int k = 0; k < np.z; k++) 
      for (int j = 0; j < np.y; j++) 
	for (int i = 0; i < np.x; i++) 
	  bnd_vec[idx((int3){i,j,k})].set_x(xoffset[i], xoffset[i+1]);
    


    // equal partitioning in Y-direction

    
    std::vector< pfloat<1> > yoffset(np.y + 1);
    for (int i = 0; i < np.x; i++) {
      
      yoffset[0] = rmin.y;
      for (int j = 1; j < np.y + 1; j++) {
	yoffset[j] = yoffset[j-1];
	yoffset[j].add(len.y);
      }
      yoffset[np.y] = rmax.y;
      
      for (int k = 0; k < np.z; k++) 
	for (int j = 0; j < np.y; j++) 
	  bnd_vec[idx((int3){i,j,k})].set_y(yoffset[j], yoffset[j+1]);
      
      
    }
    
    // equal partitioning in Z-direction
    
    std::vector< pfloat<2> > zoffset(np.z + 1);
    for (int j = 0; j < np.y; j++) 
      for (int i = 0; i < np.x; i++)  {
	
	zoffset[0] = rmin.z;
	for (int k = 1; k < np.z + 1; k++) {
	  zoffset[k] = zoffset[k-1];
	  zoffset[k].add(len.z);
	}
	zoffset[np.z] = rmax.z;

	for (int k = 0; k < np.z; k++) 
	  bnd_vec[idx((int3){i,j,k})].set_z(zoffset[k], zoffset[k+1]);

	
      }


  }

  ///////////
  
  template<int ch>
  void sort_coord_array(pfloat3 *r, int lo, int up)  {
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
  void calculate_boxdim(int np, pfloat3 pos[], int istart, int iend, pfloat3 &rlow, pfloat3 &rhigh) {
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
  
  void determine_division(std::vector<pfloat3> &pos) {
    
    this->global_domain = global_domain;
    
    int istart[NMAXPROC];
    int iend  [NMAXPROC];
    
    const int n = pos.size();
    sort_coord_array<0>(&pos[0], 0, n-1);
    
    for (int i = 0; i < nproc; i++) {
      istart[i] = (i * n)/nproc;
      if (i >= 0) iend[i-1] = istart[i] - 1;
    }
    iend[nproc - 1] = n - 1;
    
    std::vector<pfloat3> rlow(nproc), rhigh(nproc);

    pfloat3 r0, r1;
    for (int ix = 0; ix < np.x; ix++) {
      const int ix0 =  ix   *np.y*np.z;
      const int ix1 = (ix+1)*np.y*np.z;
      calculate_boxdim<0>(n, &pos[0], istart[ix0], iend[ix1-1], r0, r1);
      for (int i = ix0; i < ix1; i++){
	rlow [i].x=  r0.x;
	rhigh[i].x = r1.x;
      }
    }
    
    for(int ix = 0; ix < np.x; ix++) {
      const int ix0 =  ix   *np.y*np.z;
      const int ix1 = (ix+1)*np.y*np.z;
      const int npy = iend[ix1-1] - istart[ix0] + 1;
      sort_coord_array<1>(&pos[0], istart[ix0], iend[ix1-1]);
      for (int iy = 0; iy < np.y; iy++) {
	const int iy0 = ix0 +  iy   *np.z;
	const int iy1 = ix0 + (iy+1)*np.z;
	calculate_boxdim<1>(npy, &pos[0]+istart[ix0], istart[iy0]-istart[ix0], iend[iy1-1]-istart[ix0], r0,r1);
	for (int i = iy0; i < iy1; i++){
	  rlow [i].y = r0.y;
	  rhigh[i].y = r1.y;
	}
      }
    }
    
    for (int ix = 0; ix < np.x; ix++) {
      const int ix0 = ix*np.y*np.z;
      for (int iy = 0; iy < np.y; iy++) {
	const int iy0 = ix0 +  iy   *np.z;
	const int iy1 = ix0 + (iy+1)*np.z;
	const int npz = iend[iy1-1] - istart[iy0] + 1;
	sort_coord_array<2>(&pos[0], istart[iy0], iend[iy1-1]);
	for (int iz = 0; iz < np.z; iz++) {
	  const int iz0 = iy0 + iz;
	  calculate_boxdim<2>(npz, &pos[0]+istart[iy0], istart[iz0]-istart[iy0], iend[iz0]-istart[iy0], r0, r1);
	  rlow [iz0].z = r0.z;
	  rhigh[iz0].z = r1.z;
	}
      }
    }

    for (int p = 0; p < nproc; p++) {
      bnd_vec[p].set(rlow[p], rhigh[p]);
    }

  }


  ////////

  void dump_bnd() {
    for (int p = 0; p < nproc; p++) {
//       fprintf(stderr, " proc= %d:  rmin= %g %g %g  rmax= %g %g %g\n",
// 	      p, 
// 	      bnd_vec[p].rmin.x,
// 	      bnd_vec[p].rmin.y,
// 	      bnd_vec[p].rmin.z,
// 	      bnd_vec[p].rmax.x,
// 	      bnd_vec[p].rmax.y,
// 	      bnd_vec[p].rmax.z);
    }
  }

  /////////

  boundary& get_bnd  (int proc) {return bnd_vec[proc];};
  boundary& get_outer(int proc) {return outer  [proc];};

  ///////
  
  int which_box(const pfloat3 pos) {
    for (int proc = 0; proc < nproc; proc++) {
      if (bnd_vec[proc].isinbox(pos)) return proc;
    }
    return -1;
  }
   
};

#endif // _DISTRIBUTE_H_

