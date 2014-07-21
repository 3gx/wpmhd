#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

#include "primitives.h"
#include "pfloat.h"

#define SMALLB 1.0

#ifndef SMALLB
#define SMALLB (1.0 + 1.0e-6)
#endif

struct boundary {
  pfloat3 centre;
  pfloat3  hsize;
  bool     empty;
  
  boundary() {empty = true;}
  bool isempty() const {return empty;};
  
  boundary(const pfloat3 pos) {
    centre = pos;
    hsize.x.set(0.0);
    hsize.y.set(0.0);
    hsize.z.set(0.0);
    empty = false;
  }
  boundary(const pfloat3 pos, const float h) {
    centre = pos;
    const float h1 = h * SMALLB;
    hsize.x.set(h1);
    hsize.y.set(h1);
    hsize.z.set(h1);
    empty = false;
  }
  boundary(const float4 pos, const float scale_factor = 1.0) {
    float h = fabs(pos.w)*scale_factor*SMALLB;
    centre.x.set(pos.x);
    centre.y.set(pos.y);
    centre.z.set(pos.z);
    hsize.x.set(h);
    hsize.y.set(h);
    hsize.z.set(h);
    empty = false;
  }
  boundary(const  float3 rmin, const  float3 rmax) {set(rmin, rmax);}
  boundary(const pfloat3 r1,   const pfloat3 r2  ) {set(r1,   r2);  }
  boundary& set(pfloat3 r1, pfloat3 r2) {
    r1.x.half();
    r2.x.half();
    centre.x = r1.x; centre.x.add(r2.x);
    hsize.x  = r2.x;  hsize.x.sub(r1.x);
    
    r1.y.half();
    r2.y.half(); 
    centre.y = r1.y; centre.y.add(r2.y);
    hsize.y  = r2.y;  hsize.y.sub(r1.y);
    
    r1.z.half();
    r2.z.half();  
    centre.z = r1.z; centre.z.add(r2.z);
    hsize.z  = r2.z;  hsize.z.sub(r1.z);
    
    empty = false;
    
    return *this;
  }
  boundary& set_x(pfloat<0> x1, pfloat<0> x2) {
    x1.half();
    x2.half();
    centre.x = x1; centre.x.add(x2);
    hsize.x  = x2;  hsize.x.sub(x1);
    empty = false;    
    return *this;
  }
  boundary& set_y(pfloat<1> x1, pfloat<1> x2) {
    x1.half();
    x2.half();
    centre.y = x1; centre.y.add(x2);
    hsize.y  = x2;  hsize.y.sub(x1);
    empty = false;    
    return *this;
  }
  boundary& set_z(pfloat<2> x1, pfloat<2> x2) {
    x1.half();
    x2.half();
    centre.z = x1; centre.z.add(x2);
    hsize.z  = x2;  hsize.z.sub(x1);
    empty = false;    
    return *this;
  }

  boundary& set(const float3 rmin, const float3 rmax) {
    pfloat3 r1, r2;
    
    r1.x.set(rmin.x); 
    r1.y.set(rmin.y); 
    r1.z.set(rmin.z); 

    r2.x.set(rmax.x); 
    r2.y.set(rmax.y); 
    r2.z.set(rmax.z); 

    return set(r1, r2);
  }
  boundary& set_x(const float x1, const float x2) {
    pfloat<0> px1, px2;
    
    px1.set(x1); 
    px2.set(x2); 

    return set_x(px1, px2);
  }
  boundary& set_y(const float x1, const float x2) {
    pfloat<1> px1, px2;
    
    px1.set(x1); 
    px2.set(x2); 

    return set_y(px1, px2);
  }
  boundary& set_z(const float x1, const float x2) {
    pfloat<2> px1, px2;
    
    px1.set(x1); 
    px2.set(x2); 

    return set_z(px1, px2);
  }

  
  /***************/
  
  pfloat3 rmin() const {
    assert(!isempty());
    pfloat3 r = centre;
    r.x.sub(hsize.x);
    r.y.sub(hsize.y);
    r.z.sub(hsize.z);
    return r;
  }
  pfloat3 rmax() const {
    assert(!isempty());
    pfloat3 r = centre;
    r.x.add(hsize.x);
    r.y.add(hsize.y);
    r.z.add(hsize.z);
    return r;
  }

  float3 half_size() const {
    assert(!isempty());
    return (float3){hsize.x.getu(), hsize.y.getu(), hsize.z.getu()};
  }
  float3 size() const {
    const float3 size = half_size();
    return (float3){2.0f*size.x, 2.0f*size.y, 2.0f*size.z};
  }

  ///////////
  
  boundary& merge(const boundary &b) {
    if (b.isempty()) return *this;
    if (isempty()) {
      centre = b.centre; 
      hsize  = b.hsize;
      empty  = b.empty;
      return *this;
    }
    
    pfloat_merge(centre.x, hsize.x, b.centre.x, b.hsize.x, 
		 centre.x, hsize.x);
    pfloat_merge(centre.y, hsize.y, b.centre.y, b.hsize.y, 
		 centre.y, hsize.y);
    pfloat_merge(centre.z, hsize.z, b.centre.z, b.hsize.z, 
		 centre.z, hsize.z);
    
//     pfloat3 rmin1 =   rmin();
//     pfloat3 rmax1 =   rmax();
//     pfloat3 rmin2 = b.rmin();
//     pfloat3 rmax2 = b.rmax();
    

//     rmin1.x.min(rmin2.x);
//     rmin1.y.min(rmin2.y);
//     rmin1.z.min(rmin2.z);

//     rmax1.x.max(rmax2.x);
//     rmax1.y.max(rmax2.y);
//     rmax1.z.max(rmax2.z);

//     return set(rmin1, rmax1);
    return *this;
  }

  /////////
  
  bool overlap(const boundary &b) const {
    if (isempty()) return false;
//     assert(!isempty());
    pfloat3 dr = centre;
    dr.x.psub(b.centre.x);
    dr.y.psub(b.centre.y);
    dr.z.psub(b.centre.z);
    
    pfloat3 ds = hsize;
    ds.x.add(b.hsize.x);
    ds.y.add(b.hsize.y);
    ds.z.add(b.hsize.z);
    
//     fprintf(stderr, "dr= %g %g %g\n", 
// 	    dr.x.uval,
// 	    dr.y.uval,
// 	    dr.z.uval);
//     fprintf(stderr, "ds= %g %g %g\n", 
// 	    ds.x.uval,
// 	    ds.y.uval,
// 	    ds.z.uval);

//     bool bx = dr.x.aleq(ds.x);
//     bool by = dr.y.aleq(ds.y);
//     bool bz = dr.z.aleq(ds.z);

//     fprintf(stderr, "bx= %d\n", bx);
//     fprintf(stderr, "by= %d\n", by);
//     fprintf(stderr, "bz= %d\n", bz);


    return (dr.x.aleq(ds.x) && 
	    dr.y.aleq(ds.y) &&
	    dr.z.aleq(ds.z));
  } 
  
  bool isinbox(const pfloat3 pos) const {
    assert(!isempty());
    pfloat3 dr = centre;
    dr.x.psub(pos.x);
    dr.y.psub(pos.y);
    dr.z.psub(pos.z);
    
    pfloat3 ds = hsize;
    
    return (dr.x.aleq(ds.x) && 
	    dr.y.aleq(ds.y) &&
	    dr.z.aleq(ds.z));
  }
  
  bool isinbox(const boundary &b) const {
    assert(!isempty());
    pfloat3 dr = centre;
    dr.x.psub(b.centre.x);
    dr.y.psub(b.centre.y);
    dr.z.psub(b.centre.z);
    
    pfloat3 ds = hsize;
    ds.x.sub(b.hsize.x);
    ds.y.sub(b.hsize.y);
    ds.z.sub(b.hsize.z);
    
    return (dr.x.aleq(ds.x) && 
	    dr.y.aleq(ds.y) &&
	    dr.z.aleq(ds.z));
  }
  
  void dump(FILE *fout, bool flag = false) const {
    assert(!isempty());
    fprintf(fout, "c= %20.16lg %20.16lg %20.16lg; hs= %20.16lg %20.16lg %20.16lg", 
	    centre.x.getu(), centre.y.getu(), centre.z.getu(),
	     hsize.x.getu(),  hsize.y.getu(),  hsize.z.getu());
    if (flag) fprintf(fout, "\n");
  }
};

#endif // _BOUNDARY_H_

#ifdef _TOOLBOX_BOUNDARY_
int main(int argc, char *argv[]) {
  boundary b1;
  boundary b2;
  
  b1.centre = (pfloat3){0.3f, 0.0f, 0.0f};
  b2.centre = (pfloat3){0.8f, 0.0f, 0.0f};

  b1.hsize = (float3){0.3f/2, 0.0f, 0.0f};
  b2.hsize = (float3){0.3f/2, 0.0f, 0.0f};
  
  boundary b3 = b1;
  boundary b4 = b2;

  fprintf(stderr, "b1= "); b1.dump(stderr); fprintf(stderr, "\n");
  fprintf(stderr, "b2= "); b2.dump(stderr); fprintf(stderr, "\n");
  fprintf(stderr, "b3= "); 
//   b1.pmerge(b2);
  b3.merge(b4);

  b1.dump(stderr); 
  fprintf(stderr, "\nb4= "); 
  b3.dump(stderr); 
  fprintf(stderr, "\n");
  
  return 0;
}
#endif
