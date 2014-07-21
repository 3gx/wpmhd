#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

#include "primitives.h"
#include "periodic.h"

struct boundary {
  pfloat3 centre;
  float3  hsize;
  

  boundary() {
    centre = (pfloat3){0.0f,   0.0f,  0.0f};
    hsize  = (float3) {-1.0f, -1.0f,  0.0f};
  };
  
  bool isempty() const {
    if (hsize.x >= 0.0f && hsize.y >= 0.0f) return false;
    
    assert (hsize.x < 0 && hsize.y < 0);
    return true;
  }

  boundary(const pfloat3 pos) {
    centre = pos;
    hsize  = (float3){0,0,0};
    centre.z = 0.0;
  }
  boundary(const pfloat3 pos, const float h) {
    centre = pos;
    hsize  = (float3){h,h,0*h};
    centre.z = 0.0;
  }
  boundary(const pfloat4 pos, const float scale_factor = 1.0) {
    float h = fabs(pos.w)*scale_factor;
    centre = (pfloat3){pos.x, pos.y, 0.0};
    hsize  = (float3){h,h,0};
  }
  void set_x(float x1, float x2) {
    centre.x = (x2 + x1)*0.5f;
    hsize.x  = (x2 - x1)*0.5f;
  }
  void set_y(float x1, float x2) {
    centre.y = (x2 + x1)*0.5f;
    hsize.y  = (x2 - x1)*0.5f;
  }
  void set_z(float x1, float x2) {
    centre.z = 0*(x2 + x1)*0.5f;
    hsize.z  = 0*(x2 - x1)*0.5f;
  }


  float3 rmin() const {
    return (float3){(float)centre.x - hsize.x,
	(float)centre.y - hsize.y, 
	0.0};
  }
  
  float3 rmax() const {
    return (float3){(float)centre.x + hsize.x,
	(float)centre.y + hsize.y, 
	0.0};
  }


  


  float3 size() {return (float3){2*hsize.x, 2*hsize.y, 0.0};}
  ///////////
  
  void merge(const boundary &b) {
    if (b.isempty()) return;
    if (isempty()) {
      centre = b.centre; 
      hsize = b.hsize;
      hsize.z = 0;
      centre.z = 0.0;
      return;
    }
    
    float3 rmin1 = rmin();
    float3 rmax1 = rmax();
    
    const float3 rmin2 = b.rmin();
    const float3 rmax2 = b.rmax();
    
    rmin1.x = fmin(rmin1.x, rmin2.x);
    rmin1.y = fmin(rmin1.y, rmin2.y);
    rmin1.z = fmin(rmin1.z, rmin2.z);
    
    rmax1.x = fmax(rmax1.x, rmax2.x);
    rmax1.y = fmax(rmax1.y, rmax2.y);
    rmax1.z = fmax(rmax1.z, rmax2.z);
    
    centre = (pfloat3){0.5f*(rmax1.x + rmin1.x),
		       0.5f*(rmax1.y + rmin1.y),
		       0.0f*(rmax1.z + rmin1.z)};
    hsize  = (float3) {0.5f*(rmax1.x - rmin1.x),
		       0.5f*(rmax1.y - rmin1.y),
		       0.0f*(rmax1.z - rmin1.z)};
  }
  
  bool overlap(const boundary &b) const {
    const float3 dr = {centre.x - b.centre.x,
		       centre.y - b.centre.y,
		       centre.z - b.centre.z};
    return (fabs(dr.x) <= hsize.x + b.hsize.x && 
	    fabs(dr.y) <= hsize.y + b.hsize.y);
  } 

  bool isinbox(const pfloat3 pos) const {
    const float3 dr = {pos.x - centre.x, 
		       pos.y - centre.y, 
		       pos.z - centre.z};
    return (fabs(dr.x) <= hsize.x && 
	    fabs(dr.y) <= hsize.y);
  }
  
  bool isinbox(const boundary &b) const {
    const float3 dr = {centre.x - b.centre.x,
		       centre.y - b.centre.y,
		       centre.z - b.centre.z};
    return (fabs(dr.x) <= hsize.x - b.hsize.x && 
	    fabs(dr.y) <= hsize.y - b.hsize.y);
  }
  
  void dump(FILE *fout, bool flag = false) const {
    fprintf(fout, "c= %g %g %g; hs= %g %g %g", 
	    (float)centre.x, (float)centre.y, (float)centre.z,
	    (float)hsize.x,  (float)hsize.y,  (float)hsize.z);
    if (flag) fprintf(fout, "\n");
  }
  real volume() const {
    return hsize.x*hsize.y*hsize.z*8;
  }

#ifdef _PERIODIC_WITH_SHIFT_
  pfloat3 shift(const pfloat3 pos) const {
    float3 rmin1 = rmin();
    float3 rmax1 = rmax();
    float3 r     = {pos.x, pos.y, pos.z};
    float3 c     = {(rmin1.x + rmax1.x)*0.5f,
		    (rmin1.y + rmax1.y)*0.5f,
		    (rmin1.z + rmax1.z)*0.5f};
    
    float3 dr = {r.x - c.x, r.y - c.y, r.z - c.z};
    
    if (fabs(dr.x) > hsize.x) r.x -= sign(dr.x)*(pos.x.xpmax - pos.x.xpmin);
    if (fabs(dr.y) > hsize.y) r.y -= sign(dr.y)*(pos.y.xpmax - pos.y.xpmin);

    return (pfloat3){r.x, r.y, 0};
  };
#endif
  
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
