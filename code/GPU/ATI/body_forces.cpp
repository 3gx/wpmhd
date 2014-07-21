#include "gn.h"

#if 0
float4 system::body_forces(const pfloat3 pos) {
  const float3 fpos = {pos.x.getu(), pos.y.getu(), pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2   = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + gravity_eps2;
  const float ids   = (ds2 > 0.0f) ? 1.0f/sqrt(ds2) : 0.0f;
  const float ids2  = ids*ids;
  const float mids  = ids  * gravity_mass;
  const float mids3 = ids2 * mids;
  
  return float4(-mids3*dr.x, -mids3*dr.y, -mids3*dr.z, -mids);
};


bool system::remove_particles_within_racc(const int idx) {
  if (gravity_rin == 0.0f && gravity_rout == 0.0f) return false;
  const particle &pi = pvec[idx];
  const float3 fpos = {pi.pos.x.getu(), pi.pos.y.getu(), pi.pos.z.getu()};
  const float3 dr   = {fpos.x - gravity_pos.x,
		       fpos.y - gravity_pos.y,
		       fpos.z - gravity_pos.z};
  const float ds2   = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  
  if (ds2 <= sqr(gravity_rin) || ds2 >= sqr(gravity_rout)) {
    local_n--;
    for (int j = idx; j < local_n; j++) {
      pvec[j] = pvec[j+1];
      pmhd[j] = pmhd[j+1];
    }
      
//     std::swap(pvec    [idx], pvec    [local_n-1]);
//     std::swap(pmhd    [idx], pmhd    [local_n-1]);
//     std::swap(pmhd_dot[idx], pmhd_dot[local_n-1]);
//     pvec[idx].local_idx = idx;
//     local_n--;
  }

  return (ds2 <= 0.0f);
  
}
#endif
