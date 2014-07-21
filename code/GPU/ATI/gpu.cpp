#if 1

#include "gn.h"

#else

#include "gpu.h"
#endif


void gpu_struct::load_kernels() {

#if 1
  const char *flags = "-DNMAXPERLANE=16 -DNDIM=2 -D_PERIODIC_BOUNDARIES_=1 -O0";
#else
  const char *flags = "-DNMAXPERLANE=16 -DNDIM=3";
#endif

#ifdef __MACOSX__
  const int maxrregcount = -1; //64;
#else
  const int maxrregcount = 64;
#endif
  fprintf(stderr, "Loading compute_wngb kernel ...\n");
  compute_wngb.setContext(context);
  compute_wngb.load_source("CLkernels/compute_wngb.cl", "", flags, maxrregcount);
  compute_wngb.create("compute_wngb");

  fprintf(stderr, "Loading compute_nngb kernel ...\n");
  compute_nngb.setContext(context);
  compute_nngb.load_source("CLkernels/compute_nngb.cl", "", flags, maxrregcount);
  compute_nngb.create("compute_nngb");

  fprintf(stderr, "Loading extract_gather_list kernel ... \n");
  extract_gather_list.setContext(context);
  extract_gather_list.load_source("CLkernels/extract_ijlist.cl", "", flags, maxrregcount);
  extract_gather_list.create("extract_gather_list");

  fprintf(stderr, "Loading compute_Bmatrix kernel ...\n");
  compute_Bmatrix.setContext(context);
  compute_Bmatrix.load_source("CLkernels/compute_Bmatrix.cl", "", flags, maxrregcount);
  compute_Bmatrix.create("compute_Bmatrix");

  fprintf(stderr, "Loading compute_gradient kernel ...\n");
  compute_gradient.setContext(context);
  compute_gradient.load_source("CLkernels/compute_gradient.cl", "", flags, maxrregcount);
  compute_gradient.create("compute_gradient");

  fprintf(stderr, "Loading limit_gradient kernel ...\n");
  limit_gradient.setContext(context);
  limit_gradient.load_source("CLkernels/compute_gradient.cl", "", flags, maxrregcount);
  limit_gradient.create("limit_gradient");
  
  fprintf(stderr, "Loading limit_gradient2 kernel ...\n");
  limit_gradient2.setContext(context);
  limit_gradient2.load_source("CLkernels/compute_gradient.cl", "", flags, maxrregcount);
  limit_gradient2.create("limit_gradient2");

  fprintf(stderr, "Loading extract_ijlist kernel ... \n");
  extract_ijlist.setContext(context);
  extract_ijlist.load_source("CLkernels/extract_ijlist.cl", "", flags, maxrregcount);
  extract_ijlist.create("extract_ijlist");

  fprintf(stderr, "Loading compute_dwij kernel ... \n");
  compute_dwij.setContext(context);
  compute_dwij.load_source("CLkernels/compute_dwij.cl", "", flags, maxrregcount);
  compute_dwij.create("compute_dwij");

  fprintf(stderr, "Loading compute_states kernel ... \n");
  compute_states.setContext(context);
  compute_states.load_source("CLkernels/compute_states.cl", "", flags, maxrregcount);
  compute_states.create("compute_states");

  fprintf(stderr, "Loading compute_fluxes kernel ... \n");
  compute_fluxes.setContext(context);
  compute_fluxes.load_source("CLkernels/compute_fluxes.cl", "", flags, maxrregcount);
  compute_fluxes.create("compute_fluxes");

  fprintf(stderr, "Loading reduce_fluxes kernel ... \n");
  reduce_fluxes.setContext(context);
  reduce_fluxes.load_source("CLkernels/reduce_fluxes.cl", "", flags, maxrregcount);
  reduce_fluxes.create("reduce_fluxes");

  fprintf(stderr, "Done loading kernels \n");
  


}

void gpu_struct::init() {
  
#if 0
  context.create(false, CL_DEVICE_TYPE_CPU);
#elif 1
  context.create(false, CL_DEVICE_TYPE_GPU);
#else
  context.create(false, CL_DEVICE_TYPE_DEFAULT);

#endif
  context.createQueue(0);

  leaf_list.setContext(context);
  body_list.setContext(context);
  ilist.setContext(context);

  jlist.setContext(context);
  joffset.setContext(context);

  ppos.setContext(context);
  pvel.setContext(context);

  in_h_out_wngb.setContext(context);

  Bxx.setContext(context);
  Bxy.setContext(context);
  Bxz.setContext(context);
  Byy.setContext(context);
  Byz.setContext(context);
  Bzz.setContext(context);

  nngb.setContext(context);
  
  mhd1.setContext(context);
  mhd2.setContext(context);
  mhd3.setContext(context);
  
  mhd1_grad_x.setContext(context);
  mhd1_grad_y.setContext(context);
  mhd1_grad_z.setContext(context);
  mhd2_grad_x.setContext(context);
  mhd2_grad_y.setContext(context);
  mhd2_grad_z.setContext(context);
  mhd3_grad_x.setContext(context);
  mhd3_grad_y.setContext(context);
  mhd3_grad_z.setContext(context);

  mhd1_psi.setContext(context);
  mhd2_psi.setContext(context);
  mhd3_psi.setContext(context);

  ijlist.setContext(context);
  ijlist_offset.setContext(context);
  leaf_ngb_max.setContext(context);
  
  drij.setContext(context);
  dwij.setContext(context);

  mhd1_statesL.setContext(context);
  mhd1_statesR.setContext(context);
  mhd2_statesL.setContext(context);
  mhd2_statesR.setContext(context);
  mhd3_statesL.setContext(context);
  mhd3_statesR.setContext(context);
  fluxes1.setContext(context);
  fluxes2.setContext(context);
  fluxes3.setContext(context);
  divBij.setContext(context);
  dndt_ij.setContext(context);

  dqdt1.setContext(context);
  dqdt2.setContext(context);
  dqdt3.setContext(context);
  divB.setContext(context);
  dndt.setContext(context);

  load_kernels();

}

#ifdef _TOOLBOX_
int main(int argc, char *argv[]) {
  gpu_struct gpu;
  gpu.init();


  const int M = 1024*1024/16;
  int alloc = 1;

  fprintf(stderr, "Alloc %d\n", alloc++);
  gpu.fluxes1.cmalloc(128*M);

  fprintf(stderr, "Alloc %d\n", alloc++);
  gpu.fluxes2.cmalloc(127*M);

  fprintf(stderr, "Alloc %d\n", alloc++);
  gpu.fluxes3.cmalloc(10*M);

  fprintf(stderr, "Alloc %d\n", alloc++);
  gpu.divB.cmalloc(3*M);
//   const int nthreads = 64;
//   gpu.compute_Bmatrix.setWork(-1, nthreads, NGPUBLOCKS);
  
//   gpu.Bxx.cmalloc(1024);
//   gpu.ngb_gather.cmalloc(102400);
//   gpu.ngb_both.cmalloc(102400);

//   gpu.compute_Bmatrix.set_arg<void* >( 0, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 1, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 2, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 3, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 4, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 5, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 6, gpu.ngb_gather.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 7, gpu.ngb_both.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 8, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >( 9, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >(10, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >(11, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >(12, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<void* >(13, gpu.Bxx.p());
//   gpu.compute_Bmatrix.set_arg<float3>(14, &gpu.domain_hsize);
//   gpu.compute_Bmatrix.set_arg<float >(15, NULL, 4*nthreads);
  
//   gpu.compute_Bmatrix.execute();
//   clFinish(oclContext.get_command_queue());


//   fprintf(stderr, "GlobalWork= %u %u  LocalWork= %u %u \n",
// 	  gpu.compute_Bmatrix.hGlobalWork[0], gpu.compute_Bmatrix.hGlobalWork[1],
// 	  gpu.compute_Bmatrix.hLocalWork[0], gpu.compute_Bmatrix.hLocalWork[1]);
	      

//   gpu.ngb_gather.d2h();
//   gpu.ngb_both.d2h();
//   for (int i = 0; i < 10; i++) {
//     fprintf(stderr, "i= %d:  ngb= %d %d\n",
// 	    i,
// 	    gpu.ngb_gather[i], gpu.ngb_both[i]);
//   }
  

  fprintf(stderr, "end-of-program\n");
}
#endif
