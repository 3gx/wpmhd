CXX = mpicxx
CC  = mipcc
LD  = mpicxx
F90  = ifort

#LIBOPENCL = -framework OpenCL -fopenmp
LIBOPENCL = -lOpenCL -fopenmp

# CXX = icpc
# CC  = icc
# LD  = icpc
# F90  = ifort

.SUFFIXES: .o .f90 .cpp .cu

CUDA_SDK = /usr/local/cuda-sdk
CUDA_TK  = /usr/local/cuda

#OFLAGS = -D'drand48()=((double)rand ()/RAND_MAX)' -fpermissive -O4 -g -Wall -m64
#
# -ffast-math
OFLAGS = -O4 -m64 -g -Wall -fopenmp
#OFLAGS += -D_DEBUG_PRINT_
#OFLAGS += -D_DEBUG_ 
#OFLAGS += -D_CONSERVATIVE_
#OFLAGS += -D_SEMICONSERVATIVE_
CFLAGS = $(OFLAGS) -fPIC -I$(CUDA_TK)/include  -D__$(MACOSX)__ -I/home/egaburov/usr/include
# -I/opt/local/include
CXXFLAGS = $(CFLAGS)

CFLAGS += -I/export/opt/amdstream.opencl/ati-stream-sdk-v2.01-lnx64/include
CXXFLAGS += -I/export/opt/amdstream.opencl/ati-stream-sdk-v2.01-lnx64/include
# LIBOPENCL += -L/export/opt/amdstream.opencl/ati-stream-sdk-v2.01-lnx64/lib/x86_64

NVCC      = $(CUDA_TK)/bin/nvcc  
NVCCFLAGS = -D_DEBUG -O0 -g -I$(CUDA_SDK)/common/inc -arch=sm_12 -code=sm_12\
	--maxrregcount=64 \
# 	--device-emulation



PROG    = main
SPMHDLIB = gn

LIBOBJS = setup_particles.o \
	distribute_particles.o \
	build_octree.o \
	myMPI.o \
	import_boundary_particles.o \
	morton_order.o \
	derivs.o \
	push_particles.o \
	iterate.o  \
	dump_binary.o \
	read_binary.o \
	compute_dt.o \
	convert_to_primitives.o \
	collect_sample_coords.o \
	gpu.o \
	gpu_calls.o \
	renorm_cl.o \
	mhd_interaction_cl.o \
	compute_weights_cl.o # compute_weights.o 

CUOBJS =

LIBS = -L$(CUDA_TK)/lib  -L/opt/local/lib -L/home/egaburov/usr/lib -lgsl -lgslcblas $(LIBOPENCL)
# -lcudart 
# -L/opt/local/lib -lgsl -lgslcblas



all: $(SPMHDLIB) $(PROG) 
# $(PROG3D)

$(PROG): $(PROG).o
	$(LD) $(CXXFLAGS) $(LIBS) $^ -o $@ -L. -l$(SPMHDLIB)

# $(PROG3D): $(PROG3D).o
# 	$(LD) $(CXXFLAGS) $(LIBS) $^ -o $@ -L. -l$(SPMHDLIB)

$(SPMHDLIB): $(LIBOBJS) $(CUOBJS)
	/bin/rm -rf lib$@.a
	ar qv lib$@.a $^
	ranlib lib$@.a



.cpp.o: 
	$(CXX) $(CXXFLAGS) -c $< -o $@

.f90.o:
	$(F90) $(F90FLAGS) -c $< -o $@

%.cu_o:  %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

convert:
	ifort -O3 -o convert convert.f90

clean:
	/bin/rm -rf *.o *.cu_o *.cubin *.ptx 
	/bin/rm -rf $(PROG) lib$(SPMHDLIB).a

# $(PROG).o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
# $(PROG3D).o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
$(PROG).o:  $(SPMHDLIB)
# $(PROG3D).o:  $(SPMHDLIB)
bbuild_tree.o: gn.h boundary.h primitives.h distribute.h node.h 
distribute_particles.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h
import_boundary_particles.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h
ngb_search.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h
myMPI.o: primitives.h myMPI.h
setup_particles.o: gn.h boundary.h primitives.h distribute.h node.h  \
	orszag_tang.cpp \
	orszag_tang3d.cpp \
	advect_pulse.cpp \
	blob3d.cpp \
	mhd_rotor.cpp \
	khi.cpp \
	khi3d.cpp \
	advect3d.cpp \
	blast2d.cpp \
	blast3d.cpp \
	current_sheet.cpp \
	mri2d.cpp mri3d.cpp \
	cloud2d.cpp disk3d.cpp \
	cloud3d.cpp \
	eccentric3d.cpp \
	brio_wu.cpp \
	pair_instability.cpp
compute_weights.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
renorm.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h 
gradient.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
gradient_ppm.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
mhd_interaction.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
body_forces.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
solve_riemann.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
derivs.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
iterate.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
vain.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
dump_snapshot.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
dump_binary.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
read_binary.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
compute_dt.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
compute_defect.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
improve_weights.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
collect_sample_coords.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
convert_to_primitives.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h

renorm_cl.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h  gpu.h my_ocl.h
gradient_cl.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h gpu.h my_ocl.h
gpu_calls.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h gpu.h my_ocl.h
gpu.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h gpu.h my_ocl.h
mhd_interaction_cl.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
compute_weightss_cl.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h
push_particles.o: gn.h boundary.h primitives.h distribute.h node.h  myMPI.h pfloat.h nfloat.h  sol_kep/kep_drift.h \
	push_particles.dkd.cpp push_particles.kdk.cpp
