CXX = mpicxx
CC  = mipcc
LD  = mpicxx
F90  = ifort

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
OFLAGS = -O4 -m64 -g -Wall -mtune=core2
#OFLAGS += -D_DEBUG_PRINT_
OFLAGS += -D_DEBUG_ 
#OFLAGS += -D_CONSERVATIVE_
OFLAGS += -D_SEMICONSERVATIVE_
CFLAGS = $(OFLAGS) -fPIC -I$(CUDA_TK)/include  -D__$(MACOSX)__
# -I/opt/local/include
CXXFLAGS = $(CFLAGS)

NVCC      = $(CUDA_TK)/bin/nvcc  
NVCCFLAGS = -D_DEBUG -O0 -g -I$(CUDA_SDK)/common/inc -arch=sm_12 -code=sm_12\
	--maxrregcount=64 \
# 	--device-emulation



PROG    = main
PROG3D  = main3d
SPMHDLIB = gn

LIBOBJS = setup_particles.o \
	distribute_particles.o \
	build_octree.o \
	myMPI.o \
	import_boundary_particles.o \
	ngb_search.o \
	compute_weights.o \
	morton_order.o \
	renorm.o \
	gradient.o \
	gradient_v4sf.o \
	gradient_ppm.o \
	mhd_interaction.o \
	solve_riemann.o \
	derivs.o \
	push_particles.o \
	iterate.o  \
	vain.o \
	dump_snapshot.o \
	dump_binary.o \
	compute_dt.o \
	compute_defect.o \
	convert_to_primitives.o \
	improve_weights.o \
	collect_sample_coords.o

CUOBJS =

LIBS = -L$(CUDA_TK)/lib 
# -lcudart 
# -L/opt/local/lib -lgsl -lgslcblas



all: $(SPMHDLIB) $(PROG) $(PROG3D)

$(PROG): $(PROG).o
	$(LD) $(CXXFLAGS) $(LIBS) $^ -o $@ -L. -l$(SPMHDLIB)

$(PROG3D): $(PROG3D).o
	$(LD) $(CXXFLAGS) $(LIBS) $^ -o $@ -L. -l$(SPMHDLIB)

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

$(PROG).o:  $(SPMHDLIB)
$(PROG3D).o:  $(SPMHDLIB)
bbuild_tree.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h
distribute_particles.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h
import_boundary_particles.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h
ngb_search.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h
myMPI.o: primitives.h myMPI.h
setup_particles.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h \
	orszag_tang.cpp \
	advect_pulse.cpp \
	mhd_rotor.cpp \
	khi.cpp \
	khi3d.cpp \
	advect3d.cpp \
	blast3d.cpp \
	current_sheet.cpp
compute_weights.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
renorm.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
gradient.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
gradient_ppm.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
mhd_interaction.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
solve_riemann.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
derivs.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
push_particles.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
iterate.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
vain.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
dump_snapshot.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
dump_binary.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
compute_dt.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
compute_defect.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
improve_weights.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
collect_sample_coords.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h

gradient_v4sf.o: gn.h boundary.h primitives.h distribute.h node.h memory_pool.h myMPI.h pfloat.h nfloat.h
