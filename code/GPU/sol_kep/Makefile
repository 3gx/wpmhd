SHELL = /bin/sh
CC = gcc
FC = g77
#CFLAGS = -march=athlon64 -mmmx -msse -msse2
#ALL_CFLAGS = -Wall -O3 -fpeel-loops -DNDEBUG $(CFLAGS)
#ALL_CFLAGS = -Wall -g -O0 $(CFLAGS)

# modify the following for better optimization
ALL_CFLAGS = -Wall -O2

# use the following if you have SWIFT routines
#OBJS = Kep_Drift_test.o \
       CF_kep_drift.o \
       drift_dan.o drift_kepu.o drift_kepmd.o drift_kepu_guess.o drift_kepu_new.o drift_kepu_fchk.o drift_kepu_lag.o drift_kepu_stumpff.o drift_kepu_p3solve.o orbel_scget.o

OBJS = Kep_Drift_test.o CF_kep_drift.o 

LIBS = -lgsl -lgslcblas -lm
EXE = Kep_Drift_test

# pattern rule to compile object files from C and Fortran files
# might not work with make programs other than GNU make
%.o : %.c Makefile
	@ $(CC) $(ALL_CFLAGS) -c $< -o $@
	@ echo '[CC] -c' $< '-o' $@ 
%.o : %.f Makefile
	@ $(FC) $(ALL_CFLAGS)  -c $< -o $@
	@ echo '[FC] -c' $< '-o' $@ 

all: $(EXE)

$(EXE): $(OBJS) Makefile
	@ $(CC) $(ALL_CFLAGS) $(OBJS) -o $(EXE) $(LIBS)
	@ echo -e '\n[CC]' $(OBJS) '\n     -o' $(EXE) 

.PHONY : clean
clean:
	rm -f $(OBJS) $(EXE)
