#CC=gcc
#CFLAGS=-mavx -O3 -Wall -ffast-math -funroll-loops

#ifeq ($(enable_openmp), yes)
#CFLAGS += -fopenmp -DENABLE_OPENMP
#endif

#LIBPG5OBJ = gravity.o phantom_g5.o rsqrt.o pg5_fortran.o
#ifeq ($(enable_avx2), yes)
#CFLAGS += -fopenmp -DENABLE_AVX2
#LIBPG5OBJ = gravity_avx2.o phantom_g5.o rsqrt.o pg5_fortran.o
#endif

#ifeq ($(enable_quadrupole), yes)
LIBPG5OBJ = ../libpg5/gravity_avx2.o quad_avx2.o quad_g5c.o ../libpg5/rsqrt.o
#endif

quad.a: $(LIBPG5OBJ)
	ar -r  $@ $^
	ranlib $@

clean:
	rm -rf *.o
	rm -rf ../libpg5/*.o

distclean: clean
	rm -rf *~
	rm -rf ../libpg5/*~
	rm -rf quad.a
