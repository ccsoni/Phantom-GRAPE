#CC = g++ -march=core-avx2
#CFLAGS = -Wall -O3 -ffast-math -funroll-loops -mfma

#enable_openmp=yes
#enable_avx2=yes
#enable_symmetric=no
#enable_quadrupole=yes

#ifeq ($(enable_openmp), yes)
#LDFLAGS += -fopenmp
#CFLAGS += -fopenmp -DENABLE_OPENMP
#endif

#ifeq ($(enable_symmetric), yes)
#CFLAGS += -DSYMMETRIC
#endif

include ./setting.mk
CFLAGS += -mfma
OBJS = source.o ./quad/quad.a

source: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) -lm
source.o: source.cpp
	$(CXX) $(CFLAGS) -c source.cpp
./quad/quad.a:
	make -C ./quad -f Makefile CC="$(CC)" CFLAGS="$(CFLAGS)" quad.a

clean:
	rm -rf *.o
	make -C ./quad -f Makefile clean
distclean: clean
	rm -rf *~
	make -C ./quad -f Makefile distclean
