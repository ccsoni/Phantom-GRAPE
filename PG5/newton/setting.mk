CC = gcc
CXX = g++
CFLAGS = -O3 -ffast-math -funroll-loops

enable_openmp=no
enable_symmetric=no

#instruction_set=avx
instruction_set=avx2
#instruction_set=avx512

ifeq ($(enable_openmp), yes)
LDFLAGS += -fopenmp
CFLAGS += -fopenmp -DENABLE_OPENMP
endif

ifeq ($(enable_symmetric), yes)
CFLAGS += -DSYMMETRIC
endif

ifeq ($(instruction_set), avx)
ARCH = -mavx
LIBPG5_DIR = libpg5
endif
ifeq ($(instruction_set), avx2)
ARCH = -mavx2 -mfma
LIBPG5_DIR = libpg5
endif
ifeq ($(instruction_set), avx512)
ARCH = -mavx512f -mavx512dq
LIBPG5_DIR = libpg5_avx512
endif

CXXFLAGS = $(CFLAGS)
