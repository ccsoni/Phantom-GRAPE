CC = gcc
CXX = g++

CFLAGS = -O3 -ffast-math -funroll-loops

enable_openmp=no
enable_symmetric=no
enable_approx_rsqrt=yes

#instruction_set=avx
#instruction_set=avx2
#instruction_set=avx512
instruction_set=sve_128
#instruction_set=sve_512


ifeq ($(enable_openmp), yes)
LDFLAGS += -fopenmp
CFLAGS += -fopenmp -DENABLE_OPENMP
endif

ifeq ($(enable_symmetric), yes)
CFLAGS += -D__SYMMETRIC_EPS__
endif

ifeq ($(enable_approx_rsqrt), yes)
CFLAGS += -D__APPROX_RCP_RSQRT__
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

ifeq ($(instruction_set), sve_128)
ARCH = -march=armv8.2-a+sve -msve-vector-bits=128
LIBPG5_DIR = libpg5_sve
endif

ifeq ($(instruction_set), sve_512)
ARCH = -march=armv8.2-a+sve -msve-vector-bits=512
LIBPG5_DIR = libpg5_sve
endif

CXXFLAGS = $(CFLAGS)
