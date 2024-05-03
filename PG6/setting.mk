CC = gcc
CXX = g++
CFLAGS = -O3 -ffast-math -funroll-loops

enable_openmp=yes

#instruction_set=avx
instruction_set=avx2
#instruction_set=avx512

ifeq ($(enable_openmp), yes)
CFLAGS += -fopenmp 
endif

ifeq ($(instruction_set), avx)
ARCH = -mavx
LIBPG6_DIR = libpg6
endif
ifeq ($(instruction_set), avx2)
ARCH = -mavx2 -mfma
LIBPG6_DIR = libpg6
endif
ifeq ($(instruction_set), avx512)
ARCH = -mavx512f -mavx512dq
LIBPG6_DIR = libpg6_avx512
endif

CFLAGS += -DNORMAL
#CFLAGS += -DONLYNNB
#CFLAGS += -DONLYNEIGHBOUR


CXXFLAGS = $(CFLAGS)
