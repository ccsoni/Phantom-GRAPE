include ../setting.mk

ifeq ($(instruction_set), avx512)
instruction_set=avx2
ARCH = -mavx2 -mfma
endif

LIBPG5OBJ = gravity.o phantom_g5.o rsqrt.o pg5_fortran.o
ifeq ($(instruction_set), avx2)
LIBPG5OBJ = gravity_avx2.o phantom_g5.o rsqrt.o pg5_fortran.o
endif


libpg5.a: $(LIBPG5OBJ)
	ar -r $@ $^
	ranlib $@

clean:
	rm -rf *.o

distclean: clean
	rm -rf *~
	rm -rf libpg5.a

.c.o:
	$(CC) $(CFLAGS) $(ARCH) -c $*.c -o $*.o
