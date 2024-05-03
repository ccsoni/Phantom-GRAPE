include ../setting.mk

ifeq ($(instruction_set), avx512)
instruction_set=avx2
ARCH = -mavx2 -mfma
endif

OBJS = gravity_kernel.o phantom_g5.o pg5_table.o

enable_ntdr = yes

ifeq ($(instruction_set), avx2)
OBJS = gravity_kernel_avx2.o phantom_g5.o pg5_table.o
endif

ifeq ($(instruction_set), avx2_ntdr)
OBJS = gravity_kernel_avx2_ntdr.o phantom_g5.o pg5_table.o
endif

libpg5.a: $(OBJS)
	ar -r $@ $^
	ranlib $@

clean:
	rm -rf *.o

distclean: clean
	rm -rf *~
	rm -rf libpg5.a

.c.o:
	$(CC) $(ARCH) -c $*.c -o $*.o