include ../setting.mk 
AR = ar

HDR = gp6util.h gravity.h avx_type.h timeprof.h avx.h
OBJ = phantom_g6.o gravity.o timeprof.o
ETC = Makefile

ifeq ($(instruction_set), avx2)
OBJ = phantom_g6.o gravity_avx2.o timeprof.o
endif

libpg6.a : $(OBJ)
	$(AR) rv $@ $^
	ranlib $@

clean:
	rm -f *.o *.s 

distclean: clean
	rm -f *~ libpg6.a

