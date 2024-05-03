include ../setting.mk 
AR = ar

HDR = gp6util.h gravity.h avx_type.h timeprof.h avx.h
OBJ = phantom_g6.o gravity.o timeprof.o
ETC = Makefile

libpg6.a : $(OBJ)
	$(AR) rv $@ $^
	ranlib $@

asm:
	$(CC) $(CFLAGS) -S gravity.c

clean:
	rm -f *.o *.s

distclean: clean
	rm -f *~ libpg6.a

.c.o:
	$(CXX) $(ARCH) $(CXXFLAGS) -c $*.c -o $*.o

.cpp.o:
	$(CXX) $(ARCH) $(CXXFLAGS) -c $*.cpp -o $*.o
