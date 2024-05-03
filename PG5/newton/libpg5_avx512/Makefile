include ../setting.mk

LIBPG5OBJ = gravity.o phantom_g5.o rsqrt.o pg5_fortran.o

libpg5.a: $(LIBPG5OBJ)
	ar -r $@ $^
	ranlib $@

asm::
	$(CC) $(CFLAGS) -S gravity.c

clean:
	rm -rf *.o

distclean: clean
	rm -rf *~
	rm -rf libpg5.a

.c.o:
	$(CXX) $(ARCH) $(CXXFLAGS) -c $*.c -o $*.o

.cpp.o:
	$(CXX) $(ARCH) $(CXXFLAGS) -c $*.cpp -o $*.o