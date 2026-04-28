include ../setting.mk

LIBPG5OBJ = gravity.o phantom_g5.o pg5_table.o

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
