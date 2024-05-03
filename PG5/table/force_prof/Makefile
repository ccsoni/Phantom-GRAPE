include ../setting.mk

OBJS = force_prof.o cpu.o ../$(LIBPG5_DIR)/libpg5.a

CFLAGS += -std=c99

force_prof: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) -lm
force_prof.o: force_prof.c
	$(CC) $(CFLAGS) -c force_prof.c
cpu.o: cpu.c
	$(CC) $(CFLAGS) -c cpu.c

../$(LIBPG5_DIR)/libpg5.a:
	make -C ../$(LIBPG5_DIR) libpg5.a

clean:
	rm -rf *.o

distclean: clean
	$(MAKE) -C ../$(LIBPG5_DIR) distclean
	rm -rf force_prof *~

