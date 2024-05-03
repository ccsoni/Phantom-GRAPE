include ../setting.mk

OBJS = direct.o cpu.o ../$(LIBPG5_DIR)/libpg5.a

direct: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) -lm
direct.o: direct.c
	$(CC) $(CFLAGS) -c direct.c
cpu.o: cpu.c
	$(CC) $(CFLAGS) -c cpu.c

../$(LIBPG5_DIR)/libpg5.a:
	make -C ../$(LIBPG5_DIR) libpg5.a

clean:
	rm -rf *.o

distclean: clean
	$(MAKE) -C ../$(LIBPG5_DIR) distclean
	rm -rf direct *~

