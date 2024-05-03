include ../setting.mk

TARGET = banana01

OBJ    = $(TARGET).o io.o calc_energy.o hermite.o \
	timestep.o timeprof.o
HEADER = $(TARGET).h particle.h io.h calc_energy.h hermite.h \
	globaltime.h timestep.h step.h timeprof.h
ETC = Makefile

LIBDIR = ../$(LIBPG6_DIR)
LDFLGS = -L$(LIBDIR) -lpg6
LIBFLE = $(LIBDIR)/libpg6.a
INCDIR = $(LIBDIR)
ICFLGS = -I$(INCDIR)

ifeq ($(instruction_set), avx512) 
CC_LINK = $(CXX)
else
CC_LINK = $(CC)
endif

$(TARGET) : $(OBJ) $(HEADER) $(ETC) $(LIBFLE)
	$(CC_LINK) $(CFLAGS) -o $(TARGET) $(OBJ) -lm $(LDFLGS)
$(TARGET).o : $(TARGET).c $(HEADER) $(ETC)
	$(CC) $(CFLAGS) -c $(TARGET).c $(ICFLGS)
io.o : io.c $(HEADER) $(ETC)
	$(CC) $(CFLAGS) -c io.c
calc_energy.o : calc_energy.c $(HEADER) $(ETC)
	$(CC) $(CFLAGS) -c calc_energy.c
hermite.o : hermite.c $(HEADER) $(ETC)
	$(CC) $(CFLAGS) -c hermite.c $(ICFLGS)
timestep.o : timestep.c $(HEADER) $(ETC)
	$(CC) $(CFLAGS) -c timestep.c
timeprof.o : timeprof.c $(HEADER) $(ETC)
	$(CC) $(CFLAGS) -c timeprof.c

$(LIBFLE):
	make -C ../$(LIBPG6_DIR)

$(PHANTOM).s : $(PHANTOM).c $(HEADER) $(ETC)
	$(CC) -S $(CFLAGS) -c $(PHANTOM).c

clean :
	rm -f *.o *.s 

distclean: clean
	$(MAKE) -C ../$(LIBPG6_DIR) distclean
	rm -f *~ $(TARGET)

