#ifndef QUAD_TYPE
#define QUAD_TYPE

typedef struct jcdata0{
  float x[8], y[8], z[8], m[8];
  float q[6][8];
} Jcdata0, *cJcdata0;

typedef struct jcdata{
  float xm[2][4];
  float q[2][8];
} Jcdata, *cJcdata;

#endif
