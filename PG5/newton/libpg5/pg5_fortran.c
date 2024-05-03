#include "gp5util.h"

#define FNAME(x) (x ## _)

int FNAME(g5_get_number_of_pipelines)()
{
  return g5_get_number_of_pipelines();
}

int FNAME(g5_get_jmemsize)(){
  return g5_get_jmemsize();
}

void FNAME(g5_open)()
{
  g5_open();
}

void FNAME(g5_close)(){
  g5_close();
}

void FNAME(g5_set_eps_to_all)(double *eps)
{
  g5_set_eps_to_all(*eps);
}

void FNAME(g5_set_n)(int *n)
{
  g5_set_n(*n);
}

void FNAME(g5_set_nMC)(int *devid, int *n)
{
  g5_set_nMC(*devid, *n);
}

void FNAME(g5_set_xi)(int *ni, double (*xi)[3])
{
  g5_set_xi(*ni, xi);
}

void FNAME(g5_set_xei)(int *ni, double (*xi)[3], double *eps2)
{
  g5_set_xei(*ni, xi, eps2);
}

void FNAME(g5_set_xiMC)(int *devid, int *ni, double (*xi)[3])
{
  g5_set_xiMC(*devid, *ni, xi);
}

void FNAME(g5_set_xeiMC)(int *devid, int *ni, double (*xi)[3], double *eps2)
{
  g5_set_xeiMC(*devid, *ni, xi, eps2);
}

#ifdef SYMMETRIC
void FNAME(g5_set_xmj)(int *adr, int *nj, double (*xj)[3], double *mj, double *eps2)
{
  g5_set_xmj(*adr, *nj, xj, mj, eps2);
}
#else
void FNAME(g5_set_xmj)(int *adr, int *nj, double (*xj)[3], double *mj)
{
  g5_set_xmj(*adr, *nj, xj, mj);
}
#endif

#ifdef SYMMETRIC
void FNAME(g5_set_xmjMC)(int *devid, int *adr, int *nj, double (*xj)[3], double *mj, double *eps2)
{
  g5_set_xmjMC(*devid, *adr, *nj, xj, mj, eps2);
}
#else
void FNAME(g5_set_xmjMC)(int *devid, int *adr, int *nj, double (*xj)[3], double *mj)
{
  g5_set_xmjMC(*devid, *adr, *nj, xj, mj);
}
#endif

void FNAME(g5_run)(){
  g5_run();
}

void FNAME(g5_runMC)(int *devid)
{
  g5_runMC(*devid);
}

void FNAME(g5_get_force)(int *ni, double (*a)[3], double *p)
{
  g5_get_force(*ni, a, p);
}

void FNAME(g5_get_forceMC)(int *devid, int *ni, double (*a)[3], double *p)
{
  g5_get_forceMC(*devid, *ni, a, p);
}

void FNAME(g5_calculate_force_on_x)(double (*x)[3], double (*a)[3], double *p, int *ni)
{
  g5_calculate_force_on_x(x, a, p, *ni);
}

void FNAME(g5_calculate_force_on_xe)(double (*x)[3], double *eps2, double (*a)[3], double *p, int *ni)
{
  g5_calculate_force_on_xe(x, eps2, a, p, *ni);
}

void FNAME(g5_calculate_force_on_xMC)(int *devid, double (*x)[3], double (*a)[3], double *p, int *ni)
{
  g5_calculate_force_on_xMC(*devid, x, a, p, *ni);
}

void FNAME(g5_calculate_force_on_xeMC)(int *devid, double (*x)[3], double *eps2, double (*a)[3], double *p, int *ni)
{
  g5_calculate_force_on_xeMC(*devid, x, eps2, a, p, *ni);
}
