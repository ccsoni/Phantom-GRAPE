#ifndef QUAD_UTIL
#define QUAD_UTIL

#if defined(__cplusplus)
extern "C" {
#endif

void g5c_set_nMC(int devid, int n);
void g5c_set_xmjMC0(int devid, int adr, int nj, double (*xj)[3], double *mj, double (*qj)[6]);
void g5c_calculate_force_on_xMC0(int devid, double (*x)[3], double (*a)[3], double *p, int ni);
void g5c_runMC0(int devid);
//void g5c_add_forceMC(int devid, int ni, double (*a)[3], double *pot);
void g5c_set_xmjMC(int devid, int adr, int nj, double (*xj)[3], double *mj, double (*qj)[6]);
void g5c_calculate_force_on_xMC(int devid, double (*x)[3], double (*a)[3], double *p, int ni);
void g5c_runMC(int devid);

#if defined(__cplusplus)
}
#endif

#endif
