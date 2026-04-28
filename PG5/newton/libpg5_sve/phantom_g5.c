#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <arm_sve.h>

#include "sve_type.h"
#include "gp5util.h"

#ifndef NMAXDEV
#define NMAXDEV (72)
#endif  /* NMAXDEV */

void GravityKernel_SymmEps(pIpdata, pFodata, pJpdata, float*, int);
void GravityKernel(pIpdata, pFodata, pJpdata, int);

static double Eps;

static struct Ptcl_Mem {
  Ipdata iptcl;
  Jpdata jptcl[JMEMSIZE];
  Fodata fout;
#ifdef __SYMMETRIC_EPS__
  float epsj2[JMEMSIZE];
#endif
  int32_t nbody, pad[15];
} ptcl_mem[NMAXDEV] ALIGN64;

static float Acc_corr = 1.0;
static float Pot_corr = -1.0;

int g5_get_number_of_pipelines(void)
{
  return NI_PARALLEL;
}

int g5_get_jmemsize(void)
{
  return JMEMSIZE;
}

void g5_open(void)
{
  double rsqrt_bias();

  static bool init_call = true;
  if(init_call) {
    double bias = rsqrt_bias();
    Acc_corr = 1.0 - 3.0*bias;
    Pot_corr = -(1.0 - bias);
    init_call = false;
  }

  return;
}

void g5_close(void)
{
  return;
}

void g5_set_eps_to_all(double eps)
{
  Eps = eps;

  return;
}

void g5_set_epsMC(int devid, int ni, double *eps)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  assert(ni <= NI_PARALLEL);
  for(int32_t i=0;i<ni;i++) {
    pm->iptcl.eps2[i] = eps[i]*eps[i];
  }

  return;
}

void g5_set_eps(int ni, double *eps)
{
  g5_set_epsMC(0, ni, eps);

  return;
}

// set the number of j-particles
void g5_set_nMC(int devid, int n)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
  pm->nbody = n;

  return;
}

void g5_set_n(int n)
{
  g5_set_nMC(0, n);

  return;
}

void g5_set_xiMC(int devid, int ni, double (*xi)[3])
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  float eps2 = Eps*Eps;

  assert(ni <= NI_PARALLEL);
  for(int32_t i=0;i<ni;i++) {
    pm->iptcl.x[i] = (float)xi[i][0];
    pm->iptcl.y[i] = (float)xi[i][1];
    pm->iptcl.z[i] = (float)xi[i][2];
    pm->iptcl.eps2[i] = eps2;
  }

  return;
}

void g5_set_xeiMC(int devid, int ni, double (*xi)[3],
                  double *eps2)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  assert(ni <= NI_PARALLEL);
  for(int32_t i=0;i<ni;i++) {
    pm->iptcl.x[i] = (float)xi[i][0];
    pm->iptcl.y[i] = (float)xi[i][1];
    pm->iptcl.z[i] = (float)xi[i][2];
    pm->iptcl.eps2[i] = eps2[i];
  }

  return;
}

void g5_set_xi(int ni, double (*xi)[3])
{
  g5_set_xiMC(0, ni, xi);

  return;
}

void g5_set_xei(int ni, double (*xi)[3], double *eps2)
{
  g5_set_xeiMC(0, ni, xi, eps2);

  return;
}

#ifdef __SYMMETRIC_EPS__
void g5_set_xmjMC(int devid, int adr, int nj, double (*xj)[3], double *mj,
                  double *epsj2)
{
  svbool_t pred_all_b64 = svptrue_b64();
  svbool_t pred_all_b32 = svptrue_b32();

  svbool_t pred_vl2_b64 = svptrue_pat_b64(SV_VL2);
  svbool_t pred_vl2_b32 = svptrue_pat_b32(SV_VL2);

  svbool_t pred_vl4_b64 = svptrue_pat_b64(SV_VL4);
  svbool_t pred_vl4_b32 = svptrue_pat_b32(SV_VL4);

  struct Ptcl_Mem *pm = ptcl_mem + devid;

#if __ARM_FEATURE_SVE_BITS == 128 // I4J1
  assert(adr+nj < JMEMSIZE);
  for(int32_t j=adr;j<adr+nj;j++) {
    double tmp[4] = {xj[j][0], xj[j][1], xj[j][2], mj[j]};

    svfloat64_t _xmd0 = svld1(pred_vl2_b64, &tmp[0]); // | xj | yj |
    svfloat64_t _xmd1 = svld1(pred_vl2_b64, &tmp[2]); // | zj | mj |

    // convert to FP32 data
    svfloat32_t _xmf0 = svcvt_f32_x(pred_vl2_b64, _xmd0); // |xj|--|yj|--|
    svfloat32_t _xmf1 = svcvt_f32_x(pred_vl2_b64, _xmd1); // |zj|--|mj|--|

    // concatenate into a single register
    _xmf0 = svuzp1_f32(_xmf0, _xmf1); // |xj|yj|zj|mj|

    svst1(pred_vl4_b32, (float32_t *)&(pm->jptcl[j].x), _xmf0);
  }

#if 0
  int32_t rsdl = (NUNROLL - ((adr+nj) % NUMROLL)) % NUNROLL;
  assert(nj+rsdl < JMEMSIZE);
  for(int32_t j=nj;j<nj+rsdl;j++) {
    double tmp[4] = {0.0, 0.0, 0.0, 0.0};

    svfloat64_t _xmd0 = svld1(pred_all_b64, &tmp[0]);
    svfloat64_t _xmd1 = svld1(pred_all_b64, &tmp[2]);
    svfloat32_t _xmf0 = svcvt_f32_x(pred_all_b64, _xmd0);
    svfloat32_t _xmf1 = svcvt_f32_x(pred_all_b64, _xmd1);

    _xmf0 = svuzp1_f32(_xmf0, _xmf1);

    svst1(pred_all_b32, pm->jptcl+j);
  }
#endif

#elif (__ARM_FEATURE_SVE_BITS == 256) || (__ARM_FEATURE_SVE_BITS == 512)
  for(int32_t j=adr;j<adr+nj;j++) {
    double tmp0[4] = {xj[j][0], xj[j][1], xj[j][2], mj[j]};

    svfloat64_t _xmd0 = svld1(pred_vl4_b64, tmp0); // | xj | yj | zj | mj | ... |

    // convert to FP32 data
    svfloat32_t _xmf0 = svcvt_f32_x(pred_vl4_b64, _xmd0); // |xj|--|yj|--|zj|--|mj|

    // fill the gaps
    _xmf0 = svuzp1_f32(_xmf0, _xmf0); // |xj|yj|zj|mj|...|

    svst1(pred_vl4_b32, (float32_t *)&(pm->jptcl[j].x), _xmf0);
  }

#if 0
  int32_t rsdl = (NUNROLL - ((adr+nj) % NUMROLL)) % NUNROLL;
  assert(nj+rsdl < JMEMSIZE);
  for(int32_t j=nj;j<nj+rsdl;j++) {
    double tmp0[4] = {0.0, 0.0, 0.0, 0.0};

    svfloat64_t _xmd0 = svld1(pred_64, tmp0);
    svfloat32_t _xmf0 = svcvt_f32_x(pred_all_64, _xmd0);
    _xmf0 = svuzp1_f32(_xmf0, _xmf0);
    svst1(pred_all_b32, pm->jptcl+j);
  }
#endif

#endif // __ARM_FEATURE_SVE_BITS__

  for(int32_t j=adr;j<adr+nj;j++) {
    pm->epsj2[j] = epsj2[j];
  }

  return;
}
#else // !__SYMMETRIC_EPS__
void g5_set_xmjMC(int devid, int adr, int nj, double (*xj)[3], double *mj)
{
  svbool_t pred_all_b64 = svptrue_b64();
  svbool_t pred_all_b32 = svptrue_b32();

  svbool_t pred_vl2_b64 = svptrue_pat_b64(SV_VL2);
  svbool_t pred_vl2_b32 = svptrue_pat_b32(SV_VL2);

  svbool_t pred_vl4_b64 = svptrue_pat_b64(SV_VL4);
  svbool_t pred_vl4_b32 = svptrue_pat_b32(SV_VL4);

  struct Ptcl_Mem *pm = ptcl_mem + devid;

  static float dbgf[4];

#if __ARM_FEATURE_SVE_BITS == 128 // I4J1
  assert(adr+nj < JMEMSIZE);
  for(int32_t j=adr;j<adr+nj;j++) {
    double tmp[4] = {xj[j][0], xj[j][1], xj[j][2], mj[j]};

    svfloat64_t _xmd0 = svld1(pred_vl2_b64, &tmp[0]); // | xj | yj |
    svfloat64_t _xmd1 = svld1(pred_vl2_b64, &tmp[2]); // | zj | mj |

    // convert to FP data
    svfloat32_t _xmf0 = svcvt_f32_x(pred_vl2_b64, _xmd0); // |xj|--|yj|--|
    svfloat32_t _xmf1 = svcvt_f32_x(pred_vl2_b64, _xmd1); // |zj|--|mj|--|

    // concatenate into a single register
    _xmf0 = svuzp1_f32(_xmf0, _xmf1); // |xj|yj|zj|mj|

    svst1(pred_vl4_b32, (float32_t *)&(pm->jptcl[j].x), _xmf0);

  }

#if 0
  int32_t rsdl = (NUNROLL - ((adr+nj) % NUMROLL)) % NUNROLL;
  assert(nj+rsdl < JMEMSIZE);
  for(int32_t j=nj;j<nj+rsdl;j++) {
    double tmp[4] = {0.0, 0.0, 0.0, 0.0};

    svfloat64_t _xmd0 = svld1(pred_all_b64, &tmp[0]);
    svfloat64_t _xmd1 = svld1(pred_all_b64, &tmp[2]);
    svfloat32_t _xmf0 = svcvt_f32_x(pred_all_b64, _xmd0);
    svfloat32_t _xmf1 = svcvt_f32_x(pred_all_b64, _xmd1);

    _xmf0 = svuzp1_f32(_xmf0, _xmf1);

    svst1(pred_all_b32, pm->jptcl+j);
  }
#endif

#elif (__ARM_FEATURE_SVE_BITS == 256) || (__ARM_FEATURE_SVE_BITS == 512)
  for(int32_t j=adr;j<adr+nj;j++) {
    double tmp0[4] = {xj[j][0], xj[j][1], xj[j][2], mj[j]};

    svfloat64_t _xmd0 = svld1(pred_vl4_b64, tmp0); // | xj | yj | zj | mj | ... |

    // convert to FP32 data
    svfloat32_t _xmf0 = svcvt_f32_x(pred_vl4_b64, _xmd0); // |xj|--|yj|--|zj|--|mj|

    // fill the gaps
    _xmf0 = svuzp1_f32(_xmf0, _xmf0); // |xj|yj|zj|mj|...|

    svst1(pred_vl4_b32, pm->jptcl+j);
  }

#if 0
  int32_t rsdl = (NUNROLL - ((adr+nj) % NUMROLL)) % NUNROLL;
  assert(nj+rsdl < JMEMSIZE);
  for(int32_t j=nj;j<nj+rsdl;j++) {
    double tmp0[4] = {0.0, 0.0, 0.0, 0.0};

    svfloat64_t _xmd0 = svld1(pred_all_b64, tmp0);
    svfloat32_t _xmf0 = svcvt_f32_x(pred_all_b64, _xmd0);
    _xmf0 = svuzp1_f32(_xmf0, _xmf0);
    svst1(pred_all_b32, pm->jptcl+j);
  }
#endif

#endif // __ARM_FEATURE_SVE_BITS__

  return;
}

#endif // __SYMMETRIC_EPS__

#ifdef __SYMMETRIC_EPS__
void g5_set_xmj(int adr, int nj, double (*xj)[3], double *mj, double *epsj2)
{
  g5_set_xmjMC(0, adr, nj, xj, mj, epsj2);

  return;
}
#else // !__SYMMETRIC_EPS__
void g5_set_xmj(int adr, int nj, double (*xj)[3], double *mj)
{
  g5_set_xmjMC(0, adr, nj, xj, mj);

  return;
}
#endif // __SYMMETRIC_EPS__

void g5_runMC(int devid)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;

#ifdef __SYMMETRIC_EPS__
  GravityKernel_SymmEps(&(pm->iptcl), &(pm->fout), pm->jptcl, pm->epsj2,  pm->nbody);
#else // !__SYMMETRIC_EPS__
  GravityKernel(&(pm->iptcl), &(pm->fout), pm->jptcl, pm->nbody);
#endif //__SYMMETRIC_EPS__

  return;
}

void g5_run(void)
{
  g5_runMC(0);

  return;
}

void g5_get_forceMC(int devid, int ni, double (*a)[3], double *pot)
{
  svbool_t pred_all_64 = svptrue_b64();
  svbool_t pred_all_32 = svptrue_b32();

  assert(ni <= NI_PARALLEL);

  struct Ptcl_Mem *pm = ptcl_mem + devid;

  for(int32_t i=0;i<ni;i++) {
    a[i][0] = (double)(pm->fout.ax[i]*Acc_corr);
    a[i][1] = (double)(pm->fout.ay[i]*Acc_corr);
    a[i][2] = (double)(pm->fout.az[i]*Acc_corr);
    pot[i] = (double)(pm->fout.phi[i]*Pot_corr);
  }

  return;
}

void g5_get_force(int ni, double (*a)[3], double *pot)
{
  g5_get_forceMC(0, ni, a, pot);
}

void g5_calculate_force_on_xMC(int devid, double (*x)[3], double (*a)[3],
                               double *p, int ni)
{
  int32_t np = g5_get_number_of_pipelines();
  for(int32_t off=0;off<ni;off+=np){
    int32_t nii = np < ni-off ? np : ni-off;
    g5_set_xiMC(devid, nii, x+off);
    g5_runMC(devid);
    g5_get_forceMC(devid, nii, a+off, p+off);
  }

  return;
}

void g5_calculate_force_on_xeMC(int devid, double (*x)[3], double *eps2,
                                double (*a)[3], double *p, int ni)
{
  int32_t np = g5_get_number_of_pipelines();
  for(int32_t off=0;off<ni;off+=np){
    int32_t nii = np < ni-off ? np : ni-off;
    g5_set_xeiMC(devid, nii, x+off, eps2+off);
    g5_runMC(devid);
    g5_get_forceMC(devid, nii, a+off, p+off);
  }

  return;
}

#ifndef ENABLE_OPENMP
void g5_calculate_force_on_x(double (*x)[3], double (*a)[3], double *p, int ni)
{
  g5_calculate_force_on_xMC(0, x, a, p, ni);

  return;
}

void g5_calculate_force_on_xe(double (*x)[3], double *eps2, double (*a)[3],
                              double *p, int ni)
{
  g5_calculate_force_on_xeMC(0, x, eps2, a, p, ni);

  return;
}
#else // ENABLE_OPENMP
#include <omp.h>
void g5_calculate_force_on_x(double (*x)[3], double (*a)[3], double *p,
                             int nitot)
{
  const int np = g5_get_number_of_pipelines();
#pragma omp parallel for
  for(int32_t off=0; off<nitot; off+=np) {
    int32_t tid = omp_get_thread_num();
    int32_t ni = np < nitot-off ? np : nitot-off;
    g5_set_xiMC(tid, ni, x+off);
    {
      pIpdata ip = &ptcl_mem[tid].iptcl;
      pFodata fo = &ptcl_mem[tid].fout;
      pJpdata jp = ptcl_mem[0].jptcl;

      int nbody  = ptcl_mem[0].nbody;
#ifdef __SYMMETRIC_EPS__
      GravityKernel_SymmEps(ip, fo, jp, ptcl_mem[0].epsj2, nbody);
#else
      GravityKernel(ip, fo, jp, nbody);
#endif
    }
    g5_get_forceMC(tid, ni, a+off, p+off);
  }

  return;
}

void g5_calculate_force_on_xe(double (*x)[3], double *eps2, double (*a)[3],
                              double *p, int nitot)
{

  const int np = g5_get_number_of_pipelines();
#pragma omp parallel for
  for(int32_t off=0; off<nitot; off+=np) {
    int tid = omp_get_thread_num();
    int ni = np < nitot-off ? np : nitot-off;
    g5_set_xeiMC(tid, ni, x+off, eps2+off);
    {
      pIpdata ip = &ptcl_mem[tid].iptcl;
      pFodata fo = &ptcl_mem[tid].fout;
      pJpdata jp = ptcl_mem[0].jptcl;

      int nbody  = ptcl_mem[0].nbody;
#ifdef __SYMMETRIC_EPS__
      GravityKernel_SymmEps(ip, fo, jp, ptcl_mem[0].epsj2, nbody);
#else
      GravityKernel(ip, fo, jp, nbody);
#endif
    }
    g5_get_forceMC(tid, ni, a+off, p+off);
  }

  return;
}
#endif // ENABLE_OMP
