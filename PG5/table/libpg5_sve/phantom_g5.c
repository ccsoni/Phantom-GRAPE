#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "sve_type.h"

#include "pg5_table.h"
#include "gp5util.h"

#ifndef NMAXDEV
#define NMAXDEV (72)
#endif /* NMAXDEV */

static struct Ptcl_Mem {
  Ipdata iptcl;
  Jpdata jptcl[JMEMSIZE];
  Fodata fout;
  int32_t nbody,pad[15];
} ptcl_mem [NMAXDEV] ALIGN64;

static double Eps;
static double Eta;

static float Xscale;
static float Ascale;
static float R2cut_xscale2;
static float XMscale[4];
//static float XMscale[4] = {xscale, xscale, xscale, 1.0f};

void gravity_kernel(pIpdata, pJpdata, pFodata, int, float (*)[2], float, float);

void pg5_set_xscale(double xscale)
{
  Xscale = xscale;
  Ascale = 1.0f/Xscale;
  R2cut_xscale2 = (1<<(1<<EXP_BIT))*(2.0-1.0/(1<<FRC_BIT));
  //  R2cut_xscale2 = (1<<30) | ((TBL_SIZE - 1) << (23 - FRC_BIT));
  XMscale[0] = XMscale[1] = XMscale[2] = xscale;
  XMscale[3] = 1.0f;
  //  double ascale = 1./xscale;
  //  Ascale = (v4sf){ascale, ascale, ascale, ascale};
}

int32_t g5_get_number_of_pipelines(void)
{
  return NI_PARALLEL;
}

void g5_open()
{
#ifdef ALLOCATE_TABLE
  Force_table = valloc(TBL_SIZE * sizeof(*Force_table));
#endif
  pg5_gen_s2_force_table(SFT_FOR_PP, SFT_FOR_PM);
}

void g5_close()
{
#ifdef ALLOCATE_TABLE
  free(Force_table);
#endif
}

void g5_set_eta(double eta)
{
  Eta = eta;
}

void g5_set_eps_to_all(double eps)
{
  Eps = eps;
}

void g5_set_range(double xmin, double xmax, double mmin)
{
  // Just for the backward compatibility.
}

void g5_set_cutofff_table(double (*ffunc)(double), double fcut, double fcor,
                          double (*pfunc)(double), double pcut, double pcor)
{
  // force table is constructed in g5_open().
  // this API is just for backward compatibility.
}

void g5_set_nMC(int32_t devid, int32_t n)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
  assert(n<=JMEMSIZE);
  pm->nbody = n;
}

void g5_set_n(int32_t n)
{
  g5_set_nMC(0, n);
}

void g5_set_xiMC(int32_t devid, int32_t ni, double (*xi)[3])
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  assert(ni <= NI_PARALLEL);
  for(int32_t i=0;i<ni;i++){
    pm->iptcl.x[i] = (float)xi[i][0] * Xscale;
    pm->iptcl.y[i] = (float)xi[i][1] * Xscale;
    pm->iptcl.z[i] = (float)xi[i][2] * Xscale;
  }
}

void g5_set_xi(int32_t ni, double (*xi)[3])
{
  g5_set_xiMC(0, ni, xi);
}

void g5_set_xmjMC(int32_t devid, int32_t adr, int32_t nj, double (*xj)[3], double *mj)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
  static float dbgf[4];

#if __ARM_FEATURE_SVE_BITS == 128 // I4J1
  svbool_t pred_vl2_b64 = svptrue_pat_b64(SV_VL2);
  svbool_t pred_vl4_b32 = svptrue_pat_b32(SV_VL4);
  svfloat32_t _xmscale = svld1(pred_vl4_b32, XMscale);

  for(int32_t j=adr; j<adr+nj; j++){
    double tmp0[2] = { xj[j][0], xj[j][2] };
    double tmp1[2] = { xj[j][1], mj[j] };

    svfloat32_t _vs0  = svcvt_f32_f64_x(pred_vl2_b64, svld1_f64(pred_vl2_b64, tmp0));
    svfloat32_t _vs1  = svcvt_f32_f64_x(pred_vl2_b64, svld1_f64(pred_vl2_b64, tmp1));

    svfloat32_t _xm   = svmul_f32_x(pred_vl4_b32, svuzp1_f32(_vs0, _vs1), _xmscale);

    svst1(pred_vl4_b32, (float32_t *)&(pm->jptcl[j].x), _xm);
  }
#elif (__ARM_FEATURE_SVE_BITS == 256) || (__ARM_FEATURE_SVE_BITS == 512) //I8J1 or I8J2
  svbool_t pred_vl4_b64 = svptrue_pat_b64(SV_VL4);
  svbool_t pred_vl4_b32 = svptrue_pat_b32(SV_VL4);
  svfloat32_t _xmscale = svld1(pred_vl4_b32, XMscale);

  for(int32_t j=adr;j<adr+nj;j++) {
    double tmp0[4] = {xj[j][0], xj[j][1], xj[j][2], mj[j]};

    svfloat64_t _xmd0 = svld1(pred_vl4_b64, tmp0);
    svfloat32_t _xmf0 = svcvt_f32_x(pred_vl4_b64,_xmd0);
    _xmf0 = svuzp1_f32(_xmf0, _xmf0);
    _xmf0 = svmul_x(pred_vl4_b32, _xmf0, _xmscale);

    svst1(pred_vl4_b32, (float32_t *)&(pm->jptcl[j].x), _xmf0);
  }
#endif
}

void g5_set_xmj(int32_t adr, int32_t nj, double (*xj)[3], double *mj)
{
  return g5_set_xmjMC(0, adr, nj, xj, mj);
}

void g5_runMC(int32_t devid){
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  gravity_kernel(&pm->iptcl, pm->jptcl, &pm->fout, pm->nbody,
                 Force_table, R2cut_xscale2, Ascale);
}

void g5_run(void){
  g5_runMC(0);
}

void g5_get_forceMC(int32_t devid, int32_t ni, double (*a)[3], double *pot)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
  for(int32_t i=0;i<ni;i++){
    a[i][0] = (double)pm->fout.ax[i];
    a[i][1] = (double)pm->fout.ay[i];
    a[i][2] = (double)pm->fout.az[i];
    pot[i] = 0.0;
  }
}
void g5_get_force(int32_t ni, double (*a)[3], double *pot)
{
  g5_get_forceMC(0, ni, a, pot);
}

void g5_calculate_force_on_xMC(int32_t devid, double (*x)[3], double (*a)[3], double *p, int32_t ni)
{
  int32_t np = g5_get_number_of_pipelines();
  for(int32_t off=0; off<ni; off+=np) {
    int nii = np < ni-off ? np : ni-off;
    g5_set_xiMC(devid, nii, x+off);
    g5_runMC(devid);
    g5_get_forceMC(devid, nii, a+off, p+off);
  }
}

#ifndef ENABLE_OPENMP
void g5_calculate_force_on_x(double (*x)[3], double (*a)[3], double *p, int32_t ni)
{
  g5_calculate_force_on_xMC(0, x, a, p, ni);
}
#else // ENABLE_OPENMP
#include <omp.h>
void g5_calculate_force_on_x(double (*x)[3], double (*a)[3], double *p,
                             int32_t nitot)
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
      gravity_kernel(ip, jp, fo, ptcl_mem[0].nbody,
                     Force_table, R2cut_xscale2, Ascale);
    }
    g5_get_forceMC(tid, ni, a+off, p+off);
  }
}
#endif // ENABLE_OPENMP
