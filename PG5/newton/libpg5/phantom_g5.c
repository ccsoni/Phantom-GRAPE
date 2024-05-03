#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#include "avx_type.h"
#include "gp5util.h"

#define NUM_PIPE (4)

#ifndef MAXDEV
#define MAXDEV (24)
#endif /* MAXDEV */

static double Eps;

static struct Ptcl_Mem {
  Ipdata iptcl;
  Fodata fout;
#ifdef SYMMETRIC
  Jpdata0 jptcl0[JMEMSIZE/2];
#else
  Jpdata jptcl[JMEMSIZE];
#endif
  int nbody, pad[15];
} ptcl_mem[MAXDEV] ALIGN64;

static float Acc_correct = 1.0;
static float Pot_correct = -1.0;
static __m128 Acc_correctV;
static __m128 Pot_correctV;

int g5_get_number_of_pipelines(void) 
{
  return NUM_PIPE;
}

int g5_get_jmemsize(void) 
{
  return JMEMSIZE;
}

void g5_open(void)
{
  static int init_call = 1;
  if(init_call) {
    double rsqrt_bias();
    double bias = rsqrt_bias();
    float acc_corr = 1.0 - 3.0*bias;
    float pot_corr = -(1.0-bias);
    Acc_correct = acc_corr;
    Pot_correct = pot_corr;
    Acc_correctV = _mm_set_ps(acc_corr,acc_corr,acc_corr,acc_corr);
    Pot_correctV = _mm_set_ps(pot_corr,pot_corr,pot_corr,pot_corr);
    init_call = 0;
  }
  return;
}

void g5_close() 
{
  return;
}

void g5_set_eps_to_all(double eps) 
{
  Eps = eps;
}

void g5_set_epsMC(int devid, int ni, double *eps)
{
  int i;
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  assert(ni <= NUM_PIPE);
  for(i=0;i<ni;i++) {
    float eps2 = eps[i]*eps[i];
    pm->iptcl.eps2[i] = eps2;
  }
}

void g5_set_eps(int ni, double *eps)
{
  g5_set_epsMC(0, ni, eps);
}

void g5_set_nMC(int devid, int n)
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
  pm->nbody = n;
}

void g5_set_n(int n) 
{
  g5_set_nMC(0, n);
}

void g5_set_xiMC(int devid, int ni, double (*xi)[3])
{
  int i;
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  assert(ni <= NUM_PIPE);
  for(i=0;i<ni;i++) {
    float eps2 = Eps*Eps;
    pm->iptcl.x[i] = (float)xi[i][0];
    pm->iptcl.y[i] = (float)xi[i][1];
    pm->iptcl.z[i] = (float)xi[i][2];
    pm->iptcl.eps2[i] = eps2;
  }
}

void g5_set_xeiMC(int devid, int ni, double (*xi)[3],
		  double *eps2)
{
  int i;
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  assert(ni <= NUM_PIPE);
  for(i=0;i<ni;i++) {
    pm->iptcl.x[i] = (float)xi[i][0];
    pm->iptcl.y[i] = (float)xi[i][1];
    pm->iptcl.z[i] = (float)xi[i][2];
    pm->iptcl.eps2[i] = eps2[i];
  }
}

void g5_set_xi(int ni, double (*xi)[3]) 
{
  g5_set_xiMC(0, ni, xi);
}

void g5_set_xei(int ni, double (*xi)[3], double *eps2)
{
  g5_set_xeiMC(0, ni, xi, eps2);
}

#ifdef SYMMETRIC
void g5_set_xmjMC(int devid, int adr, int nj, double (*xj)[3], double *mj, 
		  double *epsj2) 
{
  int j;
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  assert(adr % 2 == 0);
  for(j=adr;j<adr+nj;j+=2) {
    int jadr = j / 2;
    pm->jptcl0[jadr].xm[0][0] = (float)xj[j][0];
    pm->jptcl0[jadr].xm[0][1] = (float)xj[j][1];
    pm->jptcl0[jadr].xm[0][2] = (float)xj[j][2];
    pm->jptcl0[jadr].xm[0][3] = (float)mj[j];
    pm->jptcl0[jadr].xm[1][0] = (float)xj[j+1][0];
    pm->jptcl0[jadr].xm[1][1] = (float)xj[j+1][1];
    pm->jptcl0[jadr].xm[1][2] = (float)xj[j+1][2];
    pm->jptcl0[jadr].xm[1][3] = (float)mj[j+1];
    pm->jptcl0[jadr].ep[0][0] = (float)epsj2[j];
    pm->jptcl0[jadr].ep[0][1] = (float)epsj2[j];
    pm->jptcl0[jadr].ep[0][2] = (float)epsj2[j];
    pm->jptcl0[jadr].ep[0][3] = (float)epsj2[j];
    pm->jptcl0[jadr].ep[1][0] = (float)epsj2[j+1];
    pm->jptcl0[jadr].ep[1][1] = (float)epsj2[j+1];
    pm->jptcl0[jadr].ep[1][2] = (float)epsj2[j+1];
    pm->jptcl0[jadr].ep[1][3] = (float)epsj2[j+1];
  }

  int rsdl = (NUNROLL - (nj % NUNROLL)) % NUNROLL;
  for(j=nj;j<nj+rsdl;j+=2){
      int jj, jadr = j / 2;
      for(jj = 0; jj < 2; jj++){
          int jp = jadr * 2 + jj;
          if(jp < nj)
              continue;
          pm->jptcl0[jadr].xm[jj][0] = 0.0f;
          pm->jptcl0[jadr].xm[jj][1] = 0.0f;
          pm->jptcl0[jadr].xm[jj][2] = 0.0f;
          pm->jptcl0[jadr].xm[jj][3] = 0.0f;
          pm->jptcl0[jadr].ep[jj][0] = 1.0f;
          pm->jptcl0[jadr].ep[jj][1] = 1.0f;
          pm->jptcl0[jadr].ep[jj][2] = 1.0f;
          pm->jptcl0[jadr].ep[jj][3] = 1.0f;
      }
  }

}
#else
void g5_set_xmjMC(int devid, int adr, int nj, double (*xj)[3], double *mj) 
{
  int j;
  struct Ptcl_Mem *pm = ptcl_mem + devid;

  for(j=adr;j<adr+nj;j++) {
    __m256d pd = {xj[j][0], xj[j][1], xj[j][2], mj[j]};
    __m128  ps = _mm256_cvtpd_ps(pd);
    *(__m128 *)(pm->jptcl+j) = ps;
  }

  int rsdl = (NUNROLL - (nj % NUNROLL)) % NUNROLL;
  for(j=nj;j<nj+rsdl;j++){
    __m256d pd = {0.0, 0.0, 0.0, 0.0};
    __m128  ps = _mm256_cvtpd_ps(pd);
    *(__m128 *)(pm->jptcl+j) = ps;
  }
}
#endif


#ifdef SYMMETRIC
void g5_set_xmj(int adr, int nj, double (*xj)[3], double *mj, double *epsj2) 
{
  g5_set_xmjMC(0, adr, nj, xj, mj, epsj2);
}
#else
void g5_set_xmj(int adr, int nj, double (*xj)[3], double *mj) 
{
  g5_set_xmjMC(0, adr, nj, xj, mj);
}
#endif


void g5_runMC(int devid) 
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
#ifdef SYMMETRIC
  void GravityKernel0(pIpdata, pFodata, pJpdata0, int);
#else
  void GravityKernel(pIpdata, pFodata, pJpdata, int);
#endif

#ifdef SYMMETRIC
  GravityKernel0(&(pm->iptcl), &(pm->fout), pm->jptcl0, pm->nbody);
#else
  GravityKernel(&(pm->iptcl), &(pm->fout), pm->jptcl, pm->nbody);
#endif

}

void g5_run(void) 
{
  g5_runMC(0);
}

void g5_get_forceMC(int devid, int ni, double (*a)[3], double *pot) 
{
  assert(ni <= NUM_PIPE);

  struct Ptcl_Mem *pm = ptcl_mem + devid;
  int i;

#if 1
  *(__m128 *)pm->fout.ax = 
    _mm_mul_ps(*(__m128 *)(pm->fout.ax),Acc_correctV);
  *(__m128 *)pm->fout.ay = 
    _mm_mul_ps(*(__m128 *)(pm->fout.ay),Acc_correctV);
  *(__m128 *)pm->fout.az = 
    _mm_mul_ps(*(__m128 *)(pm->fout.az),Acc_correctV);
  *(__m128 *)pm->fout.phi = 
    _mm_mul_ps(*(__m128 *)(pm->fout.phi),Pot_correctV);
#endif

  for(i=0;i<ni;i++) {
    a[i][0] = (double)(pm->fout.ax[i]);
    a[i][1] = (double)(pm->fout.ay[i]);
    a[i][2] = (double)(pm->fout.az[i]);
    pot[i]  = (double)(pm->fout.phi[i]);
  }

}

void g5_get_force(int ni, double (*a)[3], double *pot) 
{
  g5_get_forceMC(0, ni, a, pot);
}

void g5_calculate_force_on_xMC(int devid, double (*x)[3], double (*a)[3], 
			       double *p, int ni)
{
  int off;
  int np = g5_get_number_of_pipelines();
  for(off=0;off<ni;off+=np) {
    int nii = np < ni-off ? np : ni-off;
    g5_set_xiMC(devid, nii, x+off);
    g5_runMC(devid);
    g5_get_forceMC(devid, nii, a+off, p+off);
  }
}

void g5_calculate_force_on_xeMC(int devid, double (*x)[3], double *eps2,
				double (*a)[3], double *p, int ni)
{
  int off;
  int np = g5_get_number_of_pipelines();
  for(off=0;off<ni;off+=np) {
    int nii = np < ni-off ? np : ni-off;
    g5_set_xeiMC(devid, nii, x+off, eps2+off);
    g5_runMC(devid);
    g5_get_forceMC(devid, nii, a+off, p+off);
  }
}

#ifndef ENABLE_OPENMP
void g5_calculate_force_on_x(double (*x)[3], double (*a)[3], double *p, int ni)
{
  g5_calculate_force_on_xMC(0, x, a, p, ni);
}

void g5_calculate_force_on_xe(double (*x)[3], double *eps2, double (*a)[3],
			      double *p, int ni)
{
  g5_calculate_force_on_xeMC(0, x, eps2, a, p, ni);
}
#else
#include <omp.h>
void g5_calculate_force_on_x(double (*x)[3], double (*a)[3], double *p, 
			     int nitot)
{
  int off;
  const int np = g5_get_number_of_pipelines();
#pragma omp parallel for
  for(off=0; off<nitot; off+=np) {
    int tid = omp_get_thread_num();
    int ni = np < nitot-off ? np : nitot-off;
    g5_set_xiMC(tid, ni, x+off);
    {
#ifdef SYMMETRIC
      void GravityKernel0(pIpdata, pFodata, pJpdata0, int);
#else
      void GravityKernel(pIpdata, pFodata, pJpdata, int);
#endif

      pIpdata ip = &ptcl_mem[tid].iptcl;
      pFodata fo = &ptcl_mem[tid].fout;
#ifdef SYMMETRIC
      pJpdata0 jp = ptcl_mem[0].jptcl0;
#else
      pJpdata jp = ptcl_mem[0].jptcl;
#endif
      int nbody  = ptcl_mem[0].nbody;
#ifdef SYMMETRIC
      GravityKernel0(ip, fo, jp, nbody);
#else
      GravityKernel(ip, fo, jp, nbody);
#endif
    }
    g5_get_forceMC(tid, ni, a+off, p+off);
  }
}

void g5_calculate_force_on_xe(double (*x)[3], double *eps2, double (*a)[3], 
			      double *p, int nitot)
{
  int off;
  const int np = g5_get_number_of_pipelines();
#pragma omp parallel for
  for(off=0; off<nitot; off+=np) {
    int tid = omp_get_thread_num();
    int ni = np < nitot-off ? np : nitot-off;
    g5_set_xeiMC(tid, ni, x+off, eps2+off);
    {
#ifdef SYMMETRIC
      void GravityKernel0(pIpdata, pFodata, pJpdata0, int);
#else
      void GravityKernel(pIpdata, pFodata, pJpdata, int);
#endif
      pIpdata ip = &ptcl_mem[tid].iptcl;
      pFodata fo = &ptcl_mem[tid].fout;
#ifdef SYMMETRIC
      pJpdata0 jp = ptcl_mem[0].jptcl0;
#else
      pJpdata jp = ptcl_mem[0].jptcl;
#endif
      int nbody  = ptcl_mem[0].nbody;
#ifdef SYMMETRIC
      GravityKernel0(ip, fo, jp, nbody);
#else
      GravityKernel(ip, fo, jp, nbody);
#endif
    }
    g5_get_forceMC(tid, ni, a+off, p+off);
  }
}
#endif
