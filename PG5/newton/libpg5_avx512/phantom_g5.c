#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "avx.h"
#include "avx_type.h"
#include "gp5util.h"

#ifndef MAXDEV
#define MAXDEV (24)
#endif /* MAXDEV */

static double Eps;

static struct Ptcl_Mem {
    PG5::Ipdata iptcl;
    PG5::Fodata fout;
#ifdef SYMMETRIC
    PG5::Jpdata0 jptcl0[JMEMSIZE/PG5::NumberOfJParallel];
#else
    PG5::Jpdata jptcl[JMEMSIZE];
#endif
    int nbody, pad[15];
} ptcl_mem[MAXDEV] __attribute__ ((aligned(64)));

static float Acc_correct = 1.0;
static float Pot_correct = -1.0;
#ifdef PG5_I8J2
static v8sf  Acc_correctV;
static v8sf  Pot_correctV;
#else
static v4sf  Acc_correctV;
static v4sf  Pot_correctV;
#endif

int g5_get_number_of_pipelines(void) 
{
    return PG5::NumberOfIParallel;
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
	Acc_correct  = acc_corr;
	Pot_correct  = pot_corr;
	Acc_correctV = acc_corr;
	Pot_correctV = pot_corr;
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
    
    assert(ni <= PG5::NumberOfIParallel);
    pm->iptcl.eps2 = v4sf(eps[0]*eps[0], eps[1]*eps[1],
			  eps[2]*eps[2], eps[3]*eps[3]);
    // Not yet (A. Tanikawa)
}

void g5_set_eps(int ni, double *eps)
{
    g5_set_epsMC(0, ni, eps);
}

void g5_set_nMC(int devid,
		int n)
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    pm->nbody = n;
}

void g5_set_n(int n) 
{
    g5_set_nMC(0, n);
}

void g5_set_xiMC(int devid,
		 int ni,
		 double (*xi)[3])
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    float eps2 = Eps*Eps;
    
    assert(ni <= PG5::NumberOfIParallel);
#ifdef PG5_I8J2
    pm->iptcl.x = v8sf(xi[0][0], xi[1][0], xi[2][0], xi[3][0],
		       xi[4][0], xi[5][0], xi[6][0], xi[7][0]);
    pm->iptcl.y = v8sf(xi[0][1], xi[1][1], xi[2][1], xi[3][1],
		       xi[4][1], xi[5][1], xi[6][1], xi[7][1]);
    pm->iptcl.z = v8sf(xi[0][2], xi[1][2], xi[2][2], xi[3][2],
		       xi[4][2], xi[5][2], xi[6][2], xi[7][2]);
    pm->iptcl.eps2 = v8sf(eps2);
#else
    pm->iptcl.x = v4sf(xi[0][0], xi[1][0], xi[2][0], xi[3][0]);
    pm->iptcl.y = v4sf(xi[0][1], xi[1][1], xi[2][1], xi[3][1]);
    pm->iptcl.z = v4sf(xi[0][2], xi[1][2], xi[2][2], xi[3][2]);
    pm->iptcl.eps2 = v4sf(eps2);
#endif
    // Not yet (A. Tanikawa)
}

void g5_set_xeiMC(int devid,
		  int ni,
		  double (*xi)[3],
		  double *eps2)
{
    int i;
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    
    assert(ni <= PG5::NumberOfIParallel);
    pm->iptcl.x = v4sf(xi[0][0], xi[1][0], xi[2][0], xi[3][0]);
    pm->iptcl.y = v4sf(xi[0][1], xi[1][1], xi[2][1], xi[3][1]);
    pm->iptcl.z = v4sf(xi[0][2], xi[1][2], xi[2][2], xi[3][2]);
    pm->iptcl.eps2 = v4sf(eps2[0], eps2[1], eps2[2], eps2[3]);
    // Not yet (A. Tanikawa)
}

void g5_set_xi(int ni,
	       double (*xi)[3]) 
{
    g5_set_xiMC(0, ni, xi);
}

void g5_set_xei(int ni,
		double (*xi)[3],
		double *eps2)
{
    g5_set_xeiMC(0, ni, xi, eps2);
}

#ifdef SYMMETRIC
void g5_set_xmjMC(int devid,
		  int adr,
		  int nj,
		  double (*xj)[3],
		  double *mj,
		  double *epsj2) 
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    
    assert(adr % PG5::NumberOfJParallel == 0);
    for(int j = adr; j < adr + nj; j += PG5::NumberOfJParallel) {
	int jadr = j / PG5::NumberOfJParallel;
#pragma unroll(PG5::NumberOfJParallel)
	for(int jj = 0; jj < PG5::NumberOfJParallel; jj++) {
	    pm->jptcl0[jadr].posm[jj] = _mm256_cvtpd_ps(v4df(xj[j+jj][0], xj[j+jj][1],
							     xj[j+jj][2], mj[j+jj]));
	    pm->jptcl0[jadr].eps2[jj] = _mm256_cvtpd_ps(v4df(epsj2[j+jj]));;
	}
    }

    const int njparallel = PG5::NumberOfJParallel * PG5::NumberOfHandUnroll;
    const int rsdl = (njparallel - (nj % njparallel)) % njparallel;
    for(int j = nj; j < nj + rsdl; j += PG5::NumberOfJParallel){
	int jadr = j / PG5::NumberOfJParallel;
#pragma unroll(PG5::NumberOfJParallel)
	for(int jj = 0; jj < PG5::NumberOfJParallel; jj++){
	    int jp = jadr * PG5::NumberOfJParallel + jj;
	    if(jp < nj)
		continue;
	    pm->jptcl0[jadr].posm[jj] = 0.;
	    pm->jptcl0[jadr].eps2[jj] = 1.;
	}
    }
}
#else
void g5_set_xmjMC(int devid,
		  int adr,
		  int nj,
		  double (*xj)[3],
		  double *mj)
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    
    for(int j = adr; j < adr + nj; j++) {
	(pm->jptcl+j)->posm = _mm256_cvtpd_ps(v4df(xj[j][0], xj[j][1],
						   xj[j][2], mj[j]));
    }

    const int njparallel = PG5::NumberOfJParallel * PG5::NumberOfHandUnroll;
    const int rsdl = (njparallel - (nj % njparallel)) % njparallel;
    for(int j = nj; j < nj + rsdl; j++){
	(pm->jptcl+j)->posm = 0.;
    }
}
#endif

#ifdef SYMMETRIC
void g5_set_xmj(int adr,
		int nj,
		double (*xj)[3],
		double *mj,
		double *epsj2) 
{
    g5_set_xmjMC(0, adr, nj, xj, mj, epsj2);
}
#else
void g5_set_xmj(int adr,
		int nj,
		double (*xj)[3],
		double *mj) 
{
    g5_set_xmjMC(0, adr, nj, xj, mj);
}
#endif

void g5_runMC(int devid) 
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
#ifdef SYMMETRIC
    PG5:GravityKernel0(&(pm->iptcl), &(pm->fout), pm->jptcl0, pm->nbody);
#else
    PG5:GravityKernel(&(pm->iptcl), &(pm->fout), pm->jptcl, pm->nbody);
#endif
}

void g5_run(void) 
{
    g5_runMC(0);
}

void g5_get_forceMC(int devid,
		    int ni,
		    double (*a)[3],
		    double *pot) 
{
    assert(ni <= PG5::NumberOfIParallel);

    struct Ptcl_Mem *pm = ptcl_mem + devid;

    pm->fout.ax  *= Acc_correctV;
    pm->fout.ay  *= Acc_correctV;
    pm->fout.az  *= Acc_correctV;
    pm->fout.phi *= Pot_correctV;

    float ax[PG5::NumberOfIParallel] __attribute__ ((aligned(64)));
    float ay[PG5::NumberOfIParallel] __attribute__ ((aligned(64)));
    float az[PG5::NumberOfIParallel] __attribute__ ((aligned(64)));
    float pt[PG5::NumberOfIParallel] __attribute__ ((aligned(64)));

    pm->fout.ax.store(ax);
    pm->fout.ay.store(ay);
    pm->fout.az.store(az);
    pm->fout.phi.store(pt);

    for(int i = 0; i < ni; i++) {
	a[i][0] = (double)(ax[i]);
	a[i][1] = (double)(ay[i]);
	a[i][2] = (double)(az[i]);
	pot[i]  = (double)(pt[i]);
    }
    
}

void g5_get_force(int ni,
		  double (*a)[3],
		  double *pot) 
{
    g5_get_forceMC(0, ni, a, pot);
}

void g5_calculate_force_on_xMC(int devid,
			       double (*x)[3],
			       double (*a)[3], 
			       double *p,
			       int ni)
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

void g5_calculate_force_on_xeMC(int devid,
				double (*x)[3],
				double *eps2,
				double (*a)[3],
				double *p,
				int ni)
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
void g5_calculate_force_on_x(double (*x)[3],
			     double (*a)[3],
			     double *p,
			     int ni)
{
    g5_calculate_force_on_xMC(0, x, a, p, ni);
}

void g5_calculate_force_on_xe(double (*x)[3],
			      double *eps2,
			      double (*a)[3],
			      double *p,
			      int ni)
{
    g5_calculate_force_on_xeMC(0, x, eps2, a, p, ni);
}
#else
#include <omp.h>
void g5_calculate_force_on_x(double (*x)[3],
			     double (*a)[3],
			     double *p, 
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
	    PG5::pIpdata ip = &ptcl_mem[tid].iptcl;
	    PG5::pFodata fo = &ptcl_mem[tid].fout;
	    PG5::pJpdata jp = ptcl_mem[0].jptcl;
	    int nbody  = ptcl_mem[0].nbody;
	    PG5:GravityKernel(ip, fo, jp, nbody);
	}
	g5_get_forceMC(tid, ni, a+off, p+off);
    }
}

void g5_calculate_force_on_xe(double (*x)[3],
			      double *eps2,
			      double (*a)[3],
			      double *p, 
			      int nitot)
{
    int off;
    const int np = g5_get_number_of_pipelines();
#pragma omp parallel for
    for(off=0; off<nitot; off+=np) {
	int tid = omp_get_thread_num();
	int ni = np < nitot-off ? np : nitot-off;
	g5_set_xeiMC(tid, ni, x+off, eps2+off);
	{
	    PG5::pIpdata ip = &ptcl_mem[tid].iptcl;
	    PG5::pFodata fo = &ptcl_mem[tid].fout;
	    PG5::pJpdata0 jp = ptcl_mem[0].jptcl0;
	    int nbody  = ptcl_mem[0].nbody;
	    PG5:GravityKernel0(ip, fo, jp, nbody);
	}
	g5_get_forceMC(tid, ni, a+off, p+off);
    }
}
#endif
