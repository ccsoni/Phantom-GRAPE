// GRAPE-5 compatible APIs
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "avx.h"
#include "avx_type.h"
#include "pg5_table.h"
#include "gp5util.h"

#define NUM_PIPE 4
#define JMEMSIZE 65536

#define MAXDEV 24
static struct Ptcl_Mem {
    PG5::Ipdata iptcl;
    PG5::Fodata fout;
    PG5::Jpdata jptcl[JMEMSIZE];
    int Nbody, pad[15];
    int align[1024];
} ptcl_mem[MAXDEV] __attribute__ ((aligned(64)));

static double Eps;
static double Eta;

static double Xscale;
static v4sf XMscale;
static v4sf Ascale;
//static int R2cut_xscale2 = ((1<<30) | ((TBL_SIZE - 1) << (23 - FRC_BIT)));
static float R2cut_xscale2 = (1<<(1<<EXP_BIT)) * (2.0 - 1.0/(1<<FRC_BIT));


void pg5_set_xscale(double xscale)
{
    Xscale = xscale;
    XMscale = v4sf(xscale, xscale, xscale, 1.0);
    double ascale = 1. / xscale;
    Ascale  = v4sf(ascale, ascale, ascale, ascale);
}

/******** GRAPE-5 APIs ********/

int g5_get_number_of_pipelines(void)
{
    return NUM_PIPE;
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

void g5_set_range(double xmin,
		  double xmax,
		  double mmin)
{
}

void g5_set_cutoff_table(double (*ffunc)(double),
			 double fcut,
			 double fcor,
			 double (*pfunc)(double),
			 double pcut,
			 double pcor)
{
    // pg5_gen_plummer_force_table();
    // please implement the function calls pg5_gen_force_table().
}

void g5_set_nMC(int devid,
		int n)
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    assert(n<=JMEMSIZE);
    pm->Nbody = n;
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
    assert(ni <= NUM_PIPE);

    pm->iptcl.x = v4sf((float)xi[0][0] * Xscale, (float)xi[1][0] * Xscale,
		       (float)xi[2][0] * Xscale, (float)xi[3][0] * Xscale);
    pm->iptcl.y = v4sf((float)xi[0][1] * Xscale, (float)xi[1][1] * Xscale,
		       (float)xi[2][1] * Xscale, (float)xi[3][1] * Xscale);
    pm->iptcl.z = v4sf((float)xi[0][2] * Xscale, (float)xi[1][2] * Xscale,
		       (float)xi[2][2] * Xscale, (float)xi[3][2] * Xscale);
}

void g5_set_xi(int ni,
	       double (*xi)[3])
{
    g5_set_xiMC(0, ni, xi);
}

void g5_set_xmjMC(int devid,
		  int adr,
		  int nj,
		  double (*xj)[3],
		  double *mj)
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    
    for(int j = adr; j < adr + nj; j++){
	v2df pd0(xj[j][0], xj[j][2]);
	v2df pd1(xj[j][1], mj[j]   );
	v4sf ps0, ps1;
	ps0 = __builtin_ia32_cvtpd2ps(pd0);
	ps1 = __builtin_ia32_cvtpd2ps(pd1);
	ps0 = __builtin_ia32_unpcklps(ps0, ps1);
	*(v4sf *)(pm->jptcl+j) = ps0 * XMscale;
    }
    const int njparallel = PG5::NumberOfJParallel;
    const int rsdl = (njparallel - (nj % njparallel)) % njparallel;
    for(int j = nj; j < nj + rsdl; j++) {
	*(v4sf *)(pm->jptcl+j) = 0.;
    }
}

void g5_set_xmj(int adr,
		int nj,
		double (*xj)[3],
		double *mj)
{
    return g5_set_xmjMC(0, adr, nj, xj, mj);
}

void g5_runMC(int devid)
{
    struct Ptcl_Mem *pm = ptcl_mem + devid;
    PG5::GravityKernel(&pm->iptcl, pm->jptcl, &pm->fout, pm->Nbody, 
		       Force_table, v4sf(R2cut_xscale2), Ascale);
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
    struct Ptcl_Mem *pm = ptcl_mem + devid;

    float ax[PG5::NumberOfIParallel] __attribute__ ((aligned(64)));
    float ay[PG5::NumberOfIParallel] __attribute__ ((aligned(64)));
    float az[PG5::NumberOfIParallel] __attribute__ ((aligned(64)));

    pm->fout.ax.store(ax);
    pm->fout.ay.store(ay);
    pm->fout.az.store(az);

    for(int i = 0; i < ni; i++){
	a[i][0] = (double)ax[i];
	a[i][1] = (double)ay[i];
	a[i][2] = (double)az[i];
	pot[i] = 0.0;
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
    int np = g5_get_number_of_pipelines();
    for(int off = 0; off < ni; off += np) {
	int nii = np < ni-off ? np : ni-off;
	g5_set_xiMC(devid, nii, x+off);
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
#else
#include <omp.h>
void g5_calculate_force_on_x(double (*x)[3],
			     double (*a)[3],
			     double *p, 
			     int nitot)
{
    const int np = g5_get_number_of_pipelines();
#pragma omp parallel for
    for(int off = 0; off < nitot; off += np) {
	int tid = omp_get_thread_num();
	int ni = np < nitot-off ? np : nitot-off;
	g5_set_xiMC(tid, ni, x+off);
	{
	    PG5::pIpdata ip = &ptcl_mem[tid].iptcl;
	    PG5::pFodata fo = &ptcl_mem[tid].fout;
	    PG5::pJpdata jp = ptcl_mem[0].jptcl;
	    int nbody  = ptcl_mem[0].Nbody;
	    GravityKernel(ip, jp, fo, nbody,
			   Force_table, (v4sf)R2cut_xscale2, Ascale);
	}
	g5_get_forceMC(tid, ni, a+off, p+off);
    }
}
#endif
