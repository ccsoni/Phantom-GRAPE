#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <assert.h>
#include "avx.h"
#include "avx_type.h"
#include "gravity.h"

#define JMEMSIZE 262144

static double time;
static int nblen[PG6::NumberOfPipe];
static int nbl[PG6::NumberOfPipe][PG6::MaxLength];
static int nblerror;

static struct Ptcl_Mem{
    double pos[3];
    double vel[3];
    double acc[3];
    double jrk[3];
    double mss;
    double tim;
    int    idx;
    int    pad[3];
} ptcl_mem[JMEMSIZE] __attribute__ ((aligned(128)));

typedef struct Pred_Mem * pPred_Mem;
static struct Pred_Mem{
    v4df xpos;
    v4df ypos;
    v4df zpos;
    v8sf indx;
    v8sf mass;
    v8sf xvel;
    v8sf yvel;
    v8sf zvel;
} pred_mem[JMEMSIZE] __attribute__ ((aligned(256)));

typedef struct NeighbourList * pNeighbourList;
static struct NeighbourList{
    v8sf flag;
} (*neighbour)[JMEMSIZE];

typedef struct Iparticle * pIparticle;
struct Iparticle{
    v4df xpos0, xpos1;
    v4df ypos0, ypos1;
    v4df zpos0, zpos1;
    v8sf xvel01, yvel01, zvel01;
    v8sf id01, veps2;
    v4df xacc, yacc, zacc, pot;
    v8sf xjrk, yjrk, zjrk;
    v8sf rmin2, in;
    v8sf hinv;
};
#define NVAR_IP 21

void avx_debugfunc(void)
{
    for(int j = 0; j < 1024; j++){
	printf("%4d %+.13E %+.13E\n", j, ptcl_mem[j].acc[0], ptcl_mem[j].jrk[0]);
    }
    return;
}

void avx_open(int nthread)
{
    int ret;
    
    ret = posix_memalign((void **)&neighbour, 32, sizeof(struct NeighbourList) * JMEMSIZE * nthread);
    assert(ret == 0);
  
    return;
}

void avx_close(void)
{
    free(neighbour);
    return;
}

void avx_set_j_particle(int padr, int pidx, double tim, double mss,
			double *pos, double *vel, double *acc, double *jrk)
{
    ptcl_mem[padr].pos[0] = pos[0];
    ptcl_mem[padr].pos[1] = pos[1];
    ptcl_mem[padr].pos[2] = pos[2];
    ptcl_mem[padr].vel[0] = vel[0];
    ptcl_mem[padr].vel[1] = vel[1];
    ptcl_mem[padr].vel[2] = vel[2];
    ptcl_mem[padr].acc[0] = acc[0];
    ptcl_mem[padr].acc[1] = acc[1];
    ptcl_mem[padr].acc[2] = acc[2];
    ptcl_mem[padr].jrk[0] = jrk[0];
    ptcl_mem[padr].jrk[1] = jrk[1];
    ptcl_mem[padr].jrk[2] = jrk[2];
    ptcl_mem[padr].mss    = mss;
    ptcl_mem[padr].tim    = tim;
    ptcl_mem[padr].idx    = pidx;
    
    return;
}

void avx_set_ti(double tim)
{
    time = tim;
    return;
}

void avx_initialize_neighbourlist(void)
{
    nblerror = 0;
    
    return;
}

int avx_get_neighbourlist_error(void)
{
    return nblerror;
}

int avx_get_neighbourlist(int ipipe,
			  int maxlen,
			  int * nblenfunc,
			  int * nblfunc)
{
    if(nblen[ipipe] > maxlen){
	return 1;
    }else{
	*nblenfunc = nblen[ipipe];
	for(int j = 0; j < nblen[ipipe]; j++)
	    nblfunc[j] = nbl[ipipe][j];
	return 0;
    }
}

void avx_predict_j_particle(int nj)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int j = 0; j < nj; j += PG6::NumberOfJParallel){
	v4df xpos = v4df(ptcl_mem[j].pos[0], ptcl_mem[j+1].pos[0],
			 ptcl_mem[j+2].pos[0], ptcl_mem[j+3].pos[0]);
	v4df ypos = v4df(ptcl_mem[j].pos[1], ptcl_mem[j+1].pos[1],
			 ptcl_mem[j+2].pos[1], ptcl_mem[j+3].pos[1]);
	v4df zpos = v4df(ptcl_mem[j].pos[2], ptcl_mem[j+1].pos[2],
			 ptcl_mem[j+2].pos[2], ptcl_mem[j+3].pos[2]);
	v4df xvel = v4df(ptcl_mem[j].vel[0], ptcl_mem[j+1].vel[0],
			 ptcl_mem[j+2].vel[0], ptcl_mem[j+3].vel[0]);
	v4df yvel = v4df(ptcl_mem[j].vel[1], ptcl_mem[j+1].vel[1],
			 ptcl_mem[j+2].vel[1], ptcl_mem[j+3].vel[1]);
	v4df zvel = v4df(ptcl_mem[j].vel[2], ptcl_mem[j+1].vel[2],
			 ptcl_mem[j+2].vel[2], ptcl_mem[j+3].vel[2]);
	v4df xacc = v4df(ptcl_mem[j].acc[0], ptcl_mem[j+1].acc[0],
			 ptcl_mem[j+2].acc[0], ptcl_mem[j+3].acc[0]);
	v4df yacc = v4df(ptcl_mem[j].acc[1], ptcl_mem[j+1].acc[1],
			 ptcl_mem[j+2].acc[1], ptcl_mem[j+3].acc[1]);
	v4df zacc = v4df(ptcl_mem[j].acc[2], ptcl_mem[j+1].acc[2],
			 ptcl_mem[j+2].acc[2], ptcl_mem[j+3].acc[2]);
	v4df xjrk = v4df(ptcl_mem[j].jrk[0], ptcl_mem[j+1].jrk[0],
			 ptcl_mem[j+2].jrk[0], ptcl_mem[j+3].jrk[0]);
	v4df yjrk = v4df(ptcl_mem[j].jrk[1], ptcl_mem[j+1].jrk[1],
			 ptcl_mem[j+2].jrk[1], ptcl_mem[j+3].jrk[1]);
	v4df zjrk = v4df(ptcl_mem[j].jrk[2], ptcl_mem[j+1].jrk[2],
			 ptcl_mem[j+2].jrk[2], ptcl_mem[j+3].jrk[2]);
	v4df ptim = v4df(ptcl_mem[j].tim, ptcl_mem[j+1].tim,
			 ptcl_mem[j+2].tim, ptcl_mem[j+3].tim);

	int jmem = j  / PG6::NumberOfJParallel;
	v4df dt  = v4df(time) - ptim;
	v4df dt2 = dt * dt;
	v4df dt3 = dt * dt2;
	pred_mem[jmem].xpos = xpos + xvel * dt + xacc * dt2 + xjrk * dt3;
	pred_mem[jmem].ypos = ypos + yvel * dt + yacc * dt2 + yjrk * dt3;
	pred_mem[jmem].zpos = zpos + zvel * dt + zacc * dt2 + zjrk * dt3;

	v4df xvel_p = xvel + v4df(2.) * xacc * dt + v4df(3.) * xjrk * dt2;
	v4df yvel_p = yvel + v4df(2.) * yacc * dt + v4df(3.) * yjrk * dt2;
	v4df zvel_p = zvel + v4df(2.) * zacc * dt + v4df(3.) * zjrk * dt2;
	pred_mem[jmem].xvel = v8sf(_mm256_cvtpd_ps(xvel_p));
	pred_mem[jmem].yvel = v8sf(_mm256_cvtpd_ps(yvel_p));
	pred_mem[jmem].zvel = v8sf(_mm256_cvtpd_ps(zvel_p));

	pred_mem[jmem].indx = v8sf(v4sf(ptcl_mem[j].idx, ptcl_mem[j+1].idx,
					ptcl_mem[j+2].idx, ptcl_mem[j+3].idx));
	pred_mem[jmem].mass = v8sf(v4sf(ptcl_mem[j].mss, ptcl_mem[j+1].mss,
					ptcl_mem[j+2].mss, ptcl_mem[j+3].mss));

    }

    //ptcl_mem[nj].mss = 1.; // for debug
    int jmod = 0;
    if((jmod = nj % PG6::NumberOfJParallel) != 0){
	int jmem = nj / PG6::NumberOfJParallel;
	int jadr = jmem * PG6::NumberOfJParallel;
	float buf[PG6::NumberOfJParallel] __attribute__ ((aligned(32)))
	    = {0., 0., 0., 0.};
	for(int jj = 0; jj < jmod; jj++){
	    buf[jj] = ptcl_mem[jadr+jj].mss;
	}
	pred_mem[jmem].mass = v8sf(v4sf(buf[0], buf[1], buf[2], buf[3]),
				   v4sf(buf[0], buf[1], buf[2], buf[3]));
    }
    
    return;
}

#ifndef __AVX512F__

void gravity_kernel(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk)
{
    v4df pxi0(posvel[0].xpos);
    v4df pyi0(posvel[0].ypos);
    v4df pzi0(posvel[0].zpos);
    v4df pxi1(posvel[1].xpos);
    v4df pyi1(posvel[1].ypos);
    v4df pzi1(posvel[1].zpos);
    
    v8sf vxi(v4sf(posvel[0].xvel), v4sf(posvel[1].xvel));
    v8sf vyi(v4sf(posvel[0].yvel), v4sf(posvel[1].yvel));
    v8sf vzi(v4sf(posvel[0].zvel), v4sf(posvel[1].zvel));

    v8sf idi(v4sf(posvel[0].id),   v4sf(posvel[1].id));
    v8sf e2i(v4sf(posvel[0].eps2), v4sf(posvel[1].eps2));

    v4df axi = 0.;
    v4df ayi = 0.;
    v4df azi = 0.;
    v4df pti = 0.;
    v8sf jxi = 0.;
    v8sf jyi = 0.;
    v8sf jzi = 0.;

    pPred_Mem jptr = pred_mem;
    for(int j = 0; j < nj; j += PG6::NumberOfJParallel, jptr++) {
	v8sf dpx = v8sf(_mm256_cvtpd_ps(jptr->xpos - pxi0),
				 _mm256_cvtpd_ps(jptr->xpos - pxi1));
	v8sf dpy = v8sf(_mm256_cvtpd_ps(jptr->ypos - pyi0),
				 _mm256_cvtpd_ps(jptr->ypos - pyi1));
	v8sf dpz = v8sf(_mm256_cvtpd_ps(jptr->zpos - pzi0),
				 _mm256_cvtpd_ps(jptr->zpos - pzi1));

#ifndef __FMA__
	v8sf r2  = e2i + dpx * dpx + dpy * dpy + dpz * dpz;
#else
	v8sf r2  = e2i;
	r2 = _mm256_fmadd_ps(dpx, dpx, r2);
	r2 = _mm256_fmadd_ps(dpy, dpy, r2);
	r2 = _mm256_fmadd_ps(dpz, dpz, r2);
#endif
	v8sf ri  = ((jptr->indx != idi) & v8sf::rsqrt_1st_phantom(r2));
	v8sf mr  = jptr->mass * ri;

	v4df ptij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(mr, 0));
	v4df ptij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(mr, 1));
	v4df ptij  = _mm256_hadd_pd(ptij0, ptij1);
	pti += ptij;

	v8sf dvx = jptr->xvel - vxi;
	v8sf dvy = jptr->yvel - vyi;
	v8sf dvz = jptr->zvel - vzi;
#ifndef __FMA__
	v8sf rv  = dpx * dvx + dpy * dvy + dpz * dvz;
#else
	v8sf rv  = 0.;
	rv = _mm256_fmadd_ps(dpx, dvx, rv);
	rv = _mm256_fmadd_ps(dpy, dvy, rv);
	rv = _mm256_fmadd_ps(dpz, dvz, rv);
#endif	
	v8sf ri2 = ri * ri;

	rv *= v8sf(0.75) * ri2;
	mr *= ri2;

#ifndef __FMA__
	jxi += mr * dvx;
	jyi += mr * dvy;
	jzi += mr * dvz;
#else
	jxi = _mm256_fmadd_ps(mr, dvx, jxi);
	jyi = _mm256_fmadd_ps(mr, dvy, jyi);
	jzi = _mm256_fmadd_ps(mr, dvz, jzi);
#endif

	v8sf axijs = mr * dpx;
	v4df axij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(axijs, 0));
	v4df axij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(axijs, 1));
	v4df axijd = _mm256_hadd_pd(axij0, axij1);
	axi += axijd;
	v8sf ayijs = mr * dpy;
	v4df ayij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(ayijs, 0));
	v4df ayij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(ayijs, 1));
	v4df ayijd = _mm256_hadd_pd(ayij0, ayij1);
	ayi += ayijd;
	v8sf azijs = mr * dpz;
	v4df azij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(azijs, 0));
	v4df azij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(azijs, 1));
	v4df azijd = _mm256_hadd_pd(azij0, azij1);
	azi += azijd;

#ifndef __FMA__
	jxi -= rv * axijs;
	jyi -= rv * ayijs;
	jzi -= rv * azijs;
#else
	jxi = _mm256_fnmadd_ps(rv, axijs, jxi);
	jyi = _mm256_fnmadd_ps(rv, ayijs, jyi);
	jzi = _mm256_fnmadd_ps(rv, azijs, jzi);
#endif
    }

    v2df ax  = _mm256_extractf128_pd(axi, 0) + _mm256_extractf128_pd(axi, 1);
    v2df ay  = _mm256_extractf128_pd(ayi, 0) + _mm256_extractf128_pd(ayi, 1);
    v2df az  = _mm256_extractf128_pd(azi, 0) + _mm256_extractf128_pd(azi, 1);
    v2df pt  = _mm256_extractf128_pd(pti, 0) + _mm256_extractf128_pd(pti, 1);
    v8sf jxt = _mm256_hadd_ps(jxi, jxi);
    v8sf jx  = _mm256_hadd_ps(jxt, jxt);
    v8sf jyt = _mm256_hadd_ps(jyi, jyi);
    v8sf jy  = _mm256_hadd_ps(jyt, jyt);
    v8sf jzt = _mm256_hadd_ps(jzi, jzi);
    v8sf jz  = _mm256_hadd_ps(jzt, jzt);

    double buf_ax[2] __attribute__ ((aligned(16)));
    double buf_ay[2] __attribute__ ((aligned(16)));
    double buf_az[2] __attribute__ ((aligned(16)));
    double buf_pt[2] __attribute__ ((aligned(16)));
    float  buf_jx[8] __attribute__ ((aligned(32)));
    float  buf_jy[8] __attribute__ ((aligned(32)));
    float  buf_jz[8] __attribute__ ((aligned(32)));
    ax.store(buf_ax);
    ay.store(buf_ay);
    az.store(buf_az);
    pt.store(buf_pt);
    jx.store(buf_jx);
    jy.store(buf_jy);
    jz.store(buf_jz);

    accjerk[0].xacc = buf_ax[0];
    accjerk[0].yacc = buf_ay[0];
    accjerk[0].zacc = buf_az[0];
    accjerk[0].pot  = buf_pt[0];
    accjerk[0].xjrk = buf_jx[0];
    accjerk[0].yjrk = buf_jy[0];
    accjerk[0].zjrk = buf_jz[0];
    accjerk[1].xacc = buf_ax[1];
    accjerk[1].yacc = buf_ay[1];
    accjerk[1].zacc = buf_az[1];
    accjerk[1].pot  = buf_pt[1];
    accjerk[1].xjrk = buf_jx[4];
    accjerk[1].yjrk = buf_jy[4];
    accjerk[1].zjrk = buf_jz[4];

    return;
}

void gravity_kernel2(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk)
{
    v4df pxi0(posvel[0].xpos);
    v4df pyi0(posvel[0].ypos);
    v4df pzi0(posvel[0].zpos);
    v4df pxi1(posvel[1].xpos);
    v4df pyi1(posvel[1].ypos);
    v4df pzi1(posvel[1].zpos);
    
    v8sf vxi(v4sf(posvel[0].xvel), v4sf(posvel[1].xvel));
    v8sf vyi(v4sf(posvel[0].yvel), v4sf(posvel[1].yvel));
    v8sf vzi(v4sf(posvel[0].zvel), v4sf(posvel[1].zvel));

    v8sf idi(v4sf(posvel[0].id),   v4sf(posvel[1].id));
    v8sf e2i(v4sf(posvel[0].eps2), v4sf(posvel[1].eps2));

    v4df axi = 0.;
    v4df ayi = 0.;
    v4df azi = 0.;
    v4df pti = 0.;
    v8sf jxi = 0.;
    v8sf jyi = 0.;
    v8sf jzi = 0.;
    v8sf idn = -1.;
    v8sf rin = 0.;

    pPred_Mem jptr = pred_mem;
    for(int j = 0; j < nj; j += PG6::NumberOfJParallel, jptr++) {
	v8sf dpx = v8sf(_mm256_cvtpd_ps(jptr->xpos - pxi0),
				 _mm256_cvtpd_ps(jptr->xpos - pxi1));
	v8sf dpy = v8sf(_mm256_cvtpd_ps(jptr->ypos - pyi0),
				 _mm256_cvtpd_ps(jptr->ypos - pyi1));
	v8sf dpz = v8sf(_mm256_cvtpd_ps(jptr->zpos - pzi0),
				 _mm256_cvtpd_ps(jptr->zpos - pzi1));

#ifndef __FMA__
	v8sf r2  = e2i + dpx * dpx + dpy * dpy + dpz * dpz;
#else
	v8sf r2  = e2i;
	r2 = _mm256_fmadd_ps(dpx, dpx, r2);
	r2 = _mm256_fmadd_ps(dpy, dpy, r2);
	r2 = _mm256_fmadd_ps(dpz, dpz, r2);
#endif
	v8sf idj = jptr->indx;
	v8sf ri  = ((idj != idi) & v8sf::rsqrt_1st_phantom(r2));
	v8sf mr  = jptr->mass * ri;
	idn = ((ri < rin) & idj) + ((ri >= rin) & idn);
	rin = v8sf::min(rin, ri);
	
	v4df ptij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(mr, 0));
	v4df ptij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(mr, 1));
	v4df ptij  = _mm256_hadd_pd(ptij0, ptij1);
	pti += ptij;
	
	v8sf dvx = jptr->xvel - vxi;
	v8sf dvy = jptr->yvel - vyi;
	v8sf dvz = jptr->zvel - vzi;
#ifndef __FMA__
	v8sf rv  = dpx * dvx + dpy * dvy + dpz * dvz;
#else
	v8sf rv  = 0.;
	rv = _mm256_fmadd_ps(dpx, dvx, rv);
	rv = _mm256_fmadd_ps(dpy, dvy, rv);
	rv = _mm256_fmadd_ps(dpz, dvz, rv);
#endif	
	v8sf ri2 = ri * ri;
	
	rv *= v8sf(0.75) * ri2;
	mr *= ri2;
	
#ifndef __FMA__
	jxi += mr * dvx;
	jyi += mr * dvy;
	jzi += mr * dvz;
#else
	jxi = _mm256_fmadd_ps(mr, dvx, jxi);
	jyi = _mm256_fmadd_ps(mr, dvy, jyi);
	jzi = _mm256_fmadd_ps(mr, dvz, jzi);
#endif

	v8sf axijs = mr * dpx;
	v4df axij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(axijs, 0));
	v4df axij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(axijs, 1));
	v4df axijd = _mm256_hadd_pd(axij0, axij1);
	axi += axijd;
	v8sf ayijs = mr * dpy;
	v4df ayij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(ayijs, 0));
	v4df ayij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(ayijs, 1));
	v4df ayijd = _mm256_hadd_pd(ayij0, ayij1);
	ayi += ayijd;
	v8sf azijs = mr * dpz;
	v4df azij0 = _mm256_cvtps_pd(_mm256_extractf128_ps(azijs, 0));
	v4df azij1 = _mm256_cvtps_pd(_mm256_extractf128_ps(azijs, 1));
	v4df azijd = _mm256_hadd_pd(azij0, azij1);
	azi += azijd;

#ifndef __FMA__
	jxi -= rv * axijs;
	jyi -= rv * ayijs;
	jzi -= rv * azijs;
#else
	jxi = _mm256_fnmadd_ps(rv, axijs, jxi);
	jyi = _mm256_fnmadd_ps(rv, ayijs, jyi);
	jzi = _mm256_fnmadd_ps(rv, azijs, jzi);
#endif
    }

    v2df ax  = _mm256_extractf128_pd(axi, 0) + _mm256_extractf128_pd(axi, 1);
    v2df ay  = _mm256_extractf128_pd(ayi, 0) + _mm256_extractf128_pd(ayi, 1);
    v2df az  = _mm256_extractf128_pd(azi, 0) + _mm256_extractf128_pd(azi, 1);
    v2df pt  = _mm256_extractf128_pd(pti, 0) + _mm256_extractf128_pd(pti, 1);
    v8sf jxt = _mm256_hadd_ps(jxi, jxi);
    v8sf jx  = _mm256_hadd_ps(jxt, jxt);
    v8sf jyt = _mm256_hadd_ps(jyi, jyi);
    v8sf jy  = _mm256_hadd_ps(jyt, jyt);
    v8sf jzt = _mm256_hadd_ps(jzi, jzi);
    v8sf jz  = _mm256_hadd_ps(jzt, jzt);
    
    double buf_ax[2] __attribute__ ((aligned(16)));
    double buf_ay[2] __attribute__ ((aligned(16)));
    double buf_az[2] __attribute__ ((aligned(16)));
    double buf_pt[2] __attribute__ ((aligned(16)));
    float  buf_jx[8] __attribute__ ((aligned(32)));
    float  buf_jy[8] __attribute__ ((aligned(32)));
    float  buf_jz[8] __attribute__ ((aligned(32)));
    ax.store(buf_ax);
    ay.store(buf_ay);
    az.store(buf_az);
    pt.store(buf_pt);
    jx.store(buf_jx);
    jy.store(buf_jy);
    jz.store(buf_jz);
    
    accjerk[0].xacc = buf_ax[0];
    accjerk[0].yacc = buf_ay[0];
    accjerk[0].zacc = buf_az[0];
    accjerk[0].pot  = buf_pt[0];
    accjerk[0].xjrk = buf_jx[0];
    accjerk[0].yjrk = buf_jy[0];
    accjerk[0].zjrk = buf_jz[0];
    accjerk[1].xacc = buf_ax[1];
    accjerk[1].yacc = buf_ay[1];
    accjerk[1].zacc = buf_az[1];
    accjerk[1].pot  = buf_pt[1];
    accjerk[1].xjrk = buf_jx[4];
    accjerk[1].yjrk = buf_jy[4];
    accjerk[1].zjrk = buf_jz[4];
    
    float buf_rin[PG6::NumberOfVectorSingle]
	__attribute__ ((aligned(PG6::NumberOfVectorSingle*4)));
    float buf_idn[PG6::NumberOfVectorSingle]
	__attribute__ ((aligned(PG6::NumberOfVectorSingle*4)));
    rin.store(buf_rin);
    idn.store(buf_idn);
    for(int i = 0; i < PG6::NumberOfIParallel; i++) {
	float rimin = 0.;
        for(int j = 0; j < PG6::NumberOfJParallel; j++) {
            int jj = j + i * PG6::NumberOfJParallel;
 	    if(buf_rin[jj] < rimin) {
                 rimin          = buf_rin[jj];
	         accjerk[i].nnb = (int)buf_idn[jj];
            }
        }
    }

    return;
}

void gravity_kerneln(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk, int i, int ithread)
{
    return;
}

void gravity_kernel2n(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk, int i, int ithread)
{
    return;
}

#else

inline double reduce_add_pd(int imm, v8df _src) {
    v4df src = _mm512_extractf64x4_pd(_src, imm);
    double buf[4] __attribute__ ((aligned(32)));
    src.store(buf);
    return (buf[0] + buf[1] + buf[2] + buf[3]);
}

inline v8sf reduce_add_ps(int imm, v16sf src) {
    v8sf tmp2 = _mm512_extractf32x8_ps(src, imm);
    v8sf tmp1 = _mm256_hadd_ps(tmp2, tmp2);
    v8sf dst  = _mm256_hadd_ps(tmp1, tmp1);
    return dst;
}

void gravity_kernel(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk)
{
    v8df pxi0(posvel[0].xpos, posvel[1].xpos);
    v8df pyi0(posvel[0].ypos, posvel[1].ypos);
    v8df pzi0(posvel[0].zpos, posvel[1].zpos);
    v8df pxi1(posvel[2].xpos, posvel[3].xpos);
    v8df pyi1(posvel[2].ypos, posvel[3].ypos);
    v8df pzi1(posvel[2].zpos, posvel[3].zpos);

    v16sf vxi(v4sf(posvel[0].xvel), v4sf(posvel[1].xvel),
	      v4sf(posvel[2].xvel), v4sf(posvel[3].xvel));
    v16sf vyi(v4sf(posvel[0].yvel), v4sf(posvel[1].yvel),
	      v4sf(posvel[2].yvel), v4sf(posvel[3].yvel));
    v16sf vzi(v4sf(posvel[0].zvel), v4sf(posvel[1].zvel),
	      v4sf(posvel[2].zvel), v4sf(posvel[3].zvel));

    v16sf idi(v4sf(posvel[0].id),   v4sf(posvel[1].id),
	      v4sf(posvel[2].id),   v4sf(posvel[3].id));
    v16sf e2i(v4sf(posvel[0].eps2), v4sf(posvel[1].eps2),
	      v4sf(posvel[2].eps2), v4sf(posvel[3].eps2));

    v8df  axi0 = 0.;
    v8df  ayi0 = 0.;
    v8df  azi0 = 0.;
    v8df  pti0 = 0.;
    v8df  axi1 = 0.;
    v8df  ayi1 = 0.;
    v8df  azi1 = 0.;
    v8df  pti1 = 0.;
    v16sf jxi = 0.;
    v16sf jyi = 0.;
    v16sf jzi = 0.;

    pPred_Mem jptr = pred_mem;
    for(int j = 0; j < nj; j += PG6::NumberOfJParallel, jptr++) {
	v16sf dpx = v16sf(_mm512_cvtpd_ps(v8df(jptr->xpos) - pxi0),
			  _mm512_cvtpd_ps(v8df(jptr->xpos) - pxi1));
	v16sf dpy = v16sf(_mm512_cvtpd_ps(v8df(jptr->ypos) - pyi0),
			  _mm512_cvtpd_ps(v8df(jptr->ypos) - pyi1));
	v16sf dpz = v16sf(_mm512_cvtpd_ps(v8df(jptr->zpos) - pzi0),
			  _mm512_cvtpd_ps(v8df(jptr->zpos) - pzi1));
	
	v16sf r2 = e2i;
	r2 = _mm512_fmadd_ps(dpx, dpx, r2);
	r2 = _mm512_fmadd_ps(dpy, dpy, r2);
	r2 = _mm512_fmadd_ps(dpz, dpz, r2);

//#define DEBUG
#ifdef DEBUG
	v16sf ri = _mm512_mask_mov_ps(v16sf(0.),
				      _mm512_cmp_ps_mask(v16sf(jptr->indx), idi,
							 _CMP_NEQ_UQ),
				      _mm512_rsqrt28_ps(r2));
#else
	v16sf ri = _mm512_mask_mov_ps(v16sf(0.),
				      _mm512_cmp_ps_mask(v16sf(jptr->indx), idi,
							 _CMP_NEQ_UQ),
				      v16sf::rsqrt_1st_phantom(r2));
#endif
	v16sf mr = v16sf(jptr->mass) * ri;

	v8df ptij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(mr, 0));
	v8df ptij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(mr, 1));
	pti0 += ptij0;
	pti1 += ptij1;

	v16sf dvx = v16sf(jptr->xvel) - vxi;
	v16sf dvy = v16sf(jptr->yvel) - vyi;
	v16sf dvz = v16sf(jptr->zvel) - vzi;
	v16sf rv  = 0.;
	rv = _mm512_fmadd_ps(dpx, dvx, rv);
	rv = _mm512_fmadd_ps(dpy, dvy, rv);
	rv = _mm512_fmadd_ps(dpz, dvz, rv);
	v16sf ri2 = ri * ri;

#ifdef DEBUG
	rv *= v16sf(3.0) * ri2;
#else
	rv *= v16sf(0.75) * ri2;
#endif
	mr *= ri2;

	jxi = _mm512_fmadd_ps(mr, dvx, jxi);
	jyi = _mm512_fmadd_ps(mr, dvy, jyi);
	jzi = _mm512_fmadd_ps(mr, dvz, jzi);

	v16sf axijs = mr * dpx;
	v8df  axij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(axijs, 0));
	v8df  axij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(axijs, 1));
	axi0 += axij0;
	axi1 += axij1;
	v16sf ayijs = mr * dpy;
	v8df  ayij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ayijs, 0));
	v8df  ayij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ayijs, 1));
	ayi0 += ayij0;
	ayi1 += ayij1;
	v16sf azijs = mr * dpz;
	v8df  azij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(azijs, 0));
	v8df  azij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(azijs, 1));
	azi0 += azij0;
	azi1 += azij1;

	jxi = _mm512_fnmadd_ps(rv, axijs, jxi);
	jyi = _mm512_fnmadd_ps(rv, ayijs, jyi);
	jzi = _mm512_fnmadd_ps(rv, azijs, jzi);
    }

#if 1
#ifdef DEBUG
    axi0 *= v8df(-8.0);
    axi1 *= v8df(-8.0);
    ayi0 *= v8df(-8.0);
    ayi1 *= v8df(-8.0);
    azi0 *= v8df(-8.0);
    azi1 *= v8df(-8.0);
    pti0 *= v8df(+2.0);
    pti1 *= v8df(+2.0);
    jxi  *= v16sf(-8.0);
    jyi  *= v16sf(-8.0);
    jzi  *= v16sf(-8.0);
#endif
    // Should use "_mm512_mask_reduce_add" (A. Tanikawa)
    /*
    accjerk[0].xacc = _mm512_mask_reduce_add_pd(0b00001111, axi0);
    accjerk[0].yacc = _mm512_mask_reduce_add_pd(0b00001111, ayi0);
    accjerk[0].zacc = _mm512_mask_reduce_add_pd(0b00001111, azi0);
    accjerk[0].pot  = _mm512_mask_reduce_add_pd(0b00001111, pti0);
    */
    accjerk[0].xacc = reduce_add_pd(0, axi0);;
    accjerk[0].yacc = reduce_add_pd(0, ayi0);;
    accjerk[0].zacc = reduce_add_pd(0, azi0);;
    accjerk[0].pot  = reduce_add_pd(0, pti0);;
    accjerk[1].xacc = reduce_add_pd(1, axi0);;
    accjerk[1].yacc = reduce_add_pd(1, ayi0);;
    accjerk[1].zacc = reduce_add_pd(1, azi0);;
    accjerk[1].pot  = reduce_add_pd(1, pti0);;
    accjerk[2].xacc = reduce_add_pd(0, axi1);;
    accjerk[2].yacc = reduce_add_pd(0, ayi1);;
    accjerk[2].zacc = reduce_add_pd(0, azi1);;
    accjerk[2].pot  = reduce_add_pd(0, pti1);;
    accjerk[3].xacc = reduce_add_pd(1, axi1);;
    accjerk[3].yacc = reduce_add_pd(1, ayi1);;
    accjerk[3].zacc = reduce_add_pd(1, azi1);;
    accjerk[3].pot  = reduce_add_pd(1, pti1);;

    float  buf_jx[2][8] __attribute__ ((aligned(32)));
    float  buf_jy[2][8] __attribute__ ((aligned(32)));
    float  buf_jz[2][8] __attribute__ ((aligned(32)));
    (reduce_add_ps(0, jxi)).store(buf_jx[0]);
    (reduce_add_ps(1, jxi)).store(buf_jx[1]);
    (reduce_add_ps(0, jyi)).store(buf_jy[0]);
    (reduce_add_ps(1, jyi)).store(buf_jy[1]);
    (reduce_add_ps(0, jzi)).store(buf_jz[0]);
    (reduce_add_ps(1, jzi)).store(buf_jz[1]);
    accjerk[0].xjrk = buf_jx[0][0];
    accjerk[0].yjrk = buf_jy[0][0];
    accjerk[0].zjrk = buf_jz[0][0];
    accjerk[1].xjrk = buf_jx[0][4];
    accjerk[1].yjrk = buf_jy[0][4];
    accjerk[1].zjrk = buf_jz[0][4];
    accjerk[2].xjrk = buf_jx[1][0];
    accjerk[2].yjrk = buf_jy[1][0];
    accjerk[2].zjrk = buf_jz[1][0];
    accjerk[3].xjrk = buf_jx[1][4];
    accjerk[3].yjrk = buf_jy[1][4];
    accjerk[3].zjrk = buf_jz[1][4];
#else
    v4df ax0 = _mm512_extractf64x4_pd(axi0, 0);
    v4df ax1 = _mm512_extractf64x4_pd(axi0, 1);
    v4df ax2 = _mm512_extractf64x4_pd(axi1, 0);
    v4df ax3 = _mm512_extractf64x4_pd(axi1, 1);
    v4df ay0 = _mm512_extractf64x4_pd(ayi0, 0);
    v4df ay1 = _mm512_extractf64x4_pd(ayi0, 1);
    v4df ay2 = _mm512_extractf64x4_pd(ayi1, 0);
    v4df ay3 = _mm512_extractf64x4_pd(ayi1, 1);
    v4df az0 = _mm512_extractf64x4_pd(azi0, 0);
    v4df az1 = _mm512_extractf64x4_pd(azi0, 1);
    v4df az2 = _mm512_extractf64x4_pd(azi1, 0);
    v4df az3 = _mm512_extractf64x4_pd(azi1, 1);
    v4df pt0 = _mm512_extractf64x4_pd(pti0, 0);
    v4df pt1 = _mm512_extractf64x4_pd(pti0, 1);
    v4df pt2 = _mm512_extractf64x4_pd(pti1, 0);
    v4df pt3 = _mm512_extractf64x4_pd(pti1, 1);
    v8sf jx01t2 = _mm512_extractf32x8_ps(jxi, 0);
    v8sf jx01t1 = _mm256_hadd_ps(jx01t2, jx01t2);
    v8sf jx01   = _mm256_hadd_ps(jx01t1, jx01t1);
    v8sf jx23t2 = _mm512_extractf32x8_ps(jxi, 1);
    v8sf jx23t1 = _mm256_hadd_ps(jx23t2, jx23t2);
    v8sf jx23   = _mm256_hadd_ps(jx23t1, jx23t1);
    v8sf jy01t2 = _mm512_extractf32x8_ps(jyi, 0);
    v8sf jy01t1 = _mm256_hadd_ps(jy01t2, jy01t2);
    v8sf jy01   = _mm256_hadd_ps(jy01t1, jy01t1);
    v8sf jy23t2 = _mm512_extractf32x8_ps(jyi, 1);
    v8sf jy23t1 = _mm256_hadd_ps(jy23t2, jy23t2);
    v8sf jy23   = _mm256_hadd_ps(jy23t1, jy23t1);
    v8sf jz01t2 = _mm512_extractf32x8_ps(jzi, 0);
    v8sf jz01t1 = _mm256_hadd_ps(jz01t2, jz01t2);
    v8sf jz01   = _mm256_hadd_ps(jz01t1, jz01t1);
    v8sf jz23t2 = _mm512_extractf32x8_ps(jzi, 1);
    v8sf jz23t1 = _mm256_hadd_ps(jz23t2, jz23t2);
    v8sf jz23   = _mm256_hadd_ps(jz23t1, jz23t1);

    double buf_ax[4][4] __attribute__ ((aligned(32)));
    double buf_ay[4][4] __attribute__ ((aligned(32)));
    double buf_az[4][4] __attribute__ ((aligned(32)));
    double buf_pt[4][4] __attribute__ ((aligned(32)));
    float  buf_jx[2][8] __attribute__ ((aligned(32)));
    float  buf_jy[2][8] __attribute__ ((aligned(32)));
    float  buf_jz[2][8] __attribute__ ((aligned(32)));

    ax0.store(buf_ax[0]);
    ax1.store(buf_ax[1]);
    ax2.store(buf_ax[2]);
    ax3.store(buf_ax[3]);
    ay0.store(buf_ay[0]);
    ay1.store(buf_ay[1]);
    ay2.store(buf_ay[2]);
    ay3.store(buf_ay[3]);
    az0.store(buf_az[0]);
    az1.store(buf_az[1]);
    az2.store(buf_az[2]);
    az3.store(buf_az[3]);
    pt0.store(buf_pt[0]);
    pt1.store(buf_pt[1]);
    pt2.store(buf_pt[2]);
    pt3.store(buf_pt[3]);
    jx01.store(buf_jx[0]);
    jx23.store(buf_jx[1]);
    jy01.store(buf_jy[0]);
    jy23.store(buf_jy[1]);
    jz01.store(buf_jz[0]);
    jz23.store(buf_jz[1]);

    accjerk[0].xacc = buf_ax[0][0] + buf_ax[0][1] + buf_ax[0][2] + buf_ax[0][3];
    accjerk[0].yacc = buf_ay[0][0] + buf_ay[0][1] + buf_ay[0][2] + buf_ay[0][3];
    accjerk[0].zacc = buf_az[0][0] + buf_az[0][1] + buf_az[0][2] + buf_az[0][3];
    accjerk[0].pot  = buf_pt[0][0] + buf_pt[0][1] + buf_pt[0][2] + buf_pt[0][3];
    accjerk[0].xjrk = buf_jx[0][0];
    accjerk[0].yjrk = buf_jy[0][0];
    accjerk[0].zjrk = buf_jz[0][0];
    accjerk[1].xacc = buf_ax[1][0] + buf_ax[1][1] + buf_ax[1][2] + buf_ax[1][3];
    accjerk[1].yacc = buf_ay[1][0] + buf_ay[1][1] + buf_ay[1][2] + buf_ay[1][3];
    accjerk[1].zacc = buf_az[1][0] + buf_az[1][1] + buf_az[1][2] + buf_az[1][3];
    accjerk[1].pot  = buf_pt[1][0] + buf_pt[1][1] + buf_pt[1][2] + buf_pt[1][3];
    accjerk[1].xjrk = buf_jx[0][4];
    accjerk[1].yjrk = buf_jy[0][4];
    accjerk[1].zjrk = buf_jz[0][4];
    accjerk[2].xacc = buf_ax[2][0] + buf_ax[2][1] + buf_ax[2][2] + buf_ax[2][3];
    accjerk[2].yacc = buf_ay[2][0] + buf_ay[2][1] + buf_ay[2][2] + buf_ay[2][3];
    accjerk[2].zacc = buf_az[2][0] + buf_az[2][1] + buf_az[2][2] + buf_az[2][3];
    accjerk[2].pot  = buf_pt[2][0] + buf_pt[2][1] + buf_pt[2][2] + buf_pt[2][3];
    accjerk[2].xjrk = buf_jx[1][0];
    accjerk[2].yjrk = buf_jy[1][0];
    accjerk[2].zjrk = buf_jz[1][0];
    accjerk[3].xacc = buf_ax[3][0] + buf_ax[3][1] + buf_ax[3][2] + buf_ax[3][3];
    accjerk[3].yacc = buf_ay[3][0] + buf_ay[3][1] + buf_ay[3][2] + buf_ay[3][3];
    accjerk[3].zacc = buf_az[3][0] + buf_az[3][1] + buf_az[3][2] + buf_az[3][3];
    accjerk[3].pot  = buf_pt[3][0] + buf_pt[3][1] + buf_pt[3][2] + buf_pt[3][3];
    accjerk[3].xjrk = buf_jx[1][4];
    accjerk[3].yjrk = buf_jy[1][4];
    accjerk[3].zjrk = buf_jz[1][4];
#endif

    return;
}

void gravity_kernel2(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk)
{
    v8df pxi0(posvel[0].xpos, posvel[1].xpos);
    v8df pyi0(posvel[0].ypos, posvel[1].ypos);
    v8df pzi0(posvel[0].zpos, posvel[1].zpos);
    v8df pxi1(posvel[2].xpos, posvel[3].xpos);
    v8df pyi1(posvel[2].ypos, posvel[3].ypos);
    v8df pzi1(posvel[2].zpos, posvel[3].zpos);

    v16sf vxi(v4sf(posvel[0].xvel), v4sf(posvel[1].xvel),
	      v4sf(posvel[2].xvel), v4sf(posvel[3].xvel));
    v16sf vyi(v4sf(posvel[0].yvel), v4sf(posvel[1].yvel),
	      v4sf(posvel[2].yvel), v4sf(posvel[3].yvel));
    v16sf vzi(v4sf(posvel[0].zvel), v4sf(posvel[1].zvel),
	      v4sf(posvel[2].zvel), v4sf(posvel[3].zvel));

    v16sf idi(v4sf(posvel[0].id),   v4sf(posvel[1].id),
	      v4sf(posvel[2].id),   v4sf(posvel[3].id));
    v16sf e2i(v4sf(posvel[0].eps2), v4sf(posvel[1].eps2),
	      v4sf(posvel[2].eps2), v4sf(posvel[3].eps2));

    v8df  axi0 = 0.;
    v8df  ayi0 = 0.;
    v8df  azi0 = 0.;
    v8df  pti0 = 0.;
    v8df  axi1 = 0.;
    v8df  ayi1 = 0.;
    v8df  azi1 = 0.;
    v8df  pti1 = 0.;
    v16sf jxi = 0.;
    v16sf jyi = 0.;
    v16sf jzi = 0.;
    v16sf idn = -1.;
    v16sf rin = 0.;

    pPred_Mem jptr = pred_mem;
    for(int j = 0; j < nj; j += PG6::NumberOfJParallel, jptr++) {
	v16sf dpx = v16sf(_mm512_cvtpd_ps(v8df(jptr->xpos) - pxi0),
			  _mm512_cvtpd_ps(v8df(jptr->xpos) - pxi1));
	v16sf dpy = v16sf(_mm512_cvtpd_ps(v8df(jptr->ypos) - pyi0),
			  _mm512_cvtpd_ps(v8df(jptr->ypos) - pyi1));
	v16sf dpz = v16sf(_mm512_cvtpd_ps(v8df(jptr->zpos) - pzi0),
			  _mm512_cvtpd_ps(v8df(jptr->zpos) - pzi1));
	
	v16sf r2 = e2i;
	r2 = _mm512_fmadd_ps(dpx, dpx, r2);
	r2 = _mm512_fmadd_ps(dpy, dpy, r2);
	r2 = _mm512_fmadd_ps(dpz, dpz, r2);

	v16sf idj = v16sf(jptr->indx);
	v16sf ri  = _mm512_mask_mov_ps(v16sf(0.),
				       _mm512_cmp_ps_mask(v16sf(idj), idi, _CMP_NEQ_UQ),
				       v16sf::rsqrt_1st_phantom(r2));
	idn = _mm512_mask_mov_ps(idj, _mm512_cmp_ps_mask(ri, rin, _CMP_GT_OS), idn);
	rin = _mm512_mask_mov_ps(ri, _mm512_cmp_ps_mask(ri, rin, _CMP_GT_OS), rin);
				       
	v16sf mr = v16sf(jptr->mass) * ri;

	v8df ptij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(mr, 0));
	v8df ptij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(mr, 1));
	pti0 += ptij0;
	pti1 += ptij1;

	v16sf dvx = v16sf(jptr->xvel) - vxi;
	v16sf dvy = v16sf(jptr->yvel) - vyi;
	v16sf dvz = v16sf(jptr->zvel) - vzi;
	v16sf rv  = 0.;
	rv = _mm512_fmadd_ps(dpx, dvx, rv);
	rv = _mm512_fmadd_ps(dpy, dvy, rv);
	rv = _mm512_fmadd_ps(dpz, dvz, rv);
	v16sf ri2 = ri * ri;

	rv *= v16sf(0.75) * ri2;
	mr *= ri2;

	jxi = _mm512_fmadd_ps(mr, dvx, jxi);
	jyi = _mm512_fmadd_ps(mr, dvy, jyi);
	jzi = _mm512_fmadd_ps(mr, dvz, jzi);

	v16sf axijs = mr * dpx;
	v8df  axij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(axijs, 0));
	v8df  axij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(axijs, 1));
	axi0 += axij0;
	axi1 += axij1;
	v16sf ayijs = mr * dpy;
	v8df  ayij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ayijs, 0));
	v8df  ayij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ayijs, 1));
	ayi0 += ayij0;
	ayi1 += ayij1;
	v16sf azijs = mr * dpz;
	v8df  azij0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(azijs, 0));
	v8df  azij1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(azijs, 1));
	azi0 += azij0;
	azi1 += azij1;

	jxi = _mm512_fnmadd_ps(rv, axijs, jxi);
	jyi = _mm512_fnmadd_ps(rv, ayijs, jyi);
	jzi = _mm512_fnmadd_ps(rv, azijs, jzi);
    }

    // Should use "_mm512_mask_reduce_add" (A. Tanikawa)
#if 1
    /*
    accjerk[0].xacc = _mm512_mask_reduce_add_pd(0b00001111, axi0);
    accjerk[0].yacc = _mm512_mask_reduce_add_pd(0b00001111, ayi0);
    accjerk[0].zacc = _mm512_mask_reduce_add_pd(0b00001111, azi0);
    accjerk[0].pot  = _mm512_mask_reduce_add_pd(0b00001111, pti0);
    */
    accjerk[0].xacc = reduce_add_pd(0, axi0);;
    accjerk[0].yacc = reduce_add_pd(0, ayi0);;
    accjerk[0].zacc = reduce_add_pd(0, azi0);;
    accjerk[0].pot  = reduce_add_pd(0, pti0);;
    accjerk[1].xacc = reduce_add_pd(1, axi0);;
    accjerk[1].yacc = reduce_add_pd(1, ayi0);;
    accjerk[1].zacc = reduce_add_pd(1, azi0);;
    accjerk[1].pot  = reduce_add_pd(1, pti0);;
    accjerk[2].xacc = reduce_add_pd(0, axi1);;
    accjerk[2].yacc = reduce_add_pd(0, ayi1);;
    accjerk[2].zacc = reduce_add_pd(0, azi1);;
    accjerk[2].pot  = reduce_add_pd(0, pti1);;
    accjerk[3].xacc = reduce_add_pd(1, axi1);;
    accjerk[3].yacc = reduce_add_pd(1, ayi1);;
    accjerk[3].zacc = reduce_add_pd(1, azi1);;
    accjerk[3].pot  = reduce_add_pd(1, pti1);;

    float  buf_jx[2][8] __attribute__ ((aligned(32)));
    float  buf_jy[2][8] __attribute__ ((aligned(32)));
    float  buf_jz[2][8] __attribute__ ((aligned(32)));
    (reduce_add_ps(0, jxi)).store(buf_jx[0]);
    (reduce_add_ps(1, jxi)).store(buf_jx[1]);
    (reduce_add_ps(0, jyi)).store(buf_jy[0]);
    (reduce_add_ps(1, jyi)).store(buf_jy[1]);
    (reduce_add_ps(0, jzi)).store(buf_jz[0]);
    (reduce_add_ps(1, jzi)).store(buf_jz[1]);
    accjerk[0].xjrk = buf_jx[0][0];
    accjerk[0].yjrk = buf_jy[0][0];
    accjerk[0].zjrk = buf_jz[0][0];
    accjerk[1].xjrk = buf_jx[0][4];
    accjerk[1].yjrk = buf_jy[0][4];
    accjerk[1].zjrk = buf_jz[0][4];
    accjerk[2].xjrk = buf_jx[1][0];
    accjerk[2].yjrk = buf_jy[1][0];
    accjerk[2].zjrk = buf_jz[1][0];
    accjerk[3].xjrk = buf_jx[1][4];
    accjerk[3].yjrk = buf_jy[1][4];
    accjerk[3].zjrk = buf_jz[1][4];
    
    float buf_rin[PG6::NumberOfVectorSingle]
	__attribute__ ((aligned(PG6::NumberOfVectorSingle*4)));
    float buf_idn[PG6::NumberOfVectorSingle]
	__attribute__ ((aligned(PG6::NumberOfVectorSingle*4)));
    rin.store(buf_rin);
    idn.store(buf_idn);
    for(int i = 0; i < PG6::NumberOfIParallel; i++) {
	float rimin = 0.;
        for(int j = 0; j < PG6::NumberOfJParallel; j++) {
            int jj = j + i * PG6::NumberOfJParallel;
 	    if(buf_rin[jj] < rimin) {
                 rimin          = buf_rin[jj];
	         accjerk[i].nnb = (int)buf_idn[jj];
            }
        }
    }

#else
    v4df ax0 = _mm512_extractf64x4_pd(axi0, 0);
    v4df ax1 = _mm512_extractf64x4_pd(axi0, 1);
    v4df ax2 = _mm512_extractf64x4_pd(axi1, 0);
    v4df ax3 = _mm512_extractf64x4_pd(axi1, 1);
    v4df ay0 = _mm512_extractf64x4_pd(ayi0, 0);
    v4df ay1 = _mm512_extractf64x4_pd(ayi0, 1);
    v4df ay2 = _mm512_extractf64x4_pd(ayi1, 0);
    v4df ay3 = _mm512_extractf64x4_pd(ayi1, 1);
    v4df az0 = _mm512_extractf64x4_pd(azi0, 0);
    v4df az1 = _mm512_extractf64x4_pd(azi0, 1);
    v4df az2 = _mm512_extractf64x4_pd(azi1, 0);
    v4df az3 = _mm512_extractf64x4_pd(azi1, 1);
    v4df pt0 = _mm512_extractf64x4_pd(pti0, 0);
    v4df pt1 = _mm512_extractf64x4_pd(pti0, 1);
    v4df pt2 = _mm512_extractf64x4_pd(pti1, 0);
    v4df pt3 = _mm512_extractf64x4_pd(pti1, 1);
    v8sf jx01t2 = _mm512_extractf32x8_ps(jxi, 0);
    v8sf jx01t1 = _mm256_hadd_ps(jx01t2, jx01t2);
    v8sf jx01   = _mm256_hadd_ps(jx01t1, jx01t1);
    v8sf jx23t2 = _mm512_extractf32x8_ps(jxi, 1);
    v8sf jx23t1 = _mm256_hadd_ps(jx23t2, jx23t2);
    v8sf jx23   = _mm256_hadd_ps(jx23t1, jx23t1);
    v8sf jy01t2 = _mm512_extractf32x8_ps(jyi, 0);
    v8sf jy01t1 = _mm256_hadd_ps(jy01t2, jy01t2);
    v8sf jy01   = _mm256_hadd_ps(jy01t1, jy01t1);
    v8sf jy23t2 = _mm512_extractf32x8_ps(jyi, 1);
    v8sf jy23t1 = _mm256_hadd_ps(jy23t2, jy23t2);
    v8sf jy23   = _mm256_hadd_ps(jy23t1, jy23t1);
    v8sf jz01t2 = _mm512_extractf32x8_ps(jzi, 0);
    v8sf jz01t1 = _mm256_hadd_ps(jz01t2, jz01t2);
    v8sf jz01   = _mm256_hadd_ps(jz01t1, jz01t1);
    v8sf jz23t2 = _mm512_extractf32x8_ps(jzi, 1);
    v8sf jz23t1 = _mm256_hadd_ps(jz23t2, jz23t2);
    v8sf jz23   = _mm256_hadd_ps(jz23t1, jz23t1);

    double buf_ax[4][4] __attribute__ ((aligned(32)));
    double buf_ay[4][4] __attribute__ ((aligned(32)));
    double buf_az[4][4] __attribute__ ((aligned(32)));
    double buf_pt[4][4] __attribute__ ((aligned(32)));
    float  buf_jx[2][8] __attribute__ ((aligned(32)));
    float  buf_jy[2][8] __attribute__ ((aligned(32)));
    float  buf_jz[2][8] __attribute__ ((aligned(32)));

    ax0.store(buf_ax[0]);
    ax1.store(buf_ax[1]);
    ax2.store(buf_ax[2]);
    ax3.store(buf_ax[3]);
    ay0.store(buf_ay[0]);
    ay1.store(buf_ay[1]);
    ay2.store(buf_ay[2]);
    ay3.store(buf_ay[3]);
    az0.store(buf_az[0]);
    az1.store(buf_az[1]);
    az2.store(buf_az[2]);
    az3.store(buf_az[3]);
    pt0.store(buf_pt[0]);
    pt1.store(buf_pt[1]);
    pt2.store(buf_pt[2]);
    pt3.store(buf_pt[3]);
    jx01.store(buf_jx[0]);
    jx23.store(buf_jx[1]);
    jy01.store(buf_jy[0]);
    jy23.store(buf_jy[1]);
    jz01.store(buf_jz[0]);
    jz23.store(buf_jz[1]);

    accjerk[0].xacc = buf_ax[0][0] + buf_ax[0][1] + buf_ax[0][2] + buf_ax[0][3];
    accjerk[0].yacc = buf_ay[0][0] + buf_ay[0][1] + buf_ay[0][2] + buf_ay[0][3];
    accjerk[0].zacc = buf_az[0][0] + buf_az[0][1] + buf_az[0][2] + buf_az[0][3];
    accjerk[0].pot  = buf_pt[0][0] + buf_pt[0][1] + buf_pt[0][2] + buf_pt[0][3];
    accjerk[0].xjrk = buf_jx[0][0];
    accjerk[0].yjrk = buf_jy[0][0];
    accjerk[0].zjrk = buf_jz[0][0];
    accjerk[1].xacc = buf_ax[1][0] + buf_ax[1][1] + buf_ax[1][2] + buf_ax[1][3];
    accjerk[1].yacc = buf_ay[1][0] + buf_ay[1][1] + buf_ay[1][2] + buf_ay[1][3];
    accjerk[1].zacc = buf_az[1][0] + buf_az[1][1] + buf_az[1][2] + buf_az[1][3];
    accjerk[1].pot  = buf_pt[1][0] + buf_pt[1][1] + buf_pt[1][2] + buf_pt[1][3];
    accjerk[1].xjrk = buf_jx[0][4];
    accjerk[1].yjrk = buf_jy[0][4];
    accjerk[1].zjrk = buf_jz[0][4];
    accjerk[2].xacc = buf_ax[2][0] + buf_ax[2][1] + buf_ax[2][2] + buf_ax[2][3];
    accjerk[2].yacc = buf_ay[2][0] + buf_ay[2][1] + buf_ay[2][2] + buf_ay[2][3];
    accjerk[2].zacc = buf_az[2][0] + buf_az[2][1] + buf_az[2][2] + buf_az[2][3];
    accjerk[2].pot  = buf_pt[2][0] + buf_pt[2][1] + buf_pt[2][2] + buf_pt[2][3];
    accjerk[2].xjrk = buf_jx[1][0];
    accjerk[2].yjrk = buf_jy[1][0];
    accjerk[2].zjrk = buf_jz[1][0];
    accjerk[3].xacc = buf_ax[3][0] + buf_ax[3][1] + buf_ax[3][2] + buf_ax[3][3];
    accjerk[3].yacc = buf_ay[3][0] + buf_ay[3][1] + buf_ay[3][2] + buf_ay[3][3];
    accjerk[3].zacc = buf_az[3][0] + buf_az[3][1] + buf_az[3][2] + buf_az[3][3];
    accjerk[3].pot  = buf_pt[3][0] + buf_pt[3][1] + buf_pt[3][2] + buf_pt[3][3];
    accjerk[3].xjrk = buf_jx[1][4];
    accjerk[3].yjrk = buf_jy[1][4];
    accjerk[3].zjrk = buf_jz[1][4];
#endif

    return;
}

void gravity_kerneln(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk, int i, int ithread)
{
    return;
}

void gravity_kernel2n(int nj, PG6::pPrdPosVel posvel, PG6::pNewAccJrk accjerk, int i, int ithread)
{
    return;
}

#endif
