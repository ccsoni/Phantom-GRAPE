#include <stdio.h>
#include <assert.h>
#include "avx.h"
#include "avx_type.h"

namespace PhantomGrape5 {

#ifndef __AVX512F__

    void GravityKernel(PG5::pIpdata ipdata,
		       PG5::pFodata fodata,
		       PG5::pJpdata jpdata,
		       int nj)
    {
	v8sf xi = v8sf(ipdata->x);
	v8sf yi = v8sf(ipdata->y);
	v8sf zi = v8sf(ipdata->z);
	v8sf e2 = v8sf(ipdata->eps2);
	v8sf ax = 0.;
	v8sf ay = 0.;
	v8sf az = 0.;
	v8sf pt = 0.;
	
	v8sf jp;
	jp.load((float *)(&(jpdata->posm)));
	for(int j = 0; j < nj; j += PG5::NumberOfJParallel) {
	    jpdata += PG5::NumberOfJParallel;
	    
	    v8sf xj = v8sf::shuffle0(jp);
	    v8sf yj = v8sf::shuffle1(jp);
	    v8sf zj = v8sf::shuffle2(jp);
	    v8sf mj = v8sf::shuffle3(jp);
	    
	    v8sf dx = xj - xi;
	    v8sf dy = yj - yi;
	    v8sf dz = zj - zi;
	    
	    jp.load((float *)(&(jpdata->posm)));

#ifndef __FMA__	    
	    v8sf r2 = e2 + dx * dx + dy * dy + dz * dz;
#else
	    v8sf r2 = e2;
	    r2 = _mm256_fmadd_ps(dx, dx, r2);
	    r2 = _mm256_fmadd_ps(dy, dy, r2);
	    r2 = _mm256_fmadd_ps(dz, dz, r2);
#endif
	    v8sf ri = v8sf::rsqrt_0th(r2);
	    v8sf mr = mj * ri;
	    
	    ri *= ri;
	    pt -= mr;
	    mr *= ri;
#ifndef __FMA__	    
	    ax += mr * dx;
	    ay += mr * dy;
	    az += mr * dz;
#else
	    ax = _mm256_fmadd_ps(mr, dx, ax);
	    ay = _mm256_fmadd_ps(mr, dy, ay);
	    az = _mm256_fmadd_ps(mr, dz, az);
#endif
	}
	
	fodata->ax  = v8sf::reduce(ax);
	fodata->ay  = v8sf::reduce(ay);
	fodata->az  = v8sf::reduce(az);
	fodata->phi = v8sf::reduce(pt);

    }
    
    void GravityKernel0(PG5::pIpdata ipdata,
			PG5::pFodata fodata,
			PG5::pJpdata0 jpdata,
			int nj)
    {
	v8sf xi = v8sf(ipdata->x);
	v8sf yi = v8sf(ipdata->y);
	v8sf zi = v8sf(ipdata->z);
	v8sf ei = v8sf(ipdata->eps2);
	v8sf ax = 0.;
	v8sf ay = 0.;
	v8sf az = 0.;
	v8sf pt = 0.;
	
	v8sf jp, ej, e2;
	jp.load((float *)(&(jpdata->posm)));
	ej.load((float *)(&(jpdata->eps2)));
	e2 = ei + ej;
	for(int j = 0; j < nj; j += PG5::NumberOfJParallel) {
	    jpdata += 1;
	    
	    v8sf xj = v8sf::shuffle0(jp);
	    v8sf yj = v8sf::shuffle1(jp);
	    v8sf zj = v8sf::shuffle2(jp);
	    v8sf mj = v8sf::shuffle3(jp);
	    
	    v8sf dx = xj - xi;
	    v8sf dy = yj - yi;
	    v8sf dz = zj - zi;
	    
	    e2 = ei + ej;
	    jp.load((float *)(&(jpdata->posm)));
	    ej.load((float *)(&(jpdata->eps2)));
	    
#ifndef __FMA__	    
	    v8sf r2 = e2 + dx * dx + dy * dy + dz * dz;
#else
	    v8sf r2 = e2;
	    r2 = _mm256_fmadd_ps(dx, dx, r2);
	    r2 = _mm256_fmadd_ps(dy, dy, r2);
	    r2 = _mm256_fmadd_ps(dz, dz, r2);
#endif
	    v8sf ri = v8sf::rsqrt_0th(r2);
	    v8sf mr = mj * ri;
	    
	    ri *= ri;
	    pt -= mr;
	    mr *= ri;
#ifndef __FMA__	    
	    ax += mr * dx;
	    ay += mr * dy;
	    az += mr * dz;
#else
	    ax = _mm256_fmadd_ps(mr, dx, ax);
	    ay = _mm256_fmadd_ps(mr, dy, ay);
	    az = _mm256_fmadd_ps(mr, dz, az);
#endif
	}
	
	fodata->ax  = v8sf::reduce(ax);
	fodata->ay  = v8sf::reduce(ay);
	fodata->az  = v8sf::reduce(az);
	fodata->phi = v8sf::reduce(pt);
    }

#else // __AVX512F__

#ifdef PG5_I8J2
    void GravityKernel(PG5::pIpdata ipdata,
		       PG5::pFodata fodata,
		       PG5::pJpdata jpdata,
		       int nj)
    {
	v16sf xi = v16sf(ipdata->x);
	v16sf yi = v16sf(ipdata->y);
	v16sf zi = v16sf(ipdata->z);
	v16sf e2 = v16sf(ipdata->eps2);
	v16sf ax0 = 0.;
	v16sf ay0 = 0.;
	v16sf az0 = 0.;
	v16sf pt0 = 0.;
	v16sf ax1 = 0.;
	v16sf ay1 = 0.;
	v16sf az1 = 0.;
	v16sf pt1 = 0.;

	xi = _mm512_shuffle_f32x4(xi, xi, 0b01010000);
	yi = _mm512_shuffle_f32x4(yi, yi, 0b01010000);
	zi = _mm512_shuffle_f32x4(zi, zi, 0b01010000);
	e2 = _mm512_shuffle_f32x4(e2, e2, 0b01010000);

//#define DEBUG
#if DEBUG
        v16sf jt;
	jt.load((float *)(&(jpdata->posm)));
	v16sf jp0 = _mm512_shuffle_f32x4(jt, jt, 0b01000100);;
	v16sf jp1 = _mm512_shuffle_f32x4(jt, jt, 0b11101110);;
#else
	v8sf jt0, jt1;;
	jt0.load((float *)(&((jpdata                       )->posm)));
	jt1.load((float *)(&((jpdata+PG5::NumberOfJParallel)->posm)));
	v16sf jp0 = v16sf(jt0);
	v16sf jp1 = v16sf(jt1);
#endif

	jpdata += (PG5::NumberOfJParallel * PG5::NumberOfHandUnroll);

	for(int j = 0; j < nj; j += PG5::NumberOfJParallel * PG5::NumberOfHandUnroll) {
	    v16sf xj0 = v16sf::shuffle0(jp0);
	    v16sf xj1 = v16sf::shuffle0(jp1);
	    v16sf yj0 = v16sf::shuffle1(jp0);
	    v16sf yj1 = v16sf::shuffle1(jp1);
	    v16sf zj0 = v16sf::shuffle2(jp0);
	    v16sf zj1 = v16sf::shuffle2(jp1);
	    v16sf mj0 = v16sf::shuffle3(jp0);
	    v16sf mj1 = v16sf::shuffle3(jp1);

	    v16sf dx0 = xj0 - xi;
	    v16sf dy0 = yj0 - yi;
	    v16sf dz0 = zj0 - zi;
	    v16sf dx1 = xj1 - xi;
	    v16sf dy1 = yj1 - yi;
	    v16sf dz1 = zj1 - zi;

#ifdef DEBUG
	    jt.load((float *)(&(jpdata->posm)));
	    jp0 = _mm512_shuffle_f32x4(jt, jt, 0b01000100);;
	    jp1 = _mm512_shuffle_f32x4(jt, jt, 0b11101110);;
#else
	    jt0.load((float *)(&((jpdata                       )->posm)));
	    jt1.load((float *)(&((jpdata+PG5::NumberOfJParallel)->posm)));
	    jp0 = v16sf(jt0);
	    jp1 = v16sf(jt1);
#endif
	    jpdata += (PG5::NumberOfJParallel * PG5::NumberOfHandUnroll);

	    v16sf r20 = e2;
	    v16sf r21 = e2;
	    r20 = _mm512_fmadd_ps(dx0, dx0, r20);
	    r21 = _mm512_fmadd_ps(dx1, dx1, r21);
	    r20 = _mm512_fmadd_ps(dy0, dy0, r20);
	    r21 = _mm512_fmadd_ps(dy1, dy1, r21);
	    r20 = _mm512_fmadd_ps(dz0, dz0, r20);
	    r21 = _mm512_fmadd_ps(dz1, dz1, r21);

	    v16sf ri0 = v16sf::rsqrt_0th(r20);
	    v16sf ri1 = v16sf::rsqrt_0th(r21);

	    v16sf mr0 = mj0 * ri0;
	    v16sf mr1 = mj1 * ri1;
	    ri0 *= ri0;
	    ri1 *= ri1;
	    pt0 -= mr0;
	    pt1 -= mr1;
	    mr0 *= ri0;
	    mr1 *= ri1;

	    ax0 = _mm512_fmadd_ps(mr0, dx0, ax0);
	    ax1 = _mm512_fmadd_ps(mr1, dx1, ax1);
	    ay0 = _mm512_fmadd_ps(mr0, dy0, ay0);
	    ay1 = _mm512_fmadd_ps(mr1, dy1, ay1);
	    az0 = _mm512_fmadd_ps(mr0, dz0, az0);
	    az1 = _mm512_fmadd_ps(mr1, dz1, az1);
	}

	fodata->ax  = v16sf::reduce16to08(ax0 + ax1);
	fodata->ay  = v16sf::reduce16to08(ay0 + ay1);
	fodata->az  = v16sf::reduce16to08(az0 + az1);
	fodata->phi = v16sf::reduce16to08(pt0 + pt1);
    }
#else // ! PG5_I8J2

    void GravityKernel(PG5::pIpdata ipdata,
		       PG5::pFodata fodata,
		       PG5::pJpdata jpdata,
		       int nj)
    {
        v16sf xi = v16sf(ipdata->x);
	v16sf yi = v16sf(ipdata->y);
	v16sf zi = v16sf(ipdata->z);
	v16sf e2 = v16sf(ipdata->eps2);
	v16sf ax0 = 0.;
	v16sf ay0 = 0.;
	v16sf az0 = 0.;
	v16sf pt0 = 0.;
	v16sf ax1 = 0.;
	v16sf ay1 = 0.;
	v16sf az1 = 0.;
	v16sf pt1 = 0.;

	v16sf jp0, jp1;
	jp0.load((float *)(&((jpdata                       )->posm)));
	jp1.load((float *)(&((jpdata+PG5::NumberOfJParallel)->posm)));
	jpdata += (PG5::NumberOfJParallel * PG5::NumberOfHandUnroll);

	for(int j = 0; j < nj; j += PG5::NumberOfJParallel * PG5::NumberOfHandUnroll) {
	    v16sf xj0 = v16sf::shuffle0(jp0);
	    v16sf xj1 = v16sf::shuffle0(jp1);
	    v16sf yj0 = v16sf::shuffle1(jp0);
	    v16sf yj1 = v16sf::shuffle1(jp1);
	    v16sf zj0 = v16sf::shuffle2(jp0);
	    v16sf zj1 = v16sf::shuffle2(jp1);
	    v16sf mj0 = v16sf::shuffle3(jp0);
	    v16sf mj1 = v16sf::shuffle3(jp1);

	    v16sf dx0 = xj0 - xi;
	    v16sf dy0 = yj0 - yi;
	    v16sf dz0 = zj0 - zi;
	    v16sf dx1 = xj1 - xi;
	    v16sf dy1 = yj1 - yi;
	    v16sf dz1 = zj1 - zi;
	    
	    jp0.load((float *)(&((jpdata                       )->posm)));
	    jp1.load((float *)(&((jpdata+PG5::NumberOfJParallel)->posm)));
	    jpdata += (PG5::NumberOfJParallel * PG5::NumberOfHandUnroll);

	    v16sf r20 = e2;
	    v16sf r21 = e2;
	    r20 = _mm512_fmadd_ps(dx0, dx0, r20);
	    r21 = _mm512_fmadd_ps(dx1, dx1, r21);
	    r20 = _mm512_fmadd_ps(dy0, dy0, r20);
	    r21 = _mm512_fmadd_ps(dy1, dy1, r21);
	    r20 = _mm512_fmadd_ps(dz0, dz0, r20);
	    r21 = _mm512_fmadd_ps(dz1, dz1, r21);

	    v16sf ri0 = v16sf::rsqrt_0th(r20);
	    v16sf ri1 = v16sf::rsqrt_0th(r21);

	    v16sf mr0 = mj0 * ri0;
	    v16sf mr1 = mj1 * ri1;
	    ri0 *= ri0;
	    ri1 *= ri1;
	    pt0 -= mr0;
	    pt1 -= mr1;
	    mr0 *= ri0;
	    mr1 *= ri1;

	    ax0 = _mm512_fmadd_ps(mr0, dx0, ax0);
	    ax1 = _mm512_fmadd_ps(mr1, dx1, ax1);
	    ay0 = _mm512_fmadd_ps(mr0, dy0, ay0);
	    ay1 = _mm512_fmadd_ps(mr1, dy1, ay1);
	    az0 = _mm512_fmadd_ps(mr0, dz0, az0);
	    az1 = _mm512_fmadd_ps(mr1, dz1, az1);
	}

	fodata->ax  = v16sf::reduce((ax0 + ax1));
	fodata->ay  = v16sf::reduce((ay0 + ay1));
	fodata->az  = v16sf::reduce((az0 + az1));
	fodata->phi = v16sf::reduce((pt0 + pt1));
    }

#endif // PG5_I8J2

#ifdef SYMMETRIC

#ifdef PG5_I8J2
#error i8j2 is not supported for the symmetric method.
#else
    void GravityKernel0(PG5::pIpdata ipdata,
			PG5::pFodata fodata,
			PG5::pJpdata0 jpdata,
			int nj)
    {
	v16sf xi = v16sf(ipdata->x);
	v16sf yi = v16sf(ipdata->y);
	v16sf zi = v16sf(ipdata->z);
	v16sf ei = v16sf(ipdata->eps2);
	v16sf ax0 = 0.;
	v16sf ay0 = 0.;
	v16sf az0 = 0.;
	v16sf pt0 = 0.;
	v16sf ax1 = 0.;
	v16sf ay1 = 0.;
	v16sf az1 = 0.;
	v16sf pt1 = 0.;
	
	v16sf jp0, ej0, e20;
	v16sf jp1, ej1, e21;
	jp0.load((float *)(&((jpdata  )->posm)));
	jp1.load((float *)(&((jpdata+1)->posm)));
	ej0.load((float *)(&((jpdata  )->eps2)));
	ej1.load((float *)(&((jpdata+1)->eps2)));
	e20 = ei + ej0;
	e21 = ei + ej1;
	for(int j = 0; j < nj; j += PG5::NumberOfJParallel * PG5::NumberOfHandUnroll) {
	    jpdata += PG5::NumberOfHandUnroll;
	    
	    v16sf xj0 = v16sf::shuffle0(jp0);
	    v16sf xj1 = v16sf::shuffle0(jp1);
	    v16sf yj0 = v16sf::shuffle1(jp0);
	    v16sf yj1 = v16sf::shuffle1(jp1);
	    v16sf zj0 = v16sf::shuffle2(jp0);
	    v16sf zj1 = v16sf::shuffle2(jp1);
	    v16sf mj0 = v16sf::shuffle3(jp0);
	    v16sf mj1 = v16sf::shuffle3(jp1);
	    
	    v16sf dx0 = xj0 - xi;
	    v16sf dy0 = yj0 - yi;
	    v16sf dz0 = zj0 - zi;
	    v16sf dx1 = xj1 - xi;
	    v16sf dy1 = yj1 - yi;
	    v16sf dz1 = zj1 - zi;

	    e20 = ei + ej0;
	    e21 = ei + ej1;
	    jp0.load((float *)(&((jpdata  )->posm)));
	    jp1.load((float *)(&((jpdata+1)->posm)));
	    ej0.load((float *)(&((jpdata  )->eps2)));
	    ej1.load((float *)(&((jpdata+1)->eps2)));
	    
	    v16sf r20 = e20;
	    v16sf r21 = e21;
	    r20 = _mm512_fmadd_ps(dx0, dx0, r20);
	    r21 = _mm512_fmadd_ps(dx1, dx1, r21);
	    r20 = _mm512_fmadd_ps(dy0, dy0, r20);
	    r21 = _mm512_fmadd_ps(dy1, dy1, r21);
	    r20 = _mm512_fmadd_ps(dz0, dz0, r20);
	    r21 = _mm512_fmadd_ps(dz1, dz1, r21);

	    v16sf ri0 = v16sf::rsqrt_0th(r20);
	    v16sf ri1 = v16sf::rsqrt_0th(r21);
	    v16sf mr0 = mj0 * ri0;
	    v16sf mr1 = mj1 * ri1;
	    
	    ri0 *= ri0;
	    ri1 *= ri1;
	    pt0 -= mr0;
	    pt1 -= mr1;
	    mr0 *= ri0;
	    mr1 *= ri1;
	    ax0 = _mm512_fmadd_ps(mr0, dx0, ax0);
	    ax1 = _mm512_fmadd_ps(mr1, dx1, ax1);
	    ay0 = _mm512_fmadd_ps(mr0, dy0, ay0);
	    ay1 = _mm512_fmadd_ps(mr1, dy1, ay1);
	    az0 = _mm512_fmadd_ps(mr0, dz0, az0);
	    az1 = _mm512_fmadd_ps(mr1, dz1, az1);
	}
	
	fodata->ax  = v16sf::reduce(ax0 + ax1);
	fodata->ay  = v16sf::reduce(ay0 + ay1);
	fodata->az  = v16sf::reduce(az0 + az1);
	fodata->phi = v16sf::reduce(pt0 + pt1);
    }
#endif // PG5_I8J2
#endif // SYMMETRIC

#endif // AVX512F
    
};
