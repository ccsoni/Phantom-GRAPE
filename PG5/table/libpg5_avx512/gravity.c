#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "avx.h"
#include "avx_type.h"
#include "pg5_table.h"

#ifndef __AVX512F__

#ifndef __AVX2__
#error "Only supported for AVX2 and upwards"
#endif

void PG5::GravityKernel(PG5::pIpdata ipdata,
			PG5::pJpdata jpdata,
			PG5::pFodata fodata, 
			int nj,
			float fcut[][2],
			v4sf _r2cut,
			v4sf accscale)
{
    fcut -= (1<<(30-(23-FRC_BIT)));

    v8sf xi    = v8sf(ipdata->x);
    v8sf yi    = v8sf(ipdata->y);
    v8sf zi    = v8sf(ipdata->z);
    v8sf r2cut = v8sf(_r2cut);
    v8sf ax    = 0.;
    v8sf ay    = 0.;
    v8sf az    = 0.;
    
    v8sf jp;
    jp.load((float *)(&(jpdata->posm)));
    v8sf xj = v8sf::shuffle0(jp);
    v8sf yj = v8sf::shuffle1(jp);
    v8sf zj = v8sf::shuffle2(jp);
    v8sf mj = v8sf::shuffle3(jp);

    for(int j = 0; j < nj ; j += PG5::NumberOfJParallel) {
	jpdata += PG5::NumberOfJParallel;
	jp.load((float *)(&(jpdata->posm)));

	v8sf dx = xj - xi;
	v8sf dy = yj - yi;
	v8sf dz = zj - zi;

	v8sf r2 = (((v8sf(2.) + dx * dx) + dy * dy) + dz * dz);
	r2 = v8sf::min(r2cut, r2);

	__m256i r2_sr = _mm256_srli_epi32((__m256i)((__m256)(r2)), (23-FRC_BIT));
	__m256i r2_sl = _mm256_slli_epi32(r2_sr, (23-FRC_BIT));
	unsigned int idx[PG5::NumberOfVector] __attribute__((aligned(PG5::SizeOfVector)));
	*(__m256i *)idx = r2_sr;

	const long long *ptr = (long long *)fcut;
	__m256i tbl_0145 = {ptr[idx[0]], ptr[idx[1]], ptr[idx[4]], ptr[idx[5]]};
	__m256i tbl_2367 = {ptr[idx[2]], ptr[idx[3]], ptr[idx[6]], ptr[idx[7]]};

	v8sf ff = _mm256_shuffle_ps((__m256)tbl_0145, (__m256)tbl_2367, 0x88);
	v8sf df = _mm256_shuffle_ps((__m256)tbl_0145, (__m256)tbl_2367, 0xdd);
	v8sf dr2 = r2 - (__m256)r2_sl;
	ff += dr2 * df;

	v8sf mf  = mj * ff;

	xj = v8sf::shuffle0(jp);
	yj = v8sf::shuffle1(jp);
	zj = v8sf::shuffle2(jp);
	mj = v8sf::shuffle3(jp);

	ax = _mm256_fmadd_ps(mf, dx, ax);
	ay = _mm256_fmadd_ps(mf, dy, ay);
	az = _mm256_fmadd_ps(mf, dz, az);
    }
    fodata->ax = accscale * v8sf::reduce(ax);
    fodata->ay = accscale * v8sf::reduce(ay);
    fodata->az = accscale * v8sf::reduce(az);
}

#else

#define REP8(x) {x, x, x, x, x, x, x, x}
#define REP4(x) {x, x, x, x}

void PG5::GravityKernel(PG5::pIpdata ipdata,
			PG5::pJpdata jpdata,
			PG5::pFodata fodata, 
			int nj,
			float fcut[][2],
			v4sf _r2cut,
			v4sf accscale) {

    fcut -= (1<<(30-(23-FRC_BIT)));

    v16sf xi    = v16sf(ipdata->x);
    v16sf yi    = v16sf(ipdata->y);
    v16sf zi    = v16sf(ipdata->z);
    v16sf r2cut = v16sf(_r2cut);
    v16sf ax    = 0.;
    v16sf ay    = 0.;
    v16sf az    = 0.;
    
    v16sf jp;
    jp.load((float *)(&(jpdata->posm)));
    v16sf xj = v16sf::shuffle0(jp);
    v16sf yj = v16sf::shuffle1(jp);
    v16sf zj = v16sf::shuffle2(jp);
    v16sf mj = v16sf::shuffle3(jp);

    for(int j = 0; j < nj; j += PG5::NumberOfJParallel) {
	jpdata += PG5::NumberOfJParallel;
	jp.load((float *)(&(jpdata->posm)));

        v16sf dx = xj - xi;
        v16sf dy = yj - yi;
        v16sf dz = zj - zi;

        v16sf r2 = (((v16sf(2.) + dx * dx) + dy * dy) + dz * dz);
        r2 = v16sf::min(r2cut, r2);

        __m512i r2_sr = _mm512_srli_epi32((__m512i)((__m512)(r2)), (23-FRC_BIT));
        __m512i r2_sl = _mm512_slli_epi32(r2_sr, (23-FRC_BIT));

#if 1
	__m512i zer8 = REP8(0);
	__m512i idx0 = (__m512i) _mm512_unpacklo_epi32(r2_sr, zer8);
	__m512i idx1 = (__m512i) _mm512_unpackhi_epi32(r2_sr, zer8);
	__m512i tbl0 = _mm512_i64gather_epi64(idx0, (long long *)fcut, 8);
	__m512i tbl1 = _mm512_i64gather_epi64(idx1, (long long *)fcut, 8);
#else
        unsigned int idx[PG5::NumberOfVector] __attribute__((aligned(PG5::SizeOfVector)));
        *(__m512i *)idx = r2_sr;
        const long long *ptr = (long long *)fcut;
        __m512i tbl0 = {ptr[idx[ 0]], ptr[idx[ 1]], ptr[idx[ 4]], ptr[idx[ 5]],
			ptr[idx[ 8]], ptr[idx[ 9]], ptr[idx[12]], ptr[idx[13]]};
        __m512i tbl1 = {ptr[idx[ 2]], ptr[idx[ 3]], ptr[idx[ 6]], ptr[idx[ 7]],
			ptr[idx[10]], ptr[idx[11]], ptr[idx[14]], ptr[idx[15]]};
#endif

        v16sf ff  = _mm512_shuffle_ps((__m512)tbl0, (__m512)tbl1, 0x88);
        v16sf df  = _mm512_shuffle_ps((__m512)tbl0, (__m512)tbl1, 0xdd);
        v16sf dr2 = r2 - (__m512)r2_sl;
        ff += dr2 * df;

        v16sf mf  = mj * ff;

        xj = v16sf::shuffle0(jp);
        yj = v16sf::shuffle1(jp);
        zj = v16sf::shuffle2(jp);
        mj = v16sf::shuffle3(jp);

        ax = _mm512_fmadd_ps(mf, dx, ax);
        ay = _mm512_fmadd_ps(mf, dy, ay);
        az = _mm512_fmadd_ps(mf, dz, az);
    }
    fodata->ax = accscale * v16sf::reduce(ax);
    fodata->ay = accscale * v16sf::reduce(ay);
    fodata->az = accscale * v16sf::reduce(az);
}

#endif
