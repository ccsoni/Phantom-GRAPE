#pragma once

#include <immintrin.h>

struct v4sf;
struct v8sf;
struct v2df;
struct v4df;

struct v4sf {
    typedef float _v4sf __attribute__((vector_size(16))) __attribute__((aligned(16)));
    _v4sf val;

    static inline int getVectorLength() {
        return 4;
    }

    v4sf(const v4sf & rhs) : val(rhs.val) {}
    v4sf operator = (const v4sf rhs) {
        val = rhs.val;
        return (*this);
    }
    v4sf() : val(_mm_setzero_ps()) {}
    v4sf(const float x) : val(_mm_set1_ps(x)) {}
    v4sf(const float x, const float y, const float z, const float w)
        : val(_mm_set_ps(w, z, y, x)) {}
    v4sf(const _v4sf _val) : val(_val) {}
    operator _v4sf() {return val;}

    v4sf operator + (const v4sf rhs) const {
        return v4sf(val + rhs.val);
    }
    v4sf operator - (const v4sf rhs) const {
        return v4sf(val - rhs.val);
    }
    v4sf operator * (const v4sf rhs) const {
        return v4sf(val * rhs.val);
    }
    v4sf operator / (const v4sf rhs) const {
        return v4sf(_mm_div_ps(val, rhs.val));
    }

    v4sf operator += (const v4sf rhs) {
        val = val + rhs.val;
        return (*this);
    }
    v4sf operator -= (const v4sf rhs) {
        val = val - rhs.val;
        return (*this);
    }
    v4sf operator *= (const v4sf rhs) {
        val = val * rhs.val;
        return (*this);
    }
    v4sf operator /= (const v4sf rhs) {
        val = _mm_div_ps(val, rhs.val);
        return (*this);
    }

    static inline v4sf rcp_0th(const v4sf rhs) {
        return _mm_rcp_ps(rhs.val);
    }
    static inline v4sf rcp_1st(const v4sf rhs) {
        v4sf x0 = _mm_rcp_ps(rhs.val);
        v4sf h  = v4sf(1.) - rhs * x0;
        return x0 + h * x0;
    }

    static inline v4sf sqrt(const v4sf rhs) {
        return v4sf(_mm_sqrt_ps(rhs.val));
    }
    static inline v4sf rsqrt_0th(const v4sf rhs) {
        return v4sf(_mm_rsqrt_ps(rhs.val));
    }
    static inline v4sf rsqrt_1st(const v4sf rhs) {
        v4sf x0 = v4sf(_mm_rsqrt_ps(rhs.val));
        v4sf h  = v4sf(1.) - rhs * x0 * x0;
        return x0 + v4sf(0.5) * h * x0;
    }
    static inline v4sf rsqrt_1st_phantom(const v4sf rhs) {
        v4sf x0 = v4sf(_mm_rsqrt_ps(rhs.val));
        return v4sf(x0 * (rhs * x0 * x0 - v4sf(3.)));
    }

    inline void store(float *p) const {
        _mm_store_ps(p, val);
    }
    inline void load(float const *p) {
        val = _mm_load_ps(p);
    }

    void print(FILE * fp = stdout,
               const char * fmt = "%+e %+e %+e %+e\n") const {
	float a[4] __attribute__((aligned(16)));
        store(a);
        fprintf(fp, fmt, a[0], a[1], a[2], a[3]);
    }    
};

struct v8sf {
    typedef float _v8sf __attribute__((vector_size(32))) __attribute__((aligned(32)));
    _v8sf val;

    static inline int getVectorLength() {
        return 8;
    }

    v8sf(const v8sf & rhs) : val(rhs.val) {}
    v8sf operator = (const v8sf rhs) {
        val = rhs.val;
        return (*this);
    }
    v8sf() : val(_mm256_setzero_ps()) {}
    v8sf(const float x) : val(_mm256_set1_ps(x)) {}
    v8sf(const float x0, const float y0, const float z0, const float w0,
         const float x1, const float y1, const float z1, const float w1)
        : val(_mm256_set_ps(w1, z1, y1, x1, w0, z0, y0, x0)) {}
    v8sf(const _v8sf _val) : val(_val) {}
    operator _v8sf() {return val;}
    v8sf(const v4sf rhs) {
	val = _mm256_broadcast_ps((const __m128*)&rhs);
    }
    v8sf(v4sf rhs0, v4sf rhs1) {
	/*
	v8sf temp(rhs0);
	val = _mm256_insertf128_ps(temp, rhs1, 0);
	*/
	v8sf temp(rhs1);
	val = _mm256_insertf128_ps(temp, rhs0, 0);
    }

    v8sf operator + (const v8sf rhs) const {
        return val + rhs.val;
    }
    v8sf operator - (const v8sf rhs) const {
        return val - rhs.val;
    }
    v8sf operator * (const v8sf rhs) const {
        return val * rhs.val;
    }
    v8sf operator / (const v8sf rhs) const {
        return v8sf(_mm256_div_ps(val, rhs.val));
    }

    static inline v8sf max(const v8sf a, const v8sf b) {
        return v8sf(_mm256_max_ps(a.val, b.val));
    }
    static inline v8sf min(const v8sf a, const v8sf b) {
        return v8sf(_mm256_min_ps(a.val, b.val));
    }

    v8sf operator += (const v8sf rhs) {
        val = val + rhs.val;
        return (*this);
    }
    v8sf operator -= (const v8sf rhs) {
        val = val - rhs.val;
        return (*this);
    }
    v8sf operator *= (const v8sf rhs) {
        val = val * rhs.val;
        return (*this);
    }
    v8sf operator /= (const v8sf rhs) {
        val = _mm256_div_ps(val, rhs.val);
        return (*this);
    }

    v8sf operator & (const v8sf rhs) {
        return v8sf(_mm256_and_ps(val, rhs.val));
    }
    v8sf operator != (const v8sf rhs) {
        return v8sf(_mm256_cmp_ps(val, rhs.val, _CMP_NEQ_UQ));
    }
    v8sf operator < (const v8sf rhs) {
        return v8sf(_mm256_cmp_ps(val, rhs.val, _CMP_LT_OS));
    }

    static inline v8sf rcp_0th(const v8sf rhs) {
        return _mm256_rcp_ps(rhs.val);
    }
    static inline v8sf rcp_1st(const v8sf rhs) {
        v8sf x0 = _mm256_rcp_ps(rhs.val);
        v8sf h  = v8sf(1.) - rhs * x0;
        return x0 + h * x0;
    }

    static inline v8sf sqrt(const v8sf rhs) {
        return v8sf(_mm256_sqrt_ps(rhs.val));
    }
    static inline v8sf rsqrt_0th(const v8sf rhs) {
        return v8sf(_mm256_rsqrt_ps(rhs.val));
    }
    static inline v8sf rsqrt_1st(const v8sf rhs) {
        v8sf x0 = v8sf(_mm256_rsqrt_ps(rhs.val));
        v8sf h  = v8sf(1.) - rhs * x0 * x0;
        return x0 + v8sf(0.5) * h * x0;
    }
    static inline v8sf rsqrt_1st_phantom(const v8sf rhs) {
        v8sf x0 = v8sf(_mm256_rsqrt_ps(rhs.val));
        return v8sf(x0 * (rhs * x0 * x0 - v8sf(3.)));
    }
    static inline v8sf hadd(v8sf x0, v8sf x1) {
        return _mm256_hadd_ps(x0, x1);
    }

    inline void store(float *p) const {
        _mm256_store_ps(p, val);
    }
    inline void load(float const *p) {
        val = _mm256_load_ps(p);
    }
    inline void extractf128(v4sf & x0, v4sf & x1) {
        x0 = _mm256_extractf128_ps(val, 0);
        x1 = _mm256_extractf128_ps(val, 1);
    }

    void print(FILE * fp = stdout,
               const char * fmt = "%+e %+e %+e %+e\n%+e %+e %+e %+e\n") const {
	float a[8] __attribute__((aligned(32)));
        store(a);
        fprintf(fp, fmt, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    }

    static inline v8sf shuffle0(v8sf rhs) {
        return _mm256_permute_ps(rhs, 0x00);
    }

    static inline v8sf shuffle1(v8sf rhs) {
        return _mm256_permute_ps(rhs, 0x55);
    }

    static inline v8sf shuffle2(v8sf rhs) {
        return _mm256_permute_ps(rhs, 0xaa);
    }

    static inline v8sf shuffle3(v8sf rhs) {
        return _mm256_permute_ps(rhs, 0xff);
    }

    static inline v4sf reduce(v8sf rhs) {
	v4sf x0 = _mm256_extractf128_ps(rhs, 0);
	v4sf x1 = _mm256_extractf128_ps(rhs, 1);
	return v4sf(x0 + x1);
    }

};

struct v2df {
    typedef double _v2df __attribute__((vector_size(16))) __attribute__((aligned(16)));
    _v2df val;

    static inline int getVectorLength() {
        return 2;
    }

    v2df(const v2df & rhs) : val(rhs.val) {}
    v2df operator = (const v2df rhs) {
        val = rhs.val;
        return (*this);
    }
    v2df() : val(_mm_setzero_pd()) {}
    v2df(const double x) : val(_mm_set1_pd(x)) {}
    v2df(const double x0, const double y0)
        : val(_mm_set_pd(y0, x0)) {}
    v2df(const _v2df _val) : val(_val) {}
    operator _v2df() {return val;}

    v2df operator + (const v2df rhs) const {
        return v2df(val + rhs.val);
    }
    v2df operator - (const v2df rhs) const {
        return v2df(val - rhs.val);
    }
    v2df operator * (const v2df rhs) const {
        return v2df(val * rhs.val);
    }
    v2df operator / (const v2df rhs) const {
        return v2df(_mm_div_pd(val, rhs.val));
    }

    inline void store(double *p) const {
        _mm_store_pd(p, val);
    }
    inline void load(double const *p) {
        val = _mm_load_pd(p);
    }

    void print(FILE * fp = stdout,
               const char * fmt = "%+e %+e\n") const {
	double a[2] __attribute__((aligned(16)));
        store(a);
        fprintf(fp, fmt, a[0], a[1]);
    }    

};


struct v4df {
    typedef double _v4df __attribute__((vector_size(32))) __attribute__((aligned(32)));
    _v4df val;

    static inline int getVectorLength() {
        return 4;
    }

    v4df(const v4df & rhs) : val(rhs.val) {}
    v4df operator = (const v4df rhs) {
        val = rhs.val;
        return (*this);
    }
    v4df() : val(_mm256_setzero_pd()) {}
    v4df(const double x) : val(_mm256_set1_pd(x)) {}
    v4df(const double x0, const double y0, const double z0, const double w0)
        : val(_mm256_set_pd(w0, z0, y0, x0)) {}
    v4df(const _v4df _val) : val(_val) {}
    operator _v4df() {return val;}

    v4df operator + (const v4df rhs) const {
        return v4df(val + rhs.val);
    }
    v4df operator - (const v4df rhs) const {
        return v4df(val - rhs.val);
    }
    v4df operator * (const v4df rhs) const {
        return v4df(val * rhs.val);
    }
    v4df operator / (const v4df rhs) const {
        return v4df(_mm256_div_pd(val, rhs.val));
    }
    v4df operator += (const v4df rhs) {
        val = val + rhs.val;
        return (*this);
    }
    v4df operator -= (const v4df rhs) {
        val = val - rhs.val;
        return (*this);
    }
    v4df operator *= (const v4df rhs) {
        val = val * rhs.val;
        return (*this);
    }
    v4df operator /= (const v4df rhs) {
        val = v4df(_mm256_div_pd(val, rhs.val));
        return (*this);
    }

    inline void store(double *p) const {
        _mm256_store_pd(p, val);
    }
    inline void load(double const *p) {
        val = _mm256_load_pd(p);
    }

    void print(FILE * fp = stdout,
               const char * fmt = "%+e %+e %+e %+e\n") const {
	double a[4] __attribute__((aligned(32)));
        store(a);
        fprintf(fp, fmt, a[0], a[1], a[2], a[3]);
    }    
};

