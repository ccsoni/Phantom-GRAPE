#ifndef __AVX_TYPE__
#define __AVX_TYPE__

namespace PhantomGrape5 {

#ifndef __AVX512F__
    const int NumberOfIParallel  = 4;
    const int NumberOfJParallel  = 2;
    const int NumberOfHandUnroll = 1;
#else
#ifdef PG5_I8J2
    const int NumberOfIParallel  = 8;
    const int NumberOfJParallel  = 2;
    const int NumberOfHandUnroll = 2;
#else
    const int NumberOfIParallel  = 4;
    const int NumberOfJParallel  = 4;
    const int NumberOfHandUnroll = 2;
#endif
#endif

#ifdef PG5_I8J2
    typedef struct ipdata{
	v8sf x;
	v8sf y;
	v8sf z;
	v8sf eps2;
    } Ipdata, *pIpdata;
#else
    typedef struct ipdata{
	v4sf x;
	v4sf y;
	v4sf z;
	v4sf eps2;
    } Ipdata, *pIpdata;
#endif
    
    typedef struct jpdata{
	v4sf posm;
    } Jpdata, *pJpdata;
    
    typedef struct jpdata0{
	v4sf posm[NumberOfJParallel];
	v4sf eps2[NumberOfJParallel];
    } Jpdata0, *pJpdata0;
    
#ifdef PG5_I8J2
    typedef struct fodata{
	v8sf ax;
	v8sf ay;
	v8sf az;
	v8sf phi;
    } Fodata, *pFodata;
#else
    typedef struct fodata{
	v4sf ax;
	v4sf ay;
	v4sf az;
	v4sf phi;
    } Fodata, *pFodata;
#endif

    void GravityKernel(pIpdata ipdata,
		       pFodata fodata,
		       pJpdata jpdata,
		       int nj);

    void GravityKernel0(pIpdata ipdata,
			pFodata fodata,
			pJpdata0 jpdata,
			int nj);

};

namespace PG5 = PhantomGrape5;

#endif /* __AVX_TYPE__ */
