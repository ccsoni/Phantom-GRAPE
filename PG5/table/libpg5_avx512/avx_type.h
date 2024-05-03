namespace PhantomGrape5 {

#ifndef __AVX512F__
    const int NumberOfIParallel = 4;
    const int NumberOfJParallel = 2;
    const int NumberOfVector    = 8;
    const int SizeOfVector      = 32;
#else
    const int NumberOfIParallel = 4;
    const int NumberOfJParallel = 4;
    const int NumberOfVector    = 16;
    const int SizeOfVector      = 64;
#endif

    typedef struct ipdata{
        v4sf x;
        v4sf y;
        v4sf z;
        v4sf eps2;
    } Ipdata, *pIpdata;

    typedef struct jpdata{
        v4sf posm;
    } Jpdata, *pJpdata;

    typedef struct fodata{
        v4sf ax;
        v4sf ay;
        v4sf az;
        v4sf phi;
    } Fodata, *pFodata;

    void GravityKernel(pIpdata ipdata,
		       pJpdata jpdata,
		       pFodata fodata, 
		       int nj,
		       float fcut[][2],
		       v4sf _r2cut,
		       v4sf accscale);
}

namespace PG5 = PhantomGrape5;
