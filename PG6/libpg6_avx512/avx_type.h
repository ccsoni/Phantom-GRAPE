namespace PhantomGrape6 {
    const int NumberOfPipe   = 48;
    const int MaxLength      = 1024;
#ifndef __AVX512F__
    const int NumberOfIParallel = 2;
    const int NumberOfVectorSingle = 8;
    const int NumberOfVectorDouble = 4;
#else
    const int NumberOfIParallel = 4;
    const int NumberOfVectorSingle = 16;
    const int NumberOfVectorDouble = 8;
#endif
    const int NumberOfJParallel = 4;

    struct PrdPosVel{
	double xpos, ypos, zpos;
	float  xvel, yvel, zvel;
	float  id;
	float  eps2;
	float  h2;
	float  pad[5];
    };
    typedef PrdPosVel *pPrdPosVel;

    struct NewAccJrk{
	double xacc;
	double yacc;
	double zacc;
	float  pot;
	float  xjrk;
	float  yjrk;
	float  zjrk;
	int    nnb;
	float  rnnb;
	float  pad[4];
    };
    typedef NewAccJrk *pNewAccJrk;
};

namespace PG6 = PhantomGrape6;
