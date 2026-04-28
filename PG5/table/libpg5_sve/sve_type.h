#pragma once

#include <arm_sve.h>

#define ALIGN64 __attribute__ ((aligned(64)))

#if __ARM_FEATURE_SVE_BITS == 128
// I4J1
#define NI_PARALLEL (4)
#define NJ_PARALLEL (1)
#elif __ARM_FEATURE_SVE_BITS == 256
// I8J1
#define NI_PARALLEL (8)
#define NJ_PARALLEL (1)
#elif __ARM_FEATURE_SVE_BITS == 512
// I8J2
#define NI_PARALLEL (8)
#define NJ_PARALLEL (2)
#endif

// 4  * 4 * NI_PARALLEL = 16*NI_PARALLEL bytes
typedef struct ipdata {
  float x[NI_PARALLEL];
  float y[NI_PARALLEL];
  float z[NI_PARALLEL];
  float eps2[NI_PARALLEL];
} Ipdata, *pIpdata;

// 4 x 4 = 16 bytes
typedef struct jpdata {
  float x, y, z, m;
} Jpdata, *pJpdata;

// 4 * 4 * NI_PARAREL = 16*NI_PARALLEL bytes
typedef struct fodata {
  float ax[NI_PARALLEL];
  float ay[NI_PARALLEL];
  float az[NI_PARALLEL];
  float phi[NI_PARALLEL];
} Fodata, *pFodata;

#define DBG_PRINT_F32(S, _reg)                          \
  svst1(svptrue_b32(), dbgf, _reg);                     \
  printf("%s = %14.6e %14.6e %14.6e %14.6e\n",          \
         S, dbgf[0], dbgf[1], dbgf[2], dbgf[3]);        \
  fflush(stdout);

#define DBG_PRINT_F64(S, _reg)                          \
  svst1(svptrue_b64(), dbgd, _reg);                     \
  printf("%s = %14.6e %14.6e\n", S, dbgd[0], dbgd[1]);  \
  fflush(stdout);
