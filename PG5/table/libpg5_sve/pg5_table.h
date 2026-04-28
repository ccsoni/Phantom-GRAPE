#pragma once

#define SIZE_OF_MESH (16)
#define SFT_FOR_PM (3.0/SIZE_OF_MESH)
#define SFT_FOR_PP (0.5/SIZE_OF_MESH)

// Table size
#define EXP_BIT (4)
#define FRC_BIT (5)

#define TICK (1<<(23-FRC_BIT))
#define TBL_SIZE (1<<(EXP_BIT+FRC_BIT))

#ifdef GLOBAL_DEFINE
#define EXTERN
#else
#define EXTERN extern
#endif

#ifdef ALLOCATE_TABLE
#include <malloc.h>
EXTERN float (*Force_table)[2];
#else
EXTERN float Force_table[TBL_SIZE][2];
#endif

#if defined(__cplusplus)
extern "C"
{
#endif
  void pg5_gen_force_table(double (*force_func)(double), double rcut);
  void pg5_gen_s2_force_table(double sft_for_PP, double sft_for_PM);
  void pg5_set_xscale(double);
#if defined(__cplusplus)
}
#endif
