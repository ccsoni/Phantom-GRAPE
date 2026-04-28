#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "sve_type.h"
#include "pg5_table.h"

void gravity_kernel(pIpdata iptcl, pJpdata jptcl, pFodata fodata, int32_t nj, float r2cut_xscale2, float Ascale)
{
  static float dbgf[4];

#if (__ARM_FEATURE_SVE_BITS == 128) || (__ARM_FEATURE_SVE_BITS == 256)
  // NI_PARALLEL = 4 or 8 and NJ_PARALLEL = 1

  svbool_t pred_i = svptrue_b32();
  svbool_t pred_j_ld = svptrue_pat_b32(SV_VL4);

  svfloat32_t _xacc = svdup_f32(0.0f);
  svfloat32_t _yacc = svdup_f32(0.0f);
  svfloat32_t _zacc = svdup_f32(0.0f);

  svfloat32_t _xi = svld1(pred_i, iptcl->x);
  svfloat32_t _yi = svld1(pred_i, iptcl->y);
  svfloat32_t _zi = svld1(pred_i, iptcl->z);
  svfloat32_t _ei = svld1(pred_i, iptcl->eps2);

  int32_t *ptr = (int32_t *)Force_table-(1<<(31-(23-FRC_BIT)));

  for(int32_t j=0;j<nj;j++) {
    svfloat32_t _mj = svld1(pred_j_ld, (float32_t *)&(jptcl[j].x));
    svfloat32_t _xj = svdup_lane(_mj, 0);
    svfloat32_t _yj = svdup_lane(_mj, 1);
    svfloat32_t _zj = svdup_lane(_mj, 2);
    _mj = svdup_lane(_mj, 3);

    svfloat32_t _dx = svsub_x(pred_i, _xj, _xi);
    svfloat32_t _dy = svsub_x(pred_i, _yj, _yi);
    svfloat32_t _dz = svsub_x(pred_i, _zj, _zi);

    svfloat32_t _rsq = svmad_x(pred_i, _dx, _dx, 2.0f);
    _rsq = svmad_x(pred_i, _dy, _dy, _rsq);
    _rsq = svmad_x(pred_i, _dz, _dz, _rsq);
    _rsq = svmin_x(pred_i, _rsq, r2cut_xscale2);

    svuint32_t _rsq_sr = svlsr_z(pred_i, svreinterpret_u32(_rsq), 23-FRC_BIT);
    svuint32_t _rsq_sl = svlsl_z(pred_i, _rsq_sr, 23-FRC_BIT);

    svfloat32_t _ff = svreinterpret_f32(svld1_gather_u32index_s32(pred_i, ptr,   svlsl_z(pred_i, _rsq_sr, 1)));
    svfloat32_t _df = svreinterpret_f32(svld1_gather_u32index_s32(pred_i, ptr+1, svlsl_z(pred_i, _rsq_sr, 1)));

    svfloat32_t _dr2 = svsub_x(pred_i, _rsq, svreinterpret_f32(_rsq_sl));

    _ff = svmad_x(pred_i, _df, _dr2, _ff);

    _ff = svmul_x(pred_i, _ff, _mj);

    _xacc = svmad_x(pred_i, _ff, _dx, _xacc);
    _yacc = svmad_x(pred_i, _ff, _dy, _yacc);
    _zacc = svmad_x(pred_i, _ff, _dz, _zacc);
  }

  svst1(pred_i, fodata->ax, svmul_x(pred_i, _xacc, Ascale));
  svst1(pred_i, fodata->ay, svmul_x(pred_i, _yacc, Ascale));
  svst1(pred_i, fodata->az, svmul_x(pred_i, _zacc, Ascale));

#elif  __ARM_FEATURE_SVE_BITS == 512
  // NI_PARALLEL = 8 and NJ_PARALLEL = 2

  _xacc = svdup_f32(0.0f);
  _yacc = svdup_f32(0.0f);
  _zacc = svdup_f32(0.0f);

  int32_t nhalf = svcntw()>>1;
  svbool_t pred_all_b32 = svptrue_b32();

  // [0, 1, 2, ..., 14, 15]
  svuint32_t indx = svindex_u32(0, 1);
  // [0, 1, ... 6, 7, 0, 1, ... 6, 7]
  indx = svand_x(pred_all_b32, indx, nhalf-1);
  //[T, T, ... T, F, F, ... F]
  svbool_t pred_lo = svcmplt(pred_all_b32, indx, nhalf);

  static uint32_t broadcast0[16] =
    {0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4};
  static uint32_t broadcast1[16] =
    {1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5};
  static uint32_t broadcast2[16] =
    {2, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6};
  static uint32_t broadcast3[16] =
    {3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7};

  // predicate for loading two j-particles
  // [TTTTTTTTFFFFFFFF]
  svbool_t pred_j2_ld = svwhilelt_b32(0, 8);
  // predicate for loading a single j-particle
  // [TTTTFFFFFFFFFFFF]
  svbool_t pred_j1_ld = svwhilelt_b32(0, 4);

  // indices to be broadcast
  // as x-, y-, z-coordinate, and mass of the j-particles
  svuint32_t indx_x = svld1(svptrue_b32(), broadcast0);
  svuint32_t indx_y = svld1(svptrue_b32(), broadcast1);
  svuint32_t indx_z = svld1(svptrue_b32(), broadcast2);
  svuint32_t indx_m = svld1(svptrue_b32(), broadcast3);

  svfloat32_t _tmp;

  // if(ni-i>=8)
  //   pred_i = [TTTTTTTTTTTTTTTT] / pred_i_ld = [TTTTTTTTFFFFFFFF]
  // else
  //   pred_i = [TTTTTFFFTTTTTFFF] / pred_i_ld = [TTTTTFFFFFFFFFFF]
  svbool_t pred_i = svcmplt(pred_all_b32, indx, ni-i);
  svbool_t pred_i_ld = svand_x(pred_all_b32, pred_i, pred_lo);

  svfloat32_t _xi = svld1(pred_i_ld, iptcl->x+i);
  svfloat32_t _yi = svld1(pred_i_ld, iptcl->y+i);
  svfloat32_t _zi = svld1(pred_i_ld, iptcl->z+i);
  svfloat32_t _ei = svld1(pred_i_ld, iptcl->eps2+i);

  _tmp = svext(pred_all_b32, _xi, _xi, nhalf);
  _xi = svsel(pred_lo, _xi, _tmp); //[xi0, xi1, .. , xi7, xi0, xi1, .. , xi7]
  _tmp = svext(pred_all_b32, _yi, _yi, nhalf);
  _yi = svsel(pred_lo, _yi, _tmp); //[yi0, yi1, .. , yi7, yi0, yi1, .. , yi7]
  _tmp = svext(pred_all_b32, _zi, _zi, nhalf);
  _zi = svsel(pred_lo, _zi, _tmp); //[zi0, zi1, .. , zi7, zi0, zi1, .. , zi7]
  _tmp = svext(pred_all_b32, _ei, _ei, nhalf);
  _ei = svsel(pred_lo, _ei, _tmp); //[ei0, ei1, .. , ei7, ei0, ei1, .. , ei7]

  for(int32_t j=0;j<nj;j+=2) {
    // predicate pred_j :
    // [TTTTTTTTFFFFFFFF] for nj-j < 2
    // [TTTTTTTTTTTTTTTT] otherwise
    svbool_t pred_j = svsel_b32(svcmpge(pred_all_b32, nj-j, 2),
                                pred_all_b32, pred_lo);

    svbool_t pred_ij = svand_x(pred_all_b32, pred_i, pred_j);

    svbool_t pred_j_ld;
    if(nj-j<2) {
      pred_j_ld = pred_j1_ld;
    }else{
      pred_j_ld = pred_j2_ld;
    }

    // _xmj=[xj0, yj0, zj0, mj0, xj1, yj1, zj1, mj1, 0, 0, 0, 0, 0, 0, 0, 0]
    // or
    // _xmj=[xj0, yj0, zj0, mj0,   0,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0]
    svfloat32_t _xmj = svld1(pred_j_ld, (float32_t *)(jptcl+j));

    // _Xj=[Xj0, Xj0, ... , Xj0, Xj0, Xj1,Xj1, ... ,Xj1,Xj1]
    svfloat32_t _xj = svtbl_f32(_xmj, indx_x);
    svfloat32_t _yj = svtbl_f32(_xmj, indx_y);
    svfloat32_t _zj = svtbl_f32(_xmj, indx_z);
    svfloat32_t _mj = svtbl_f32(_xmj, indx_m);

    svfloat32_t _dx = svsub_x(pred_i, _xj, _xi);
    svfloat32_t _dy = svsub_x(pred_i, _yj, _yi);
    svfloat32_t _dz = svsub_x(pred_i, _zj, _zi);

    svfloat32_t _rsq = svmad_x(pred_i, _dx, _dx, 2.0f);
    _rsq = svmad_x(pred_i, _dy, _dy, _rsq);
    _rsq = svmad_x(pred_i, _dz, _dz, _rsq);
    _rsq = svmin_x(pred_i, _rsq, r2cut_xscale2);

    svuint32_t _rsq_sr = svlsr_z(pred_i, svreinterpret_u32(_rsq), 23-FRC_BIT);
    svuint32_t _rsq_sl = svlsl_z(pred_i, _rsq_sr, 23-FRC_BIT);

    svfloat32_t _ff = svreinterpret_f32(svld1_gather_u32index_s32(pred_i, ptr,   svlsl_z(pred_i, _rsq_sr, 1)));
    svfloat32_t _df = svreinterpret_f32(svld1_gather_u32index_s32(pred_i, ptr+1, svlsl_z(pred_i, _rsq_sr, 1)));

    svfloat32_t _dr2 = svsub_x(pred_i, _rsq, svreinterpret_f32(_rsq_sl));

    _ff = svmad_x(pred_i, _df, _dr2, _ff);

    _ff = svmul_x(pred_i, _ff, _mj);

    _xacc = svmad_x(pred_i, _ff, _dx, _xacc);
    _yacc = svmad_x(pred_i, _ff, _dy, _yacc);
    _zacc = svmad_x(pred_i, _ff, _dz, _zacc);
  }

  _xacc = svmul_x(pred_i, _xacc, Ascale);
  _yacc = svmul_x(pred_i, _yacc, Ascale);
  _zacc = svmul_x(pred_i, _zacc, Ascale);

  svbool_t pred_st = svptrue_pat_b32(SV_VL8);
  _xacc = svadd_x(pred_st, _xacc, svext_f32(_xacc, _xacc, 8));
  _yacc = svadd_x(pred_st, _yacc, svext_f32(_yacc, _yacc, 8));
  _zacc = svadd_x(pred_st, _zacc, svext_f32(_zacc, _zacc, 8));

#endif

}
