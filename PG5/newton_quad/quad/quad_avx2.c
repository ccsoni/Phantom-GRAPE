#include "../libpg5/avx_type.h"
#include "quad_type.h"

#include "../libpg5/sse.h"
#include "../libpg5/avx.h"
#include "../libpg5/avx2.h"

void c_GravityKernel(pIpdata ipdata, pFodata fodata, cJcdata jcdata, int nj){
  int j;
  
  static float five[8] __attribute__((aligned(32))) = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  static float half[8] __attribute__((aligned(32))) = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  
  PREFETCH(jcdata[0]);

  VZEROALL;
  
  // load i-particle
  VLOADPS(*ipdata->x, XMM00);
  VLOADPS(*ipdata->y, XMM01);
  VLOADPS(*ipdata->z, XMM02);
  VLOADPS(jcdata->xm[0][0], YMM14);
  VLOADPS(jcdata->q[0][0], YMM08);
  VLOADPS(jcdata->q[1][0], YMM15);
  jcdata++;
  VPERM2F128(YMM00, YMM00, YMM00, 0x00);
  VPERM2F128(YMM01, YMM01, YMM01, 0x00);
  VPERM2F128(YMM02, YMM02, YMM02, 0x00);
  
  // load jcell's coordinate
  VSHUFPS(YMM14, YMM14, YMM03, 0x00);
  VSHUFPS(YMM14, YMM14, YMM04, 0x55);
  VSHUFPS(YMM14, YMM14, YMM05, 0xaa);

  for(j = 0; j < nj; j += 2){
    // r_ij,x -> YMM03
    VSUBPS(YMM00, YMM03, YMM03);
    // r_ij,y -> YMM04
    VSUBPS(YMM01, YMM04, YMM04);
    VLOADPS(*ipdata->eps2, XMM01);
    VPERM2F128(YMM01, YMM01, YMM01, 0x00);
    // r_ij,z -> YMM05
    VSUBPS(YMM02, YMM05, YMM05);

    // r_ij^2 -> YMM01
    VFMADDPS(YMM01, YMM03, YMM03); // eps2 + r_ij,x^2
    VSHUFPS(YMM14, YMM14, YMM02, 0xff);
    VFMADDPS(YMM01, YMM04, YMM04); // r_ij,x^2 + r_ij,y^2
    VFMADDPS(YMM01, YMM05, YMM05); // r_ij,x^2 + r_ij,y^2 + r_ij,z^2
    // 1 / r_ij -> YMM01
    
    VRSQRTPS(YMM01, YMM01);
    // 1 / r_ij^2 -> YMM00

    VMULPS(YMM01, YMM01, YMM00);
    // phi_p(mj / r_ij) -> YMM02
    //VLOADPS(*jcdata->m, YMM02);
    VMULPS(YMM01, YMM02, YMM02);

    // q00, q01, q02, q11, q12, q22 -> YMM06, 07, 08, 13, 14, 15, respectively
    /*
    VLOADPS(*jcdata->q[0], YMM06);
    VLOADPS(*jcdata->q[1], YMM07);
    VLOADPS(*jcdata->q[2], YMM08);
    VLOADPS(*jcdata->q[3], YMM13);
    VLOADPS(*jcdata->q[4], YMM14);
    VLOADPS(*jcdata->q[5], YMM15);
    */

    // q00, q01, q02, q11, q12, q22 -> YMM06, YMM07, YMM08, YMM13, YMM14, YMM15 respectively
    VSHUFPS(YMM08, YMM08, YMM06, 0x00); // 00
    VSHUFPS(YMM08, YMM08, YMM07, 0x55); // 55
    VSHUFPS(YMM15, YMM15, YMM13, 0x00); // 00
    VSHUFPS(YMM15, YMM15, YMM14, 0x55); // 55
    VSHUFPS(YMM08, YMM08, YMM08, 0xaa); // aa
    VSHUFPS(YMM15, YMM15, YMM15, 0xaa); // aa

    // q00 * r_ij,x -> YMM06
    VMULPS(YMM03, YMM06, YMM06);
    VMULPS(YMM13, YMM04, YMM13);
    VMULPS(YMM15, YMM05, YMM15);
    // YMM06 + q01 * r_ij,y -> YMM06
    VFMADDPS(YMM06, YMM04, YMM07);
    VFMADDPS(YMM13, YMM03, YMM07);
    VFMADDPS(YMM15, YMM03, YMM08);
    // YMM06 + q02 * r_ij,z -> YMM06
    VFMADDPS(YMM06, YMM05, YMM08);
    VFMADDPS(YMM13, YMM05, YMM14);
    VFMADDPS(YMM15, YMM04, YMM14);
    // 0.5 -> YMM14
    VLOADPS(half, YMM14);
    // q11 * r_ij,y -> YMM13
    // YMM13 + q01 * r_ij,x -> YMM13
    // YMM13 + q12 * r_ij,z -> YMM13

    // q22 * r_ij,z -> YMM15
    // YMM15 + q02 * r_ij,x -> YMM15
    // YMM15 + q12 * r_ij,y -> YMM15

    // YMM06(qdr[0]) * YMM03(r_ij,x) -> YMM07
    VMULPS(YMM03, YMM06, YMM07);
    VMULPS(YMM00, YMM00, YMM08); // {1/(r_ij)^2}^2 = 1/(r_ij)^4 -> YMM08
    // YMM07 + YMM13(qdr[1]) * YMM04(r_ij,y) -> YMM07
    VFMADDPS(YMM07, YMM04, YMM13);
    VMULPS(YMM01, YMM08, YMM08); // 1/r_ij * 1/(r_ij)^4 = 1/(r_ij)^5 ->YMM08
    // YMM07 + YMM15(qdr[2]) * YMM05(r_ij,z) -> YMM07(drqdr)
    VFMADDPS(YMM07, YMM05, YMM15);

    // 1/(r_ij)^5 -> YMM08
    VLOADPS(*ipdata->y, XMM01);
    VPERM2F128(YMM01, YMM01, YMM01, 0x00);

    // 1/(r_ij)^5 * drqdr * 0.5 (=phi_q) -> YMM07
    VMULPS(YMM07, YMM08, YMM07);
    VMULPS(YMM07, YMM14, YMM07);

    // YMM09 += phi_p(YMM02) + phi_q(YMM07)
    VADDPS(YMM02, YMM07, YMM14);
    VADDPS(YMM14, YMM09, YMM09);
    // 5.0 -> YMM14
    VLOADPS(five, YMM14);
    // 5.0 * phi_q + phi_p -> YMM01
    VFMADDPS(YMM02, YMM07, YMM14);

    VLOADPS(jcdata->xm[0][0], YMM14);

    // (phi_p + 5.0 * phi_q) / (r_ij)^2 ->YMM00
    //VMULPS(YMM00, YMM01, YMM00); // AVX
    VMULPS(YMM02, YMM00, YMM00); 

    VLOADPS(*ipdata->z, XMM02);
    // YMM10, YMM11, YMM12 += ax, ay, az
    VFMADDPS(YMM10, YMM00, YMM03);
    VPERM2F128(YMM02, YMM02, YMM02, 0x00);
    VSHUFPS(YMM14, YMM14, YMM03, 0x00);
    VFMADDPS(YMM11, YMM00, YMM04);
    VSHUFPS(YMM14, YMM14, YMM04, 0x55);
    VFMADDPS(YMM12, YMM00, YMM05);
    VSHUFPS(YMM14, YMM14, YMM05, 0xaa);
    VLOADPS(*ipdata->x, XMM00);
    VFNMADDPS(YMM12, YMM08, YMM15);
    VLOADPS(jcdata->q[1][0], YMM15);
    VFNMADDPS(YMM10, YMM08, YMM06);
    VPERM2F128(YMM00, YMM00, YMM00, 0x00);
    VFNMADDPS(YMM11, YMM08, YMM13);
    
    VLOADPS(jcdata->q[0][0], YMM08);
    jcdata++;
  }
  VEXTRACTF128(YMM10, XMM00, 0x01);
  VEXTRACTF128(YMM11, XMM01, 0x01);
  VEXTRACTF128(YMM12, XMM02, 0x01);
  VEXTRACTF128(YMM09, XMM03, 0x01);
  VADDPS(YMM10, YMM00, YMM10);
  VADDPS(YMM11, YMM01, YMM11);
  VADDPS(YMM12, YMM02, YMM12);
  VADDPS(YMM09, YMM03, YMM09);

  VSTORPS(XMM10,  *fodata->ax);
  VSTORPS(XMM11,  *fodata->ay);
  VSTORPS(XMM12,  *fodata->az);
  VSTORPS(XMM09, *fodata->phi);
}

void c_GravityKernel0(pIpdata ipdata, pFodata fodata, cJcdata0 jcdata, int nj)
{
  int j;
  static float five[8] __attribute__((aligned(32))) = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  static float half[8] __attribute__((aligned(32))) = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  
  PREFETCH(jcdata[0]);

  VZEROALL;
  
  for(j = 0; j < nj; j += 2){
    // load i-particle
    VLOADPS(*ipdata->x, XMM00);
    VLOADPS(*ipdata->y, XMM01);
    VLOADPS(*ipdata->z, XMM02);
    VPERM2F128(YMM00, YMM00, YMM00, 0x00);
    VPERM2F128(YMM01, YMM01, YMM01, 0x00);
    VPERM2F128(YMM02, YMM02, YMM02, 0x00);

    // load jcell's coordinate
    VLOADPS(*jcdata->x, YMM03);
    VLOADPS(*jcdata->y, YMM04);
    VLOADPS(*jcdata->z, YMM05);

    // r_ij,x -> YMM03
    VSUBPS(YMM00, YMM03, YMM03);

    // r_ij,y -> YMM04
    VSUBPS(YMM01, YMM04, YMM04);

    // r_ij,z -> YMM05
    VSUBPS(YMM02, YMM05, YMM05);

    // r_ij^2 -> YMM01
    VLOADPS(*ipdata->eps2, XMM01);
    VPERM2F128(YMM01, YMM01, YMM01, 0x00);
    VFMADDPS(YMM01, YMM03, YMM03); // eps2 + r_ij,x^2
    /*
    VMULPS(YMM03, YMM03, YMM01); // r_ij,x^2 -> YMM01
    VADDPS(YMM01, YMM15, YMM01); // YMM01 + eps2
    */
    VFMADDPS(YMM01, YMM04, YMM04); // r_ij,x^2 + r_ij,y^2
    VFMADDPS(YMM01, YMM05, YMM05); // r_ij,x^2 + r_ij,y^2 + r_ij,z^2

    /*
    VMULPS(YMM03, YMM03, YMM00); // r_ij,x^2 -> YMM00
    VMULPS(YMM04, YMM04, YMM01); // r_ij,y^2 -> YMM01
    VMULPS(YMM05, YMM05, YMM02); // r_ij,z^2 -> YMM02
    VADDPS(YMM00, YMM01, YMM01); // r_ij,x^2 + r_ij,y^2
    VADDPS(YMM01, YMM02, YMM01); // r_ij,x^2 + r_ij,y^2 + r_ij,z^2
    VADDPS(YMM01, YMM15, YMM01); // r_ij^2
    */

    // 1 / r_ij -> YMM01
    VRSQRTPS(YMM01, YMM01);
    // 1 / r_ij^2 -> YMM00
    VMULPS(YMM01, YMM01, YMM00);

    // phi_p(mj / r_ij) -> YMM02
    VLOADPS(*jcdata->m, YMM02);
    VMULPS(YMM01, YMM02, YMM02);

    // q00, q01, q02, q11, q12, q22 -> YMM06, 07, 08, 13, 14, 15, respectively
    VLOADPS(*jcdata->q[0], YMM06);
    VLOADPS(*jcdata->q[1], YMM07);
    VLOADPS(*jcdata->q[2], YMM08);
    VLOADPS(*jcdata->q[3], YMM13);
    VLOADPS(*jcdata->q[4], YMM14);
    VLOADPS(*jcdata->q[5], YMM15);

    // q00 * r_ij,x -> YMM06
    VMULPS(YMM03, YMM06, YMM06);
    // YMM06 + q01 * r_ij,y -> YMM06
    VFMADDPS(YMM06, YMM04, YMM07);
    /*
    VMULPS(YMM04, YMM07, YMM00);
    VADDPS(YMM00, YMM06, YMM06);
    */
    // YMM06 + q02 * r_ij,z -> YMM06
    VFMADDPS(YMM06, YMM05, YMM08);
    /*
    VMULPS(YMM05, YMM08, YMM00);
    VADDPS(YMM00, YMM06, YMM06);
    */

    // q11 * r_ij,y -> YMM13
    VMULPS(YMM13, YMM04, YMM13);
    // YMM13 + q01 * r_ij,x -> YMM13
    VFMADDPS(YMM13, YMM03, YMM07);
    /*
    VMULPS(YMM07, YMM03, YMM00);
    VADDPS(YMM00, YMM13, YMM13);
    */
    // YMM13 + q12 * r_ij,z -> YMM13
    VFMADDPS(YMM13, YMM05, YMM14);
    /*
    VMULPS(YMM14, YMM05, YMM00);
    VADDPS(YMM00, YMM13, YMM13);
    */

    // q22 * r_ij,z -> YMM15
    VMULPS(YMM15, YMM05, YMM15);
    // YMM15 + q02 * r_ij,x -> YMM15
    VFMADDPS(YMM15, YMM03, YMM08);
    /*
    VMULPS(YMM08, YMM03, YMM00);
    VADDPS(YMM00, YMM15, YMM15);
    */
    // YMM15 + q12 * r_ij,y -> YMM15
    VFMADDPS(YMM15, YMM04, YMM14);
    /*
    VMULPS(YMM14, YMM04, YMM00);
    VADDPS(YMM00, YMM15, YMM15);
    */

    /*
    // 1 / r_ij^2 -> YMM00
    VMULPS(YMM01, YMM01, YMM00);
    */

    // YMM06(qdr[0]) * YMM03(r_ij,x) -> YMM07
    VMULPS(YMM03, YMM06, YMM07);
    // YMM07 + YMM13(qdr[1]) * YMM04(r_ij,y) -> YMM07
    VFMADDPS(YMM07, YMM04, YMM13);
    /*
    VMULPS(YMM04, YMM13, YMM08);
    VADDPS(YMM07, YMM08, YMM07);
    */
    // YMM07 + YMM15(qdr[2]) * YMM05(r_ij,z) -> YMM07(drqdr)
    VFMADDPS(YMM07, YMM05, YMM15);
    /*
    VMULPS(YMM05, YMM15, YMM08);
    VADDPS(YMM07, YMM08, YMM07);
    */

    // 1/(r_ij)^5 -> YMM08
    VMULPS(YMM00, YMM00, YMM08); // {1/(r_ij)^2}^2 = 1/(r_ij)^4 -> YMM08
    VMULPS(YMM01, YMM08, YMM08); // 1/r_ij * 1/(r_ij)^4 = 1/(r_ij)^5 ->YMM08

    // 0.5 -> YMM14
    VLOADPS(half, YMM14);

    // 1/(r_ij)^5 * drqdr * 0.5 (=phi_q) -> YMM07
    VMULPS(YMM07, YMM08, YMM07);
    VMULPS(YMM07, YMM14, YMM07);

    // YMM09 += phi_p(YMM02) + phi_q(YMM07)
    VADDPS(YMM02, YMM07, YMM14);
    VADDPS(YMM14, YMM09, YMM09);
    /*
    VADDPS(YMM02, YMM09, YMM09);
    VADDPS(YMM07, YMM09, YMM09);
    */

    // 5.0 -> YMM14
    VLOADPS(five, YMM14);

    // 5.0 * phi_q + phi_p -> YMM01
    VFMADDPS(YMM02, YMM07, YMM14);
    /*
    VMULPS(YMM07, YMM14, YMM01);
    VADDPS(YMM01, YMM02, YMM01);
    */

    // (phi_p + 5.0 * phi_q) / (r_ij)^2 ->YMM00
    //VMULPS(YMM00, YMM01, YMM00); AVX用
    VMULPS(YMM02, YMM00, YMM00); 

    // ax, ay, az -> YMM10, YMM11, YMM12
    VFMADDPS(YMM10, YMM00, YMM03);
    VFNMADDPS(YMM10, YMM08, YMM06);
    VFMADDPS(YMM11, YMM00, YMM04);
    VFNMADDPS(YMM11, YMM08, YMM13);
    VFMADDPS(YMM12, YMM00, YMM05);
    VFNMADDPS(YMM12, YMM08, YMM15);

    /*
    VMULPS(YMM00, YMM03, YMM03); // mr3i * dx
    VMULPS(YMM00, YMM04, YMM04); // mr3i * dy
    VMULPS(YMM00, YMM05, YMM05); // mr3i * dz
    VMULPS(YMM06, YMM08, YMM06); // qdx * dr5i;
    VMULPS(YMM13, YMM08, YMM13); // qdy * dr5i;
    VMULPS(YMM15, YMM08, YMM15); // qdz * dr5i

    VADDPS(YMM03, YMM10, YMM10);
    VSUBPS(YMM06, YMM10, YMM10);
    VADDPS(YMM04, YMM11, YMM11);
    VSUBPS(YMM13, YMM11, YMM11);
    VADDPS(YMM05, YMM12, YMM12);
    VSUBPS(YMM15, YMM12, YMM12);
    */
    /*
    VSUBPS(YMM06, YMM03, YMM03);
    VADDPS(YMM03, YMM10, YMM10);
    VSUBPS(YMM13, YMM04, YMM04);
    VADDPS(YMM04, YMM11, YMM11);
    VSUBPS(YMM15, YMM05, YMM05);
    VADDPS(YMM05, YMM12, YMM12);
    */
    jcdata++;
  }
  VEXTRACTF128(YMM10, XMM00, 0x01);
  VADDPS(YMM10, YMM00, YMM10);
  VEXTRACTF128(YMM11, XMM01, 0x01);
  VADDPS(YMM11, YMM01, YMM11);
  VEXTRACTF128(YMM12, XMM02, 0x01);
  VADDPS(YMM12, YMM02, YMM12);
  VEXTRACTF128(YMM09, XMM03, 0x01);
  VADDPS(YMM09, YMM03, YMM09);

  VSTORPS(XMM10,  *fodata->ax);
  VSTORPS(XMM11,  *fodata->ay);
  VSTORPS(XMM12,  *fodata->az);
  VSTORPS(XMM09, *fodata->phi);
}
