#include <math.h>
#include <stdio.h>
#include "sve_type.h"

#define GLOBAL_DEFINE
#include "pg5_table.h"
#undef GLOBAL_DEFINE

static double reff(double sft_length, double radius)
{
  double ret;
  double eta  = 2.0*radius/sft_length;

  if(eta < 1.0){
    ret = (224.+eta*eta*(-224.+eta*(70.+eta*(48.-eta*21.))))
      * 2.e0/(35.0*(sft_length*sft_length*sft_length));
  }else if(eta < 2.0){
    ret = (12./(eta*eta)-224.+eta*(896.+eta*(-840.+eta*(224.+eta*(70.+eta*(-48.+eta*7.)))))) *2.e0/(35.0*eta*(sft_length*sft_length*sft_length));
  }else{
    ret = 1.0/pow(radius,3);
  }
  return ret;
}

static double Sft_for_PP = SFT_FOR_PP;
static double Sft_for_PM = SFT_FOR_PM;
static float R2scale;

union pack32 {
  float f;
  uint32_t u;
};

static double s2_force(double r)
{
  return (reff(Sft_for_PP, r)  - reff(Sft_for_PM, r));
}

void pg5_gen_s2_force_table(double sft_for_PP, double sft_for_PM)
{
  Sft_for_PP = sft_for_PP;
  Sft_for_PM = sft_for_PM;
  pg5_gen_force_table(s2_force, sft_for_PM);
}

// force_func() caluclate r^{-3} from r in N-body unit
// rcut : cut-off radius in N-body unit
void pg5_gen_force_table(double (*force_func)(double r), double rcut)
{

  int32_t i;
  const float r2max = (rcut * rcut);
  union pack32 m32;
  const float fmax = (1<<(1<<EXP_BIT)) * (2.0 - 1.0/(1<<FRC_BIT));
  const float r2scale = (fmax - 2.0f) / r2max;
  // so that 0 <= r2/r2max <= 2^17 - 3
  // or 2 <= s < 2^17, where s = r2/r2max + 2.0
  // in binary, 0_100000000_00000000... <= s <= 0_100001111_11110000...

  R2scale = r2scale;

  // skip this for the debug mode.
#ifndef DEBUG
  pg5_set_xscale((double)sqrtf(r2scale));
#endif

  // table value and...
  const uint32_t tick = (1<<(23-FRC_BIT));
  m32.f=2.0f;
  for(int32_t i=0; i<TBL_SIZE; i++, m32.u+=tick) {
    float f = m32.f;
    float r2 = (f - 2.0f) / r2scale;
    float r = sqrtf(r2);
    Force_table[i][0] = force_func(r);
  }

  // slope
  m32.f = 2.0f;
  for(int32_t i=0; i<TBL_SIZE-1; i++) {
    float x0 = m32.f;
    m32.u += tick;
    float x1 = m32.f;
    float y0 = Force_table[i][0];
    float y1 = (i==TBL_SIZE-1) ? 0.0 : Force_table[i+1][0];
    Force_table[i][1] = (y1-y0)/(x1-x0);
  }

  Force_table[TBL_SIZE-1][1] = 0.0f;
}

#ifdef DEBUG
#include <stdio.h>
#include <assert.h>

static void dump_table(FILE *fp)
{
  const uint32_t tick = (1<<(23-FRC_BIT));
  union pack32 m32;
  float r2scale = R2scacle;

  for(int32_t i=0,m32.f=2.0f;i<TBL_SIZE;i++, m32.u += tick) {
    float f = m32.f;
    float r2 = (f - 2.0f) / r2scale;
    float r = sqrtf(r2);
    float force = r * Force_table[i][0];
    fprintf(fp, "%e %e %e %e\n", r, force, Force_table[i][0], Force_table[i][1]);
  }
}

static float refer_table(float r)
{
  union pack32 m32;
  float x = r*r*R2scacle + 2.0f;
  m32.f = x;
  int idx = (m32.u >> (23-FRC_BIT)) & (TBL_SIZE-1);
  assert(idx < TBL_SIZE);
  m32.u >>= (23-FRC_BIT);
  m32.u <<= (23-FRC_BIT);
  float dx = (x - m32.f);
  return Force_table[idx][0] + dx*Force_table[idx][1];
}

#define N_SAMPLE_POINT (8192)
int main(int argc, char **argv)
{
  float tick = SFT_FOR_PM/N_SAMPLE_POINT;

  FILE *fp1 = fopen("force_table.dat", "w");
  FILE *fp2 = fopen("force_out.dat", "w");

  pg5_gen_force_table(s2_force, SFT_FOR_PM);
  dum_talbe(fp1);

  for(int32_t i=0;i<=N_SAMPLE_POINT;i++) {
    float r = i*tick;
    double force = r * refer_table(r);
    double realf = r * s2_force(r);
    double ferror = (force - realf);
    fprintf(fp2,"%12.4e %12.4e %12.4e %12.4e %12.4e\n",
            r, force, realf, ferror, ferror/force);
  }
  fclose(fp1);
  fclose(fp2);

  return 0;
}
#endif // DEBUG
