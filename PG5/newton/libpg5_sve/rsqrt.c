#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <arm_sve.h>

double rsqrt(double x)
{
  float xf = (float)x;
  svbool_t p = svptrue_pat_b32(SV_VL1);
  svfloat32_t vx = svdup_f32(xf);
  svfloat32_t vr = svrsqrte_f32(vx);

  return svlastb_f32(p, vr);
}

double drsqrt(double x)
{
  return 1.0/sqrt(x);
}

double rsqrt_bias()
{
  union {int i; float f;} buf[2];

  double sum_weight = 0.0;
  double sum_error  = 0.0;
  double sum_sqr_error = 0.0;

  for(int32_t i=0;i<(1<<23);i++) {
    buf[0].i = (127<<23) + i;
    buf[1].i = (128<<23) + i;

    double x0 = (double)buf[0].f;
    double x1 = (double)buf[1].f;
    double y0 = rsqrt(x0);
    double y1 = rsqrt(x1);
    double err0 = y0*y0*x0 - 1.0;
    double err1 = y1*y1*x1 - 1.0;

    double weight = y0*y0;
    sum_weight += weight;
    sum_error += weight*(err0 + err1);
    sum_sqr_error += weight*(err0*err0 + err1*err1);
  }

  sum_weight *= 2.0;
  double bias = 0.5*sum_error/sum_weight;
  double mse  = 0.5*sqrt(sum_sqr_error/ sum_weight);

  fprintf(stderr, "rsqrt: MSE = %e,  Bias = %e\n", mse, bias);
  
  return bias;
}


#if 0
int32_t main()
{
  double rsqrt_correct = 1.0 - rsqrt_bias();
}
#endif
