#include <stdio.h>
//#include <sys/time.h>
#include <time.h>

#if 0
/* elapsed time in real world */
void get_cputime_(double *laptime, double *splittime)
{
  struct timeval x;
  double sec,microsec;

  gettimeofday(&x, NULL);
  sec = x.tv_sec;
  microsec = x.tv_usec;

  *laptime = sec + microsec / 1000000.0 - *splittime;
  *splittime = sec + microsec / 1000000.0;
}
#else
void get_cputime(double *laptime, double *splittime)
{
  struct timespec ts_split;
  struct timespec ts_now;

  clock_gettime(CLOCK_MONOTONIC, &ts_now);

  *laptime = (double)ts_now.tv_sec + (double)ts_now.tv_nsec/1.0e9 - (*splittime);
  *splittime = (double)ts_now.tv_sec + (double)ts_now.tv_nsec/1.0e9;
}
#endif
