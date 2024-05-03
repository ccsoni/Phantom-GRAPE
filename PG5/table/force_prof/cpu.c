#include <stdio.h>
#include <sys/time.h>

/* elapsed time in real world */
void get_cputime(double *laptime, double *splittime)
{
    struct timeval x;
    double sec,microsec;

    gettimeofday(&x, NULL);
    sec = x.tv_sec;
    microsec = x.tv_usec;

    *laptime = sec + microsec / 1000000.0 - *splittime;
    *splittime = sec + microsec / 1000000.0;
}

