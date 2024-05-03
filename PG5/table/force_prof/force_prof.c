#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gp5util.h"

void get_cputime(double *laptime, double *splittime);

#define SQR(x) ((x)*(x))

struct ptcl
{
  double mass, eps;
  double xpos, ypos, zpos;
  double xvel, yvel, zvel;
  double xacc, yacc, zacc;
  double pot;
};

int main(int argc, char **argv)
{
  struct ptcl *part;

  if(argc != 3) {
    fprintf(stderr, "Usage :: %s  <# of particles> <lbox> \n",argv[0]);
    exit(EXIT_FAILURE);
  }

  int npart = atoi(argv[1]);
  double lbox = atof(argv[2]);

  part = (struct ptcl *) malloc(sizeof(struct ptcl)*npart);

  for(int i=0;i<npart;i++) {
    part[i].mass = 1.0/(double)npart;
    part[i].xpos = (double)rand()/RAND_MAX*lbox;
    part[i].ypos = (double)rand()/RAND_MAX*lbox;
    part[i].zpos = (double)rand()/RAND_MAX*lbox;
  }

  part[0].mass = 1.0e5;
  part[0].xpos = part[0].ypos = part[0].zpos = 0.5*lbox;

  int nj = npart;
  double (*xj)[3], *mj;
  xj = (double (*)[3])malloc(sizeof(double)*nj*3);
  mj = (double *)malloc(sizeof(double)*nj);
  
  for(int j=0;j<nj;j++) {
    xj[j][0] = part[j].xpos;
    xj[j][1] = part[j].ypos;
    xj[j][2] = part[j].zpos;
    mj[j]    = part[j].mass;
  }

  int ni = npart;
  double (*xi)[3], (*ai)[3], *pi, *epsi2;
  xi = (double (*)[3])malloc(sizeof(double)*ni*3);
  ai = (double (*)[3])malloc(sizeof(double)*ni*3);
  pi = (double *)malloc(sizeof(double)*ni);
  epsi2 = (double *)malloc(sizeof(double)*ni);

  for(int i=0;i<ni;i++) {
    xi[i][0] = part[i].xpos;
    xi[i][1] = part[i].ypos;
    xi[i][2] = part[i].zpos;
    epsi2[i] = 1.0e-1;

    ai[i][0] = ai[i][1] = ai[i][3] = 0.0;
    pi[i] = 0.0;
  }
  
  double lap, split;

  g5_open();

  g5_set_xmjMC(0, 0, nj, xj, mj);
  g5_set_nMC(0,nj);

  get_cputime(&lap, &split);
  g5_calculate_force_on_xMC(0, xi, ai, pi, ni);
  get_cputime(&lap, &split);

  double ninter_per_sec = (double)ni*(double)nj/lap;

  fprintf(stderr, " %14.6e interactions / sec\n", ninter_per_sec);
  fprintf(stderr, " %14.6e Gflops \n", ninter_per_sec*38.0*1.0e-9);

  for(int i=0;i<ni;i++) {
    part[i].xacc = ai[i][0];
    part[i].yacc = ai[i][1];
    part[i].zacc = ai[i][2];
    part[i].pot  = pi[i];
  }

  for(int i=1;i<ni;i++) {
    double rad = 
      sqrt(SQR(part[i].xpos-0.5*lbox) + SQR(part[i].ypos-0.5*lbox) + SQR(part[i].zpos-0.5*lbox));

    double acc = sqrt(SQR(part[i].xacc)+SQR(part[i].yacc)+SQR(part[i].zacc));

    printf("%14.6e %14.6e %14.6e\n", rad, acc, part[i].pot);

  }

  free(part);
}
