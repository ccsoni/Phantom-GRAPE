#include "quad_type.h"
#include "quad_util.h"

#define avx_type.h ../libpg5/avx_type.h
#define gp5util.h ../libpg5/gp5util.h
#include "../libpg5/phantom_g5.c"

struct Cell_Mem {
  int ncell, pad[15];
  Jcdata0 jcell0[JMEMSIZE/2];
  Jcdata jcell[JMEMSIZE/2];
} cell_mem[MAXDEV] ALIGN64;

void g5c_set_nMC(int devid, int n){
  struct Cell_Mem *cm = cell_mem + devid;
  cm->ncell = n;
};

void g5c_set_xmjMC0(int devid, int adr, int nj, double (*xj)[3], double *mj, double (*qj)[6]){
  int j;
  //struct Ptcl_Mem *pm = ptcl_mem + devid;
  struct Cell_Mem *cm = cell_mem + devid;
  
  assert(adr % 2 == 0);
  for(j=adr;j<adr+nj;j+=2) {
    int jadr = j / 2;
    cm->jcell0[jadr].x[0] = (float)xj[j][0];
    cm->jcell0[jadr].x[1] = (float)xj[j][0];
    cm->jcell0[jadr].x[2] = (float)xj[j][0];
    cm->jcell0[jadr].x[3] = (float)xj[j][0];
    cm->jcell0[jadr].y[0] = (float)xj[j][1];
    cm->jcell0[jadr].y[1] = (float)xj[j][1];
    cm->jcell0[jadr].y[2] = (float)xj[j][1];
    cm->jcell0[jadr].y[3] = (float)xj[j][1];
    cm->jcell0[jadr].z[0] = (float)xj[j][2];
    cm->jcell0[jadr].z[1] = (float)xj[j][2];
    cm->jcell0[jadr].z[2] = (float)xj[j][2];
    cm->jcell0[jadr].z[3] = (float)xj[j][2];
    cm->jcell0[jadr].m[0] = (float)mj[j];
    cm->jcell0[jadr].m[1] = (float)mj[j];
    cm->jcell0[jadr].m[2] = (float)mj[j];
    cm->jcell0[jadr].m[3] = (float)mj[j];


    cm->jcell0[jadr].x[4] = (float)xj[j+1][0];
    cm->jcell0[jadr].x[5] = (float)xj[j+1][0];
    cm->jcell0[jadr].x[6] = (float)xj[j+1][0];
    cm->jcell0[jadr].x[7] = (float)xj[j+1][0];
    cm->jcell0[jadr].y[4] = (float)xj[j+1][1];
    cm->jcell0[jadr].y[5] = (float)xj[j+1][1];
    cm->jcell0[jadr].y[6] = (float)xj[j+1][1];
    cm->jcell0[jadr].y[7] = (float)xj[j+1][1];
    cm->jcell0[jadr].z[4] = (float)xj[j+1][2];
    cm->jcell0[jadr].z[5] = (float)xj[j+1][2];
    cm->jcell0[jadr].z[6] = (float)xj[j+1][2];
    cm->jcell0[jadr].z[7] = (float)xj[j+1][2];
    cm->jcell0[jadr].m[4] = (float)mj[j+1];
    cm->jcell0[jadr].m[5] = (float)mj[j+1];
    cm->jcell0[jadr].m[6] = (float)mj[j+1];
    cm->jcell0[jadr].m[7] = (float)mj[j+1];


    cm->jcell0[jadr].q[0][0] = (float)qj[j][0];
    cm->jcell0[jadr].q[0][1] = (float)qj[j][0];
    cm->jcell0[jadr].q[0][2] = (float)qj[j][0];
    cm->jcell0[jadr].q[0][3] = (float)qj[j][0];

    cm->jcell0[jadr].q[1][0] = (float)qj[j][1];
    cm->jcell0[jadr].q[1][1] = (float)qj[j][1];
    cm->jcell0[jadr].q[1][2] = (float)qj[j][1];
    cm->jcell0[jadr].q[1][3] = (float)qj[j][1];

    cm->jcell0[jadr].q[2][0] = (float)qj[j][2];
    cm->jcell0[jadr].q[2][1] = (float)qj[j][2];
    cm->jcell0[jadr].q[2][2] = (float)qj[j][2];
    cm->jcell0[jadr].q[2][3] = (float)qj[j][2];

    cm->jcell0[jadr].q[3][0] = (float)qj[j][3];
    cm->jcell0[jadr].q[3][1] = (float)qj[j][3];
    cm->jcell0[jadr].q[3][2] = (float)qj[j][3];
    cm->jcell0[jadr].q[3][3] = (float)qj[j][3];

    cm->jcell0[jadr].q[4][0] = (float)qj[j][4];
    cm->jcell0[jadr].q[4][1] = (float)qj[j][4];
    cm->jcell0[jadr].q[4][2] = (float)qj[j][4];
    cm->jcell0[jadr].q[4][3] = (float)qj[j][4];

    cm->jcell0[jadr].q[5][0] = (float)qj[j][5];
    cm->jcell0[jadr].q[5][1] = (float)qj[j][5];
    cm->jcell0[jadr].q[5][2] = (float)qj[j][5];
    cm->jcell0[jadr].q[5][3] = (float)qj[j][5];


    cm->jcell0[jadr].q[0][4] = (float)qj[j+1][0];
    cm->jcell0[jadr].q[0][5] = (float)qj[j+1][0];
    cm->jcell0[jadr].q[0][6] = (float)qj[j+1][0];
    cm->jcell0[jadr].q[0][7] = (float)qj[j+1][0];

    cm->jcell0[jadr].q[1][4] = (float)qj[j+1][1];
    cm->jcell0[jadr].q[1][5] = (float)qj[j+1][1];
    cm->jcell0[jadr].q[1][6] = (float)qj[j+1][1];
    cm->jcell0[jadr].q[1][7] = (float)qj[j+1][1];

    cm->jcell0[jadr].q[2][4] = (float)qj[j+1][2];
    cm->jcell0[jadr].q[2][5] = (float)qj[j+1][2];
    cm->jcell0[jadr].q[2][6] = (float)qj[j+1][2];
    cm->jcell0[jadr].q[2][7] = (float)qj[j+1][2];
    
    cm->jcell0[jadr].q[3][4] = (float)qj[j+1][3];
    cm->jcell0[jadr].q[3][5] = (float)qj[j+1][3];
    cm->jcell0[jadr].q[3][6] = (float)qj[j+1][3];
    cm->jcell0[jadr].q[3][7] = (float)qj[j+1][3];
    
    cm->jcell0[jadr].q[4][4] = (float)qj[j+1][4];
    cm->jcell0[jadr].q[4][5] = (float)qj[j+1][4];
    cm->jcell0[jadr].q[4][6] = (float)qj[j+1][4];
    cm->jcell0[jadr].q[4][7] = (float)qj[j+1][4];
    
    cm->jcell0[jadr].q[5][4] = (float)qj[j+1][5];
    cm->jcell0[jadr].q[5][5] = (float)qj[j+1][5];
    cm->jcell0[jadr].q[5][6] = (float)qj[j+1][5];
    cm->jcell0[jadr].q[5][7] = (float)qj[j+1][5];
  }

  int rsdl = (NUNROLL - (nj % NUNROLL)) % NUNROLL;
  for(j=nj;j<nj+rsdl;j+=2){
    int jj, jadr = j / 2;
    for(jj = 0; jj < 2; jj++){
      int jp = jadr * 2 + jj;
      if(jp < nj)
	continue;
      cm->jcell0[jadr].x[4*jj+0] = 0.0f;
      cm->jcell0[jadr].x[4*jj+1] = 0.0f;
      cm->jcell0[jadr].x[4*jj+2] = 0.0f;
      cm->jcell0[jadr].x[4*jj+3] = 0.0f;
      cm->jcell0[jadr].y[4*jj+0] = 0.0f;
      cm->jcell0[jadr].y[4*jj+1] = 0.0f;
      cm->jcell0[jadr].y[4*jj+2] = 0.0f;
      cm->jcell0[jadr].y[4*jj+3] = 0.0f;
      cm->jcell0[jadr].z[4*jj+0] = 0.0f;
      cm->jcell0[jadr].z[4*jj+1] = 0.0f;
      cm->jcell0[jadr].z[4*jj+2] = 0.0f;
      cm->jcell0[jadr].z[4*jj+3] = 0.0f;
      cm->jcell0[jadr].m[4*jj+0] = 0.0f;
      cm->jcell0[jadr].m[4*jj+1] = 0.0f;
      cm->jcell0[jadr].m[4*jj+2] = 0.0f;
      cm->jcell0[jadr].m[4*jj+3] = 0.0f;
	  
      cm->jcell0[jadr].q[0][4*jj+0] = 0.0f;
      cm->jcell0[jadr].q[0][4*jj+1] = 0.0f;
      cm->jcell0[jadr].q[0][4*jj+2] = 0.0f;
      cm->jcell0[jadr].q[0][4*jj+3] = 0.0f;
      cm->jcell0[jadr].q[1][4*jj+0] = 0.0f;
      cm->jcell0[jadr].q[1][4*jj+1] = 0.0f;
      cm->jcell0[jadr].q[1][4*jj+2] = 0.0f;
      cm->jcell0[jadr].q[1][4*jj+3] = 0.0f;
      cm->jcell0[jadr].q[2][4*jj+0] = 0.0f;
      cm->jcell0[jadr].q[2][4*jj+1] = 0.0f;
      cm->jcell0[jadr].q[2][4*jj+2] = 0.0f;
      cm->jcell0[jadr].q[2][4*jj+3] = 0.0f;
      cm->jcell0[jadr].q[3][4*jj+0] = 0.0f;
      cm->jcell0[jadr].q[3][4*jj+1] = 0.0f;
      cm->jcell0[jadr].q[3][4*jj+2] = 0.0f;
      cm->jcell0[jadr].q[3][4*jj+3] = 0.0f;
      cm->jcell0[jadr].q[4][4*jj+0] = 0.0f;
      cm->jcell0[jadr].q[4][4*jj+1] = 0.0f;
      cm->jcell0[jadr].q[4][4*jj+2] = 0.0f;
      cm->jcell0[jadr].q[4][4*jj+3] = 0.0f;
      cm->jcell0[jadr].q[5][4*jj+0] = 0.0f;
      cm->jcell0[jadr].q[5][4*jj+1] = 0.0f;
      cm->jcell0[jadr].q[5][4*jj+2] = 0.0f;
      cm->jcell0[jadr].q[5][4*jj+3] = 0.0f;
    }
  }
}

void g5c_calculate_force_on_xMC0(int devid, double (*x)[3], double (*a)[3], double *p, int ni){
  int off;
  int np = g5_get_number_of_pipelines();
  for(off=0;off<ni;off+=np) {
    int nii = np < ni-off ? np : ni-off;
    g5_set_xiMC(devid, nii, x+off);
    g5c_runMC0(devid);
    g5_get_forceMC(devid, nii, a+off, p+off);
    //g5c_add_forceMC(devid, nii, a+off, p+off);
  }
}


void g5c_runMC0(int devid) 
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
  struct Cell_Mem *cm = cell_mem + devid;
  void c_GravityKernel0(pIpdata, pFodata, cJcdata0, int);
  c_GravityKernel0(&(pm->iptcl), &(pm->fout), cm->jcell0, cm->ncell);
}

void g5c_set_xmjMC(int devid, int adr, int nj, double (*xj)[3], double *mj, double (*qj)[6]) 
{
  int j;
  //double zero = 0.0;
  //struct Ptcl_Mem *pm = ptcl_mem + devid;
  struct Cell_Mem *cm = cell_mem + devid;

  assert(adr % 2 == 0);
  for(j=adr;j<adr+nj;j+=2) {
    int jadr = j / 2;
    /*
    pm->jcell[jadr].xm[0][0] = xj[j][0];
    pm->jcell[jadr].xm[0][1] = xj[j][1];
    pm->jcell[jadr].xm[0][2] = xj[j][2];
    pm->jcell[jadr].xm[0][3] = mj[j];
    pm->jcell[jadr].q[0][0] = qj[j][0];
    pm->jcell[jadr].q[0][1] = qj[j][1];
    pm->jcell[jadr].q[0][2] = qj[j][2];
    pm->jcell[jadr].q[0][3] = 0.0;
    pm->jcell[jadr].q[0][4] = qj[j+1][0];
    pm->jcell[jadr].q[0][5] = qj[j+1][1];
    pm->jcell[jadr].q[0][6] = qj[j+1][2];
    pm->jcell[jadr].q[0][7] = 0.0;

    pm->jcell[jadr].xm[1][0] = xj[j+1][0];
    pm->jcell[jadr].xm[1][1] = xj[j+1][1];
    pm->jcell[jadr].xm[1][2] = xj[j+1][2];
    pm->jcell[jadr].xm[1][3] = mj[j+1];
    pm->jcell[jadr].q[1][0] = qj[j][3];
    pm->jcell[jadr].q[1][1] = qj[j][4];
    pm->jcell[jadr].q[1][2] = qj[j][5];
    pm->jcell[jadr].q[1][3] = 0.0;
    pm->jcell[jadr].q[1][4] = qj[j+1][3];
    pm->jcell[jadr].q[1][5] = qj[j+1][4];
    pm->jcell[jadr].q[1][6] = qj[j+1][5];
    pm->jcell[jadr].q[1][7] = 0.0;
    */
    
    __m256d pd = {xj[j][0], xj[j][1], xj[j][2], mj[j]};
    __m128  ps = _mm256_cvtpd_ps(pd);
    *(__m128 *)(cm->jcell[jadr].xm[0]) = ps;
    
    __m256d pd1 = {xj[j+1][0], xj[j+1][1], xj[j+1][2], mj[j+1]};
    __m128  ps1 = _mm256_cvtpd_ps(pd1);
    *(__m128 *)(cm->jcell[jadr].xm[1]) = ps1;

    __m256d pd2 = {qj[j][0], qj[j][1], qj[j][2], 0.0};
    __m128 ps2 = _mm256_cvtpd_ps(pd2);
    *(__m128 *)(cm->jcell[jadr].q[0]) = ps2;

    __m256d pd3 = {qj[j+1][0], qj[j+1][1], qj[j+1][2], 0.0};
    __m128 ps3 = _mm256_cvtpd_ps(pd3);
    *(__m128 *)(cm->jcell[jadr].q[0]+4) = ps3;

    __m256d pd4 = {qj[j][3], qj[j][4], qj[j][5], 0.0};
    __m128 ps4 = _mm256_cvtpd_ps(pd4);
    *(__m128 *)(cm->jcell[jadr].q[1]) = ps4;

    __m256d pd5 = {qj[j+1][3], qj[j+1][4], qj[j+1][5], 0.0};
    __m128 ps5 = _mm256_cvtpd_ps(pd5);
    *(__m128 *)(cm->jcell[jadr].q[1]+4) = ps5;
    
  }

  int rsdl = (NUNROLL - (nj % NUNROLL)) % NUNROLL;
  /*
  __m256d pd6 = {0.0, 0.0, 0.0, 0.0};
  __m128  ps6 = _mm256_cvtpd_ps(pd6);
  */
  __m128  ps6 = {0.0f, 0.0f, 0.0f, 0.0f};
  for(j=nj;j<nj+rsdl;j+=2){
    int jj, jadr = j / 2;
    for(jj = 0; jj < 2; jj++){
      int jp = jadr * 2 + jj;
      if(jp < nj)
	continue;
      *(__m128 *)(cm->jcell[jadr].xm[jj]) = ps6;
      //*(__m128 *)(pm->jcell[jadr].xm[1]) = ps6;
      *(__m128 *)(cm->jcell[jadr].q[0]+4*jj) = ps6;
      //*(__m128 *)(pm->jcell[jadr].q[0]+4) = ps6;
      *(__m128 *)(cm->jcell[jadr].q[1]+4*jj) = ps6;
      //*(__m128 *)(pm->jcell[jadr].q[1]+4) = ps6;
    }
  }
  
}

void g5c_calculate_force_on_xMC(int devid, double (*x)[3], double (*a)[3], double *p, int ni){
  int off;
  int np = g5_get_number_of_pipelines();
  for(off=0;off<ni;off+=np) {
    int nii = np < ni-off ? np : ni-off;
    g5_set_xiMC(devid, nii, x+off);
    g5c_runMC(devid);
    g5_get_forceMC(devid, nii, a+off, p+off);
    //g5c_add_forceMC(devid, nii, a+off, p+off);
  }
}

void g5c_runMC(int devid) 
{
  struct Ptcl_Mem *pm = ptcl_mem + devid;
  struct Cell_Mem *cm = cell_mem + devid;
  void c_GravityKernel(pIpdata, pFodata, cJcdata, int);
  c_GravityKernel(&(pm->iptcl), &(pm->fout), cm->jcell, cm->ncell);
}
