#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "gp5util.h"
#include "./quad/quad_util.h"
#include <immintrin.h>
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif


#define SQR(x) ((x)*(x))
// #define DEBUG 1

#define G 1.0
#define M 1.0
#define R 1.0

class ilist;

// Contains particle data
class particle{
public:
  double pos[3]; // Position
  double v[3]; // Velocity
  double a[3]; // Force
  double phi; // Potential
  double m; // Mass
  particle *next;
};

// Contains one j-particle data
class jlist{
public:
  double (*xj)[3]; // Position
  double (*mj); // Mass
  int nj; // Number of particles

public:
  jlist(int n){ // constructor
    xj = new double[n][3]; 
    mj = new double[n];
  }

  ~jlist(){ // destructor
    delete[] xj;
    delete[] mj;
  }
};

class jcell{ // Contais one j-cell data
public:
  int nj; // Number of cells
  double (*xj)[3]; // Mass center
  double (*mj); // Total mass of a cell
  double (*qj)[6]; // Quadrupole tensor

public:
  jcell(int n){ // constructor
    xj = new double[n][3];
    mj = new double[n];
    qj = new double[n][6];
  }

  ~jcell(){ // destructor
    delete[] xj;
    delete[] mj;
    delete[] qj;
  }
};

// Contains cell data
class node{
public:
  double cpos[3]; // center
  double l; // length
  node * child[8]; // pointers to children
  particle * pfirst; // pointer to the first particle in the cell
  int nparticle; // number of particle in the cell
  double pos[3]; // center of mass
  double mass; // total mass
  double m_quad[6]; // quadrupole tensor

public:
  node(){
    cpos[0] = cpos[1] = cpos[2] = 0.0;
    l = 0.0;
    child[0] = child[1] = child[2] = child[3] = child[4] = child[5] = child[6] = child[7] = NULL;
    pfirst = NULL;
    nparticle = 0;
    pos[0] = pos[1] = pos[2] = 0.0;
    mass = 0.0;
  }
  void assign_root(double root_pos[3], double length, particle *p, int np);
  void create_tree_recursive(node * & heap_top, int & heap_remainder, int &n, int nleaf, int ncrit, int flag, ilist** i_list);
  void assign_child(int subindex, node * & heap_top, int & heap_remainder);
  int make_interaction_list(double ipos[3], double theta2, double xj[][3], double mj[], int nj, double li, int nleaf, jcell* j_cell, int &nj_cell);
  void calc_dist(double ipos[3], double theta2, double li, float dd[8]);
  void create_tree_nogc(node * & heap_top, int & heap_remainder, int &n, int nleaf0, node **leaflist);
  void create_para(node** heap_top, int * heap_remainder, int* ni, int nleaf, int ncrit, ilist*** i_list, int n);
  void set_cm_quantities(int nleaf);
  void setcm_solo(int nleaf);
  void set_quad_solo(int nleaf);
  void set_quad_moment(int nleaf);
};

//a cell that contains particles which share same interactions
class ilist{
public:
  int ni; // number of particles
  double l; // Cell's length
  double (*xi)[3]; // Position
  double (*ai)[3]; // Force
  double (*pi); // Potential
  particle*(*pp); // Pointer to particle
  double cpos[3]; // Cell's center

public:
  ilist(int nparticle){ // constructor
    ni = nparticle;
    xi = new double [nparticle][3];
    ai = new double [nparticle][3];
    pi = new double [nparticle];
    pp = new particle* [nparticle];
  }

  ~ilist(){ // destructor
    delete[] xi;
    delete[] ai;
    delete[] pi;
    delete[] pp;
  }
  void set_ilist(node* inode);
};

inline double getTime( double *now_time){
  struct timeval tv;
  gettimeofday( &tv, NULL);
  double diff_time;

  diff_time = (tv.tv_sec + (double)tv.tv_usec*1e-6) - *now_time;
  *now_time = (tv.tv_sec + (double)tv.tv_usec*1e-6);

  return diff_time;
}

void ilist::set_ilist(node* inode){
  particle *pnext = inode->pfirst;
  l = inode->l;
  cpos[0] = inode->cpos[0];
  cpos[1] = inode->cpos[1];
  cpos[2] = inode->cpos[2];
  
  for(int i = 0; i < inode->nparticle; i++){
    xi[i][0] = pnext->pos[0];
    xi[i][1] = pnext->pos[1];
    xi[i][2] = pnext->pos[2];
    pp[i] = pnext;
    pnext = pnext->next;
  }
}

void node::assign_root(double root_pos[3], double length, particle *p, int np){
  pos[0] = root_pos[0];
  pos[1] = root_pos[1];
  pos[2] = root_pos[2];
  l = length;
  pfirst = p;
  nparticle = np;
  for(int i = 0; i < np - 1; i++){
    p->next = p + 1;
    p++;
  }
  p->next = NULL;
}

int childindex(const double pos[3], const double cpos[3]){
  int subindex = 0;

  subindex <<= 1;
  if(pos[0] > cpos[0]){
    subindex += 1;
  }

  subindex <<= 1;
  if(pos[1] > cpos[1]){
    subindex += 1;
  }

  subindex <<= 1;
  if(pos[2] > cpos[2]){
    subindex += 1;
  }

  return subindex;
}

void node::assign_child(int subindex, node * & heap_top, int & heap_remainder){
  if(heap_remainder <= 0){
    fprintf(stderr, "create_tree: no more free node. exit\n");
    exit(1);
  }
  child[subindex] = heap_top;
  heap_top++;
  heap_remainder--;
  
  child[subindex]->cpos[0] = cpos[0] + ((subindex & 4)*0.5 - 1.0) * l * 0.25;
  child[subindex]->cpos[1] = cpos[1] + ((subindex & 2) - 1.0) * l * 0.25;
  child[subindex]->cpos[2] = cpos[2] + ((subindex & 1)*2.0 - 1.0) * l * 0.25;
  child[subindex]->l = l * 0.5;
  child[subindex]->nparticle = 0;
}


inline void set_quad_leaf(node *c){
  int i, j, k;
  int kk;
  double d[3];
  double dd;
  double q;

  // initialize leaf's quadrupole
  kk = 0;
  for(i = 0; i < 6; i++){
    c->m_quad[i] = 0.0;
  }
  // calculate leaf's quadrupole
  particle *pnext = c->pfirst;
  for(i = 0; i < c->nparticle; i++){
    d[0] = pnext->pos[0] - c->pos[0];
    d[1] = pnext->pos[1] - c->pos[1];
    d[2] = pnext->pos[2] - c->pos[2];
    dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    kk = 0;
    for(j = 0; j < 2; j++){
      for(k = j; k < 3; k++){
	q = 3.0 * d[j] * d[k];
	if(j == k){
	  q -= dd;
	}
	c->m_quad[kk] += pnext->m * q;
	kk++;
      }
    }
    pnext = pnext->next;
  }
  c->m_quad[5] = -(c->m_quad[0] + c->m_quad[3]);
}

void node::setcm_solo(int nleaf){
  int i;
  double mchild;
  if(nparticle > nleaf){
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 0.0;
    mass = 0.0;
    
    for(i = 0; i < 8; i++){
      if(child[i]->nparticle == 0){
	continue;
      }
      child[i]->setcm_solo(nleaf);
      mchild = child[i]->mass;
      pos[0] += mchild * child[i]->pos[0];
      pos[1] += mchild * child[i]->pos[1];
      pos[2] += mchild * child[i]->pos[2];
      mass += mchild;	
    }
    
    pos[0] /= mass;
    pos[1] /= mass;
    pos[2] /= mass;
  }
}

void node::set_quad_solo(int nleaf){
  int i, j, k;
  int kk;
  double q, dd;
  double d[3];

  if(nparticle > nleaf){
    for(i = 0; i < 6; i++){
      m_quad[i] = 0.0;
    }
    
    for(i = 0; i < 8; i++){
      if(child[i]->nparticle == 0){
	continue;
      }
      child[i]->set_quad_solo(nleaf);
      d[0] = child[i]->pos[0] - pos[0];
      d[1] = child[i]->pos[1] - pos[1];
      d[2] = child[i]->pos[2] - pos[2];
      dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      kk = 0;
      for(j = 0; j < 2; j++){
	for(k = j; k < 3; k++){
	  q = 3.0 * d[j] * d[k];
	  if(j == k){
	    q -= dd;
	  }
	  m_quad[kk] += child[i]->mass * q;
	  if(child[i]->nparticle > 1){
	    m_quad[kk] += child[i]->m_quad[kk];
	  }
	  kk++;
	}
      }
    }
    m_quad[5] = -(m_quad[0] + m_quad[3]);
  }
}

void node::set_quad_moment(int nleaf){
  int i, j, k;
  int kk;
  double q, dd;
  double d[3];

  if(nparticle > nleaf){
    for(i = 0; i < 6; i++){
      m_quad[i] = 0.0;
    }
    
#pragma omp parallel for private(d, dd, q, j, k, kk)
    for(i = 0; i < 8; i++){
      if(child[i]->nparticle == 0){
	continue;
      }
      child[i]->set_quad_solo(nleaf);
      d[0] = child[i]->pos[0] - pos[0];
      d[1] = child[i]->pos[1] - pos[1];
      d[2] = child[i]->pos[2] - pos[2];
      dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      kk = 0;
      for(j = 0; j < 2; j++){
	for(k = j; k < 3; k++){
	  q = 3.0 * d[j] * d[k];
	  if(j == k){
	    q -= dd;
	  }
	  m_quad[kk] += child[i]->mass * q;
	  if(child[i]->nparticle > 1){
	    m_quad[kk] += child[i]->m_quad[kk];
	  }
	  kk++;
	}
      }
    }
    m_quad[5] = -(m_quad[0] + m_quad[3]);
  }
}

void node::set_cm_quantities(int nleaf){
  int i;
  double mchild;
  if(nparticle > nleaf){
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 0.0;
    mass = 0.0;

#pragma omp parallel for private(mchild)
    for(i = 0; i < 8; i++){
      if(child[i]->nparticle == 0){
	continue;
      }
      child[i]->setcm_solo(nleaf);
      mchild = child[i]->mass;
      pos[0] += mchild * child[i]->pos[0];
      pos[1] += mchild * child[i]->pos[1];
      pos[2] += mchild * child[i]->pos[2];
      mass += mchild;	
    }
    
    pos[0] /= mass;
    pos[1] /= mass;
    pos[2] /= mass;
  }
}

void node::create_tree_nogc(node * & heap_top, int & heap_remainder, int &n, int nleaf0, node **leaflist){
  child[0] = child[1] = child[2] = child[3] = child[4] = child[5] = child[6] = child[7] = NULL;
  int i;
  particle *p = pfirst;
  particle *pnext;
  int subindex;
  
  for(i = 0; i < 8; i++){
    assign_child(i, heap_top, heap_remainder);
  }

  for(i = 0; i < nparticle; i++){
    pnext = p->next;
    subindex = childindex(p->pos, cpos);
    child[subindex]->nparticle++;
    p->next = child[subindex]->pfirst;
    child[subindex]->pfirst = p;
    p = pnext;
  }
  
  for(i = 0; i < 8; i++){
    if(child[i]->nparticle > nleaf0){
      child[i]->create_tree_nogc(heap_top, heap_remainder, n, nleaf0, leaflist);
    } else {
      if(child[i]->nparticle == 0){
	continue;
      }

      leaflist[n] = child[i];
      n++;
    }
  }
}

void node::create_para(node** heap_top, int * heap_remainder, int* ni, int nleaf, int ncrit, ilist*** i_list, int n){
  int i, k;
  int flag = 0;
  int nleaf0 = 300000;
  node *(*leaflist);
  leaflist = new node*[n];
  int ncell_leaf = 0;
  particle *pnext;

  create_tree_nogc(heap_top[0], heap_remainder[0], ncell_leaf, nleaf0, leaflist);

#pragma omp parallel for private(pnext, flag, k) schedule(dynamic)
  for(i = 0; i < ncell_leaf; i++){
    int devid = omp_get_thread_num();
    if(leaflist[i]->nparticle <= nleaf){
      i_list[devid][ni[devid]] = new ilist(leaflist[i]->nparticle);
      i_list[devid][ni[devid]]->set_ilist(leaflist[i]);
      ni[devid]++;
      
      /* calculate cell's gravity center*/
      pnext = leaflist[i]->pfirst;
      leaflist[i]->mass = 0.0;
      leaflist[i]->pos[0] = 0.0;
      leaflist[i]->pos[1] = 0.0;
      leaflist[i]->pos[2] = 0.0;

      for(k = 0; k < leaflist[i]->nparticle; k++){
	// calculate the child's mass and position
	leaflist[i]->mass += pnext->m;
	leaflist[i]->pos[0] += pnext->m * pnext->pos[0];
	leaflist[i]->pos[1] += pnext->m * pnext->pos[1];
	leaflist[i]->pos[2] += pnext->m * pnext->pos[2];
	pnext = pnext->next;
      }
      // finally, divide the pos by mass
      leaflist[i]->pos[0] /= leaflist[i]->mass;
      leaflist[i]->pos[1] /= leaflist[i]->mass;
      leaflist[i]->pos[2] /= leaflist[i]->mass;
      
      if(leaflist[i]->nparticle > 1){
	set_quad_leaf(leaflist[i]);
      }
      continue;
    }
    leaflist[i]->create_tree_recursive(heap_top[devid], heap_remainder[devid], ni[devid], nleaf, ncrit, flag, i_list[devid]);
  }
  
  setcm_solo(nleaf0);
  set_quad_solo(nleaf0);
  delete[] leaflist;
}

void node::create_tree_recursive(node * & heap_top, int & heap_remainder, int &n, int nleaf, int ncrit, int flag, ilist** i_list){
  child[0] = child[1] = child[2] = child[3] = child[4] = child[5] = child[6] = child[7] = NULL;
  int i, j;
  particle *p = pfirst;
  particle *pnext;
  int subindex;
  
  if(flag != 1 && nparticle <= ncrit){
    i_list[n] = new ilist(nparticle);
    i_list[n]->set_ilist(this);
    flag = 1;
    n++;
  }
  
  for(i = 0; i < 8; i++){
    assign_child(i, heap_top, heap_remainder);
  }
  for(i = 0; i < nparticle; i++){
    pnext = p->next;
    subindex = childindex(p->pos, cpos);
    child[subindex]->nparticle++;
    p->next = child[subindex]->pfirst;
    child[subindex]->pfirst = p;
    p = pnext;
  }
  
  mass = 0.0;
  pos[0] = 0.0;
  pos[1] = 0.0;
  pos[2] = 0.0;

  for(i = 0; i < 8; i++){
    if(child[i]->nparticle > nleaf){
      child[i]->create_tree_recursive(heap_top, heap_remainder, n, nleaf, ncrit, flag, i_list);
    } else {
      if(child[i]->nparticle == 0){
	continue;
      }
      if(flag != 1){
	i_list[n] = new ilist(child[i]->nparticle);
	i_list[n]->set_ilist(child[i]);
	n++;
      }
      pnext = child[i]->pfirst;
      child[i]->mass = 0.0;
      child[i]->pos[0] = 0.0;
      child[i]->pos[1] = 0.0;
      child[i]->pos[2] = 0.0;
	
      for(j = 0; j < child[i]->nparticle; j++){
	// calculate the child's mass and position
	child[i]->mass += pnext->m;
	child[i]->pos[0] += pnext->m * pnext->pos[0];
	child[i]->pos[1] += pnext->m * pnext->pos[1];
	child[i]->pos[2] += pnext->m * pnext->pos[2];
	pnext = pnext->next;
      }
      // finally, divide the pos by mass
      child[i]->pos[0] /= child[i]->mass;
      child[i]->pos[1] /= child[i]->mass;
      child[i]->pos[2] /= child[i]->mass;
      if(child[i]->nparticle > 1){
	set_quad_leaf(child[i]);
      }
    }
    mass += child[i]->mass;
    pos[0] += child[i]->mass * child[i]->pos[0];
    pos[1] += child[i]->mass * child[i]->pos[1];
    pos[2] += child[i]->mass * child[i]->pos[2];
  }
  pos[0] /= mass;
  pos[1] /= mass;
  pos[2] /= mass;

  int k;
  int kk;
  double q, dd;
  double d[3];

  for(i = 0; i < 6; i++){
    m_quad[i] = 0.0;
  }

  if(nparticle > nleaf){
    for(i = 0; i < 8; i++){
      if(child[i]->nparticle == 0){
	continue;
      }
      d[0] = child[i]->pos[0] - pos[0];
      d[1] = child[i]->pos[1] - pos[1];
      d[2] = child[i]->pos[2] - pos[2];
      dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      kk = 0;
      for(j = 0; j < 2; j++){
	for(k = j; k < 3; k++){
	  q = 3.0 * d[j] * d[k];
	  if(j == k){
	    q -= dd;
	  }
	  m_quad[kk] += child[i]->mass * q;
	  if(child[i]->nparticle > 1){
	    m_quad[kk]+=child[i]->m_quad[kk];
	  }
	  kk++;
	}
      }
    }
    m_quad[5] = -(m_quad[0] + m_quad[3]);
  }
}

void node::calc_dist(double ipos[3], double theta2, double li, float dd[8]){
  float fipos[3];
  fipos[0] = ipos[0];
  fipos[1] = ipos[1];
  fipos[2] = ipos[2];
  float child_x[8];
  float child_y[8];
  float child_z[8];
  float theta = theta2;
  float ll = li;
  float *ptr;

  __m256 zero = {0.0};
  __m256 ix;
  __m256 iy;
  __m256 iz;
  __m256 x = {0.0};
  __m256 y = {0.0};
  __m256 z = {0.0};
  __m256 d2 = {0.0};
  __m256 ftheta = _mm256_broadcast_ss(&theta);
  __m256 li_half = _mm256_broadcast_ss(&ll);
  __m256 mask;

  ix = _mm256_broadcast_ss(&fipos[0]);
  iy = _mm256_broadcast_ss(&fipos[1]);
  iz = _mm256_broadcast_ss(&fipos[2]);
  for(int i = 0; i < 8; i++){
    child_x[i] = child[i]->pos[0];
    child_y[i] = child[i]->pos[1];
    child_z[i] = child[i]->pos[2];
  }
  x = _mm256_load_ps(child_x);
  y = _mm256_load_ps(child_y);
  z = _mm256_load_ps(child_z);
  // dx,dy,dz
  x = _mm256_sub_ps(x, ix);
  y = _mm256_sub_ps(y, iy);
  z = _mm256_sub_ps(z, iz);

  // dx^2
  x = _mm256_mul_ps(x, x);
  y = _mm256_mul_ps(y, y);
  z = _mm256_mul_ps(z, z);
    
  x = _mm256_sqrt_ps(x);
  y = _mm256_sqrt_ps(y);
  z = _mm256_sqrt_ps(z);

  x = _mm256_sub_ps(x, li_half);
  y = _mm256_sub_ps(y, li_half);
  z = _mm256_sub_ps(z, li_half);

  mask = _mm256_cmp_ps(x, zero, 1);
  x = _mm256_blendv_ps(x, zero, mask);
  mask = _mm256_cmp_ps(y, zero, 1);
  y = _mm256_blendv_ps(y, zero, mask);
  mask = _mm256_cmp_ps(z, zero, 1);
  z = _mm256_blendv_ps(z, zero, mask);
    
  // d^2
  d2 = _mm256_mul_ps(x, x);
  d2 = _mm256_fmadd_ps(y, y, d2);
  d2 = _mm256_fmadd_ps(z, z, d2);

  // d^2 * theta2
  d2 = _mm256_mul_ps(d2, ftheta);

  ptr = (float*)&d2;
  dd[0] = ptr[0];
  dd[1] = ptr[1];
  dd[2] = ptr[2];
  dd[3] = ptr[3];
  dd[4] = ptr[4];
  dd[5] = ptr[5];
  dd[6] = ptr[6];
  dd[7] = ptr[7];
}

int node::make_interaction_list(double ipos[3], double theta2, double xj[][3], double mj[], int nj, double li, int nleaf, jcell* j_cell, int &nj_cell){
  particle* pnext;
  float dd[8] = {0.0};
  double child_l2;
  int i;

  if(nparticle <= nleaf){
    pnext = pfirst;
    for(i = 0; i < nparticle; i++){
      xj[nj][0] = pnext->pos[0];
      xj[nj][1] = pnext->pos[1];
      xj[nj][2] = pnext->pos[2];
      mj[nj] = pnext->m;
      pnext = pnext->next;
      nj++;
    }
  } else {
    calc_dist(ipos, theta2, li, dd);
    child_l2 = 0.25 * l * l;
    for(i = 0; i < 8; i++){
      if(child[i]->nparticle == 0){
	continue;
      }
      if(child[i]->nparticle == 1){
	xj[nj][0] = child[i]->pos[0];
	xj[nj][1] = child[i]->pos[1];
	xj[nj][2] = child[i]->pos[2];
	mj[nj] = child[i]->mass;
	nj++;
	continue;
      }
      if(dd[i] > child_l2){
        j_cell->xj[nj_cell][0] = child[i]->pos[0];
	j_cell->xj[nj_cell][1] = child[i]->pos[1];
	j_cell->xj[nj_cell][2] = child[i]->pos[2];
	j_cell->mj[nj_cell] = child[i]->mass;
	for(int j = 0; j < 6; j++){
	  j_cell->qj[nj_cell][j] = child[i]->m_quad[j];
	}
	nj_cell++;
      } else {
	nj = child[i]->make_interaction_list(ipos, theta2, xj, mj, nj, li, nleaf, j_cell, nj_cell);
      }
    } 
  }

  return nj;
}

double calculate_size(particle *p, const int n){
  double rsize = 1.0;
  double ppos[3];
  for(int i = 0; i < n; i++){
    ppos[0]=p[i].pos[0];
    ppos[1]=p[i].pos[1];
    ppos[2]=p[i].pos[2];

    if(fabs(ppos[0]) > rsize){
      rsize *= 2;
    }
    if(fabs(ppos[1]) > rsize){
      rsize *= 2;
    }
    if(fabs(ppos[2]) > rsize){
      rsize *= 2;
    }
  }
  return rsize;
}

int compare_double(const void *a, const void *b){
  if(*(double*)a > *(double*)b){
    return 1;
  } else if(*(double*)a < *(double*)b){
    return -1;
  } else return 0;
}

void calc_force(const int n, const int nnodes, particle pb[], node **bn, const double eps2, const double theta2, double *sum_time_calc, double *sum_time_search, double *sum_time_create, const int ncrit, const int nleaf, double *sum_time_alloc, const int devnum, const double epsinv){
  int *ni;
  ni = new int[devnum];
  int i, j, /*k,*/ devid;
  double now_time;
  double li;

  ilist ***i_list = new ilist**[devnum];
  for(i = 0; i < devnum; i++){
    ni[i] = 0;
    i_list[i] = new ilist*[n];
  }
  
  /*
    ilist **i_list;
    i_list = new ilist*[n];
  */

  double rsize = calculate_size(pb, n);
  double origin[3] = {0.0};
  bn[0]->assign_root(origin, rsize * 2, pb, n);

  int *heap_remainder = new int[devnum];
  heap_remainder[0] = nnodes - 1;
  for(i = 1; i < devnum; i++){
    heap_remainder[i] = nnodes;
  }

  node** heap_top = new node*[devnum];
  heap_top[0] = bn[0] + 1;
  for(i = 1; i < devnum; i++){
    heap_top[i] = bn[i];
  }
  
  //int heap_remainder = nnodes - 1;
  //node * btmp = bn + 1;
  
  getTime(&now_time);
  //ni = bn->create_tree_recursive(btmp, heap_remainder, 0, nleaf, ncrit, 0, i_list);
  bn[0]->create_para(heap_top, heap_remainder, ni, nleaf, ncrit, i_list, n);
  *sum_time_create += getTime(&now_time);

  double dx[3];
  double r2;
  
  getTime(&now_time);
  jlist **j_list;
  j_list = new jlist*[devnum];
  
  for(i = 0; i < devnum; i++){
    j_list[i] = new jlist(n);
  }

  jcell **j_cell;
  j_cell = new jcell*[devnum];
  for(i = 0; i < devnum; i++){
    j_cell[i] = new jcell(n);
  }
  *sum_time_alloc += getTime(&now_time); 

  ilist** t_ilist = new ilist*[n];
  int t_ni = 0;
  for(i = 0; i < devnum; i++){
    for(j = 0; j < ni[i]; j++){
      t_ilist[t_ni] = i_list[i][j];
      t_ni++;
    }
  }

  double t_force;
  double t_search;
  /*
  long long int num_interaction = 0;
  long long int num_approximate = 0;
  */
#pragma omp parallel for private(li, j, /*k,*/ dx, r2, devid/*,kk, kkk, ja, drab, dr5i, mr3i, drqdr, qdr, phi_p, phi_q*/, now_time) schedule(dynamic), reduction(max:t_force) reduction(max:t_search) /*reduction(+:num_interaction), reduction(+:num_approximate)*/
  for(i = 0; i < t_ni; i++){
    if(i == 0){
      t_force = 0.0;
      t_search = 0.0;
    }
    devid = omp_get_thread_num();
    getTime(&now_time);
    li = t_ilist[i]->l * 0.5;
    for(j = 0; j < 3; j++){
      dx[j] = bn[0]->pos[j] - t_ilist[i]->cpos[j];
      if(dx[j] < -li){
	dx[j] += li;
      } else if(dx[j] > li){
	dx[j] -= li;
      } else {
	dx[j] = 0;
      }
    }
    r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
    j_cell[devid]->nj = 0;
    if(r2*theta2 < bn[0]->l * bn[0]->l){
      j_list[devid]->nj = bn[0]->make_interaction_list(t_ilist[i]->cpos, theta2, j_list[devid]->xj, j_list[devid]->mj, 0, li, nleaf, j_cell[devid], j_cell[devid]->nj);
    } else {
      j_list[devid]->nj = 1;
      j_list[devid]->xj[0][0] = bn[0]->pos[0];
      j_list[devid]->xj[0][1] = bn[0]->pos[1];
      j_list[devid]->xj[0][2] = bn[0]->pos[2];
      j_list[devid]->mj[0] = bn[0]->mass;
    }
    t_search += getTime(&now_time);

    // monopole
    getTime(&now_time);
    g5_set_xmjMC(devid, 0, j_list[devid]->nj, j_list[devid]->xj, j_list[devid]->mj);
    g5_set_nMC(devid, j_list[devid]->nj);
    g5_calculate_force_on_xMC(devid, t_ilist[i]->xi, t_ilist[i]->ai, t_ilist[i]->pi, t_ilist[i]->ni);
    for(j = 0; j < t_ilist[i]->ni; j++){
      t_ilist[i]->pp[j]->a[0] = t_ilist[i]->ai[j][0];
      t_ilist[i]->pp[j]->a[1] = t_ilist[i]->ai[j][1];
      t_ilist[i]->pp[j]->a[2] = t_ilist[i]->ai[j][2];
      t_ilist[i]->pp[j]->phi = -(t_ilist[i]->pi[j]);
    }
    
    // quadrupole
    g5c_set_xmjMC(devid, 0, j_cell[devid]->nj, j_cell[devid]->xj, j_cell[devid]->mj, j_cell[devid]->qj);
    g5c_set_nMC(devid, j_cell[devid]->nj);
    g5c_calculate_force_on_xMC(devid, t_ilist[i]->xi, t_ilist[i]->ai, t_ilist[i]->pi, t_ilist[i]->ni);
   
    for(j = 0; j < t_ilist[i]->ni; j++){
      t_ilist[i]->pp[j]->a[0] += t_ilist[i]->ai[j][0];
      t_ilist[i]->pp[j]->a[1] += t_ilist[i]->ai[j][1];
      t_ilist[i]->pp[j]->a[2] += t_ilist[i]->ai[j][2];
      t_ilist[i]->pp[j]->phi += epsinv * t_ilist[i]->pp[j]->m + t_ilist[i]->pi[j];
    }
    /*
    num_interaction += t_ilist[i]->ni * j_list[devid]->nj;
    num_approximate += t_ilist[i]->ni * j_cell[devid]->nj;
    */
    t_force += getTime(&now_time);
    delete t_ilist[i];   
  }
  /*
  printf("#number of interactions = %lld\n", num_interaction);
  printf("#number of approximate = %lld\n", num_approximate);
  */
  *sum_time_search += t_search;
  *sum_time_calc += t_force;
  delete[] t_ilist;

  getTime(&now_time);
  for(i = 0; i < devnum; i++){
    delete j_list[i];
  }
  delete[] j_list;
  *sum_time_alloc += getTime(&now_time);
}

void leap_frog(const int n, const int nnodes, particle pb[], node **bn, const double dt, const double eps2, const double theta2, double *t_force, double *t_search, double *t_create, const int ncrit, const int nleaf, double *t_cf, double *sum_time_alloc, const int devnum, const double epsinv, const double halfdt){
  int i;
  double (*vhalf)[3];
  double now_time;
  vhalf = new double[n][3];

  // vhalf
  for(i = 0; i < n; i++){
    vhalf[i][0] = pb[i].v[0] + pb[i].a[0] * halfdt;
    vhalf[i][1] = pb[i].v[1] + pb[i].a[1] * halfdt;
    vhalf[i][2] = pb[i].v[2] + pb[i].a[2] * halfdt;
    pb[i].pos[0] += vhalf[i][0] * dt;
    pb[i].pos[1] += vhalf[i][1] * dt;
    pb[i].pos[2] += vhalf[i][2] * dt;
  }
    
  // a
  getTime(&now_time);
  calc_force(n, nnodes, pb, bn, eps2, theta2, t_force, t_search, t_create, ncrit, nleaf, sum_time_alloc, devnum, epsinv);
  *t_cf += getTime(&now_time);
    
  // v
  for(i = 0; i < n; i++){
    pb[i].v[0] = vhalf[i][0] + pb[i].a[0] * halfdt;
    pb[i].v[1] = vhalf[i][1] + pb[i].a[1] * halfdt;
    pb[i].v[2] = vhalf[i][2] + pb[i].a[2] * halfdt;
  }
  
  delete[] vhalf;
}

double kinetic_energy(particle *pp, const int n){
  double ke = 0.0;
  for(int i = 0; i < n; i++){
    ke += pp[i].m * (pp[i].v[0] * pp[i].v[0] + pp[i].v[1] * pp[i].v[1] + pp[i].v[2] * pp[i].v[2]);
  }
  return 0.5 * ke;
}

double potential_energy(particle *pp, const int n){
  double pe = 0.0;
  for(int i = 0; i < n; i++){
    pe += pp[i].phi * pp[i].m;
  }
  return 0.5 * pe;
}

void direct_force(const int n, particle pp[], const double eps2) {
  double epsinv;
  int i;
  double (*x)[3];
  x = new double[n][3];
  double (*a)[3];
  a = new double[n][3];
  double *p;
  p = new double[n];
  double *m;
  m = new double[n];
  double *aerror;
  aerror = new double[n];
  
  for(i = 0; i < n; i++){
    x[i][0] = pp[i].pos[0];
    x[i][1] = pp[i].pos[1];
    x[i][2] = pp[i].pos[2];
    p[i] = pp[i].phi;
    m[i] = pp[i].m;
  }

  g5_set_xmj(0, n, x, m);
  g5_set_eps_to_all(eps2);
  g5_set_n(n);
  g5_calculate_force_on_x(x, a, p, n);


  for(i = 0;i < n; i++){
    p[i] = -p[i];
  }
  
  if(eps2 != 0.0){
    epsinv = 1.0 / eps2;

    for(i = 0; i < n; i++){
      p[i] += m[i] * epsinv;
    }
  }

  for(i = 0; i < n; i++){
    aerror[i] = sqrt((SQR(a[i][0] - pp[i].a[0]) + SQR(a[i][1] - pp[i].a[1]) + SQR(a[i][2] - pp[i].a[2])) / (SQR(a[i][0]) + SQR(a[i][1]) + SQR(a[i][2])));
  }

  
  qsort(aerror, n, sizeof(double), compare_double);  
  for(i = 0; i < n; i++){
    fprintf(stdout, "%.20e %.20e\n", aerror[i], (i+1) / (double)(n));
  }
  

  /*
  for(i = 0; i < n; i++){
    fprintf(stdout, "%.20e %.20e %.20e %.20e\n", x[i][0], x[i][1], x[i][2], aerror[i]);
  }
  */
  
  delete[] x;
  delete[] a;
  delete[] p;
  delete[] m;
}

void readnbody(int *nj, particle *pp, char *fname){
  int i, dummy /*fi*/;
  int nn;
  double dummyd;
  FILE *fp;
  
  fp = fopen(fname, "r");
  if (fp == NULL)
    {
      perror("readnbody");
      exit(1);
    }
  /*fi =*/ fscanf(fp, "%d\n", &nn);
  if(nn != *nj){
    fprintf(stderr, "not matching number of particle\n");
    exit(1);
  }
  /*fi =*/ fscanf(fp, "%d\n", &dummy);
  /*fi =*/ fscanf(fp, "%lf\n", &dummyd);
  // /*fi =*/fscanf(fp, "%lf\n", pe);
  //pp = new particle[*nj];
  /*fi = */fprintf(stderr, "nj: %d\n", *nj);
  for (i = 0; i < *nj; i++)
    {
      /*fi = */fscanf(fp, "%lf\n", &pp[i].m);
    }
  for (i = 0; i < *nj; i++)
    {
      /*fi = */fscanf(fp, "%lf %lf %lf\n",
		      &pp[i].pos[0], &pp[i].pos[1], &pp[i].pos[2]);
    }
  for (i = 0; i < *nj; i++)
    {
      /*fi = */fscanf(fp, "%lf %lf %lf\n",
		      &pp[i].v[0], &pp[i].v[1], &pp[i].v[2]);
    }
}

void calc_force_nopg(const int n, const int nnodes, particle pb[], node **bn, const double eps2, const double theta2, double *sum_time_calc, double *sum_time_search, double *sum_time_create, const int ncrit, const int nleaf, double *sum_time_alloc, const int devnum, const double epsinv){
  double (*a)[3];
  a = new double[n][3];

  double *aerror;
  aerror = new double[n];

  for(int i = 0; i < n; i++){
    a[i][0] = pb[i].a[0];
    a[i][1] = pb[i].a[1];
    a[i][2] = pb[i].a[2];
  }

  int *ni;
  ni = new int[devnum];
  int i, j, k, devid;
  //double now_time;
  double li;
  double eps_sqr = eps2 * eps2;

  ilist ***i_list = new ilist**[devnum];
  for(i = 0; i < devnum; i++){
    ni[i] = 0;
    i_list[i] = new ilist*[n];
  }

  double rsize = calculate_size(pb, n);
  double origin[3] = {0.0};
  bn[0]->assign_root(origin, rsize * 2, pb, n);
 
  int *heap_remainder = new int[devnum];
  heap_remainder[0] = nnodes - 1;
  for(i = 1; i < devnum; i++){
    heap_remainder[i] = nnodes;
  }

  node** heap_top = new node*[devnum];
  heap_top[0] = bn[0] + 1;
  for(i = 1; i < devnum; i++){
    heap_top[i] = bn[i];
  }
  
  //int heap_remainder = nnodes - 1;
  //node * btmp = bn + 1;
  
  bn[0]->create_para(heap_top, heap_remainder, ni, nleaf, ncrit, i_list, n);

  double dx[3];
  double r2;
  
  jlist **j_list;
  j_list = new jlist*[devnum];
  
  for(i = 0; i < devnum; i++){
    j_list[i] = new jlist(n);
  }

  jcell **j_cell;
  j_cell = new jcell*[devnum];
  for(i = 0; i < devnum; i++){
    j_cell[i] = new jcell(n);
  }

  ilist** t_ilist = new ilist*[n];
  int t_ni = 0;
  for(i = 0; i < devnum; i++){
    for(j = 0; j < ni[i]; j++){
      t_ilist[t_ni] = i_list[i][j];
      t_ni++;
    }
  }

  int ja, kk;
  double drab, dr5i, mr3i, drqdr, phi_p, phi_q;
  double qdr[3];

#pragma omp parallel for private(li, j, k, dx, r2, devid, kk, /*kkk,*/ ja, drab, dr5i, mr3i, drqdr, qdr, phi_p, phi_q)
  for(i = 0; i < t_ni; i++){
    devid = omp_get_thread_num();

    li = t_ilist[i]->l * 0.5;
    for(j = 0; j < 3; j++){
      dx[j] = bn[0]->pos[j] - t_ilist[i]->cpos[j];
      if(dx[j] < -li){
	dx[j] += li;
      } else if(dx[j] > li){
	dx[j] -= li;
      } else {
	dx[j] = 0;
      }
    }
    r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
    j_cell[devid]->nj = 0;
    if(r2*theta2 < bn[0]->l * bn[0]->l){
      j_list[devid]->nj = bn[0]->make_interaction_list(t_ilist[i]->cpos, theta2, j_list[devid]->xj, j_list[devid]->mj, 0, li, nleaf, j_cell[devid], j_cell[devid]->nj);
    } else {
      j_list[devid]->nj = 1;
      j_list[devid]->xj[0][0] = bn[0]->pos[0];
      j_list[devid]->xj[0][1] = bn[0]->pos[1];
      j_list[devid]->xj[0][2] = bn[0]->pos[2];
      j_list[devid]->mj[0] = bn[0]->mass;
    }
    
    for(ja = 0; ja < t_ilist[i]->ni; ja++){
      t_ilist[i]->ai[ja][0] = 0.0;
      t_ilist[i]->ai[ja][1] = 0.0;
      t_ilist[i]->ai[ja][2] = 0.0;
      t_ilist[i]->pi[ja] = 0.0;
      for(k = 0; k < j_list[devid]->nj; k++){
	for(kk = 0; kk < 3; kk++){
	  dx[kk] = j_list[devid]->xj[k][kk] - t_ilist[i]->xi[ja][kk];
	}
	r2 = 1 / (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + eps_sqr);
	drab = sqrt(r2);
	mr3i = j_list[devid]->mj[k] * r2 * drab;
	t_ilist[i]->ai[ja][0] += mr3i * dx[0];
	t_ilist[i]->ai[ja][1] += mr3i * dx[1];
	t_ilist[i]->ai[ja][2] += mr3i * dx[2];
	t_ilist[i]->pi[ja] += j_list[devid]->mj[k] * drab;
      }
    }
    
    for(ja = 0; ja < t_ilist[i]->ni; ja++){
      for(k = 0; k < j_cell[devid]->nj; k++){
	for(kk = 0; kk < 3; kk++){
	  dx[kk] = j_cell[devid]->xj[k][kk] - t_ilist[i]->xi[ja][kk];
	}
	r2 = 1.0 / (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + eps_sqr);
	drab = sqrt(r2);
	phi_p = j_cell[devid]->mj[k] * drab;

	qdr[0] = j_cell[devid]->qj[k][0] * dx[0] + j_cell[devid]->qj[k][1] * dx[1] + j_cell[devid]->qj[k][2] * dx[2];
	qdr[1] = j_cell[devid]->qj[k][1] * dx[0] + j_cell[devid]->qj[k][3] * dx[1] + j_cell[devid]->qj[k][4] * dx[2];
	qdr[2] = j_cell[devid]->qj[k][2] * dx[0] + j_cell[devid]->qj[k][4] * dx[1] + j_cell[devid]->qj[k][5] * dx[2];
	  
	drqdr = dx[0] * qdr[0];
	drqdr += dx[1] * qdr[1];
	drqdr += dx[2] * qdr[2];

	dr5i = r2 * r2 * drab;
	phi_q = 0.5 * dr5i * drqdr;
	t_ilist[i]->pi[ja] += phi_p + phi_q;
	mr3i = (phi_p + 5.0 * phi_q) * r2;
	for(kk = 0; kk < 3; kk++){
	  t_ilist[i]->ai[ja][kk] += dx[kk] * mr3i - qdr[kk] * dr5i;
	}
      }
    }
      
    for(j = 0; j < t_ilist[i]->ni; j++){
      t_ilist[i]->pp[j]->a[0] = t_ilist[i]->ai[j][0];
      t_ilist[i]->pp[j]->a[1] = t_ilist[i]->ai[j][1];
      t_ilist[i]->pp[j]->a[2] = t_ilist[i]->ai[j][2];
      t_ilist[i]->pp[j]->phi = -(t_ilist[i]->pi[j]) + epsinv * t_ilist[i]->pp[j]->m;
    }
    delete t_ilist[i];   
  }
  delete[] t_ilist;

  for(i = 0; i < devnum; i++){
    delete j_list[i];
  }
  delete[] j_list;

  for(i = 0; i < n; i++){
    aerror[i] = sqrt((SQR(a[i][0] - pb[i].a[0]) + SQR(a[i][1] - pb[i].a[1]) + SQR(a[i][2] - pb[i].a[2])) / (SQR(a[i][0]) + SQR(a[i][1]) + SQR(a[i][2])));
  }
  
  qsort(aerror, n, sizeof(double), compare_double);
  for(i = 0; i < n; i++){
    fprintf(stdout, "%.20e %.20e\n", aerror[i], (i+1) / (double)(n));
  }
  
  delete[] a;
}


int main(int argc, char** argv) {
  int i;

  //double xmax = 10.0, xmin = -10.0, mmin;
  
  // number of particles
  int n;
  int nnodes;
  int ncrit;
  int nleaf;
  int devnum = omp_get_max_threads();
  printf("#number of threads: %d\n", devnum);

  // system energy, virial ratio
  double e_ini, e, r_v, einv;
  // time, timestep, end time, data interval
  double t = 0.0, dt, halfdt, t_end, t_out;

  double eps2; // squared softening parameter
  double epsinv = 0.0; // inverse of softening

  double theta2;

  // kinetic energy, potential energy
  double ke, pe;

  double now_time;
  double now_time_makeini;
  double t_force = 0.0;
  double t_lf = 0.0;
  double t_search = 0.0;
  double t_create = 0.0;
  double t_total = 0.0;
  double sum_time_alloc = 0.0;

  if(argc < 2){
    fprintf(stderr, "usage: ./source FILENAME");
  }
  fprintf(stderr, "N = ");
  scanf("%d", &n);
  printf("#N = %d\n", n);
  static particle *pp;
  pp = new particle[n];
  readnbody(&n, pp, argv[1]);

  // setting simulation parameters
  fprintf(stderr, "dt = ");
  scanf("%lf", &dt);
  halfdt = 0.5 * dt;
  printf("#dt = %g\n", dt);

  fprintf(stderr, "t_end = ");
  scanf("%lf", &t_end);
  printf("#t_end = %g\n", t_end);

  fprintf(stderr, "t_out = ");
  scanf("%lf", &t_out);
  printf("#t_out = %g\n", t_out);

  fprintf(stderr, "eps = ");
  scanf("%lf", &eps2);
  printf("#eps = %g\n", eps2);
  if(eps2 != 0.0){
    epsinv = 1.0 / eps2;
  }  

  fprintf(stderr, "theta = ");
  scanf("%lf", &theta2);
  printf("#theta = %f\n", theta2);
  theta2 = theta2 * theta2;

  fprintf(stderr, "ncrit = ");
  scanf("%d", &ncrit);
  printf("#ncrit = %d\n", ncrit);

  fprintf(stderr, "nleaf = ");
  scanf("%d", &nleaf);
  printf("#nleaf = %d\n", nleaf);

  getTime(&now_time);
  
  ke = kinetic_energy(pp, n);
  //mmin = pp[0].m;
  
  nnodes = n * 2 / devnum;
  static node **bn = new node*[devnum];
  for(i = 0; i < devnum; i++){
    bn[i] = new node[nnodes];
  }

  g5_open();
  //g5_set_range(xmin, xmax, mmin);
  g5_set_eps_to_all(eps2);

  getTime(&now_time_makeini);
  calc_force(n, nnodes, pp, bn, eps2, theta2, &t_force, &t_search, &t_create, ncrit, nleaf, &sum_time_alloc, devnum, epsinv);
  t_lf += getTime(&now_time_makeini);
  
  pe = potential_energy(pp, n);
  e = ke + pe;
  einv = 1.0 / e;

  while (t < t_end){
    // realtime analysis
    if(fmod(t, t_out) == 0.0){
      ke = kinetic_energy(pp, n);
      pe = potential_energy(pp, n);
      e_ini = ke + pe;
      r_v = ke / fabs(pe);
      printf("#%f %f %e\n", t, r_v, fabs((e_ini - e) * einv));
    }
    // time integration
    leap_frog(n, nnodes, pp, bn, dt, eps2, theta2, &t_force, &t_search, &t_create, ncrit, nleaf, &t_lf, &sum_time_alloc, devnum, epsinv, halfdt);
    t += dt;
  }
  g5_close();

  ke = kinetic_energy(pp, n);
  pe = potential_energy(pp, n);
  r_v = ke / fabs(pe);
  e_ini = ke + pe;

  
  printf("#%f %f %e\n", t, r_v, fabs((e_ini - e) * einv));
  // integration error check
  printf("#%f %f\n", e, e_ini);

  printf("#total calculation time: %f sec\n", t_lf);
  printf("#force calculation: %f sec\n", t_force);
  printf("#accumulate_force_from_tree: %f sec\n", t_search);
  printf("#create_tree_recursive: %f sec\n", t_create);
  //printf("#alloc_time : %f\n", sum_time_alloc);
  t_total = getTime(&now_time);
  //printf("#total time: %f\n", t_total);
  //printf("#calculation time per step: %e\n",t_lf / (t_end / dt + 1));
  //direct_force(n, pp, eps2);
  //calc_force_nopg(n, nnodes, pp, bn, eps2, theta2, &t_force, &t_search, &t_create, ncrit, nleaf, &sum_time_alloc, devnum, epsinv);
  /*
  for(i = 0; i < n; i++){
    printf("%.8f %.8f %.8f\n", pp[i].pos[0], pp[i].pos[1], pp[i].pos[2]);
  }
  */
  /*
  for(i = 0; i < n; i++){
    printf("%.8f %.8f %.8f\n", pp[i].a[0], pp[i].a[1], pp[i].a[2]);
  }
  */
  return 0;
}
