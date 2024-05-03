void avx_debugfunc(void);

void avx_open(int);

void avx_close(void);

void avx_set_j_particle(int,
			int,
			double,
			double,
			double *,
			double *,
			double *,
			double *);

void avx_set_ti(double);

void avx_initialize_neighbourlist(void);

int avx_get_neighbourlist_error(void);

int avx_get_neighbourlist(int,
			  int,
			  int *,
			  int *);

void avx_predict_j_particle(int);

void gravity_kernels(int,
		     double,
		     PG6::pPrdPosVel,
		     PG6::pNewAccJrk);

void gravity_kernel(int,
		    PG6::pPrdPosVel,
		    PG6::pNewAccJrk);

void gravity_kernel2(int,
		     PG6::pPrdPosVel,
		     PG6::pNewAccJrk);

void gravity_kerneln(int,
		     PG6::pPrdPosVel,
		     PG6::pNewAccJrk,
		     int,
		     int);

void gravity_kernel2n(int,
		      PG6::pPrdPosVel,
		      PG6::pNewAccJrk,
		      int,
		      int);
