#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <gsl/gsl_rng.h>

#include <unistd.h>
#include <sys/times.h>

#define PI ((double) 3.14159265358979323846264338327950288419716939937510)

void drift_dan__(double *mu,
		 double *x0, double *y0, double *z0,
		 double *vx0, double *vy0,double *vz0,
		 double *dt0, int *iflg);

void drift_shep(double mu, double *r0, double *v0, double dt);
void drift_shep_nonrecurs(double mu, double *r0, double *v0, double dt);

double F(double E, double e, double M) {
	/* F(E) = E - e*sin(E) - M */
	return E - e*sin(E) - M;
}

double Fp(double E, double e){
	/* derivative of the above 
	 * F'(E) = 1 - e*cos(E) */
	return 1 - e*cos(E);
}

double Fpp(double E, double e) {
	/* second derivative of F 
	 * F''(E) = e*sin(E) */
	return e*sin(E);
}

double Fppp(double E, double e) {
	/* third derivative of F 
	 * F'''(E) = e*cos(E) */
	return e*cos(E);
}

static double dsignum(double x){
	if (x>0) return 1.0;
	else return -1.0;
}

double E_from_M_e(double M, double e){
	/* extension of Halley's method, based on Danby eq. 6.6.7*/
	double tol = 2e-6;
	double E, dE1, dE2, dE3;
	double f, fp, fpp, fppp;
//	int niter;

	E = M + dsignum(sin(M))*0.85*e;
//	niter = 0;
	do {
		f = F(E, e, M);
		fp = Fp(E, e);
		fpp = Fpp(E, e);
		fppp= Fppp(E, e);

		dE1 = -f/fp;
		dE2 = -f/(fp+1/2.0*dE1*fpp);
		dE3 = -f/(fp+1/2.0*dE2*fpp+1/6.0*dE2*dE2*fppp);
		E += dE3;
//		niter++;
	} while (fabs(dE3) > tol) ;

//	printf(" %d ", niter);
	return E;
}

void get_Cart_coords2(double mu, double a, 
		     double e, double cosI, double RAAN, double Ta, double Ap,
		     double *r, double *v, double *Period){
	/* this is for hyperbolic orbits.
	 * it gets true anomaly as an argument */
	double sinI;
	double cosO, sinO;
	double cosw, sinw;
	double cosf, sinf;
	double cosfw, sinfw;
	double rcoef, vcoef;

	sinI = sqrt(1-cosI*cosI);
	cosO = cos(RAAN); sinO = sin(RAAN);
	cosw = cos(Ap); sinw = sin(Ap);

	cosf = cos(Ta);
	sinf = sin(Ta);

	sinfw = sinf*cosw + cosf*sinw;
	cosfw = cosf*cosw - sinf*sinw;

	rcoef = a*(e*e-1)/(1+e*cosf);

	r[0] = rcoef*(cosO*cosfw - sinO*sinfw*cosI);
	r[1] = rcoef*(sinO*cosfw + cosO*sinfw*cosI);
	r[2] = rcoef*sinfw*sinI;

	vcoef = sqrt(mu/a/(e*e-1));

	v[0] = -vcoef*(cosO*(sinfw+e*sinw) + sinO*(cosfw+e*cosw)*cosI);
	v[1] = -vcoef*(sinO*(sinfw+e*sinw) - cosO*(cosfw+e*cosw)*cosI);
	v[2] =  vcoef*(cosfw+e*cosw)*sinI;

	*Period = 2*PI*sqrt(mu/(a*a*a)); /* for hyperbolic orbit, this is
			not actually period of course, but some timescale */
}

void get_Cart_coords(double mu, double a, 
		     double e, double cosI, double RAAN, double Ma, double Ap,
		     double *r, double *v, double *Period){
	double sinI;
	double cosO, sinO;
	double cosw, sinw;
	double cosf, sinf;
	double cosfw, sinfw;
	double rcoef, vcoef;
	double Ea;

	sinI = sqrt(1-cosI*cosI);
	cosO = cos(RAAN); sinO = sin(RAAN);
	cosw = cos(Ap); sinw = sin(Ap);
	Ea = E_from_M_e(Ma, e);
	cosf = (cos(Ea) - e)/(1-e*cos(Ea));
	sinf = sqrt(1-cosf*cosf);
	if ( sin(Ea)<0.0 ) sinf *= -1;

	sinfw = sinf*cosw + cosf*sinw;
	cosfw = cosf*cosw - sinf*sinw;

	rcoef = a*(1-e*cos(Ea));

	r[0] = rcoef*(cosO*cosfw - sinO*sinfw*cosI);
	r[1] = rcoef*(sinO*cosfw + cosO*sinfw*cosI);
	r[2] = rcoef*sinfw*sinI;

	vcoef = sqrt(mu/a/(1-e*e));

	v[0] = -vcoef*(cosO*(sinfw+e*sinw) + sinO*(cosfw+e*cosw)*cosI);
	v[1] = -vcoef*(sinO*(sinfw+e*sinw) - cosO*(cosfw+e*cosw)*cosI);
	v[2] =  vcoef*(cosfw+e*cosw)*sinI;

	*Period = 2*PI*sqrt(mu/(a*a*a));
}

#define DRIFT(mu, r, v)	drift_shep(mu, r, v, dt)
//#define DRIFT(mu, r, v)	drift_shep_nonrecurs(mu, r, v, dt)
//#define DRIFT(mu, r, v) drift_dan__(&mu, r, r+1, r+2, v, v+1, v+2, &dt, &iflg)

int main(int argc, char *argv[]){
	/* test values, lots of sampling for near parabolic orbits */
//	double eccs[] = {
//		0.001, 0.01,
//		0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9,
//		0.99, 0.999, 0.9999, 0.99999,
//		1.0-1e-6, 1.0-1e-7, 1.0-1e-8, 1.0-1e-9,
//		1.0+1e-9, 1.0+1e-8, 1.0+1e-7, 1.0+1e-6,
//		1.00001, 1.0001, 1.001, 1.01,
//		1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 3.0
//	};
	/* fewer test values */
	double eccs[] = {
		0.001, 0.01,
		0.1, 0.2, 0.5, 0.8, 0.9,
		0.99, 0.999, 
		1.001, 1.01,
		1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 3.0
	};

	int n_eccs;
	double a, ecc, Ma, Ta, RAAN, Ap, cosI;

	double X;
	int i, j, k, l, d;
	const gsl_rng_type *rng_type = gsl_rng_gfsr4;
        gsl_rng *rng = gsl_rng_alloc(rng_type);
	int rseed = 42;

	double r0[3], v0[3];
	double r[3], v[3];
	double Period, dt;
	double mu = 1.0; /* G*M_central */
	int iflg;

	double r0mag, beta, v0mag2;

	struct tms tmsbuf, tmsbufref;

        gsl_rng_set(rng, rseed);
//	/* semi-major axis */
	a = 1.0;

	n_eccs = sizeof(eccs)/sizeof(eccs[0]);
	for(i = 0; i<n_eccs; i++){
		ecc = eccs[i];
		printf("### ecc = %.8g\n", ecc);
		times(&tmsbufref);
		if (ecc<1) { /* elliptic */
			for(j=1; j<23; j++){ for(l=0; l<30; l++){
				Ma = (j+0.01)/23.0 * 2*PI;
				X = gsl_rng_uniform(rng); cosI = 2*X-1;
				X = gsl_rng_uniform(rng); RAAN = X*2*PI;
				X = gsl_rng_uniform(rng); Ap = X*2*PI;
				get_Cart_coords(a, mu, ecc, cosI, RAAN, Ma, Ap, r, v, &Period);
				for(d=0; d<3; d++){
					r0[d] = r[d];
					v0[d] = v[d];
				}
                        
                        
				/* integrate back and forth */
				X = gsl_rng_uniform(rng);
				dt = Period/7.0*(X+0.5);
//				r0mag = sqrt(r0[0]*r0[0] +r0[1]*r0[1] +r0[2]*r0[2]); 
//				v0mag2 = (v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]); 
//				beta = 2*mu/r0mag - v0mag2;
//				printf("beta = %e ", beta);
//				printf("Ma = %e ", Ma);
//				printf("dt = %e ", dt);
//				printf("\n");
//				printf("r0: %24.16e %24.16e %24.16e\n", r0[0], r0[1], r0[2]);
//				printf("v0: %24.16e %24.16e %24.16e\n", v0[0], v0[1], v0[2]);
				for(k=0; k<12000; k++){
					DRIFT(mu, r, v);
//					printf(" r: %24.16e %24.16e %24.16e\n", r[0], r[1], r[2]);
//					printf(" v: %24.16e %24.16e %24.16e\n", v[0], v[1], v[2]);
//					printf("\n");
				}
				for(d=0; d<3; d++){ v[d] = -v[d]; }
				for(k=0; k<12000; k++){
					DRIFT(mu, r, v);
//					printf(" r: %24.16e %24.16e %24.16e\n", r[0], r[1], r[2]);
//					printf(" v: %24.16e %24.16e %24.16e\n", v[0], v[1], v[2]);
//					printf("\n");
				}
//				printf(" r: %24.16e %24.16e %24.16e\n", r[0], r[1], r[2]);
//				printf(" v: %24.16e %24.16e %24.16e\n", v[0], v[1], v[2]);
				printf(" |dr| =  %e |dv| =  %e\n",
					fabs(hypot(hypot(r[0]-r0[0],r[1]-r0[1]),r[2]-r0[2])), 
					fabs(hypot(hypot(v[0]+v0[0],v[1]+v0[1]),v[2]+v0[2])));
			}}
		
		} else { /* hyperbolic */
			for(l=0; l<300; l++){
				Ta = -(7+0.01*gsl_rng_uniform(rng))*PI/13;
				X = gsl_rng_uniform(rng); cosI = 2*X-1;
				X = gsl_rng_uniform(rng); RAAN = X*2*PI;
				X = gsl_rng_uniform(rng); Ap = X*2*PI;
				get_Cart_coords2(a, mu, ecc, cosI, RAAN, Ta, Ap, r, v, &Period);
				for(d=0; d<3; d++){
					r0[d] = r[d];
					v0[d] = v[d];
				}
                        
				/* integrate back and forth */
				X = gsl_rng_uniform(rng);
				dt = Period/7.0*(X+0.5);
//				r0mag = sqrt(r0[0]*r0[0] +r0[1]*r0[1] +r0[2]*r0[2]); 
//				v0mag2 = (v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]); 
//				beta = 2*mu/r0mag - v0mag2;
//				printf("beta = %e ", beta);
//				printf("Ta = %e ", Ta);
//				printf("dt = %e ", dt);
//				printf("\n");
//				printf("r0: %24.16e %24.16e %24.16e\n", r0[0], r0[1], r0[2]);
//				printf("v0: %24.16e %24.16e %24.16e\n", v0[0], v0[1], v0[2]);
				for(k=0; k<120; k++){
					DRIFT(mu, r, v);
//					printf(" r: %24.16e %24.16e %24.16e\n", r[0], r[1], r[2]);
//					printf(" v: %24.16e %24.16e %24.16e\n", v[0], v[1], v[2]);
//					printf("\n");
				}
				for(d=0; d<3; d++){ v[d] = -v[d]; }
				for(k=0; k<120; k++){
					DRIFT(mu, r, v);
//					printf(" r: %24.16e %24.16e %24.16e\n", r[0], r[1], r[2]);
//					printf(" v: %24.16e %24.16e %24.16e\n", v[0], v[1], v[2]);
//					printf("\n");
				}
//				printf(" r: %24.16e %24.16e %24.16e\n", r[0], r[1], r[2]);
//				printf(" v: %24.16e %24.16e %24.16e\n", v[0], v[1], v[2]);
				printf(" |dr| =  %e |dv| =  %e\n",
					fabs(hypot(hypot(r[0]-r0[0],r[1]-r0[1]),r[2]-r0[2])), 
					fabs(hypot(hypot(v[0]+v0[0],v[1]+v0[1]),v[2]+v0[2])));
			}

		}
		times(&tmsbuf);
		printf("Usr time = %.6e ", (double)
		       	(tmsbuf.tms_utime-tmsbufref.tms_utime)/sysconf(_SC_CLK_TCK));
		printf("Sys time = %.6e\n", (double)
		       	(tmsbuf.tms_stime-tmsbufref.tms_stime)/sysconf(_SC_CLK_TCK));
		printf("Usr time (ch) = %.6e ", (double)
		       	(tmsbuf.tms_cutime-tmsbufref.tms_cutime)/sysconf(_SC_CLK_TCK));
		printf("Sys time (ch)= %.6e seconds\n", (double)
			(tmsbuf.tms_cstime-tmsbufref.tms_cstime)/sysconf(_SC_CLK_TCK));

		printf("\n");

	}

	return 0;
}
