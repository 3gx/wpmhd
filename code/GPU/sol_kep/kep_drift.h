#ifndef _KEP_DRIFT_H_
#define _KEP_DRIFT_H_

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// #include <cunistd>
#include <sys/times.h>


template<class REAL = double>
struct kep_drift {

#define DRIFT_FAIL -1
#define DRIFT_SUCCESS 0
  
#define PI ((REAL) 3.14159265358979323846264338327950288419716939937510)
  
  static REAL dsignum(REAL x){
    if (x>0) return 1.0;
    else return -1.0;
  }
  
  static REAL G_func2(REAL q) {
    int l = 3;
    int d = 15;
    int n = 0;
    REAL A, B, G;
    
    if(q==0.0) return 1.0;	/* this isn't necessary when first
				   Newt-Raph iteration is done by hand */
    
    A = B = G = 1.0;
    
    while (fabs(B/G)>1e-15) {
      l += 2;
      d += 4*l;
      n += 10*l;
      
      A = d/(d-n*A*q);
      B *= A-1.0;
      G += B;
      
      l += 2;
      d += 4*l;
      n -= 8*l;
      
      A = d/(d-n*A*q);
      B *= A-1.0;
      G += B;
    }
    
    return G;
  }
  
  static REAL init_u(REAL dt, REAL mu, REAL r0, REAL r0v0, REAL beta){
    /* XXX this routine is NOT used XXX */
    /* XXX note that this function uses sqrt() XXX */
    REAL T, phi, psi, bp2;
    REAL a, b, c, d, k;
    
    T = sqrt(mu/(r0*r0*r0))*dt;
    phi = r0v0/sqrt(r0);
    psi = T - phi*T*T/2 - (1-beta-3*phi*phi*phi)*T*T*T/6
      + phi*(10-9*beta-15*phi*phi)*T*T*T*T/24;  
    
    psi /= 4.0;
    bp2 = -beta*psi*psi;
	
    a = 1.0; 
    b = c = psi;
    d = 1.0;
    k = 1.0;
    do {
      a = 1.0/ (1 + bp2/(2*k-1)/(2*k+1)*a);
      b = (a-1.0)*b;
      c = c + b;
      k++;
    } while (fabs(b/c) > 1e-10 && k<6.0);
    
    return c;
  }
  
  static int lkep_drift3(REAL mu, REAL *r0, REAL *v0, REAL dt){
    /* XXX this routine is NOT used XXX */
    /* using laguarre's method */
    REAL r0mag, v0mag2;
    REAL r0v0;	/* r dot v */
    REAL rcalc, dtcalc, terr;
    REAL u;	/* (?) universal variable */
    REAL beta;	/* (?) vis-a-vis integral */
    REAL P;	/* period (for elliptic orbits only) */
    REAL dU;
    int n;
    REAL q;
    REAL U0w2, U1w2;
    REAL U, U0, U1, U2, U3;
    REAL f, g, F, G;
    REAL r1[3], v1[3];
    int d;
    REAL ln = 5.0;
    int no_iter;

    REAL du, dqdu, drdu, fn, fnp, fnpp;

    r0mag  = sqrt(r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2]);
    v0mag2 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    r0v0   = r0[0]*v0[0] + r0[1]*v0[1] + r0[2]*v0[2];
    beta   = 2*mu/r0mag - v0mag2;

    if (beta > 0) {	/* elliptic orbit */
      //		printf("Yes, there is a Santa Claus\n");
      P = 2*PI*mu/sqrt(beta*beta*beta);
      n = floor((dt + P/2.0 -2*r0v0/beta)/P);
      //		printf("n= %d\n", n);
      dU = 2*n*PI/sqrt(beta*beta*beta*beta*beta);
    } else {
      dU = 0.0;
    }

    //	u = 0;	/* a "better" guess is possible, see footnote at Battin p.219 */
    u = dt/(4*r0mag); 			/* N-R step by hand */
    //	u = dt/(4*r0mag-2*dt*r0v0/r0mag);
    //	u = init_u(dt, mu, r0mag, r0v0, beta);

    no_iter = 0;
    do {
      //		printf("u = %e\tq=%e\n", u,q);
      q = beta*u*u/(1+beta*u*u);
      if (q > 0.5 || no_iter > 12) return DRIFT_FAIL;
      dqdu = 2*beta*u/(1+beta*u*u)/(1+beta*u*u);
      U0w2 = 1 - 2*q;
      U1w2 = 2*(1-q)*u;
      U = 16.0/15 * U1w2*U1w2*U1w2*U1w2*U1w2 * G_func2(q) + dU;
      U0 = 2*U0w2*U0w2 - 1;
      U1 = 2*U0w2*U1w2;
      U2 = 2*U1w2*U1w2;
      U3 = beta*U + U1*U2/3.0;
      rcalc = r0mag*U0 + r0v0*U1 + mu*U2;
      drdu   = 4*(1-q)*(r0v0*U0 + (mu-beta*r0mag)*U1);
      dtcalc = r0mag*U1 + r0v0*U2 + mu*U3;

      fn    = dtcalc-dt;
      fnp   = 4*(1-q)*rcalc;
      fnpp  = 4*(drdu*(1-q) - rcalc*dqdu);

      du = -ln*fn / (fnp + dsignum(fnp)*
		     sqrt(fabs((ln-1)*(ln-1)*fnp*fnp - 
			       ln*(ln-1)*fn*fnpp) ) );
      u += du;
      no_iter++;

      terr = fabs((dt-dtcalc)/dt);
    } while (terr > 1e-10);

    f = 1 - (mu/r0mag)*U2;
    g = r0mag*U1 + r0v0*U2;
    F = -mu*U1/(rcalc*r0mag);
    G = 1 - (mu/rcalc)*U2;

    for(d=0; d<3; d++){
      r1[d] = f*r0[d] + g*v0[d];
      v1[d] = F*r0[d] + G*v0[d];
    }
    for(d=0; d<3; d++){
      r0[d] = r1[d];
      v0[d] = v1[d];
    }

    return DRIFT_SUCCESS;
  }
  
  static int nkep_drift3(REAL mu, REAL *r0, REAL *v0, REAL dt){
    REAL r0mag, v0mag2;
    REAL r0v0;	/* r dot v */
    REAL rcalc, dtcalc, terr;
    REAL u;	/* (?) universal variable */
    REAL beta;	/* (?) vis-a-vis integral */
    REAL P;	/* period (for elliptic orbits only) */
    REAL dU;
    int n;
    REAL q;
    REAL U0w2, U1w2;
    REAL U, U0, U1, U2, U3;
    REAL f, g, F, G;
    REAL r1[3], v1[3];
    int d;
    int no_iter;

    REAL du1, du2, du3, dqdu, d2qdu2, drdu, d2rdu2, fn, fnp, fnpp, fnppp;

    r0mag  = sqrt(r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2]);
    v0mag2 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    r0v0   = r0[0]*v0[0] + r0[1]*v0[1] + r0[2]*v0[2];
    beta   = 2*mu/r0mag - v0mag2;

    if (beta > 0) {	/* elliptic orbit */
      P = 2*PI*mu/sqrt(beta*beta*beta);
      n = floor((dt + P/2.0 -2*r0v0/beta)/P);
      dU = 2*n*PI/sqrt(beta*beta*beta*beta*beta);
    } else {
      dU = 0.0;
    }

    u = 0;	/* a "better" guess is possible, see footnote at Battin p.219 */
    //	u = dt/(4*r0mag); 				  /* N-R step by hand */
    //	u = dt/(4*r0mag-2*dt*r0v0/r0mag);
    //	u = init_u(dt, mu, r0mag, r0v0, beta);

    no_iter = 0;
    do {
      q = beta*u*u/(1+beta*u*u);
      if (q > 0.5 || no_iter > 12) return DRIFT_FAIL;
      dqdu = 2*beta*u/(1+beta*u*u)/(1+beta*u*u);
      d2qdu2 = 2*beta/(1+beta*u*u) 
	- 8*beta*beta*u*u / (1+beta*u*u)/(1+beta*u*u)/(1+beta*u*u);
      U0w2 = 1 - 2*q;
      U1w2 = 2*(1-q)*u;
      U = 16.0/15 * U1w2*U1w2*U1w2*U1w2*U1w2 * G_func2(q) + dU;
      U0 = 2*U0w2*U0w2 - 1;
      U1 = 2*U0w2*U1w2;
      U2 = 2*U1w2*U1w2;
      U3 = beta*U + U1*U2/3.0;
      rcalc = r0mag*U0 + r0v0*U1 + mu*U2;
      drdu   = 4*(1-q)*(r0v0*U0 + (mu-beta*r0mag)*U1);
      d2rdu2 = -4*dqdu*(r0v0*U0 + (mu-beta*r0mag)*U1)
	+ (4*(1-q)*4*(1-q))*(-beta*r0v0*U1 + (mu-beta*r0mag)*U0);
      dtcalc = r0mag*U1 + r0v0*U2 + mu*U3;

      fn    = dtcalc-dt;
      fnp   = 4*(1-q)*rcalc;
      fnpp  = 4*(drdu*(1-q) - rcalc*dqdu);
      fnppp = -8*drdu*dqdu - 4*rcalc*d2qdu2 + 4*(1-q)*d2rdu2;

      du1  = -fn/fnp;
      du2  = -fn/(fnp + du1*fnpp/2);
      du3  = -fn/(fnp + du2*fnpp/2 + du2*du2*fnppp/6);

      u += du3;
      no_iter++;

      terr = fabs((dt-dtcalc)/dt);
    } while (terr > 1e-15);
	
    f = 1 - (mu/r0mag)*U2;
    g = r0mag*U1 + r0v0*U2;
    F = -mu*U1/(rcalc*r0mag);
    G = 1 - (mu/rcalc)*U2;

    for(d=0; d<3; d++){
      r1[d] = f*r0[d] + g*v0[d];
      v1[d] = F*r0[d] + G*v0[d];
    }
    for(d=0; d<3; d++){
      r0[d] = r1[d];
      v0[d] = v1[d];
    }

    return DRIFT_SUCCESS;
  }
  
  static REAL c0(REAL z){
    if (z<0) {
      REAL sqrtz = sqrt(-z);
      return cosh(sqrtz);
    } else {
      REAL sqrtz = sqrt(z);
      return cos(sqrtz);
    }
  }

  static REAL c1(REAL z){
    if (z<0) {
      REAL sqrtz = sqrt(-z);
      return sinh(sqrtz)/sqrtz;
    } else if (z>0) {
      REAL sqrtz = sqrt(z);
      return sin(sqrtz)/sqrtz;
    } else {
      return 1.0;
    }
  }

  static REAL at1(REAL z){
    if (z<0) {
      REAL sqrtz = sqrt(-z);
      return 0.5*log((1+sqrtz)/(1-sqrtz))/sqrtz;
    } else if (z>0){
      REAL sqrtz = sqrt(z);
      return atan(sqrtz)/sqrt(z);
    } else {
      return 1.0;
    }
  }

  static int nkep_drift3_pert(REAL mu, REAL B2, REAL *r0, REAL *v0, REAL dt){
    REAL r0mag2, r0mag, v0mag2, r0dot,rdot;
    REAL r0v0;	/* r dot v */
    REAL rcalc, dtcalc, terr;
    REAL u;	/* (?) universal variable */
    REAL beta;	/* (?) vis-a-vis integral */
    REAL P;	/* period (for elliptic orbits only) */
    REAL dU;
    int n;
    REAL q;
    REAL U0w2, U1w2;
    REAL U, U0, U1, U2, U3;
    REAL f, g;
    REAL r1[3], v1[3];
    int d;
    int no_iter;

    REAL pthe[3], pthemag2, ppsimag2, theta2;
    REAL pthe_cro_r0[3], pthe_cro_r[3];
    REAL ypsi, xi;

    REAL du1, du2, du3, dqdu, d2qdu2, drdu, d2rdu2, fn, fnp, fnpp, fnppp;

    r0mag2   = r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2];
    r0mag    = sqrt(r0mag2);
    r0v0     = r0[0]*v0[0] + r0[1]*v0[1] + r0[2]*v0[2];
    r0dot    = r0v0/r0mag;	/* time derivative of distance, v_r */
    pthe[0]  = r0[1]*v0[2] - r0[2]*v0[1];
    pthe[1]  = r0[2]*v0[0] - r0[0]*v0[2];
    pthe[2]  = r0[0]*v0[1] - r0[1]*v0[0];
    pthemag2 = pthe[0]*pthe[0] + pthe[1]*pthe[1] + pthe[2]*pthe[2];
    ppsimag2 = pthemag2 - 2*B2;
    v0mag2   = r0dot*r0dot + ppsimag2/r0mag2;
    beta     = 2*mu/r0mag - v0mag2;

    if (beta > 0) {	/* elliptic orbit */
      P = 2*PI*mu/sqrt(beta*beta*beta);
      n = floor((dt + P/2.0 -2*r0v0/beta)/P);
      dU = 2*n*PI/sqrt(beta*beta*beta*beta*beta);
    } else {
      dU = 0.0;
    }

    u = 0;	/* a "better" guess is possible, see footnote at Battin p.219 */
    //	u = dt/(4*r0mag); 			/* N-R step by hand */
    //	u = dt/(4*r0mag-2*dt*r0v0/r0mag);
    //	u = init_u(dt, mu, r0mag, r0v0, beta);


    no_iter = 0;
    do {
      q = beta*u*u/(1+beta*u*u);
      if (q > 0.5 || no_iter > 12) return DRIFT_FAIL;
      dqdu = 2*beta*u/(1+beta*u*u)/(1+beta*u*u);
      d2qdu2 = 2*beta/(1+beta*u*u) 
	- 8*beta*beta*u*u / (1+beta*u*u)/(1+beta*u*u)/(1+beta*u*u);
      U0w2 = 1 - 2*q;
      U1w2 = 2*(1-q)*u;
      U = 16.0/15 * U1w2*U1w2*U1w2*U1w2*U1w2 * G_func2(q) + dU;
      U0 = 2*U0w2*U0w2 - 1;
      U1 = 2*U0w2*U1w2;
      U2 = 2*U1w2*U1w2;
      U3 = beta*U + U1*U2/3.0;
      rcalc = r0mag*U0 + r0v0*U1 + mu*U2;
      drdu   = 4*(1-q)*(r0v0*U0 + (mu-beta*r0mag)*U1);
      d2rdu2 = -4*dqdu*(r0v0*U0 + (mu-beta*r0mag)*U1)
	+ (4*(1-q)*4*(1-q))*(-beta*r0v0*U1 + (mu-beta*r0mag)*U0);
      dtcalc = r0mag*U1 + r0v0*U2 + mu*U3;

      fn    = dtcalc-dt;
      fnp   = 4*(1-q)*rcalc;
      fnpp  = 4*(drdu*(1-q) - rcalc*dqdu);
      fnppp = -8*drdu*dqdu - 4*rcalc*d2qdu2 + 4*(1-q)*d2rdu2;

      du1  = -fn/fnp;
      du2  = -fn/(fnp + du1*fnpp/2);
      du3  = -fn/(fnp + du2*fnpp/2 + du2*du2*fnppp/6);

      u += du3;
      no_iter++;

      terr = fabs((dt-dtcalc)/dt);
    } while (terr > 1e-15);

    f = 1 - (mu/r0mag)*U2;
    g = r0mag*U1 + r0v0*U2;

    xi = g / (r0mag2*f + r0v0*g);
    ypsi = xi*at1(ppsimag2*xi*xi);

    rdot = (r0v0*U0 + (mu-beta*r0mag)*U1)/rcalc;
    theta2 = pthemag2*ypsi*ypsi;

    pthe_cro_r0[0] = pthe[1]*r0[2] - pthe[2]*r0[1];
    pthe_cro_r0[1] = pthe[2]*r0[0] - pthe[0]*r0[2];
    pthe_cro_r0[2] = pthe[0]*r0[1] - pthe[1]*r0[0];

    {
      REAL ttt1 = rcalc/r0mag;
      REAL ttt2 = c0(theta2);      /* XXX probably it is more efficient   */
      REAL ttt3 = ypsi*c1(theta2); /* to calculate c1 and c0 together XXX */
      r1[0] = ttt1*(ttt2*r0[0] + ttt3*pthe_cro_r0[0]);
      r1[1] = ttt1*(ttt2*r0[1] + ttt3*pthe_cro_r0[1]);
      r1[2] = ttt1*(ttt2*r0[2] + ttt3*pthe_cro_r0[2]);
    }

    pthe_cro_r[0] = pthe[1]*r1[2] - pthe[2]*r1[1];
    pthe_cro_r[1] = pthe[2]*r1[0] - pthe[0]*r1[2];
    pthe_cro_r[2] = pthe[0]*r1[1] - pthe[1]*r1[0];

    v1[0] = rdot/rcalc*r1[0] + 1/(rcalc*rcalc)*pthe_cro_r[0];
    v1[1] = rdot/rcalc*r1[1] + 1/(rcalc*rcalc)*pthe_cro_r[1];
    v1[2] = rdot/rcalc*r1[2] + 1/(rcalc*rcalc)*pthe_cro_r[2];

    for(d=0; d<3; d++){
      r0[d] = r1[d];
      v0[d] = v1[d];
    }

    return DRIFT_SUCCESS;
  }



  static void drift_shep(REAL mu, REAL *r0, REAL *v0, REAL dt){
    /* this is purely N-R like acceleration, so does not use sqrt */
    if ( nkep_drift3(mu, r0, v0, dt) == DRIFT_FAIL){
      drift_shep(mu, r0, v0, dt/2);
      drift_shep(mu, r0, v0, dt/2);
    }
  }
  
  static void drift_shep_pert(REAL mu, REAL B2, REAL *r0, REAL *v0, REAL dt){
    /* this is purely N-R like acceleration, so does not use sqrt */
    if ( nkep_drift3_pert(mu, B2, r0, v0, dt) == DRIFT_FAIL){
      drift_shep_pert(mu, B2, r0, v0, dt/2);
      drift_shep_pert(mu, B2, r0, v0, dt/2);
    }
  }

  static void drift_shep_nonrecurs(REAL mu, REAL *r0, REAL *v0, REAL dt){
    /* this is purely N-R like acceleration (nkep_drift3() ),
     * so does not use sqrt */
    /* non-recursive version, more suitable to port to GPUS.
     * this can be made more similar to recursive version by
     * using block time step like scheme. */
    REAL at = 0.0;
    REAL step = dt;
    REAL dest = dt;
    int stat;

    while (at < dest) { /* potentially we will go over given dt */
      stat = nkep_drift3(mu, r0, v0, step);
      if (stat == DRIFT_FAIL) {
	step /= 2;
      } else {
	at += step;
	step = dest-at;
      }
    }
  }

};

#endif // _KEP_DRIFT_H_
