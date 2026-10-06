/* QUADPACK in C, from SciPy's scipy/integrate/__quadpack.c (a translation of the Fortran QUADPACK that
   scipy.integrate.quad calls). Tessera: the prototypes of scipy/integrate/__quadpack.h without the Python
   module glue (local modification, see THIRD_PARTY_NOTICES.md). */
#ifndef TSR_SCIPY_QUADPACK_H
#define TSR_SCIPY_QUADPACK_H
#ifdef __cplusplus
extern "C" {
#endif

void dqagie(double(*fcn)(double* x), const double bound, const int inf,
            const double epsabs, const double epsrel, const int limit, double* result,
            double* abserr, int* neval, int* ier, double* alist, double* blist,
            double* rlist, double* elist, int* iord, int* last);

void dqagpe(double(*fcn)(double* x), const double a, const double b, int npts2,
            double* points, const double epsabs, const double epsrel, const int limit,
            double* result, double* abserr, int* neval, int* ier, double* alist,
            double* blist, double* rlist, double* elist, double* pts, int* level,
            int* ndin, int* iord, int* last);

void dqagse(double(*fcn)(double* x), const double a, const double b,
            const double epsabs, const double epsrel, const int limit, double* result,
            double* abserr, int* neval, int* ier, double* alist, double* blist,
            double* rlist, double* elist, int* iord, int* last);

void dqawce(double(*fcn)(double* x), const double a, const double b, const double c,
            const double epsabs, const double epsrel, const int limit, double* result,
            double* abserr, int* neval, int* ier, double* alist, double* blist,
            double* rlist, double* elist, int* iord, int* last);

void dqawfe(double(*fcn)(double* x), const double a, const double omega, const int integr,
            const double epsabs, const int limlst, const int limit, const int maxp1,
            double* result, double* abserr, int* neval, int* ier, double* rslst,
            double* erlst, int* ierlst, int* lst, double* alist, double* blist,
            double* rlist, double* elist, int* iord, int* nnlog, double* chebmo);

void dqawoe(double(*fcn)(double* x), const double a, const double b, const double omega,
            const int integr, const double epsabs, const double epsrel, const int limit,
            const int icall, const int maxp1, double* result, double* abserr, int* neval,
            int* ier, int* last, double* alist, double* blist, double* rlist,
            double* elist, int* iord, int* nnlog, int* momcom, double* chebmo);

void dqawse(double(*fcn)(double* x), const double a, const double b, const double alfa,
            const double beta, const int integr, const double epsabs, const double epsrel,
            const int limit, double* result, double* abserr, int* neval, int* ier,
            double* alist, double* blist, double* rlist, double* elist, int* iord,
            int* last);

#ifdef __cplusplus
}
#endif
#endif
