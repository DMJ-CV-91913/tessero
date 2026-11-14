/*
 * linalg.lmdif and linalg.lmdif_result: MINPACK's lmdif and lmder, the Levenberg-Marquardt method with a QR-factored
 * step that scipy.optimize.leastsq, curve_fit and least_squares (method 'lm') use, for Optimize\LeastSquares.
 *
 * enorm, lmpar, qrfac and qrsolv at the end of this file are SciPy 1.17.1's C translation of MINPACK
 * (scipy/optimize/__minpack.c), copied unchanged; its licence and MINPACK's follow. lmdif is the same code (LMDIF
 * there, with nprint 0, mode 1 and fdjac2 inlined), cut at each function evaluation: no kernel function takes a
 * callback (tessero.h), so the caller evaluates the model at the points lmdif asks for and calls again with the
 * values. The iteration state travels between the calls as a float64 array, checked on every call. LMDER differs
 * from LMDIF only in where the Jacobian comes from and in counting it in njev, not nfev: here the caller's (leastsq
 * with Dfun) or SciPy's 2-point differences (least_squares with method 'lm', whose jac is approx_derivative's).
 *
 * Built with multiply-add contraction off (csrc/Makefile NOCONTRACT_C): each product rounded on its own, as SciPy's
 * x86-64 wheels compute it, so the fit follows SciPy's.
 */

/*
 * Copyright (C) 2024 SciPy developers
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * b. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * c. Names of the SciPy Developers may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *
 * This file is a C translation of the Fortran code written by Jorge J. More,
 * Burton S. Garbow, and Kenneth E. Hillstrom, with the original description
 * below. Original Fortran docstrings are included at the top of each function.
 *
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Minpack includes software for solving nonlinear equations and nonlinear
 * least squares problems.  Five algorithmic paths each include a core
 * subroutine and an easy-to-use driver.  The algorithms proceed either from
 * an analytic specification of the Jacobian matrix or directly from the
 * problem functions.  The paths include facilities forsystems of equations
 * with a banded Jacobian matrix, for least squares problems with a large
 * amount of data, and for checking the consistency of the Jacobian matrix
 * with the functions.
 *
 *
 * Disclaimer:
 * Minpack Copyright Notice (1999) University of Chicago.  All rights reserved
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the
 * following conditions are met:
 *
 * 1. Redistributions of source code must retain the above
 * copyright notice, this list of conditions and the following
 * disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials
 * provided with the distribution.
 *
 * 3. The end-user documentation included with the
 * redistribution, if any, must include the following
 * acknowledgment:
 *
 *    "This product includes software developed by the
 *    University of Chicago, as Operator of Argonne National
 *    Laboratory.
 *
 * Alternately, this acknowledgment may appear in the software
 * itself, if and wherever such third-party acknowledgments
 * normally appear.
 *
 * 4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS"
 * WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDER, THE
 * UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND
 * THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE
 * OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY
 * OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
 * USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF
 * THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4)
 * DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION
 * UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL
 * BE CORRECTED.
 *
 * 5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT
 * HOLDER, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF
 * ENERGY, OR THEIR EMPLOYEES: BE LIABLE FOR ANY INDIRECT,
 * INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF
 * ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
 * PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER
 * SUCH LIABILITY IS ASSERTED ON THE BASIS OF CONTRACT, TORT
 * (INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR OTHERWISE,
 * EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE
 * POSSIBILITY OF SUCH LOSS OR DAMAGES.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * The original Fortran code can be found at https://www.netlib.org/minpack/
 *
 * References:
 *
 * [1]: J. J. More, B. S. Garbow, and K. E. Hillstrom, User Guide for
 *      MINPACK-1, Argonne National Laboratory Report ANL-80-74, Argonne,
 *      Illinois, 1980.
 *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "fn.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double enorm(const int,const double*);
static void lmpar(const int,double *,const int,const int*,const double*,const double*,const double,double*,double*,double*,double*,double*);
static void qrfac(const int,const int,double*,const int,const int,int*,double*,double*,double*);
static void qrsolv(const int,double*,const int,const int*,const double*,const double*,double*,double*,double*);

static const double dpmpar[3] = {
    2.220446049250313e-16,    /* np.finfo(np.float64).eps  */
    2.2250738585072014e-308,  /* np.finfo(np.float64).tiny */
    1.7976931348623157e+308,  /* np.finfo(np.float64).max  */
};

/* ---- the state array: a header of S_HEAD doubles (integers held exactly), then x, diag, qtf, wa1, wa2, wa3, the
   difference steps h and ipvt (n each), fvec, wa4, y and w (m each) and fjac (m x n, column-major as MINPACK's).
   m is 0 until the first values arrive when no y was given. resume: 0 finished, 1 the values at x0, 2 the values
   at the n points of the forward-difference Jacobian (jac 1: the Jacobian at x), 3 the values at the trial point
   x + p (wa2). jac: 0 lmdif's differences, 1 the caller's Jacobian (lmder), 2 approx_derivative's 2-point
   differences (lmder as least_squares runs it). */
#define LM_MAGIC 20261114.0
enum { S_MAGIC, S_M, S_N, S_MAXFEV, S_NFEV, S_INFO, S_ITER, S_RESUME, S_HASY, S_HASW,
       S_FTOL, S_XTOL, S_GTOL, S_EPSFCN, S_FACTOR,
       S_ACTRED, S_DELTA, S_DIRDER, S_FNORM, S_FNORM1, S_GNORM, S_PAR, S_PNORM, S_PRERED, S_RATIO, S_XNORM,
       S_JAC, S_NJEV,
       S_HEAD = 32 };

typedef struct {
    int m, n;
    double *hd, *x, *diag, *qtf, *wa1, *wa2, *wa3, *h, *ipv, *fvec, *wa4, *y, *w, *fjac;
} lm_t;

/* m * n stays an int index (MINPACK's) and the whole state below 2^31 doubles */
static int lm_fits(int64_t m, int64_t n)
{
    if (n < 1 || n > INT_MAX || m < 0 || m > INT_MAX) return 0;
    if (m > 0 && n > (int64_t)INT_MAX / m) return 0;
    return S_HEAD + 8 * n + m * (n + 4) <= ((int64_t)1 << 31);
}

static int64_t lm_len(int64_t m, int64_t n) { return S_HEAD + 8 * n + m * (n + 4); }

static void lm_bind(lm_t *L, double *s, int m, int n)
{
    double *p = s + S_HEAD;
    L->m = m; L->n = n; L->hd = s;
    L->x = p; p += n; L->diag = p; p += n; L->qtf = p; p += n; L->wa1 = p; p += n; L->wa2 = p; p += n;
    L->wa3 = p; p += n; L->h = p; p += n; L->ipv = p; p += n;
    L->fvec = p; p += m; L->wa4 = p; p += m; L->y = p; p += m; L->w = p; p += m; L->fjac = p;
}

/* v is an integer in [lo, hi] (held exactly in a double) */
static int lm_int(double v, double lo, double hi) { return v == floor(v) && v >= lo && v <= hi; }

/* a header lmdif wrote: 1, or 0 with the error set */
static int lm_check(const double *s, int64_t len, int *m, int *n)
{
    if (len < S_HEAD || s[S_MAGIC] != LM_MAGIC || !lm_int(s[S_N], 1, INT_MAX) || !lm_int(s[S_M], 0, INT_MAX)
        || !lm_fits((int64_t)s[S_M], (int64_t)s[S_N]) || len != lm_len((int64_t)s[S_M], (int64_t)s[S_N])
        || !lm_int(s[S_RESUME], 0, 3) || !lm_int(s[S_MAXFEV], 1, INT_MAX) || !lm_int(s[S_NFEV], 0, INT_MAX)
        || !lm_int(s[S_ITER], 0, INT_MAX - 1) || !lm_int(s[S_INFO], 0, 8) || !lm_int(s[S_HASY], 0, 1)
        || !lm_int(s[S_HASW], 0, 1) || !lm_int(s[S_JAC], 0, 2) || !lm_int(s[S_NJEV], 0, INT_MAX) || (s[S_M] == 0 && (s[S_RESUME] != 1 || s[S_HASY] != 0))
        || (s[S_M] > 0 && s[S_M] < s[S_N])) {
        fn_set_error("lmdif: state must be the array lmdif returned");
        return 0;
    }
    *m = (int)s[S_M];
    *n = (int)s[S_N];
    return 1;
}

/* ipvt from the state: integers in [0, n) once qrfac has set them (resume 3, finished), else 0 .. n-1 */
static int lm_ipvt(const lm_t *L, int *ipvt, int set)
{
    for (int j = 0; j < L->n; j++) {
        if (!set) { ipvt[j] = j; continue; }
        if (!lm_int(L->ipv[j], 0, L->n - 1)) { fn_set_error("lmdif: state must be the array lmdif returned"); return 0; }
        ipvt[j] = (int)L->ipv[j];
    }
    return 1;
}

/* residual i of evaluation j of F: f - y, times w (curve_fit's transform * (f(xdata, *p) - ydata)), or f */
static double lm_r(const lm_t *L, const double *F, int fcols, int k, int j, int i)
{
    double r = fcols ? F[(int64_t)i * k + j] : F[(int64_t)j * L->m + i];
    if (L->hd[S_HASY] != 0.0) {
        r = r - L->y[i];
        if (L->hd[S_HASW] != 0.0) r = L->w[i] * r;
    }
    return r;
}

/* LMDIF from the evaluation the state waits for to the next one (or the end); F holds k evaluations of m values */
static void lm_run(lm_t *L, int *ipvt, const double *F, int fcols, int k)
{
    double *hd = L->hd;
    const int m = L->m, n = L->n, ldfjac = m, mode = 1;
    double *x = L->x, *fvec = L->fvec, *fjac = L->fjac, *diag = L->diag, *qtf = L->qtf;
    double *wa1 = L->wa1, *wa2 = L->wa2, *wa3 = L->wa3, *wa4 = L->wa4;
    const double ftol = hd[S_FTOL], xtol = hd[S_XTOL], gtol = hd[S_GTOL], factor = hd[S_FACTOR];
    const int maxfev = (int)hd[S_MAXFEV], jac = (int)hd[S_JAC];
    int i, j, l, resume = (int)hd[S_RESUME];
    int nfev = (int)hd[S_NFEV], info = (int)hd[S_INFO], iter = (int)hd[S_ITER], njev = (int)hd[S_NJEV];
    double actred = hd[S_ACTRED], delta = hd[S_DELTA], dirder = hd[S_DIRDER], fnorm = hd[S_FNORM];
    double fnorm1 = hd[S_FNORM1], gnorm = hd[S_GNORM], par = hd[S_PAR], pnorm = hd[S_PNORM];
    double prered = hd[S_PRERED], ratio = hd[S_RATIO], xnorm = hd[S_XNORM];
    double ssum, temp, temp1, temp2;
    const double epsmch = dpmpar[0];

    if (resume == 2) goto R2;
    if (resume == 3) goto R3;

    // Evaluate the function at the starting point and calculate its norm.
    for (i = 0; i < m; i++) fvec[i] = lm_r(L, F, fcols, k, 0, i);
    nfev = 1;

    fnorm = enorm(m, fvec);

    // Initialize Levenberg-Marquardt parameter and iteration counter.
    par = 0.0;
    iter = 1;

    // Beginning of the outer loop
OUTER:
    // Calculate the jacobian matrix (fdjac2: the caller evaluates the function at x + h e_j, j = 0 .. n-1).
    if (jac == 2)
    {
        // approx_derivative, 2-point: h = EPS**0.5 * sign(x) * max(1, |x|), sign(0) = 1 (_compute_absolute_step)
        const double rstep = sqrt(epsmch);
        for (j = 0; j < n; j++) { L->h[j] = (rstep * (x[j] >= 0.0 ? 1.0 : -1.0)) * fmax(1.0, fabs(x[j])); }
    }
    else if (jac == 0)
    {
        const double eps = sqrt(fmax(hd[S_EPSFCN], epsmch));
        for (j = 0; j < n; j++)
        {
            temp = x[j];
            double h = eps*fabs(temp);
            if (h == 0.0) { h = eps; }
            L->h[j] = h;
        }
    }
    resume = 2;
    goto SAVE;

R2:
    if (jac == 1)
    {
        // the caller's Jacobian, (m, n) rows (fcols: (n, m)), times curve_fit's transform when sigma was given
        for (j = 0; j < n; j++)
        {
            for (i = 0; i < m; i++)
            {
                const double d = fcols ? F[(int64_t)j * m + i] : F[(int64_t)i * n + j];
                fjac[i + ldfjac*j] = hd[S_HASW] != 0.0 ? L->w[i] * d : d;
            }
        }
        njev += 1;
    }
    else if (jac == 2)
    {
        // (f(x + h e_j) - f(x)) / dx with dx = (x_j + h_j) - x_j, as _dense_difference; lmder counts it in njev
        for (j = 0; j < n; j++)
        {
            const double dx = (x[j] + L->h[j]) - x[j];
            for (i = 0; i < m; i++)
            {
                fjac[i + ldfjac*j] = (lm_r(L, F, fcols, k, j, i) - fvec[i])/dx;
            }
        }
        njev += 1;
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            const double h = L->h[j];
            for (i = 0; i < m; i++)
            {
                fjac[i + ldfjac*j] = (lm_r(L, F, fcols, k, j, i) - fvec[i])/h;
            }
        }
        nfev += n;
    }

    // Compute the qr factorization of the jacobian.
    qrfac(m, n, fjac, ldfjac, 1, ipvt, wa1, wa2, wa3);

    // On the first iteration and if mode is 1, scale according to the norms
    // of the columns if the initial jacobian.

    if (iter == 1)
    {
        if (mode != 2)
        {
            for (j = 0; j < n; j++)
            {
                diag[j] = wa2[j];
                if (wa2[j] == 0.0) { diag[j] = 1.0; }
            }
            // 50
        }
        // 60

        // On the first iteration, calculate the norm of the scaled x and
        // initialize the step bound delta.
        for (j = 0; j < n; j++)
        {
            wa3[j] = diag[j]*x[j];
        }
        // 70

        xnorm = enorm(n, wa3);
        delta = factor*xnorm;
        delta = (delta == 0.0 ? factor : delta);
    }
    // 80

    // Form (q.T)*fvec and store the first n components in qtf.
    for (i = 0; i < m; i++)
    {
        wa4[i] = fvec[i];
    }
    // 90

    for (j = 0; j < n; j++)
    {
        if (fjac[j + ldfjac*j] != 0.0)
        {
            ssum = 0.0;
            for (i = j; i < m; i++)
            {
                ssum += fjac[i + ldfjac*j]*wa4[i];
            }
            // 100
            temp = -ssum / fjac[j + ldfjac*j];
            for (i = j; i < m; i++)
            {
                wa4[i] += fjac[i + ldfjac*j]*temp;
            }
            // 110
        }
        // 120

        fjac[j + ldfjac*j] = wa1[j];
        qtf[j] = wa4[j];
    }
    // 130

    // Compute the norm of the scaled gradient.

    gnorm = 0.0;
    if (fnorm != 0.0)
    {
        for (j = 0; j < n; j++) {
            l = ipvt[j];
            if (wa2[l] != 0.0)
            {
                ssum = 0.0;
                for (i = 0; i <= j; i++)
                {
                    ssum += fjac[i + ldfjac*j] * (qtf[i] / fnorm);
                }
                // 140
                gnorm = fmax(gnorm, fabs(ssum / wa2[l]));
            }
            // 150
        }
        // 160
    }
    // 170

    // Test for convergence of the gradient norm.
    if (gnorm <= gtol) { info = 4; }
    if (info != 0) { goto DONE; }

    // Rescale if necessary
    if (mode != 2)
    {
        for (j = 0; j < n; j++)
        {
            diag[j] = fmax(diag[j], wa2[j]);
        }
        // 180
    }
    // 190

    // Beginning of the inner loop
INNER:
    // Determine the Levenberg-Marquardt parameter
    lmpar(n, fjac, ldfjac, ipvt, diag, qtf, delta, &par, wa1, wa2, wa3, wa4);

    // Store the direction p and x+p. Calculate the norm of p.
    for (j = 0; j < n; j++)
    {
        wa1[j] = -wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j]*wa1[j];
    }
    // 210
    pnorm = enorm(n, wa3);

    // On the first iteration, adjust the initial step bound.
    if (iter == 1) { delta = fmin(delta, pnorm); }

    // Evaluate the function at x+p (the caller) and calculate its norm.
    resume = 3;
    goto SAVE;

R3:
    for (i = 0; i < m; i++) wa4[i] = lm_r(L, F, fcols, k, 0, i);
    nfev += 1;
    fnorm1 = enorm(m, wa4);

    // Compute the scaled actual reduction.
    actred = (0.1*fnorm1 < fnorm ? 1.0 - pow(fnorm1/fnorm, 2.0) : -1.0);

    // Compute the scaled predicted reduction and the scaled directional
    // derivative.
    for (j = 0; j < n; j++)
    {
        wa3[j] = 0.0;
        l = ipvt[j];
        temp = wa1[l];
        for (i = 0; i <= j; i++)
        {
            wa3[i] += fjac[i + ldfjac*j]*temp;
        }
        // 220
    }
    // 230

    temp1 = enorm(n, wa3)/fnorm;
    temp2 = (sqrt(par)*pnorm)/fnorm;
    prered = temp1*temp1 + temp2*temp2/0.5;
    dirder = -(temp1*temp1 + temp2*temp2);

    // Compute the ratio of the actual to the predicted reduction.
    ratio = ( prered != 0.0 ? actred/prered : 0.0);

    // Update the step bound
    if (ratio <= 0.25)
    {
        temp = (actred < 0.0 ? (0.5*dirder)/(dirder + 0.5*actred): 0.5);
        if ((0.1*fnorm1 >= fnorm) || (temp < 0.1)) { temp = 0.1; }
        delta = temp*fmin(delta, pnorm/0.1);
        par /= temp;
    } else {
        // 240
        if ((par == 0.0) || (ratio >= 0.75))
        {
            delta = pnorm/0.5;
            par *= 0.5;
        }
        // 250
    }
    // 260

    // Test for succesful iteration.
    if (ratio >= 0.0001)
    {
        // Successful iteration. Update x, fvec, and their norms.
        for (j = 0; j < n; ++j)
        {
            x[j] = wa2[j];
            wa2[j] = diag[j] * x[j];
        }
        // 270
        for (i = 0; i < m; i++)
        {
            fvec[i] = wa4[i];
        }
        // 280
        xnorm = enorm(n, wa2);
        fnorm = fnorm1;
        iter++;
    }
    // 290

    // Tests for convergence.
    if ((fabs(actred) <= ftol) && (prered <= ftol) && (0.5*ratio <= 1.0)) { info = 1; }
    if (delta <= xtol*xnorm) { info = 2; }
    if ((fabs(actred) <= ftol) && (prered <= ftol) && (0.5*ratio <= 1.0) && (info == 2)) { info = 3; }
    if (info != 0){ goto DONE; }

    // Tests for termination and stringent tolerances.
    if (nfev >= maxfev) { info = 5; }
    if ((fabs(actred) <= epsmch) && (prered <= epsmch) && (0.5*ratio <= 1.0)) { info = 6; }
    if (delta <= epsmch*xnorm) { info = 7; }
    if (gnorm <= epsmch) { info = 8; }
    if (info != 0) { goto DONE; }

    // End of the inner loop. Repeat if iteration unsuccessful.
    if (ratio < 0.0001) { goto INNER; }

    // End of the outer loop.
    goto OUTER;

DONE:
    resume = 0;
SAVE:
    hd[S_RESUME] = resume; hd[S_NFEV] = nfev; hd[S_INFO] = info; hd[S_ITER] = iter;
    hd[S_ACTRED] = actred; hd[S_DELTA] = delta; hd[S_DIRDER] = dirder; hd[S_FNORM] = fnorm; hd[S_FNORM1] = fnorm1;
    hd[S_GNORM] = gnorm; hd[S_PAR] = par; hd[S_PNORM] = pnorm; hd[S_PRERED] = prered; hd[S_RATIO] = ratio;
    hd[S_XNORM] = xnorm; hd[S_NJEV] = njev;
    for (j = 0; j < n; j++) L->ipv[j] = ipvt[j];
}

static const tsr_arg *lm_arg(const tsr_arg *args, int nargs, int i) { return i < nargs && args[i].kind != 0 ? &args[i] : NULL; }

static int lm_num(const tsr_arg *args, int nargs, int i, double dflt, const char *name, double *out)
{
    const tsr_arg *a = lm_arg(args, nargs, i);
    if (!a) { *out = dflt; return TSR_OK; }
    if (a->kind != 1 && a->kind != 4) { fn_set_error("lmdif: %s must be a number", name); return TSR_ETYPE; }
    *out = (a->kind == 1 && (a->flags & 1)) ? (double)a->ival : a->num;
    return TSR_OK;
}

/* the points the state waits for, as rows of a (k, n) result: x0 or x + p (1 row), x + h_j e_j (n rows), or x for
   the caller's Jacobian (1 row) */
static int lm_points(tsr_result *r, const lm_t *L)
{
    const int n = L->n, resume = (int)L->hd[S_RESUME], k = resume == 2 && L->hd[S_JAC] != 1.0 ? n : 1;
    int64_t shape[2] = {k, n};
    double *p = (double *)fn_result_array(r, TSR_F64, 2, shape);
    if (!p) return TSR_ENOMEM;
    if (resume == 2 && k == n) {
        for (int j = 0; j < n; j++) {
            memcpy(p + (int64_t)j * n, L->x, sizeof(double) * (size_t)n);
            p[(int64_t)j * n + j] = L->x[j] + L->h[j];
        }
    } else {
        memcpy(p, resume == 3 ? L->wa2 : L->x, sizeof(double) * (size_t)n);
    }
    return TSR_OK;
}

/* the first call: the state for x0, y, sigma and the tolerances, waiting for the values at x0 */
static int lm_start(const tsr_arg *args, int nargs, tsr_result *res)
{
    const tsr_arg *ax = lm_arg(args, nargs, 3), *ay = lm_arg(args, nargs, 4), *as = lm_arg(args, nargs, 5);
    if (!ax || ax->kind != 3) { fn_set_error("lmdif: x0 (the starting parameters) must be an array"); return TSR_EARG; }
    if ((ay && ay->kind != 3) || (as && as->kind != 3)) { fn_set_error("lmdif: y and sigma must be arrays"); return TSR_EARG; }
    if (as && !ay) { fn_set_error("lmdif: sigma needs y"); return TSR_EARG; }
    double ftol, xtol, gtol, epsfcn, factor, maxfev_d, jac_d;
    int rc;
    if ((rc = lm_num(args, nargs, 6, 1.49012e-8, "ftol", &ftol)) || (rc = lm_num(args, nargs, 7, 1.49012e-8, "xtol", &xtol))
        || (rc = lm_num(args, nargs, 8, 0.0, "gtol", &gtol)) || (rc = lm_num(args, nargs, 9, 0.0, "maxfev", &maxfev_d))
        || (rc = lm_num(args, nargs, 10, 0.0, "epsfcn", &epsfcn)) || (rc = lm_num(args, nargs, 11, 100.0, "factor", &factor))
        || (rc = lm_num(args, nargs, 12, 0.0, "jac", &jac_d)))
        return rc;
    if (!lm_int(jac_d, 0, 2)) { fn_set_error("lmdif: jac must be 0 (differences), 1 (the caller's) or 2 (2-point)"); return TSR_EARG; }
    const int64_t n = tsr_array_size(&ax->arr), m = ay ? tsr_array_size(&ay->arr) : 0;
    if (n < 1) { fn_set_error("lmdif: x0 must hold at least one parameter"); return TSR_EARG; }
    if (ay && m < n) {
        fn_set_error("lmdif: improper input: %lld parameters must not exceed %lld values", (long long)n, (long long)m);
        return TSR_EARG;
    }
    if (!lm_fits(m, n) || n > (INT_MAX / 200) - 1) { fn_set_error("lmdif: too many parameters or values"); return TSR_EARG; }
    if (maxfev_d != floor(maxfev_d) || maxfev_d > INT_MAX) { fn_set_error("lmdif: maxfev must be an integer"); return TSR_EARG; }
    /* leastsq's 200 * (n + 1), with Dfun 100 * (n + 1); least_squares' 100 * n */
    const int maxfev = maxfev_d > 0 ? (int)maxfev_d : jac_d == 0 ? 200 * (int)(n + 1) : jac_d == 1 ? 100 * (int)(n + 1) : 100 * (int)n;
    if ((ftol < 0.0) || (xtol < 0.0) || (gtol < 0.0) || (factor <= 0.0)) {
        fn_set_error("lmdif: improper input parameters (ftol, xtol, gtol >= 0, factor > 0)");
        return TSR_EARG;
    }
    int64_t ns, ny = 0, nsg = 0;
    double *x0 = fn_arg_doubles(ax, &ns);
    double *y = (x0 && ay) ? fn_arg_doubles(ay, &ny) : NULL;
    double *sg = (x0 && as && (!ay || y)) ? fn_arg_doubles(as, &nsg) : NULL;
    if (!x0 || (ay && !y) || (as && !sg)) {
        fn_free_doubles(x0, ns); fn_free_doubles(y, ny); fn_free_doubles(sg, nsg);
        fn_set_error("lmdif: out of memory"); return TSR_ENOMEM;
    }
    if (as && nsg != 1 && nsg != m) {
        fn_free_doubles(x0, ns); fn_free_doubles(y, ny); fn_free_doubles(sg, nsg);
        fn_set_error("lmdif: sigma must hold 1 or %lld values", (long long)m); return TSR_EARG;
    }
    const int64_t len = lm_len(m, n);
    double *s = (double *)fn_result_array(&res[0], TSR_F64, 1, &len);
    if (!s) { fn_free_doubles(x0, ns); fn_free_doubles(y, ny); fn_free_doubles(sg, nsg); return TSR_ENOMEM; }
    memset(s, 0, sizeof(double) * (size_t)len);
    lm_t L;
    lm_bind(&L, s, (int)m, (int)n);
    s[S_MAGIC] = LM_MAGIC; s[S_M] = (double)m; s[S_N] = (double)n; s[S_MAXFEV] = maxfev; s[S_RESUME] = 1;
    s[S_HASY] = ay ? 1.0 : 0.0; s[S_HASW] = as ? 1.0 : 0.0;
    s[S_FTOL] = ftol; s[S_XTOL] = xtol; s[S_GTOL] = gtol; s[S_EPSFCN] = epsfcn; s[S_FACTOR] = factor; s[S_JAC] = jac_d;
    memcpy(L.x, x0, sizeof(double) * (size_t)n);
    for (int64_t i = 0; i < m && ay; i++) L.y[i] = y[i];
    for (int64_t i = 0; i < m && as; i++) L.w[i] = 1.0 / sg[nsg == 1 ? 0 : i];     /* curve_fit's 1.0 / sigma */
    for (int64_t j = 0; j < n; j++) L.ipv[j] = (double)j;
    fn_free_doubles(x0, ns); fn_free_doubles(y, ny); fn_free_doubles(sg, nsg);
    fn_result_int(&res[1], 1);
    return lm_points(&res[2], &L);
}

/* linalg.lmdif(state, f=None, fcols=False, x0=None, y=None, sigma=None, ftol=1.49012e-08, xtol=1.49012e-08,
   gtol=0.0, maxfev=0, epsfcn=0.0, factor=100.0, jac=0) -> state, task, points. Without a state: starts at x0,
   minimising the sum of squares of (f - y) * (1 / sigma) (of f without y), with leastsq's defaults (maxfev 0:
   200 * (n + 1); 100 * (n + 1) with jac 1, least_squares' 100 * n with jac 2). With one: f holds the function values
   at the points the last call returned, one row per point ((k, m); with fcols, (m, k)). task is the number of points
   to evaluate next (the rows of points, (k, n)), -1 for the Jacobian at the one point (jac 1: f is (m, n), with
   fcols (n, m)), 0 when finished (lmdif_result then gives the fit). */
int r_lmdif(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *as = lm_arg(args, nargs, 0), *af = lm_arg(args, nargs, 1);
    if (!as) return lm_start(args, nargs, res);
    int64_t len;
    const double *s0 = fn_arg_f64_view(as, &len);
    int m, n;
    if (!s0) { fn_set_error("lmdif: state must be the array lmdif returned"); return TSR_EARG; }
    if (!lm_check(s0, len, &m, &n)) return TSR_EARG;
    const int resume = (int)s0[S_RESUME];
    if (resume == 0) { fn_set_error("lmdif: the fit has finished (lmdif_result gives it)"); return TSR_EARG; }
    if (!af || af->kind != 3) { fn_set_error("lmdif: f (the function values at the points) must be an array"); return TSR_EARG; }
    double fc = 0.0;
    int rc = lm_num(args, nargs, 2, 0.0, "fcols", &fc);
    if (rc) return rc;
    const int jac = (int)s0[S_JAC], k = resume == 2 ? n : 1;         /* jac 1: the Jacobian, n columns of m */
    int64_t nf;
    double *F = fn_arg_doubles(af, &nf);
    if (!F) { fn_set_error("lmdif: out of memory"); return TSR_ENOMEM; }
    int64_t mm = m;
    if (m == 0) {                                          /* no y: m is the length of the first values */
        mm = nf;
        if (mm < n || !lm_fits(mm, n)) {
            fn_free_doubles(F, nf);
            fn_set_error("lmdif: improper input: %d parameters must not exceed %lld values", n, (long long)mm);
            return TSR_EARG;
        }
    }
    if (nf != (int64_t)k * mm) {
        fn_free_doubles(F, nf);
        fn_set_error("lmdif: f must hold %d x %lld values", k, (long long)mm);
        return TSR_EARG;
    }
    const int64_t nlen = lm_len(mm, n);
    double *s = (double *)fn_result_array(&res[0], TSR_F64, 1, &nlen);
    int *ipvt = s ? (int *)malloc(sizeof(int) * (size_t)n) : NULL;
    if (!ipvt) { fn_free_doubles(F, nf); fn_set_error("lmdif: out of memory"); return TSR_ENOMEM; }
    if (m == 0) {
        memset(s, 0, sizeof(double) * (size_t)nlen);
        memcpy(s, s0, sizeof(double) * (size_t)(S_HEAD + 8 * (int64_t)n));
        s[S_M] = (double)mm;
    } else {
        memcpy(s, s0, sizeof(double) * (size_t)nlen);
    }
    lm_t L;
    lm_bind(&L, s, (int)mm, n);
    if (!lm_ipvt(&L, ipvt, resume == 3)) { free(ipvt); fn_free_doubles(F, nf); return TSR_EARG; }
    lm_run(&L, ipvt, F, fc != 0.0, k);
    free(ipvt);
    fn_free_doubles(F, nf);
    fn_result_int(&res[1], s[S_RESUME] == 0 ? 0 : s[S_RESUME] != 2 ? 1 : jac == 1 ? -1 : n);
    return lm_points(&res[2], &L);
}

/* linalg.lmdif_result(state) -> x, fvec, info, nfev, cov, sumsq, njev: the finished fit. cov is leastsq's cov_x, the
   inverse of R^T R from the last QR factorisation (R's inverse by LAPACK dtrti2's column order, rows permuted by
   ipvt, times its transpose), NaN when info is not 1 .. 4 or R is singular; sumsq is curve_fit's
   np.sum(fvec ** 2) (NumPy's pairwise summation). */
int r_lmdif_result(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *as = lm_arg(args, nargs, 0);
    int64_t len;
    const double *s = as ? fn_arg_f64_view(as, &len) : NULL;
    int m, n;
    if (!s) { fn_set_error("lmdif_result: state must be the array lmdif returned"); return TSR_EARG; }
    if (!lm_check(s, len, &m, &n)) return TSR_EARG;
    if (s[S_RESUME] != 0) { fn_set_error("lmdif_result: the fit has not finished (call lmdif until task is 0)"); return TSR_EARG; }
    lm_t L;
    lm_bind(&L, (double *)s, m, n);
    int *ipvt = (int *)malloc(sizeof(int) * (size_t)n);
    double *R = ipvt ? (double *)malloc(sizeof(double) * (size_t)n * (size_t)n * 2) : NULL;   /* R, then P */
    double *sq = R ? (double *)malloc(sizeof(double) * (size_t)m) : NULL;
    if (!sq) { free(ipvt); free(R); fn_set_error("lmdif_result: out of memory"); return TSR_ENOMEM; }
    if (!lm_ipvt(&L, ipvt, 1)) { free(ipvt); free(R); free(sq); return TSR_EARG; }
    const int64_t sn[1] = {n}, sm[1] = {m}, sc[2] = {n, n};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, 1, sn);
    double *fv = x ? (double *)fn_result_array(&res[1], TSR_F64, 1, sm) : NULL;
    double *cov = fv ? (double *)fn_result_array(&res[4], TSR_F64, 2, sc) : NULL;
    if (!cov) { free(ipvt); free(R); free(sq); return TSR_ENOMEM; }
    memcpy(x, L.x, sizeof(double) * (size_t)n);
    memcpy(fv, L.fvec, sizeof(double) * (size_t)m);
    const int info = (int)s[S_INFO];
    int ok = info >= 1 && info <= 4;
    /* r = triu(fjac.T[:n, :]), column-major n x n */
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) R[i + (int64_t)j * n] = i <= j ? L.fjac[i + (int64_t)j * m] : 0.0;
    for (int j = 0; j < n && ok; j++) if (R[j + (int64_t)j * n] == 0.0) ok = 0;
    if (ok) {
        /* dtrti2 (upper, non-unit): column j of the inverse from the inverted leading j x j block (dtrmv), scaled */
        for (int j = 0; j < n; j++) {
            double *a = R + (int64_t)j * n;
            a[j] = 1.0 / a[j];
            const double ajj = -a[j];
            for (int jj = 0; jj < j; jj++) {
                if (a[jj] != 0.0) {
                    const double temp = a[jj];
                    for (int i = 0; i < jj; i++) a[i] += temp * R[i + (int64_t)jj * n];
                    a[jj] *= R[jj + (int64_t)jj * n];
                }
            }
            for (int i = 0; i < j; i++) a[i] *= ajj;
        }
        /* invR[perm] = invR.copy(); cov_x = invR @ invR.T */
        double *P = R + (int64_t)n * n;                    /* n x n, row-major */
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++) P[(int64_t)ipvt[r] * n + c] = R[r + (int64_t)c * n];
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++) {
                double acc = 0.0;
                for (int c = 0; c < n; c++) acc += P[(int64_t)a * n + c] * P[(int64_t)b * n + c];
                cov[(int64_t)a * n + b] = acc;
            }
    } else {
        for (int64_t i = 0; i < (int64_t)n * n; i++) cov[i] = NAN;
    }
    for (int i = 0; i < m; i++) sq[i] = L.fvec[i] * L.fvec[i];
    const double sumsq = tsr_psum(sq, m);
    free(ipvt); free(R); free(sq);
    fn_result_int(&res[2], info);
    fn_result_int(&res[3], (int64_t)s[S_NFEV]);
    fn_result_num(&res[5], sumsq);
    fn_result_int(&res[6], (int64_t)s[S_NJEV]);
    return TSR_OK;
}

/* ---- SciPy 1.17.1, scipy/optimize/__minpack.c (unchanged) ---- */

double enorm(const int n, const double* x)
{
    //     **********
    //
    //     function enorm
    //
    //     given an n-vector x, this function calculates the
    //     euclidean norm of x.
    //
    //     the euclidean norm is computed by accumulating the sum of
    //     squares in three different sums. the sums of squares for the
    //     small and large components are scaled so that no overflows
    //     occur. non-destructive underflows are permitted. underflows
    //     and overflows do not occur in the computation of the unscaled
    //     sum of squares for the intermediate components.
    //     the definitions of small, intermediate and large components
    //     depend on two constants, rdwarf and rgiant. the main
    //     restrictions on these constants are that rdwarf**2 not
    //     underflow and rgiant**2 not overflow. the constants
    //     given here are suitable for every known computer.
    //
    //     the function statement is
    //
    //       double precision function enorm(n,x)
    //
    //     where
    //
    //       n is a positive integer input variable.
    //
    //       x is an input array of length n.
    //
    //     subprograms called
    //
    //       fortran-supplied ... dabs,dsqrt
    //
    //     argonne national laboratory. minpack project. march 1980.
    //     burton s. garbow, kenneth e. hillstrom, jorge j. more
    //
    //     **********
    int i;
    double agiant, s1, s2, s3, xabs, x1max, x3max;

    const double rgiant = pow(2.0, 63.5);
    const double rdwarf = 0.5 / rgiant;

    s1 = 0.0;
    s2 = 0.0;
    s3 = 0.0;
    x1max = 0.0;
    x3max = 0.0;
    agiant = rgiant / n;

    for (i = 0; i < n; i++)
    {
        xabs = fabs(x[i]);
        if ((xabs <= rdwarf) || (xabs >= agiant))
        {
            if (xabs > rdwarf)
            {
                // sum for large components
                if (xabs > x1max)
                {
                    s1 = 1.0 + s1*pow(x1max / xabs, 2.0);
                    x1max = xabs;
                } else {
                    // 10
                    s1 += pow(xabs / x1max, 2.0);
                }
                // 20
            } else {
                // 30

                //sum for small components
                if (xabs > x3max)
                {
                    s3 = 1.0 + s3*pow(x3max / xabs, 2.0);
                    x3max = xabs;
                } else {
                    // 40
                    if (xabs != 0.0)
                    {
                        s3 += pow(xabs / x3max, 2.0);
                    }
                }
                // 50
            }
        } else {
            // 70

            // sum for intermediate components
            s2 += xabs*xabs;
        }
        // 80
        // 90
    }

    // calculation of norm.
    if (s1 != 0.0)
    {
        return x1max*sqrt(s1+(s2/x1max)/x1max);
    }
    // 100
    if (s2 != 0.0)
    {
        if (s2 >= x3max)
        {
            return sqrt(s2*(1.0 + (x3max/s2)*(x3max*s3)));
        } else {
            return sqrt(x3max*((s2/x3max)+(x3max*s3)));
        }
    }
    // 110

    return x3max*sqrt(s3);
}


void lmpar(const int n, double *r, const int ldr, const int* ipvt, const double* diag,
           const double* qtb, const double delta, double* par, double* x, double* sdiag,
           double* wa1, double* wa2)
{
    //     **********
    //
    //     subroutine lmpar
    //
    //     given an m by n matrix a, an n by n nonsingular diagonal
    //     matrix d, an m-vector b, and a positive number delta,
    //     the problem is to determine a value for the parameter
    //     par such that if x solves the system
    //
    //           a*x = b ,     sqrt(par)*d*x = 0 ,
    //
    //     in the least squares sense, and dxnorm is the euclidean
    //     norm of d*x, then either par is zero and
    //
    //           (dxnorm-delta) .le. 0.1*delta ,
    //
    //     or par is positive and
    //
    //           abs(dxnorm-delta) .le. 0.1*delta .
    //
    //     this subroutine completes the solution of the problem
    //     if it is provided with the necessary information from the
    //     qr factorization, with column pivoting, of a. that is, if
    //     a*p = q*r, where p is a permutation matrix, q has orthogonal
    //     columns, and r is an upper triangular matrix with diagonal
    //     elements of nonincreasing magnitude, then lmpar expects
    //     the full upper triangle of r, the permutation matrix p,
    //     and the first n components of (q transpose)*b. on output
    //     lmpar also provides an upper triangular matrix s such that
    //
    //            t   t                   t
    //           p *(a *a + par*d*d)*p = s *s .
    //
    //     s is employed within lmpar and may be of separate interest.
    //
    //     only a few iterations are generally needed for convergence
    //     of the algorithm. if, however, the limit of 10 iterations
    //     is reached, then the output par will contain the best
    //     value obtained so far.
    //
    //     the subroutine statement is
    //
    //       subroutine lmpar(n,r,ldr,ipvt,diag,qtb,delta,par,x,sdiag,
    //                        wa1,wa2)
    //
    //     where
    //
    //       n is a positive integer input variable set to the order of r.
    //
    //       r is an n by n array. on input the full upper triangle
    //         must contain the full upper triangle of the matrix r.
    //         on output the full upper triangle is unaltered, and the
    //         strict lower triangle contains the strict upper triangle
    //         (transposed) of the upper triangular matrix s.
    //
    //       ldr is a positive integer input variable not less than n
    //         which specifies the leading dimension of the array r.
    //
    //       ipvt is an integer input array of length n which defines the
    //         permutation matrix p such that a*p = q*r. column j of p
    //         is column ipvt(j) of the identity matrix.
    //
    //       diag is an input array of length n which must contain the
    //         diagonal elements of the matrix d.
    //
    //       qtb is an input array of length n which must contain the first
    //         n elements of the vector (q transpose)*b.
    //
    //       delta is a positive input variable which specifies an upper
    //         bound on the euclidean norm of d*x.
    //
    //       par is a nonnegative variable. on input par contains an
    //         initial estimate of the levenberg-marquardt parameter.
    //         on output par contains the final estimate.
    //
    //       x is an output array of length n which contains the least
    //         squares solution of the system a*x = b, sqrt(par)*d*x = 0,
    //         for the output par.
    //
    //       sdiag is an output array of length n which contains the
    //         diagonal elements of the upper triangular matrix s.
    //
    //       wa1 and wa2 are work arrays of length n.
    //
    //     subprograms called
    //
    //       minpack-supplied ... dpmpar,enorm,qrsolv
    //
    //       fortran-supplied ... dabs,dmax1,dmin1,dsqrt
    //
    //     argonne national laboratory. minpack project. march 1980.
    //     burton s. garbow, kenneth e. hillstrom, jorge j. more
    //
    //     **********
    int i, iter, j, k, l, nsing;
    double dxnorm, dwarf, fp, gnorm, parc, parl, paru, ssum, temp;

    dwarf = dpmpar[1];

    // Compute and store in x the Gauss-Newton direction. If the jacobian
    // is rank-deficient, obtain a least squares solution.

    nsing = n;  // Used as an unattainable index, signaling nonsingularity.
    for (j = 0; j < n; j++) {
	    wa1[j] = qtb[j];
	    if ((r[j + j*ldr] == 0.0) && (nsing == n)) { nsing = j; }
	    if (nsing < n) { wa1[j] = 0.0; }
    }
    // 10
    if (nsing > 0) {
        for (k = 1; k <= nsing; k++) {
            j = nsing - k;
            wa1[j] /= r[j + j*ldr];
            temp = wa1[j];
            if (j >= 1) {
                for (i = 0; i < j; i++) {
                    wa1[i] -= r[i + j*ldr]*temp;
                }
                // 20
            }
            // 30
        }
        // 40
    }
    // 50
    for (j = 0; j < n; j++) {
	    l = ipvt[j];
	    x[l] = wa1[j];
    }
    // 60

    // Initialize the iteration counter.
    // Evaluate the function at the origin and test for acceptance
    // of the Gauss-Newton direction.
    iter = 0;
    for (j = 0; j < n; j++)
    {
	    wa2[j] = diag[j] * x[j];
    }
    // 70

    dxnorm = enorm(n, wa2);
    fp = dxnorm - delta;
    if (fp <= 0.1*delta)
    {
        // 220
        if (iter == 0) { *par = 0.0; }
        return;
    }

    // If the jacobian is not rank deficient, the newton
    // step provides a lower bound, parl, for the zero of
    // the function. otherwise set this bound to zero.
    parl = 0.0;
    if (nsing == n) {
        for (j = 0; j < n; j++)
        {
            l = ipvt[j];
            wa1[j] = diag[l]*(wa2[l]/dxnorm);
        }
        // 80
        for (j = 0; j < n; j++)
        {
            ssum = 0.0;
            if (j > 0) {
                for (i = 0; i < j; i++) {
                    ssum += r[i + j*ldr]*wa1[i];
                }
                // 90
            }
            // 100
            wa1[j] = (wa1[j] - ssum) / r[j + j*ldr];
        }
        // 110
        temp = enorm(n, wa1);
        parl = ((fp / delta) / temp) / temp;
    }
    // 120

    // Calculate an upper bound, paru, for the zero of the function.
    for (j = 0; j < n; j++) {
        ssum = 0.0;
        for (i = 0; i <= j; i++)
        {
            ssum += r[i + j*ldr] * qtb[i];
        }
        // 130
        l = ipvt[j];
        wa1[j] = ssum / diag[l];
    }
    // 140

    gnorm = enorm(n, wa1);
    paru = gnorm / delta;
    if (paru == 0.0) { paru = dwarf / fmin(delta, 0.1); }

    // If the input par lies outside of the interval (parl, paru),
    // set par to the closer endpoint.
    *par = fmin(fmax(*par, parl), paru);
    if (*par == 0.0) { *par = gnorm / dxnorm; }

    // Beginning of an iteration.
    // 150
    while (1)
    {
        iter++;

        // Evaluate the function at the current value of par.
        if (*par == 0.) { *par = fmax(dwarf, paru*0.001); }
        temp = sqrt(*par);

        for (j = 0; j < n; j++) {
            wa1[j] = temp * diag[j];
        }
        // 160

        qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2);

        for (j = 0; j < n; j++) {
            wa2[j] = diag[j] * x[j];
        }
        // 170

        dxnorm = enorm(n, wa2);
        temp = fp;
        fp = dxnorm - delta;
        // If the function is small enough, accept the current value
        // of par. also test for the exceptional cases where parl
        // is zero or the number of iterations has reached 10.

        if ((fabs(fp) <= 0.1*delta) || ((parl == 0.0) && (fp <= temp) && (temp < 0.0)) || (iter == 10)) {
            // 220
            if (iter == 0) { *par = 0.0; }
            return;
        }

        // Compute the Newton correction.
        for (j = 0; j < n; j++)
        {
            l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        // 180

        for (j = 0; j < n; j++)
        {
            wa1[j] /= sdiag[j];
            temp = wa1[j];
            if (j != n - 1)
            {
                for (i = j + 1; i < n; i++)
                {
                    wa1[i] -= r[i + j*ldr]*temp;
                }
            }
            // 200
        }
        // 210

        temp = enorm(n, wa1);
        parc = ((fp/delta)/temp)/temp;

        // Depending on the sign of the function, update parl or paru
        if (fp > 0.) { parl = fmax(parl, *par); }
        if (fp < 0.) { paru = fmin(paru, *par); }

        // compute an improved estimate for par.
        *par = fmax(parl, *par + parc);

        // end of an iteration.
    }

    if (iter == 0)
    {
        *par = 0.0;
    }
    return;
}


void qrfac(const int m, const int n, double* a, const int lda, const int pivot,
           int* ipvt, double* rdiag, double* acnorm, double* wa)
{
    //     **********
    //
    //     subroutine qrfac
    //
    //     this subroutine uses householder transformations with column
    //     pivoting (optional) to compute a qr factorization of the
    //     m by n matrix a. that is, qrfac determines an orthogonal
    //     matrix q, a permutation matrix p, and an upper trapezoidal
    //     matrix r with diagonal elements of nonincreasing magnitude,
    //     such that a*p = q*r. the householder transformation for
    //     column k, k = 1,2,...,min(m,n), is of the form
    //
    //                           t
    //           i - (1/u(k))*u*u
    //
    //     where u has zeros in the first k-1 positions. the form of
    //     this transformation and the method of pivoting first
    //     appeared in the corresponding linpack subroutine.
    //
    //     the subroutine statement is
    //
    //       subroutine qrfac(m,n,a,lda,pivot,ipvt,lipvt,rdiag,acnorm,wa)
    //
    //     where
    //
    //       m is a positive integer input variable set to the number
    //         of rows of a.
    //
    //       n is a positive integer input variable set to the number
    //         of columns of a.
    //
    //       a is an m by n array. on input a contains the matrix for
    //         which the qr factorization is to be computed. on output
    //         the strict upper trapezoidal part of a contains the strict
    //         upper trapezoidal part of r, and the lower trapezoidal
    //         part of a contains a factored form of q (the non-trivial
    //         elements of the u vectors described above).
    //
    //       lda is a positive integer input variable not less than m
    //         which specifies the leading dimension of the array a.
    //
    //       pivot is a logical input variable. if pivot is set true,
    //         then column pivoting is enforced. if pivot is set false,
    //         then no column pivoting is done.
    //
    //       ipvt is an integer output array of length lipvt. ipvt
    //         defines the permutation matrix p such that a*p = q*r.
    //         column j of p is column ipvt(j) of the identity matrix.
    //         if pivot is false, ipvt is not referenced.
    //
    //       lipvt is a positive integer input variable. if pivot is false,
    //         then lipvt may be as small as 1. if pivot is true, then
    //         lipvt must be at least n.
    //
    //       rdiag is an output array of length n which contains the
    //         diagonal elements of r.
    //
    //       acnorm is an output array of length n which contains the
    //         norms of the corresponding columns of the input matrix a.
    //         if this information is not needed, then acnorm can coincide
    //         with rdiag.
    //
    //       wa is a work array of length n. if pivot is false, then wa
    //         can coincide with rdiag.
    //
    //     subprograms called
    //
    //       minpack-supplied ... dpmpar,enorm
    //
    //       fortran-supplied ... dmax1,dsqrt,min0
    //
    //     argonne national laboratory. minpack project. march 1980.
    //     burton s. garbow, kenneth e. hillstrom, jorge j. more
    //
    //     **********
    int i, j, k, kmax, minmn;
    double ajnorm, epsmch, ssum, temp;

    epsmch = dpmpar[0];
    // Compute the initial column norms and initialize several arrays
    for (j = 0; j < n; j++)
    {
        acnorm[j] = enorm(m, &a[j*lda]);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
        if (pivot) { ipvt[j] = j; }
    }
    // 10

    // Reduce a to r with householder transformations.
    minmn = (m > n ? n : m);
    for (j = 0; j < minmn; j++)
    {
        if (pivot)
        {
            // Bring the column of largest norm into the pivot position.
            kmax = j;
            // Search for the largest norm.
            for (k = j; k < n; k++)
            {
                if (rdiag[k] > rdiag[kmax]) { kmax = k; }
            }
            if (kmax != j)
            {
                // Swap columns j and kmax
                for (i = 0; i < m; i++)
                {
                    temp = a[i + lda*j];
                    a[i + lda*j] = a[i + lda*kmax];
                    a[i + lda*kmax] = temp;
                }
                // Swap the column norm values of j and kmax.
                rdiag[kmax] = rdiag[j];
                wa[kmax] = wa[j];
                // Swap the pivot entries
                k = ipvt[j];
                ipvt[j] = ipvt[kmax];
                ipvt[kmax] = k;
            }
        }
        // 40

        // Compute Householder transformation to reduce the j-th column
        // of "a" to a multiple of the j-th unit vector.
        ajnorm = enorm(m - j, &a[lda*j + j]);
        if (ajnorm == 0.0)
        {
            // 100
            rdiag[j] = -ajnorm;
            continue;
        }

        if (a[lda*j + j] < 0.0) { ajnorm = -ajnorm; }
        // Rescale column j.
        for (i = j; i < m; i++)
        {
            a[i + lda*j] /= ajnorm;
        }
        // 50

        a[j + lda*j] += 1.0;

        // Apply the transformation to the remaining columns and
        // update the norms

        // Any column left?
        if (j == n - 1)
        {
            // 100
            rdiag[j] = -ajnorm;
            continue;
        }

        for (k = j+1; k < n; k++)
        {
            ssum = 0.0;
            for (i = j; i < m; i++)
            {
                ssum += a[i + lda*j]*a[i + lda*k];
            }
            // 60
            temp = ssum / a[j + lda*j];
            for (i = j; i < m; i++)
            {
                a[i + lda*k] -= temp*a[i + lda*j];
            }
            // 70
            if ((pivot) && (rdiag[k] != 0.0))
            {
                temp = a[j + lda*k] / rdiag[k];
                rdiag[k] *= sqrt(fmax(0.0, 1.0 - temp*temp));
                if (0.05*pow(rdiag[k]/wa[k], 2.0) <= epsmch)
                {
                    rdiag[k] = enorm(m-j, &a[(j+1) + lda*k]);
                    wa[k] = rdiag[k];
                }
            }
            // 80
        }
        // 90
        rdiag[j] = -ajnorm;
    }
    // 110

    return;
}


void qrsolv(const int n, double* r, const int ldr, const int* ipvt,
            const double* diag, const double* qtb, double* x,
            double* sdiag, double* wa)
{
    //     **********
    //
    //     subroutine qrsolv
    //
    //     given an m by n matrix a, an n by n diagonal matrix d,
    //     and an m-vector b, the problem is to determine an x which
    //     solves the system
    //
    //           a*x = b ,     d*x = 0 ,
    //
    //     in the least squares sense.
    //
    //     this subroutine completes the solution of the problem
    //     if it is provided with the necessary information from the
    //     qr factorization, with column pivoting, of a. that is, if
    //     a*p = q*r, where p is a permutation matrix, q has orthogonal
    //     columns, and r is an upper triangular matrix with diagonal
    //     elements of nonincreasing magnitude, then qrsolv expects
    //     the full upper triangle of r, the permutation matrix p,
    //     and the first n components of (q transpose)*b. the system
    //     a*x = b, d*x = 0, is then equivalent to
    //
    //                  t       t
    //           r*z = q *b ,  p *d*p*z = 0 ,
    //
    //     where x = p*z. if this system does not have full rank,
    //     then a least squares solution is obtained. on output qrsolv
    //     also provides an upper triangular matrix s such that
    //
    //            t   t               t
    //           p *(a *a + d*d)*p = s *s .
    //
    //     s is computed within qrsolv and may be of separate interest.
    //
    //     the subroutine statement is
    //
    //       subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
    //
    //     where
    //
    //       n is a positive integer input variable set to the order of r.
    //
    //       r is an n by n array. on input the full upper triangle
    //         must contain the full upper triangle of the matrix r.
    //         on output the full upper triangle is unaltered, and the
    //         strict lower triangle contains the strict upper triangle
    //         (transposed) of the upper triangular matrix s.
    //
    //       ldr is a positive integer input variable not less than n
    //         which specifies the leading dimension of the array r.
    //
    //       ipvt is an integer input array of length n which defines the
    //         permutation matrix p such that a*p = q*r. column j of p
    //         is column ipvt(j) of the identity matrix.
    //
    //       diag is an input array of length n which must contain the
    //         diagonal elements of the matrix d.
    //
    //       qtb is an input array of length n which must contain the first
    //         n elements of the vector (q transpose)*b.
    //
    //       x is an output array of length n which contains the least
    //         squares solution of the system a*x = b, d*x = 0.
    //
    //       sdiag is an output array of length n which contains the
    //         diagonal elements of the upper triangular matrix s.
    //
    //       wa is a work array of length n.
    //
    //     subprograms called
    //
    //       fortran-supplied ... dabs,dsqrt
    //
    //     argonne national laboratory. minpack project. march 1980.
    //     burton s. garbow, kenneth e. hillstrom, jorge j. more
    //
    //     **********
    int i, j, k, l, nsing;
    double ccos, ccotan, qtbpj, ssin, ssum, ttan, temp;

    // Copy r and (q transpose)*b to preserve input and initialize s.
    // In particular, save the diagonal elements of r in x.
    for (j = 0; j < n; j++) {
	    for (i = j; i < n; i++) {
	        r[i + j*ldr] = r[j + i*ldr];
	    }
        // 10
	    x[j] = r[j + j*ldr];
	    wa[j] = qtb[j];
    }
    // 20

    // Eliminate the diagonal matrix d using a givens rotation.
    for (j = 0; j < n; j++)
    {
        // Prepare the row of d to be eliminated, locating the
        // diagonal element using p from the qr factorization.
        l = ipvt[j];
        if (diag[l] != 0.0)
        {
            for (k = j; k < n; k++)
            {
                sdiag[k] = 0.0;
            }
            sdiag[j] = diag[l];

            // The transformations to eliminate the row of d
            // modify only a single element of (q.T b) beyond
            // the first n, which is initially 0.
            qtbpj = 0.0;
            for (k = j; k < n; k++)
            {
                // Determine a givens rotation which eliminates the
                // appropriate element in the current row of d
                if (sdiag[k] != 0.0)
                {
                    if (fabs(r[ldr*k + k]) < fabs(sdiag[k]))
                    {
                        ccotan = r[ldr*k + k] / sdiag[k];
                        ssin = 0.5 / sqrt(0.25 + 0.25*pow(ccotan, 2.0));
                        ccos = ssin*ccotan;
                    } else {
                        ttan = sdiag[k] / r[ldr*k + k];
                        ccos = 0.5 / sqrt(0.25 + 0.25*pow(ttan, 2.0));
                        ssin = ccos*ttan;
                    }

                    // Compute the modified diagonal element of r and the
                    // modified element of ((q.T b),0).
                    r[ldr*k + k] *= ccos;
                    r[ldr*k + k] += ssin*sdiag[k];
                    temp = ccos*wa[k] + ssin*qtbpj;
                    qtbpj = -ssin*wa[k] + ccos*qtbpj;
                    wa[k] = temp;

                    // Accumulate the transformation in the row of s.
                    if (k != n - 1)
                    {
                        for (i = k+1; i < n; i++)
                        {
                            temp = ccos*r[i + ldr*k] + ssin*sdiag[i];
                            sdiag[i] = -ssin*r[i + ldr*k] + ccos*sdiag[i];
                            r[i + ldr*k] = temp;
                        }
                        // 60
                    }
                }
                // 70
            }
            // 80
        }
        // 90

        // Store the diagonal element of s and restore the corresponding
        // diagonal element of r.
        sdiag[j] = r[ldr*j + j];
        r[ldr*j + j] = x[j];
    }
    // 100

    // Solve the triangular system for z. If the system is
    // singular, then obtain a least squares solution.

    nsing = n;
    for (j = 0; j < n; j++)
    {
        if ((sdiag[j] == 0.0) && (nsing == n)) { nsing = j; }
        if (nsing < n) { wa[j] = 0.; }
    }
    // 110
    if (nsing > 0) {
        for (k = 1; k <= nsing; k++) {
            j = nsing - k;
            ssum = 0.0;
            if (nsing >= j+1) {
                for (i = j+1; i < nsing; ++i) {
                    ssum += r[i + j*ldr] * wa[i];
                }
            }
            // 130
            wa[j] = (wa[j] - ssum) / sdiag[j];
        }
        // 140
    }
    // 150

    // Permute the components of z back to components of x.
    for (j = 0; j < n; j++)
    {
    	l = ipvt[j];
	    x[l] = wa[j];
    }

    return;
}
