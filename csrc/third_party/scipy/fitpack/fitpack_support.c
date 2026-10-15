/*
 * Minimal f2c runtime support for the vendored (f2c-translated) Dierckx FITPACK
 * sources in csrc/third_party/scipy/fitpack/.  Only the handful of libf2c
 * intrinsics actually referenced by the FITPACK smoothing routines are provided
 * here so the vendored tree is self-contained (no libf2c dependency), mirroring
 * the hand-maintained quadpack.c precedent.
 *
 * Referenced intrinsics: d_sign, pow_dd (math), and the list-directed WRITE
 * stubs (s_wsle/do_lio/e_wsle) which appear ONLY in surfit's invalid-input
 * diagnostic path (ier=10, never reached for validated inputs).
 */
#include <math.h>
#include "f2c.h"

double d_sign(doublereal *a, doublereal *b)
{
    double x = (*a >= 0 ? *a : -*a);
    return (*b >= 0 ? x : -x);
}

double pow_dd(doublereal *ap, doublereal *bp)
{
    return pow(*ap, *bp);
}

/* List-directed WRITE stubs: no-ops. These fire only in FITPACK's
 * invalid-parameter diagnostic branches (which return ier>=10); the PHP
 * facades validate inputs before calling, so they are never executed. */
integer s_wsle(cilist *a) { (void)a; return 0; }
integer do_lio(integer *type, integer *n, char *s, ftnlen len) { (void)type; (void)n; (void)s; (void)len; return 0; }
integer e_wsle(void) { return 0; }
