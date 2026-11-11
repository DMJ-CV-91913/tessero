/*
 * The C tests' floating-point references for spatial_distance.c (convex hull, Delaunay, k-d tree) and la_fd_normal.c
 * (curve_fit's normal equations), built like those files with multiply-add contraction off (csrc/Makefile). The checks compare the library's geometry with brute force
 * computed here at run time: an orientation or in-circle sign for every triangle and point, the exhaustive k-nearest
 * search. Both sides must round each product and sum separately, as every x86-64 build does: a compiler that fuses
 * a*b - c*d into one multiply-add (aarch64 GCC and Apple clang do by default) judges near-collinear points,
 * near-cocircular points and equal distances otherwise on one side only, and the test then reports a difference in
 * its own arithmetic, not in the code under test. No expected value is stored or changed: each reference is
 * recomputed on every run, with the rounding of the code it checks, so the tests still check the algorithms
 * (strict turns and containment, empty circumcircles, the same neighbours in the same order as an exhaustive search).
 */
#include <math.h>
#include <stdint.h>

/* > 0: a, b, c turn counter-clockwise */
double g2t_orient(const double *P, int64_t a, int64_t b, int64_t c)
{
    return (P[2 * b] - P[2 * a]) * (P[2 * c + 1] - P[2 * a + 1]) - (P[2 * b + 1] - P[2 * a + 1]) * (P[2 * c] - P[2 * a]);
}

/* > 0: point p lies inside the circumcircle of the counter-clockwise triangle (ax, ay), (bx, by), (cx, cy) */
double g2t_incircle(double ax, double ay, double bx, double by, double cx, double cy, const double *P, int64_t p)
{
    const double dx = ax - P[2 * p], dy = ay - P[2 * p + 1], ex = bx - P[2 * p], ey = by - P[2 * p + 1],
                 fx = cx - P[2 * p], fy = cy - P[2 * p + 1];
    return (dx * dx + dy * dy) * (ex * fy - ey * fx) - (ex * ex + ey * ey) * (dx * fy - dy * fx)
           + (fx * fx + fy * fy) * (dx * ey - dy * ex);
}

/* linalg.fd_normal's definition (la_fd_normal.c, also built with contraction off: it reproduces
   Optimize\LeastSquares's PHP loop bit for bit, and PHP rounds every product separately): the forward-difference
   Jacobian J (m x np) of the residuals r from rh (np x m) at the steps h, then A = J^T J and g = J^T r, each sum in
   the order of k with zero Jacobian entries skipped; A and g arrive zeroed */
void fd_normal_ref(const double *r, const double *rh, const double *h, int m, int np, double *J, double *A, double *g)
{
    for (int k = 0; k < m; k++) for (int j = 0; j < np; j++) J[k * np + j] = (rh[j * m + k] - r[k]) / h[j];
    for (int k = 0; k < m; k++)
        for (int i = 0; i < np; i++) {
            if (J[k * np + i] == 0.0) continue;
            g[i] += J[k * np + i] * r[k];
            for (int j = i; j < np; j++) A[i * np + j] += J[k * np + i] * J[k * np + j];
        }
    for (int i = 0; i < np; i++) for (int j = 0; j < i; j++) A[i * np + j] = A[j * np + i];
}

/* the Euclidean distance between row i of P and row q of Q (m columns each), summed in column order */
double kd_ref_dist(const double *P, const double *Q, int64_t m, int64_t i, int64_t q)
{
    double s = 0.0;
    for (int64_t j = 0; j < m; j++) { const double df = P[i * m + j] - Q[q * m + j]; s += df * df; }
    return sqrt(s);
}
