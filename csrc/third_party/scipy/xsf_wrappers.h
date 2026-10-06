/* Tessera: the subset of SciPy's xsf_wrappers.h that _cosine.c needs (identical Horner evaluation). */
#ifndef TSR_SCIPY_XSF_WRAPPERS_SHIM_H
#define TSR_SCIPY_XSF_WRAPPERS_SHIM_H
static inline double cephes_polevl_wrap(double x, const double coef[], int N)
{
    const double *p = coef;
    double ans = *p++;
    int i = N;
    do { ans = ans * x + *p++; } while (--i);
    return ans;
}
#endif
