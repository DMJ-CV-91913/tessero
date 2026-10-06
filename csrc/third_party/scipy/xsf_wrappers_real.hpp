/*
 * Real-valued wrappers extracted from SciPy 1.17.1 scipy/special/xsf_wrappers.cpp (BSD-3-Clause,
 * see LICENSE.txt) by tools/parity/extract_xsf_wrappers.py. Py_ssize_t parameters became long;
 * complex wrappers are omitted. Do not edit by hand.
 */
#pragma once

#include <xsf/airy.h>
#include <xsf/amos.h>
#include <xsf/bessel.h>
#include <xsf/beta.h>
#include <xsf/binom.h>
#include <xsf/cdflib.h>
#include <xsf/digamma.h>
#include <xsf/ellip.h>
#include <xsf/erf.h>
#include <xsf/exp.h>
#include <xsf/expint.h>
#include <xsf/fresnel.h>
#include <xsf/gamma.h>
#include <xsf/hyp2f1.h>
#include <xsf/kelvin.h>
#include <xsf/lambertw.h>
#include <xsf/log.h>
#include <xsf/log_exp.h>
#include <xsf/loggamma.h>
#include <xsf/mathieu.h>
#include <xsf/par_cyl.h>
#include <xsf/sici.h>
#include <xsf/specfun.h>
#include <xsf/sph_bessel.h>
#include <xsf/sph_harm.h>
#include <xsf/sphd_wave.h>
#include <xsf/stats.h>
#include <xsf/struve.h>
#include <xsf/trig.h>
#include <xsf/wright_bessel.h>
#include <xsf/zeta.h>
#include <xsf/cephes/cbrt.h>
#include <xsf/cephes/erfinv.h>
#include <xsf/cephes/expn.h>
#include <xsf/cephes/fresnl.h>
#include <xsf/cephes/hyperg.h>
#include <xsf/cephes/igam.h>
#include <xsf/cephes/igami.h>
#include <xsf/cephes/jv.h>
#include <xsf/cephes/lanczos.h>
#include <xsf/cephes/poch.h>
#include <xsf/cephes/rgamma.h>
#include <xsf/cephes/round.h>
#include <xsf/cephes/scipy_iv.h>
#include <xsf/cephes/spence.h>
#include <xsf/cephes/trig.h>
#include <xsf/cephes/unity.h>
#include <xsf/cephes/yn.h>
#include <limits>

namespace scipy_xsfw {

inline double hypU_wrap(double a, double b, double x) { return xsf::hypu(a, b, x); }
inline double hyp1f1_wrap(double a, double b, double x) { return xsf::hyp1f1(a, b, x); }
inline void special_itairy(double x, double *apt, double *bpt, double *ant, double *bnt) {
    xsf::itairy(x, *apt, *bpt, *ant, *bnt);
}
inline double xsf_exp1(double x) { return xsf::exp1(x); }
inline double xsf_expi(double x) { return xsf::expi(x); }
inline double special_itstruve0(double x) { return xsf::itstruve0(x); }
inline double special_it2struve0(double x) { return xsf::it2struve0(x); }
inline double special_itmodstruve0(double x) { return xsf::itmodstruve0(x); }
inline double special_ber(double x) { return xsf::ber(x); }
inline double special_bei(double x) { return xsf::bei(x); }
inline double special_ker(double x) { return xsf::ker(x); }
inline double special_kei(double x) { return xsf::kei(x); }
inline double special_berp(double x) { return xsf::berp(x); }
inline double special_beip(double x) { return xsf::beip(x); }
inline double special_kerp(double x) { return xsf::kerp(x); }
inline double special_keip(double x) { return xsf::keip(x); }
inline void it1j0y0_wrap(double x, double *j0int, double *y0int) { xsf::it1j0y0(x, *j0int, *y0int); }
inline void it2j0y0_wrap(double x, double *j0int, double *y0int) { xsf::it2j0y0(x, *j0int, *y0int); }
inline void it1i0k0_wrap(double x, double *i0int, double *k0int) { xsf::it1i0k0(x, *i0int, *k0int); }
inline void it2i0k0_wrap(double x, double *i0int, double *k0int) { xsf::it2i0k0(x, *i0int, *k0int); }
inline double cem_cva_wrap(double m, double q) { return xsf::cem_cva(m, q); }
inline double sem_cva_wrap(double m, double q) { return xsf::sem_cva(m, q); }
inline void cem_wrap(double m, double q, double x, double *csf, double *csd) { xsf::cem(m, q, x, *csf, *csd); }
inline void sem_wrap(double m, double q, double x, double *csf, double *csd) { xsf::sem(m, q, x, *csf, *csd); }
inline void mcm1_wrap(double m, double q, double x, double *f1r, double *d1r) { xsf::mcm1(m, q, x, *f1r, *d1r); }
inline void msm1_wrap(double m, double q, double x, double *f1r, double *d1r) { xsf::msm1(m, q, x, *f1r, *d1r); }
inline void mcm2_wrap(double m, double q, double x, double *f2r, double *d2r) { xsf::mcm2(m, q, x, *f2r, *d2r); }
inline void msm2_wrap(double m, double q, double x, double *f2r, double *d2r) { xsf::msm2(m, q, x, *f2r, *d2r); }
inline double pmv_wrap(double m, double v, double x) { return xsf::pmv(m, v, x); }
inline void pbwa_wrap(double a, double x, double *wf, double *wd) { xsf::pbwa(a, x, *wf, *wd); }
inline void pbdv_wrap(double v, double x, double *pdf, double *pdd) { xsf::pbdv(v, x, *pdf, *pdd); }
inline void pbvv_wrap(double v, double x, double *pvf, double *pvd) { xsf::pbvv(v, x, *pvf, *pvd); }
inline double prolate_segv_wrap(double m, double n, double c) { return xsf::prolate_segv(m, n, c); }
inline double oblate_segv_wrap(double m, double n, double c) { return xsf::oblate_segv(m, n, c); }
inline double prolate_aswfa_nocv_wrap(double m, double n, double c, double x, double *s1d) {
    double s1f;
    xsf::prolate_aswfa_nocv(m, n, c, x, s1f, *s1d);

    return s1f;
}
inline double oblate_aswfa_nocv_wrap(double m, double n, double c, double x, double *s1d) {
    double s1f;
    xsf::oblate_aswfa_nocv(m, n, c, x, s1f, *s1d);

    return s1f;
}
inline void prolate_aswfa_wrap(double m, double n, double c, double cv, double x, double *s1f, double *s1d) {
    xsf::prolate_aswfa(m, n, c, cv, x, *s1f, *s1d);
}
inline void oblate_aswfa_wrap(double m, double n, double c, double cv, double x, double *s1f, double *s1d) {
    xsf::oblate_aswfa(m, n, c, cv, x, *s1f, *s1d);
}
inline double prolate_radial1_nocv_wrap(double m, double n, double c, double x, double *r1d) {
    double r1f;
    xsf::prolate_radial1_nocv(m, n, c, x, r1f, *r1d);

    return r1f;
}
inline double prolate_radial2_nocv_wrap(double m, double n, double c, double x, double *r2d) {
    double r2f;
    xsf::prolate_radial2_nocv(m, n, c, x, r2f, *r2d);

    return r2f;
}
inline void prolate_radial1_wrap(double m, double n, double c, double cv, double x, double *r1f, double *r1d) {
    xsf::prolate_radial1(m, n, c, cv, x, *r1f, *r1d);
}
inline void prolate_radial2_wrap(double m, double n, double c, double cv, double x, double *r2f, double *r2d) {
    xsf::prolate_radial2(m, n, c, cv, x, *r2f, *r2d);
}
inline double oblate_radial1_nocv_wrap(double m, double n, double c, double x, double *r1d) {
    double r1f;
    xsf::oblate_radial1_nocv(m, n, c, x, r1f, *r1d);

    return r1f;
}
inline double oblate_radial2_nocv_wrap(double m, double n, double c, double x, double *r2d) {
    double r2f;
    xsf::oblate_radial2_nocv(m, n, c, x, r2f, *r2d);

    return r2f;
}
inline void oblate_radial1_wrap(double m, double n, double c, double cv, double x, double *r1f, double *r1d) {
    xsf::oblate_radial1(m, n, c, cv, x, *r1f, *r1d);
}
inline void oblate_radial2_wrap(double m, double n, double c, double cv, double x, double *r2f, double *r2d) {
    xsf::oblate_radial2(m, n, c, cv, x, *r2f, *r2d);
}
inline void special_airy(double x, double *ai, double *aip, double *bi, double *bip) { xsf::airy(x, *ai, *aip, *bi, *bip); }
inline void special_airye(double z, double *ai, double *aip, double *bi, double *bip) { xsf::airye(z, *ai, *aip, *bi, *bip); }
inline double cephes_ellpk_wrap(double x) { return xsf::cephes::ellpk(x); }
inline int cephes_fresnl_wrap(double xxa, double *ssa, double *cca) { return xsf::cephes::fresnl(xxa, ssa, cca); }
inline double xsf_binom(double n, double k) { return xsf::binom(n, k); }
inline double special_digamma(double z) { return xsf::digamma(z); }
inline double special_rgamma(double x) { return xsf::rgamma(x); }
inline float special_expitf(float x) { return xsf::expit(x); }
inline double special_expit(double x) { return xsf::expit(x); }
inline float special_log_expitf(float x) { return xsf::log_expit(x); }
inline double special_log_expit(double x) { return xsf::log_expit(x); }
inline float special_logitf(float x) { return xsf::logit(x); }
inline double special_logit(double x) { return xsf::logit(x); }
inline double special_loggamma(double x) { return xsf::loggamma(x); }
inline double cephes_expm1_wrap(double x) { return xsf::cephes::expm1(x); }
inline double cephes_expn_wrap(long n, double x) { return xsf::cephes::expn(static_cast<int>(n), x); }
inline double cephes_log1p_wrap(double x) { return xsf::cephes::log1p(x); }
inline double cephes_jv_wrap(double v, double x) { return xsf::cephes::jv(v, x); }
inline int cephes_ellpj_wrap(double u, double m, double *sn, double *cn, double *dn, double *ph) {
    return xsf::cephes::ellpj(u, m, sn, cn, dn, ph);
}
inline int xsf_sici(double x, double *si, double *ci) { return xsf::sici(x, *si, *ci); }
inline int xsf_shichi(double x, double *si, double *ci) { return xsf::shichi(x, *si, *ci); }
inline double cephes__struve_asymp_large_z(double v, double z, long is_h, double *err) {
    return xsf::cephes::detail::struve_asymp_large_z(v, z, static_cast<int>(is_h), err);
}
inline double cephes__struve_bessel_series(double v, double z, long is_h, double *err) {
    return xsf::cephes::detail::struve_bessel_series(v, z, static_cast<int>(is_h), err);
}
inline double cephes__struve_power_series(double v, double z, long is_h, double *err) {
    return xsf::cephes::detail::struve_power_series(v, z, static_cast<int>(is_h), err);
}
inline double cephes_yn_wrap(long n, double x) { return xsf::cephes::yn(static_cast<int>(n), x); }
inline double cephes_polevl_wrap(double x, const double coef[], int N) { return xsf::cephes::polevl(x, coef, N); }
inline double cephes_p1evl_wrap(double x, const double coef[], int N) { return xsf::cephes::p1evl(x, coef, N); }
inline double special_wright_bessel(double a, double b, double x) { return xsf::wright_bessel(a, b, x); }
inline double special_log_wright_bessel(double a, double b, double x) { return xsf::log_wright_bessel(a, b, x); }
inline double special_scaled_exp1(double x) { return xsf::scaled_exp1(x); }
inline double special_ellipk(double m) { return xsf::ellipk(m); }
inline double xsf_beta(double a, double b) { return xsf::beta(a, b); }
inline double xsf_betaln(double a, double b) { return xsf::betaln(a, b); }
inline double xsf_cbrt(double x) { return xsf::cephes::cbrt(x); }
inline double xsf_gamma(double x) { return xsf::gamma(x); }
inline double xsf_gammaln(double x) { return xsf::gammaln(x); }
inline double xsf_gammasgn(double x) { return xsf::gammasgn(x); }
inline double xsf_hyp2f1(double a, double b, double c, double x) { return xsf::hyp2f1(a, b, c, x); }
inline double cephes_igam(double a, double x) { return xsf::cephes::igam(a, x); }
inline double cephes_igamc(double a, double x) { return xsf::cephes::igamc(a, x); }
inline double cephes_igami(double a, double p) { return xsf::cephes::igami(a, p); }
inline double cephes_igamci(double a, double p) { return xsf::cephes::igamci(a, p); }
inline double cephes_igam_fac(double a, double x) { return xsf::cephes::detail::igam_fac(a, x); }
inline double cephes_lanczos_sum_expg_scaled(double x) { return xsf::cephes::lanczos_sum_expg_scaled(x); }
inline double cephes_poch(double x, double m) { return xsf::cephes::poch(x, m); }
inline double cephes_rgamma(double x) { return xsf::cephes::rgamma(x); }
inline double xsf_zetac(double x) { return xsf::zetac(x); }
inline double cephes_lgam1p(double x) { return xsf::cephes::lgam1p(x); }
inline double cephes_expn(int n, double x) { return xsf::cephes::expn(n, x); }
inline double xsf_ellipe(double x) { return xsf::ellipe(x); }
inline double xsf_erf(double x) { return xsf::erf(x); }
inline double xsf_erfc(double x) { return xsf::erfc(x); }
inline double xsf_erfcx(double x) { return xsf::erfcx(x); }
inline double xsf_dawsn(double x) { return xsf::dawsn(x); }
inline double xsf_erfi(double x) { return xsf::erfi(x); }
inline double xsf_voigt_profile(double x, double sigma, double gamma) { return xsf::voigt_profile(x, sigma, gamma); }
inline double cephes_ellpk(double x) { return xsf::ellipkm1(x); }
inline double cephes_ellie(double phi, double m) { return xsf::ellipeinc(phi, m); }
inline double xsf_ellipkinc(double phi, double m) { return xsf::ellipkinc(phi, m); }
inline double cephes_poch_wrap(double x, double m) { return xsf::cephes::poch(x, m); }
inline double cephes_erfcinv(double y) { return xsf::cephes::erfcinv(y); }
inline double cephes_round(double x) { return xsf::cephes::round(x); }
inline double cephes_spence(double x) { return xsf::cephes::spence(x); }
inline double xsf_struve_h(double v, double z) { return xsf::struve_h(v, z); }
inline double xsf_struve_l(double v, double z) { return xsf::struve_l(v, z); }
inline double xsf_expm1(double x) { return xsf::expm1(x); }
inline double xsf_exp2(double x) { return xsf::exp2(x); }
inline double xsf_exp10(double x) { return xsf::exp10(x); }
inline double xsf_log1p(double x) { return xsf::log1p(x); }
inline double xsf_xlogy(double x, double y) { return xsf::xlogy(x, y); }
inline double xsf_xlog1py(double x, double y) { return xsf::xlog1py(x, y); }
inline double xsf_i0(double x) { return xsf::cyl_bessel_i0(x); }
inline double xsf_i0e(double x) { return xsf::cyl_bessel_i0e(x); }
inline double xsf_i1(double x) { return xsf::cyl_bessel_i1(x); }
inline double xsf_i1e(double x) { return xsf::cyl_bessel_i1e(x); }
inline double xsf_iv(double v, double x) { return xsf::cyl_bessel_i(v, x); }
inline double xsf_j0(double x) { return xsf::cyl_bessel_j0(x); }
inline double xsf_j1(double x) { return xsf::cyl_bessel_j1(x); }
inline double xsf_k0(double x) { return xsf::cyl_bessel_k0(x); }
inline double xsf_k0e(double x) { return xsf::cyl_bessel_k0e(x); }
inline double xsf_k1(double x) { return xsf::cyl_bessel_k1(x); }
inline double xsf_k1e(double x) { return xsf::cyl_bessel_k1e(x); }
inline double xsf_y0(double x) { return xsf::cyl_bessel_y0(x); }
inline double xsf_y1(double x) { return xsf::cyl_bessel_y1(x); }
inline double cephes_yn(int n, double x) { return xsf::cephes::yn(n, x); }
inline double special_cyl_bessel_j(double v, double x) { return xsf::cyl_bessel_j(v, x); }
inline double special_cyl_bessel_je(double v, double z) { return xsf::cyl_bessel_je(v, z); }
inline double special_cyl_bessel_y(double v, double x) { return xsf::cyl_bessel_y(v, x); }
inline double special_cyl_bessel_ye(double v, double z) { return xsf::cyl_bessel_ye(v, z); }
inline double special_cyl_bessel_i(double v, double z) { return xsf::cyl_bessel_i(v, z); }
inline double special_cyl_bessel_ie(double v, double z) { return xsf::cyl_bessel_ie(v, z); }
inline double special_cyl_bessel_k_int(long n, double z) { return xsf::cyl_bessel_k(static_cast<double>(n), z); }
inline double special_cyl_bessel_k(double v, double z) { return xsf::cyl_bessel_k(v, z); }
inline double special_cyl_bessel_ke(double v, double z) { return xsf::cyl_bessel_ke(v, z); }
inline double xsf_besselpoly(double a, double lambda, double nu) { return xsf::besselpoly(a, lambda, nu); }
inline double special_sph_bessel_j(long n, double x) { return xsf::sph_bessel_j(n, x); }
inline double special_sph_bessel_j_jac(long n, double x) { return xsf::sph_bessel_j_jac(n, x); }
inline double special_sph_bessel_y(long n, double x) { return xsf::sph_bessel_y(n, x); }
inline double special_sph_bessel_y_jac(long n, double x) { return xsf::sph_bessel_y_jac(n, x); }
inline double special_sph_bessel_i(long n, double x) { return xsf::sph_bessel_i(n, x); }
inline double special_sph_bessel_i_jac(long n, double x) { return xsf::sph_bessel_i_jac(n, x); }
inline double special_sph_bessel_k(long n, double x) { return xsf::sph_bessel_k(n, x); }
inline double special_sph_bessel_k_jac(long n, double x) { return xsf::sph_bessel_k_jac(n, x); }
inline double xsf_bdtr(double k, int n, double p) { return xsf::bdtr(k, n, p); }
inline double xsf_bdtri(double k, int n, double y) { return xsf::bdtri(k, n, y); }
inline double xsf_bdtrc(double k, int n, double p) { return xsf::bdtrc(k, n, p); }
inline double xsf_chdtr(double df, double x) { return xsf::chdtr(df, x); }
inline double xsf_chdtrc(double df, double x) { return xsf::chdtrc(df, x); }
inline double xsf_chdtri(double df, double y) { return xsf::chdtri(df, y); }
inline double xsf_gdtr(double a, double b, double x) { return xsf::gdtr(a, b, x); }
inline double xsf_gdtrc(double a, double b, double x) { return xsf::gdtrc(a, b, x); }
inline double special_gdtria(double p, double b, double x) { 
    if (x == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if ((b == 0) & (p == 0)) {
        if (std::isinf(x) && (x > 0)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 0.0;
    }
    return xsf::gammaincinv(b, p) / x;
}
inline double special_gdtrix(double a, double b, double p) { 
    if ((a == 0) && (b == 0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // if a or b is positive infinite, return NaN
    if ((std::isinf(a) || std::isinf(b)) && (a >= 0 && b >= 0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return xsf::gammaincinv(b, p) / a;
}
inline double xsf_gdtrib(double a, double p, double x) { return xsf::gdtrib(a, p, x); }
inline double xsf_kolmogorov(double x) { return xsf::kolmogorov(x); }
inline double xsf_kolmogc(double x) { return xsf::kolmogc(x); }
inline double xsf_kolmogi(double x) { return xsf::kolmogi(x); }
inline double xsf_kolmogci(double x) { return xsf::kolmogci(x); }
inline double xsf_kolmogp(double x) { return xsf::kolmogp(x); }
inline double xsf_nbdtr(int k, int n, double p) { return xsf::nbdtr(k, n, p); }
inline double xsf_nbdtrc(int k, int n, double p) { return xsf::nbdtrc(k, n, p); }
inline double xsf_nbdtri(int k, int n, double p) { return xsf::nbdtri(k, n, p); }
inline double xsf_ndtr(double x) { return xsf::ndtr(x); }
inline double xsf_log_ndtr(double x) { return xsf::log_ndtr(x); }
inline double xsf_ndtri(double x) { return xsf::ndtri(x); }
inline double xsf_owens_t(double h, double a) { return xsf::owens_t(h, a); }
inline double xsf_pdtr(double k, double m) { return xsf::pdtr(k, m); }
inline double xsf_pdtrc(double k, double m) { return xsf::pdtrc(k, m); }
inline double xsf_pdtri(int k, double y) { return xsf::pdtri(k, y); }
inline double xsf_smirnov(int n, double x) { return xsf::smirnov(n, x); }
inline double xsf_smirnovc(int n, double x) { return xsf::smirnovc(n, x); }
inline double xsf_smirnovi(int n, double x) { return xsf::smirnovi(n, x); }
inline double xsf_smirnovci(int n, double x) { return xsf::smirnovci(n, x); }
inline double xsf_smirnovp(int n, double x) { return xsf::smirnovp(n, x); }
inline double xsf_tukeylambdacdf(double x, double lmbda) { return xsf::tukeylambdacdf(x, lmbda); }
inline double cephes_bdtr_wrap(double k, long n, double p) { return xsf::bdtr(k, static_cast<int>(n), p); }
inline double cephes_bdtri_wrap(double k, long n, double y) { return xsf::bdtri(k, static_cast<int>(n), y); }
inline double cephes_bdtrc_wrap(double k, long n, double p) { return xsf::bdtrc(k, static_cast<int>(n), p); }
inline double cephes_nbdtr_wrap(long k, long n, double p) {
    return xsf::cephes::nbdtr(static_cast<int>(k), static_cast<int>(n), p);
}
inline double cephes_nbdtrc_wrap(long k, long n, double p) {
    return xsf::cephes::nbdtrc(static_cast<int>(k), static_cast<int>(n), p);
}
inline double cephes_nbdtri_wrap(long k, long n, double p) {
    return xsf::cephes::nbdtri(static_cast<int>(k), static_cast<int>(n), p);
}
inline double cephes_ndtr_wrap(double x) { return xsf::cephes::ndtr(x); }
inline double cephes_ndtri_wrap(double x) { return xsf::cephes::ndtri(x); }
inline double cephes_pdtri_wrap(long k, double y) { return xsf::cephes::pdtri(static_cast<int>(k), y); }
inline double cephes_smirnov_wrap(long n, double x) { return xsf::cephes::smirnov(static_cast<int>(n), x); }
inline double cephes_smirnovc_wrap(long n, double x) { return xsf::cephes::smirnovc(static_cast<int>(n), x); }
inline double cephes_smirnovi_wrap(long n, double x) { return xsf::cephes::smirnovi(static_cast<int>(n), x); }
inline double cephes_smirnovci_wrap(long n, double x) { return xsf::cephes::smirnovci(static_cast<int>(n), x); }
inline double cephes_smirnovp_wrap(long n, double x) { return xsf::cephes::smirnovp(static_cast<int>(n), x); }
inline double xsf_sinpi(double x) { return xsf::sinpi(x); }
inline double xsf_cospi(double x) { return xsf::cospi(x); }
inline double xsf_cosm1(double x) { return xsf::cosm1(x); }
inline double xsf_sindg(double x) { return xsf::sindg(x); }
inline double xsf_cosdg(double x) { return xsf::cosdg(x); }
inline double xsf_tandg(double x) { return xsf::tandg(x); }
inline double xsf_cotdg(double x) { return xsf::cotdg(x); }
inline double xsf_radian(double d, double m, double s) { return xsf::radian(d, m, s); }

} // namespace scipy_xsfw
