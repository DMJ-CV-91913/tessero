/*
 * numpy.random.Generator methods in the function registry (ADR 0011): the variates come from NumPy's own
 * distribution code (third_party/numpy/random, vendored unchanged) driven by the kernel's PCG64, so that
 *
 *     Generator::defaultRng(42)->gamma(2.0, size: 5)   ==   numpy.random.default_rng(42).gamma(2.0, size=5)
 *
 * bit for bit. Parameter checks follow numpy/random/_common.pyx (check_constraint) and the method bodies of
 * numpy/random/_generator.pyx; all elements are checked before anything is drawn, as NumPy does.
 */
#include "fn.h"
#include "internal.h"
#include "numpy/random/distributions.h"
#include <stdio.h>
#include <string.h>

/* POISSON_LAM_MAX of numpy/random/_common.pyx: int64 max - sqrt(int64 max) * 10 */
#define POISSON_LAM_MAX (9223372036854775807.0 - 3037000499.97605 * 10)

static double d_random(bitgen_t *b, const double *p) { (void)p; return random_standard_uniform(b); }
static float f_random(bitgen_t *b, const double *p) { (void)p; return random_standard_uniform_f(b); }
static double d_std_normal(bitgen_t *b, const double *p) { (void)p; return random_standard_normal(b); }
static float f_std_normal(bitgen_t *b, const double *p) { (void)p; return random_standard_normal_f(b); }
/* standard_exponential(method='zig'|'inv'): the enum parameter follows the (no) array arguments */
static double d_std_exp(bitgen_t *b, const double *p)
{
    if (p[1] == 0.0) return random_standard_exponential(b);   /* p[0] dtype, p[1] method */
    double x;
    random_standard_exponential_inv_fill(b, 1, &x);
    return x;
}
static float f_std_exp(bitgen_t *b, const double *p)
{
    if (p[1] == 0.0) return random_standard_exponential_f(b);
    float x;
    random_standard_exponential_inv_fill_f(b, 1, &x);
    return x;
}
static double d_std_gamma(bitgen_t *b, const double *p) { return random_standard_gamma(b, p[0]); }
static float f_std_gamma(bitgen_t *b, const double *p) { return random_standard_gamma_f(b, (float)p[0]); }
static double d_gamma(bitgen_t *b, const double *p) { return random_gamma(b, p[0], p[1]); }
static double d_beta(bitgen_t *b, const double *p) { return random_beta(b, p[0], p[1]); }
static double d_exponential(bitgen_t *b, const double *p) { return random_exponential(b, p[0]); }
/* uniform: the range high - low is formed (and checked) first, as numpy does */
static double d_uniform(bitgen_t *b, const double *p) { return random_uniform(b, p[0], p[1] - p[0]); }
static double d_normal(bitgen_t *b, const double *p) { return random_normal(b, p[0], p[1]); }
static double d_f(bitgen_t *b, const double *p) { return random_f(b, p[0], p[1]); }
static double d_noncentral_f(bitgen_t *b, const double *p) { return random_noncentral_f(b, p[0], p[1], p[2]); }
static double d_chisquare(bitgen_t *b, const double *p) { return random_chisquare(b, p[0]); }
static double d_noncentral_chisquare(bitgen_t *b, const double *p) { return random_noncentral_chisquare(b, p[0], p[1]); }
static double d_std_cauchy(bitgen_t *b, const double *p) { (void)p; return random_standard_cauchy(b); }
static double d_std_t(bitgen_t *b, const double *p) { return random_standard_t(b, p[0]); }
static double d_vonmises(bitgen_t *b, const double *p) { return random_vonmises(b, p[0], p[1]); }
static double d_pareto(bitgen_t *b, const double *p) { return random_pareto(b, p[0]); }
static double d_weibull(bitgen_t *b, const double *p) { return random_weibull(b, p[0]); }
static double d_power(bitgen_t *b, const double *p) { return random_power(b, p[0]); }
static double d_laplace(bitgen_t *b, const double *p) { return random_laplace(b, p[0], p[1]); }
static double d_gumbel(bitgen_t *b, const double *p) { return random_gumbel(b, p[0], p[1]); }
static double d_logistic(bitgen_t *b, const double *p) { return random_logistic(b, p[0], p[1]); }
static double d_lognormal(bitgen_t *b, const double *p) { return random_lognormal(b, p[0], p[1]); }
static double d_rayleigh(bitgen_t *b, const double *p) { return random_rayleigh(b, p[0]); }
static double d_wald(bitgen_t *b, const double *p) { return random_wald(b, p[0], p[1]); }
static double d_triangular(bitgen_t *b, const double *p) { return random_triangular(b, p[0], p[1], p[2]); }

static int64_t i_binomial(bitgen_t *b, const double *p)
{
    binomial_t bin = {0};                  /* NumPy caches the setup per Generator; the variates do not depend on it */
    return random_binomial(b, p[1], (int64_t)p[0], &bin);
}
static int64_t i_negative_binomial(bitgen_t *b, const double *p) { return random_negative_binomial(b, p[0], p[1]); }
static int64_t i_poisson(bitgen_t *b, const double *p) { return random_poisson(b, p[0]); }
static int64_t i_zipf(bitgen_t *b, const double *p) { return random_zipf(b, p[0]); }
static int64_t i_geometric(bitgen_t *b, const double *p) { return random_geometric(b, p[0]); }
static int64_t i_hypergeometric(bitgen_t *b, const double *p)
{
    return random_hypergeometric(b, (int64_t)p[0], (int64_t)p[1], (int64_t)p[2]);
}
static int64_t i_logseries(bitgen_t *b, const double *p) { return random_logseries(b, p[0]); }

/* ---------------------------------------------------------------- method-specific checks */

static const char *chk_uniform(const double *p)
{
    const double range = p[1] - p[0];
    if (!isfinite(range)) return "Range exceeds valid bounds";
    if (!isnan(range) && signbit(range)) return "high - low < 0";
    return NULL;
}

static const char *chk_triangular(const double *p)
{
    if (p[0] > p[1]) return "left > mode";
    if (p[1] > p[2]) return "mode > right";
    if (p[0] == p[2]) return "left == right";
    return NULL;
}

/* NumPy converts integer parameters (binomial n, hypergeometric counts) to int64 first: NaN and values outside
   int64 fail there (ValueError / OverflowError), before any range check */
static const char *int64_param_error(double v)
{
    if (v != v) return "cannot convert float NaN to integer";
    if (!(v > -9223372036854775808.0 && v < 9223372036854775808.0)) return "Python int too large to convert to C long";
    return NULL;
}

static const char *chk_binomial(const double *p)
{
    const char *e = int64_param_error(p[0]);
    if (e) return e;
    if (!(p[1] >= 0) || !(p[1] <= 1)) return "p < 0, p > 1 or p is NaN";
    if ((int64_t)p[0] < 0) return "n < 0";
    return NULL;
}

static const char *chk_negative_binomial(const double *p)
{
    if (isnan(p[0])) return "n must not be NaN";
    if (p[0] <= 0) return "n <= 0";
    if (!(p[1] > 0) || !(p[1] <= 1)) return "p <= 0, p > 1 or p contains NaNs";
    if ((1 - p[1]) / p[1] * (p[0] + 10 * sqrt(p[0])) > POISSON_LAM_MAX)
        return "n too large or p too small, see Generator.negative_binomial Notes";
    return NULL;
}

static const char *chk_hypergeometric(const double *p)
{
    for (int k = 0; k < 3; k++) {
        const char *e = int64_param_error(p[k]);
        if (e) return e;
    }
    const int64_t g = (int64_t)p[0], b = (int64_t)p[1], s = (int64_t)p[2];
    if (g >= 1000000000 || b >= 1000000000) return "both ngood and nbad must be less than 1000000000";
    if (g + b < s) return "ngood + nbad < nsample";
    if (g < 0) return "ngood < 0";
    if (b < 0) return "nbad < 0";
    if (s < 0) return "nsample < 0";
    return NULL;
}

static const fn_def NP_RANDOM[] = {
    RANDOM("random.random", 0, "", "dtype=float64|float32", d_random, NULL, f_random, "", NULL,
           "Floats in the half-open interval [0.0, 1.0) (Generator.random)."),
    RANDOM("random.standard_normal", 0, "", "dtype=float64|float32", d_std_normal, NULL, f_std_normal, "", NULL,
           "Samples from the standard normal distribution, ziggurat method (Generator.standard_normal)."),
    RANDOM("random.standard_exponential", 0, "", "dtype=float64|float32, method=zig|inv", d_std_exp, NULL, f_std_exp, "", NULL,
           "Samples from the standard exponential distribution (Generator.standard_exponential)."),
    RANDOM("random.standard_gamma", 1, "shape", "dtype=float64|float32", d_std_gamma, NULL, f_std_gamma, "n", NULL,
           "Samples from the standard gamma distribution (Generator.standard_gamma)."),
    RANDOM("random.gamma", 2, "shape, scale=1.0", "", d_gamma, NULL, NULL, "nn", NULL,
           "Samples from a gamma distribution (Generator.gamma)."),
    RANDOM("random.beta", 2, "a, b", "", d_beta, NULL, NULL, "pp", NULL, "Samples from a beta distribution (Generator.beta)."),
    RANDOM("random.exponential", 1, "scale=1.0", "", d_exponential, NULL, NULL, "n", NULL,
           "Samples from an exponential distribution (Generator.exponential)."),
    RANDOM("random.uniform", 2, "low=0.0, high=1.0", "", d_uniform, NULL, NULL, "..", chk_uniform,
           "Samples from a uniform distribution over [low, high) (Generator.uniform)."),
    RANDOM("random.normal", 2, "loc=0.0, scale=1.0", "", d_normal, NULL, NULL, ".n", NULL,
           "Samples from a normal (Gaussian) distribution (Generator.normal)."),
    RANDOM("random.f", 2, "dfnum, dfden", "", d_f, NULL, NULL, "pp", NULL, "Samples from an F distribution (Generator.f)."),
    RANDOM("random.noncentral_f", 3, "dfnum, dfden, nonc", "", d_noncentral_f, NULL, NULL, "ppn", NULL,
           "Samples from the noncentral F distribution (Generator.noncentral_f)."),
    RANDOM("random.chisquare", 1, "df", "", d_chisquare, NULL, NULL, "p", NULL,
           "Samples from a chi-square distribution (Generator.chisquare)."),
    RANDOM("random.noncentral_chisquare", 2, "df, nonc", "", d_noncentral_chisquare, NULL, NULL, "pn", NULL,
           "Samples from a noncentral chi-square distribution (Generator.noncentral_chisquare)."),
    RANDOM("random.standard_cauchy", 0, "", "", d_std_cauchy, NULL, NULL, "", NULL,
           "Samples from a standard Cauchy distribution (Generator.standard_cauchy)."),
    RANDOM("random.standard_t", 1, "df", "", d_std_t, NULL, NULL, "p", NULL,
           "Samples from a standard Student's t distribution (Generator.standard_t)."),
    RANDOM("random.vonmises", 2, "mu, kappa", "", d_vonmises, NULL, NULL, ".n", NULL,
           "Samples from a von Mises distribution (Generator.vonmises)."),
    RANDOM("random.pareto", 1, "a", "", d_pareto, NULL, NULL, "p", NULL,
           "Samples from a Pareto II (Lomax) distribution (Generator.pareto)."),
    RANDOM("random.weibull", 1, "a", "", d_weibull, NULL, NULL, "n", NULL, "Samples from a Weibull distribution (Generator.weibull)."),
    RANDOM("random.power", 1, "a", "", d_power, NULL, NULL, "p", NULL,
           "Samples in [0, 1] from a power distribution with positive exponent a - 1 (Generator.power)."),
    RANDOM("random.laplace", 2, "loc=0.0, scale=1.0", "", d_laplace, NULL, NULL, ".n", NULL,
           "Samples from the Laplace or double exponential distribution (Generator.laplace)."),
    RANDOM("random.gumbel", 2, "loc=0.0, scale=1.0", "", d_gumbel, NULL, NULL, ".n", NULL,
           "Samples from a Gumbel distribution (Generator.gumbel)."),
    RANDOM("random.logistic", 2, "loc=0.0, scale=1.0", "", d_logistic, NULL, NULL, ".n", NULL,
           "Samples from a logistic distribution (Generator.logistic)."),
    RANDOM("random.lognormal", 2, "mean=0.0, sigma=1.0", "", d_lognormal, NULL, NULL, ".n", NULL,
           "Samples from a log-normal distribution (Generator.lognormal)."),
    RANDOM("random.rayleigh", 1, "scale=1.0", "", d_rayleigh, NULL, NULL, "n", NULL,
           "Samples from a Rayleigh distribution (Generator.rayleigh)."),
    RANDOM("random.wald", 2, "mean, scale", "", d_wald, NULL, NULL, "pp", NULL,
           "Samples from a Wald, or inverse Gaussian, distribution (Generator.wald)."),
    RANDOM("random.triangular", 3, "left, mode, right", "", d_triangular, NULL, NULL, "...", chk_triangular,
           "Samples from the triangular distribution over [left, right] (Generator.triangular)."),
    RANDOM("random.binomial", 2, "n, p", "", NULL, i_binomial, NULL, "..", chk_binomial,
           "Samples from a binomial distribution (Generator.binomial)."),
    RANDOM("random.negative_binomial", 2, "n, p", "", NULL, i_negative_binomial, NULL, "..", chk_negative_binomial,
           "Samples from a negative binomial distribution (Generator.negative_binomial)."),
    RANDOM("random.poisson", 1, "lam=1.0", "", NULL, i_poisson, NULL, "L", NULL,
           "Samples from a Poisson distribution (Generator.poisson)."),
    RANDOM("random.zipf", 1, "a", "", NULL, i_zipf, NULL, "1", NULL, "Samples from a Zipf distribution (Generator.zipf)."),
    RANDOM("random.geometric", 1, "p", "", NULL, i_geometric, NULL, "g", NULL,
           "Samples from the geometric distribution (Generator.geometric)."),
    RANDOM("random.hypergeometric", 3, "ngood, nbad, nsample", "", NULL, i_hypergeometric, NULL, "...", chk_hypergeometric,
           "Samples from a hypergeometric distribution (Generator.hypergeometric)."),
    RANDOM("random.logseries", 1, "p", "", NULL, i_logseries, NULL, "l", NULL,
           "Samples from a logarithmic series distribution (Generator.logseries)."),
};
const fn_table TSR_NP_RANDOM_TABLE = {NP_RANDOM, (int)(sizeof NP_RANDOM / sizeof NP_RANDOM[0])};

/* ---------------------------------------------------------------- the draw loop */

static int check_cons(char c, double v, const char *name, char *msg, size_t cap)
{
    const char *why = NULL;
    switch (c) {
    case RC_NON_NEGATIVE: if (!isnan(v) && signbit(v)) why = "< 0"; break;
    case RC_POSITIVE: if (v <= 0) why = "<= 0"; break;
    case RC_POSITIVE_NOT_NAN: if (isnan(v)) why = "must not be NaN"; else if (v <= 0) why = "<= 0"; break;
    case RC_BOUNDED_0_1: if (!(v >= 0) || !(v <= 1)) why = "is outside [0, 1] or NaN"; break;
    case RC_BOUNDED_GT_0_1: if (!(v > 0) || !(v <= 1)) why = "<= 0, > 1 or NaN"; break;
    case RC_BOUNDED_LT_0_1: if (!(v >= 0) || !(v < 1)) why = "< 0, >= 1 or NaN"; break;
    case RC_GT_1: if (!(v > 1)) why = "<= 1 or NaN"; break;
    case RC_POISSON:
        if (!(v >= 0)) why = "< 0 or NaN";
        else if (!(v <= POISSON_LAM_MAX)) why = "value too large";
        break;
    default: break;
    }
    if (!why) return 0;
    snprintf(msg, cap, "%s %s", name, why);
    return 1;
}

/* the name of array argument k ("a, b=1.0" -> "b") */
static void arg_name(const char *args, int k, char *out, size_t cap)
{
    const char *p = args;
    for (int i = 0; i < k && p; i++) { p = strchr(p, ','); if (p) p++; }
    out[0] = '\0';
    if (!p) return;
    while (*p == ' ') p++;
    size_t n = 0;
    while (p[n] && p[n] != ',' && p[n] != '=' && n + 1 < cap) { out[n] = p[n]; n++; }
    out[n] = '\0';
}

int tsr_random_run(const fn_def *d, uint64_t *state, const double *params, int32_t ndim, const int64_t *shape,
                   int nop, void **data, const int64_t *strides, int out_dtype)
{
    const int nin = d->nin;
    if (nop != nin + 1 || nin > 8) return TSR_EARG;
    if (out_dtype == TSR_F32 ? !d->rdraw_f : (out_dtype == TSR_I64 ? !d->rdraw_i : !d->rdraw)) return TSR_ETYPE;
    int np = 0;
    if (d->params && *d->params) { np = 1; for (const char *c = d->params; *c; c++) if (*c == ',') np++; }
    const int64_t size = tsr_shape_size(ndim, shape);
    if (size < 0) return TSR_ENOMEM;
    bitgen_t bg;
    tsr_pcg64_bitgen(state, &bg);
    double p[16];
    for (int k = 0; k < np && nin + k < 16; k++) p[nin + k] = params ? params[k] : 0.0;
    int64_t idx[TSR_MAXDIM];
    /* pass 1: check every element's parameters (numpy raises before drawing) */
    if (nin > 0) {
        memset(idx, 0, sizeof idx);
        for (int64_t e = 0; e < size; e++) {
            for (int o = 0; o < nin; o++) {
                const char *q = (const char *)data[o];
                for (int32_t a = 0; a < ndim; a++) q += idx[a] * strides[o * ndim + a];
                p[o] = *(const double *)q;
            }
            char msg[160];
            for (int o = 0; o < nin; o++) {
                char name[32];
                arg_name(d->args, o, name, sizeof name);
                if (d->cons && d->cons[o] && check_cons(d->cons[o], p[o], name, msg, sizeof msg)) {
                    fn_set_error("%s", msg);
                    return TSR_EARG;
                }
            }
            if (d->rcheck) {
                const char *why = d->rcheck(p);
                if (why) { fn_set_error("%s", why); return TSR_EARG; }
            }
            for (int32_t a = ndim - 1; a >= 0; a--) { if (++idx[a] < shape[a]) break; idx[a] = 0; }
        }
    }
    /* pass 2: draw, serially and in C order (the stream order is NumPy's) */
    memset(idx, 0, sizeof idx);
    for (int64_t e = 0; e < size; e++) {
        for (int o = 0; o <= nin; o++) {
            char *q = (char *)data[o];
            for (int32_t a = 0; a < ndim; a++) q += idx[a] * strides[o * ndim + a];
            if (o < nin) { p[o] = *(const double *)q; continue; }
            if (out_dtype == TSR_I64) *(int64_t *)q = d->rdraw_i(&bg, p);
            else if (out_dtype == TSR_F32) *(float *)q = d->rdraw_f(&bg, p);
            else *(double *)q = d->rdraw(&bg, p);
        }
        for (int32_t a = ndim - 1; a >= 0; a--) { if (++idx[a] < shape[a]) break; idx[a] = 0; }
    }
    return TSR_OK;
}
