/*
 * Loop ufuncs: element-wise functions defined as tables of typed inner loops,
 * as in NumPy. Shared by both bindings: ext-tessero's Tessero\Ext\Math and the
 * FFI package's Tessero\Math call the same code, so their results are
 * identical to the bit.
 *
 * A loop processes one run of n elements: loop(args, n, steps), args[] the
 * input pointers then the output pointer, steps[] their byte strides. The
 * engine (tsr_ufunc) coalesces dimensions with tsr_iter and calls the loop
 * once per inner run, splitting large arrays across OpenMP threads.
 * Operands arrive already broadcast (zero strides) and already in the loop's
 * input dtype; tsr_ufunc_resolve() tells the caller which dtype that is.
 *
 * Vectorisation: every loop has a contiguous fast path over restrict pointers
 * compiled for AVX-512, AVX2 and baseline (TSR_CLONES) and marked `omp simd`.
 * No -ffast-math: NaN and signed-zero semantics are kept.
 *
 * Results do not depend on the thread count: each element is computed by the
 * same code whichever thread owns it.
 */
#include "internal.h"
#include "ops.h"

#include <math.h>
#include <string.h>

typedef void (*tsr_loop_fn)(char **args, int64_t n, const int64_t *steps);

/* ================================================================ inner loops */

#if defined(_OPENMP)
#  define TSR_SIMD _Pragma("omp simd")
#else
#  define TSR_SIMD
#endif

/* one input -> one output; EXPR uses `v` */
#define LOOP1(NAME, TI, TO, EXPR)                                                             \
    TSR_CLONES static void NAME(char **args, int64_t n, const int64_t *st)                    \
    {                                                                                         \
        const char *ip = args[0];                                                             \
        char *op = args[1];                                                                   \
        if (st[0] == (int64_t)sizeof(TI) && st[1] == (int64_t)sizeof(TO)) {                   \
            if ((const void *)ip != (const void *)op) {                                       \
                const TI *restrict x = (const TI *)ip;                                        \
                TO *restrict y = (TO *)op;                                                    \
                TSR_SIMD for (int64_t i = 0; i < n; i++) { const TI v = x[i]; y[i] = (TO)(EXPR); } \
            } else {                                                                          \
                TI *x = (TI *)op;                                                             \
                for (int64_t i = 0; i < n; i++) { const TI v = x[i]; ((TO *)x)[i] = (TO)(EXPR); } \
            }                                                                                 \
            return;                                                                           \
        }                                                                                     \
        for (int64_t i = 0; i < n; i++) {                                                     \
            const TI v = *(const TI *)(ip + i * st[0]);                                       \
            *(TO *)(op + i * st[1]) = (TO)(EXPR);                                             \
        }                                                                                     \
    }

/* two inputs -> one output; EXPR uses `a`, `b`. Fast paths: all contiguous, b broadcast scalar. */
#define LOOP2(NAME, T, TO, EXPR)                                                              \
    TSR_CLONES static void NAME(char **args, int64_t n, const int64_t *st)                    \
    {                                                                                         \
        const char *ap = args[0], *bp = args[1];                                              \
        char *op = args[2];                                                                   \
        const int64_t s = (int64_t)sizeof(T), so = (int64_t)sizeof(TO);                       \
        if (st[0] == s && st[1] == s && st[2] == so && (const void *)ap != (const void *)op   \
            && (const void *)bp != (const void *)op) {                                        \
            const T *restrict xa = (const T *)ap, *restrict xb = (const T *)bp;               \
            TO *restrict y = (TO *)op;                                                        \
            TSR_SIMD for (int64_t i = 0; i < n; i++) { const T a = xa[i], b = xb[i]; y[i] = (TO)(EXPR); } \
            return;                                                                           \
        }                                                                                     \
        if (st[0] == s && st[1] == 0 && st[2] == so && (const void *)ap != (const void *)op) { \
            const T *restrict xa = (const T *)ap;                                             \
            const T b = *(const T *)bp;                                                       \
            TO *restrict y = (TO *)op;                                                        \
            TSR_SIMD for (int64_t i = 0; i < n; i++) { const T a = xa[i]; y[i] = (TO)(EXPR); } \
            return;                                                                           \
        }                                                                                     \
        for (int64_t i = 0; i < n; i++) {                                                     \
            const T a = *(const T *)(ap + i * st[0]), b = *(const T *)(bp + i * st[1]);       \
            *(TO *)(op + i * st[2]) = (TO)(EXPR);                                             \
        }                                                                                     \
    }

static const double DEG = 180.0 / M_PI, RAD = M_PI / 180.0;
static const float DEGF = (float)(180.0 / M_PI), RADF = (float)(M_PI / 180.0);
#define LOGE2 0.693147180559945309417232121458176568
#define LOG2E 1.442695040888963407359924681001892137

/* numpy's npy_logaddexp / npy_logaddexp2, including the x == y and NaN branches */
static inline double lae(double x, double y)
{
    if (x == y) return x + LOGE2;             /* also handles infinities of the same sign */
    const double t = x - y;
    if (t > 0) return x + log1p(exp(-t));
    if (t <= 0) return y + log1p(exp(t));
    return t;                                 /* NaN */
}
static inline float laef(float x, float y)
{
    if (x == y) return x + (float)LOGE2;
    const float t = x - y;
    if (t > 0) return x + log1pf(expf(-t));
    if (t <= 0) return y + log1pf(expf(t));
    return t;
}
static inline double lae2(double x, double y)
{
    if (x == y) return x + 1.0;
    const double t = x - y;
    if (t > 0) return x + LOG2E * log1p(exp2(-t));
    if (t <= 0) return y + LOG2E * log1p(exp2(t));
    return t;
}
static inline float lae2f(float x, float y)
{
    if (x == y) return x + 1.0f;
    const float t = x - y;
    if (t > 0) return x + (float)LOG2E * log1pf(exp2f(-t));
    if (t <= 0) return y + (float)LOG2E * log1pf(exp2f(t));
    return t;
}
/* C fmod on integers (sign of the dividend, as numpy.fmod); x % 0 -> 0 and MIN % -1 -> 0 instead of UB */
#define IFMOD(T, MIN) ((b == 0 || (b == (T)-1 && a == (MIN))) ? (T)0 : (T)(a % b))

/* cbrt is the C library's, as NumPy's float64 cbrt is on every CPU without AVX-512 (and in the pinned
   reference, ADR 0012); glibc returns -3.0000000000000004 for cbrt(-27), and so does NumPy there */
LOOP1(l_cbrt_d, double, double, cbrt(v))         LOOP1(l_cbrt_f, float, float, cbrtf(v))
LOOP1(l_exp2_d, double, double, exp2(v))            LOOP1(l_exp2_f, float, float, exp2f(v))
LOOP1(l_trunc_d, double, double, trunc(v))          LOOP1(l_trunc_f, float, float, truncf(v))
LOOP1(l_asinh_d, double, double, asinh(v))          LOOP1(l_asinh_f, float, float, asinhf(v))
LOOP1(l_acosh_d, double, double, acosh(v))          LOOP1(l_acosh_f, float, float, acoshf(v))
LOOP1(l_atanh_d, double, double, atanh(v))          LOOP1(l_atanh_f, float, float, atanhf(v))
LOOP1(l_deg_d, double, double, v * DEG)             LOOP1(l_deg_f, float, float, v * DEGF)
LOOP1(l_rad_d, double, double, v * RAD)             LOOP1(l_rad_f, float, float, v * RADF)
LOOP1(l_erf_d, double, double, erf(v))              LOOP1(l_erf_f, float, float, erff(v))
LOOP1(l_erfc_d, double, double, erfc(v))            LOOP1(l_erfc_f, float, float, erfcf(v))
LOOP1(l_gamma_d, double, double, tgamma(v))         LOOP1(l_gamma_f, float, float, tgammaf(v))
LOOP1(l_signbit_d, double, uint8_t, signbit(v) != 0) LOOP1(l_signbit_f, float, uint8_t, signbit(v) != 0)
LOOP2(l_fmax_d, double, double, fmax(a, b))         LOOP2(l_fmax_f, float, float, fmaxf(a, b))
LOOP2(l_fmin_d, double, double, fmin(a, b))         LOOP2(l_fmin_f, float, float, fminf(a, b))
LOOP2(l_copysign_d, double, double, copysign(a, b)) LOOP2(l_copysign_f, float, float, copysignf(a, b))
LOOP2(l_nextafter_d, double, double, nextafter(a, b)) LOOP2(l_nextafter_f, float, float, nextafterf(a, b))
/* numpy.heaviside: 0 for x1<0, x2 for x1==0, 1 for x1>0; NaN propagates from x1 */
LOOP2(l_heaviside_d, double, double, (isnan(a) ? a : (a < 0.0 ? 0.0 : (a > 0.0 ? 1.0 : b))))
LOOP2(l_heaviside_f, float, float, (isnan(a) ? a : (a < 0.0f ? 0.0f : (a > 0.0f ? 1.0f : b))))
LOOP2(l_lae_d, double, double, lae(a, b))           LOOP2(l_lae_f, float, float, laef(a, b))
LOOP2(l_lae2_d, double, double, lae2(a, b))         LOOP2(l_lae2_f, float, float, lae2f(a, b))
LOOP2(l_fmod_d, double, double, fmod(a, b))         LOOP2(l_fmod_f, float, float, fmodf(a, b))
LOOP2(l_fmod_i64, int64_t, int64_t, IFMOD(int64_t, INT64_MIN))
LOOP2(l_fmod_i32, int32_t, int32_t, IFMOD(int32_t, INT32_MIN))
LOOP2(l_fmod_u8, uint8_t, uint8_t, (b == 0 ? (uint8_t)0 : (uint8_t)(a % b)))
/* numpy.gcd / numpy.lcm: integer-only, non-negative; computed in the input width (lcm wraps like numpy).
   Magnitudes are taken through the unsigned type so INT_MIN does not trip signed-overflow UB. */
static inline int64_t tsr_gcd64(int64_t a, int64_t b){ uint64_t x=a<0?-(uint64_t)a:(uint64_t)a, y=b<0?-(uint64_t)b:(uint64_t)b; while(y){uint64_t t=x%y; x=y; y=t;} return (int64_t)x; }
static inline int32_t tsr_gcd32(int32_t a, int32_t b){ uint32_t x=a<0?-(uint32_t)a:(uint32_t)a, y=b<0?-(uint32_t)b:(uint32_t)b; while(y){uint32_t t=x%y; x=y; y=t;} return (int32_t)x; }
static inline uint8_t tsr_gcd8(uint8_t a, uint8_t b){ while(b){uint8_t t=a%b; a=b; b=t;} return a; }
static inline int64_t tsr_lcm64(int64_t a, int64_t b){ if(!a||!b) return 0; uint64_t x=a<0?-(uint64_t)a:(uint64_t)a, y=b<0?-(uint64_t)b:(uint64_t)b; return (int64_t)((x/(uint64_t)tsr_gcd64(a,b))*y); }
static inline int32_t tsr_lcm32(int32_t a, int32_t b){ if(!a||!b) return 0; uint32_t x=a<0?-(uint32_t)a:(uint32_t)a, y=b<0?-(uint32_t)b:(uint32_t)b; return (int32_t)((x/(uint32_t)tsr_gcd32(a,b))*y); }
static inline uint8_t tsr_lcm8(uint8_t a, uint8_t b){ if(!a||!b) return 0; return (uint8_t)((a/tsr_gcd8(a,b))*b); }
LOOP2(l_gcd_i64, int64_t, int64_t, tsr_gcd64(a, b)) LOOP2(l_gcd_i32, int32_t, int32_t, tsr_gcd32(a, b)) LOOP2(l_gcd_u8, uint8_t, uint8_t, tsr_gcd8(a, b))
LOOP2(l_lcm_i64, int64_t, int64_t, tsr_lcm64(a, b)) LOOP2(l_lcm_i32, int32_t, int32_t, tsr_lcm32(a, b)) LOOP2(l_lcm_u8, uint8_t, uint8_t, tsr_lcm8(a, b))
/* numpy.float_power: like power but always in float64 (float32 inputs promote to float64) */
LOOP2(l_fpow_d, double, double, pow(a, b))
LOOP2(l_fpow_fd, float, double, pow((double)a, (double)b))

/* lgamma writes the global `signgam` in most libms: use the reentrant form and never share it across threads */
static double lgamma_safe(double x)
{
#if defined(__GLIBC__) || defined(__APPLE__) || defined(__FreeBSD__)
    int sign;
    return lgamma_r(x, &sign);
#else
    return lgamma(x);
#endif
}
static float lgammaf_safe(float x)
{
#if defined(__GLIBC__) || defined(__APPLE__) || defined(__FreeBSD__)
    int sign;
    return lgammaf_r(x, &sign);
#else
    return lgammaf(x);
#endif
}
LOOP1(l_lgamma_d, double, double, lgamma_safe(v))   LOOP1(l_lgamma_f, float, float, lgammaf_safe(v))


/* ================================================================ registry */

typedef struct {
    int in;                 /* dtype of every input */
    int out;                /* dtype written */
    tsr_loop_fn fn;
} uloop;

/* how integer and bool inputs are handled */
enum {
    INT_TO_FLOAT = 0,       /* cast to float64 and use the float64 loop (sin-like functions) */
    INT_IDENTITY,           /* the function is the identity on integers (trunc): copy */
    INT_NATIVE,             /* integers go to a kernel binary op instead (fmax -> maximum) */
    INT_LOOPS,              /* the table has integer loops (fmod); bool uses int64 */
};

typedef struct {
    const char *name;
    int nin;
    int ints;
    int native_op;          /* for INT_NATIVE: the tsr_binary op */
    int parallel;           /* 0 when the element function is not thread-safe */
    uloop loops[6];         /* terminated by fn == NULL */
    const char *summary;
} kufunc;

#define F2(dl, fl) {{TSR_F64, TSR_F64, dl}, {TSR_F32, TSR_F32, fl}, {0, 0, NULL}}
#define F2B(dl, fl) {{TSR_F64, TSR_BOOL, dl}, {TSR_F32, TSR_BOOL, fl}, {0, 0, NULL}}

static const kufunc UF[] = {
    {"cbrt", 1, INT_TO_FLOAT, 0, 1, F2(l_cbrt_d, l_cbrt_f), "cube root"},
    {"exp2", 1, INT_TO_FLOAT, 0, 1, F2(l_exp2_d, l_exp2_f), "2^x"},
    {"trunc", 1, INT_IDENTITY, 0, 1, F2(l_trunc_d, l_trunc_f), "round towards zero"},
    {"arcsinh", 1, INT_TO_FLOAT, 0, 1, F2(l_asinh_d, l_asinh_f), "inverse hyperbolic sine"},
    {"arccosh", 1, INT_TO_FLOAT, 0, 1, F2(l_acosh_d, l_acosh_f), "inverse hyperbolic cosine"},
    {"arctanh", 1, INT_TO_FLOAT, 0, 1, F2(l_atanh_d, l_atanh_f), "inverse hyperbolic tangent"},
    {"degrees", 1, INT_TO_FLOAT, 0, 1, F2(l_deg_d, l_deg_f), "radians to degrees"},
    {"radians", 1, INT_TO_FLOAT, 0, 1, F2(l_rad_d, l_rad_f), "degrees to radians"},
    {"erf", 1, INT_TO_FLOAT, 0, 1, F2(l_erf_d, l_erf_f), "error function (scipy.special.erf)"},
    {"erfc", 1, INT_TO_FLOAT, 0, 1, F2(l_erfc_d, l_erfc_f), "complementary error function"},
    {"gamma", 1, INT_TO_FLOAT, 0, 1, F2(l_gamma_d, l_gamma_f), "gamma function (scipy.special.gamma)"},
    {"lgamma", 1, INT_TO_FLOAT, 0, 1, F2(l_lgamma_d, l_lgamma_f), "log|gamma(x)| (scipy.special.gammaln)"},
    {"signbit", 1, INT_TO_FLOAT, 0, 1, F2B(l_signbit_d, l_signbit_f), "sign bit is set (bool; true for -0.0)"},
    {"fmax", 2, INT_NATIVE, OP_MAX, 1, F2(l_fmax_d, l_fmax_f), "maximum ignoring NaN"},
    {"fmin", 2, INT_NATIVE, OP_MIN, 1, F2(l_fmin_d, l_fmin_f), "minimum ignoring NaN"},
    {"copysign", 2, INT_TO_FLOAT, 0, 1, F2(l_copysign_d, l_copysign_f), "magnitude of x with the sign of y"},
    {"nextafter", 2, INT_TO_FLOAT, 0, 1, F2(l_nextafter_d, l_nextafter_f), "next representable value after x towards y"},
    {"heaviside", 2, INT_TO_FLOAT, 0, 1, F2(l_heaviside_d, l_heaviside_f), "0 if x1<0, x2 if x1==0, 1 if x1>0 (numpy.heaviside)"},
    {"logaddexp", 2, INT_TO_FLOAT, 0, 1, F2(l_lae_d, l_lae_f), "log(e^x + e^y) without overflow"},
    {"logaddexp2", 2, INT_TO_FLOAT, 0, 1, F2(l_lae2_d, l_lae2_f), "log2(2^x + 2^y) without overflow"},
    {"fmod", 2, INT_LOOPS, 0, 1,
     {{TSR_F64, TSR_F64, l_fmod_d}, {TSR_F32, TSR_F32, l_fmod_f}, {TSR_I64, TSR_I64, l_fmod_i64},
      {TSR_I32, TSR_I32, l_fmod_i32}, {TSR_U8, TSR_U8, l_fmod_u8}, {0, 0, NULL}},
     "C remainder with the sign of x (numpy.fmod); integer x % 0 gives 0"},
    {"gcd", 2, INT_LOOPS, 0, 1,
     {{TSR_I64, TSR_I64, l_gcd_i64}, {TSR_I32, TSR_I32, l_gcd_i32}, {TSR_U8, TSR_U8, l_gcd_u8}, {0, 0, NULL}},
     "greatest common divisor (numpy.gcd; integer inputs only)"},
    {"lcm", 2, INT_LOOPS, 0, 1,
     {{TSR_I64, TSR_I64, l_lcm_i64}, {TSR_I32, TSR_I32, l_lcm_i32}, {TSR_U8, TSR_U8, l_lcm_u8}, {0, 0, NULL}},
     "least common multiple (numpy.lcm; integer inputs only)"},
    {"floatPower", 2, INT_TO_FLOAT, 0, 1,
     {{TSR_F64, TSR_F64, l_fpow_d}, {TSR_F32, TSR_F64, l_fpow_fd}, {0, 0, NULL}},
     "x1**x2 computed in float64 (numpy.float_power)"},
};
#define N_UF ((int)(sizeof(UF) / sizeof(UF[0])))

int tsr_ufunc_count(void) { return N_UF; }
const char *tsr_ufunc_name(int id) { return (id >= 0 && id < N_UF) ? UF[id].name : NULL; }
const char *tsr_ufunc_summary(int id) { return (id >= 0 && id < N_UF) ? UF[id].summary : NULL; }
int tsr_ufunc_nin(int id) { return (id >= 0 && id < N_UF) ? UF[id].nin : TSR_EARG; }

int tsr_ufunc_find(const char *name)
{
    if (!name) return TSR_EARG;
    for (int i = 0; i < N_UF; i++)
        if (strcmp(UF[i].name, name) == 0) return i;
    return TSR_EARG;
}

static const uloop *find_loop(const kufunc *u, int dt)
{
    for (int i = 0; i < 6 && u->loops[i].fn; i++)
        if (u->loops[i].in == dt) return &u->loops[i];
    return NULL;
}

static int is_intlike(int dt) { return dt == TSR_I64 || dt == TSR_I32 || dt == TSR_U8 || dt == TSR_BOOL; }

int tsr_ufunc_resolve(int id, int dtype, int *loop_dtype, int *out_dtype, int *native_op)
{
    if (id < 0 || id >= N_UF || dtype < 0 || dtype >= TSR_NDTYPES) return TSR_EARG;
    const kufunc *u = &UF[id];
    if (dtype == TSR_C128) return TSR_ETYPE;
    if (is_intlike(dtype)) {
        switch (u->ints) {
        case INT_IDENTITY:
            if (loop_dtype) *loop_dtype = dtype;
            if (out_dtype) *out_dtype = dtype;
            return 1;
        case INT_NATIVE:
            if (loop_dtype) *loop_dtype = dtype;
            if (out_dtype) *out_dtype = dtype;
            if (native_op) *native_op = u->native_op;
            return 2;
        case INT_LOOPS:
            if (dtype == TSR_BOOL) dtype = TSR_I64;
            break;
        default:
            dtype = TSR_F64;
        }
    }
    const uloop *lp = find_loop(u, dtype);
    if (!lp) return TSR_ETYPE;
    if (loop_dtype) *loop_dtype = lp->in;
    if (out_dtype) *out_dtype = lp->out;
    return 0;
}

/* ================================================================ engine */

static void run_loop(tsr_loop_fn fn, int parallel, int nop, const tsr_iter *it, char **base)
{
    const int64_t n = it->shape[it->nd - 1];
    const int64_t outer = it->size / n;
    int64_t steps[4];
    for (int o = 0; o < nop; o++) steps[o] = it->st[o][it->nd - 1];
    int threads = parallel ? tsr_get_threads() : 1;
    if (it->size < TSR_PAR_MIN) threads = 1;

    if (outer == 1) {
        if (threads <= 1) { fn(base, n, steps); return; }
        const int64_t chunk = (n + threads - 1) / threads;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
        for (int t = 0; t < threads; t++) {
            const int64_t lo = (int64_t)t * chunk, len = lo >= n ? 0 : (n - lo < chunk ? n - lo : chunk);
            if (len <= 0) continue;
            char *p[4];
            for (int o = 0; o < nop; o++) p[o] = base[o] + lo * steps[o];
            fn(p, len, steps);
        }
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
#endif
    for (int64_t r = 0; r < outer; r++) {
        char *p[4];
        for (int o = 0; o < nop; o++) p[o] = base[o];
        int64_t rem = r;
        for (int d = it->nd - 2; d >= 0; d--) {
            const int64_t i = rem % it->shape[d];
            rem /= it->shape[d];
            for (int o = 0; o < nop; o++) p[o] += i * it->st[o][d];
        }
        fn(p, n, steps);
    }
}

int tsr_ufunc(int id, int loop_dtype, int32_t ndim, const int64_t *shape,
              const void *a, const int64_t *sa, const void *b, const int64_t *sb, void *out, const int64_t *so)
{
    if (id < 0 || id >= N_UF || !out || !a || (ndim > 0 && (!shape || !sa || !so))) return TSR_EARG;
    if (ndim < 0 || ndim > TSR_MAXDIM) return ndim > TSR_MAXDIM ? TSR_EDIM : TSR_EARG;
    const kufunc *u = &UF[id];
    if (u->nin == 2 && (!b || (ndim > 0 && !sb))) return TSR_EARG;
    const uloop *lp = find_loop(u, loop_dtype);
    if (!lp) return TSR_ETYPE;

    static const int64_t zero[TSR_MAXDIM] = {0};
    const int64_t *strides[3];
    char *base[3];
    int nop = 0;
    base[nop] = (char *)a; strides[nop++] = ndim ? sa : zero;
    if (u->nin == 2) { base[nop] = (char *)b; strides[nop++] = ndim ? sb : zero; }
    base[nop] = (char *)out; strides[nop++] = ndim ? so : zero;

    tsr_iter it;
    const int r = tsr_iter_init(&it, nop, ndim, shape, strides);
    if (r != TSR_OK) return r;
    if (it.size == 0) return TSR_OK;
    run_loop(lp->fn, u->parallel, nop, &it, base);
    return TSR_OK;
}
