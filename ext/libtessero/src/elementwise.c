/* Element-wise kernels: binary, comparison, unary, cast/copy, fill, arange, where. */
#include <math.h>
#include "internal.h"
#include "vmath.h"                 /* vectorisable exp/log (fdlibm algorithms, branch-free) */
#include "ops.h"

/* ------------------------------------------------------------------ binary (same dtype in/out) */

#define FMAX(x, y) ((x) != (x) ? (x) : ((y) != (y) ? (y) : ((x) > (y) ? (x) : (y))))
#define FMIN(x, y) ((x) != (x) ? (x) : ((y) != (y) ? (y) : ((x) < (y) ? (x) : (y))))

/* One inner run: n elements, a/b/o with byte strides. Three shapes of loop per op:
   all contiguous, contiguous with scalar b (stride 0), and general strided. */
#define RUN(T, EXPR)                                                                         \
    do {                                                                                     \
        if (sa == (int64_t)sizeof(T) && sb == (int64_t)sizeof(T) && so == (int64_t)sizeof(T)) { \
            const T *A = (const T *)a; const T *B = (const T *)b; T *O = (T *)o;             \
            for (int64_t i = 0; i < n; i++) { T x = A[i]; T y = B[i]; O[i] = (T)(EXPR); }     \
        } else if (sa == (int64_t)sizeof(T) && sb == 0 && so == (int64_t)sizeof(T)) {        \
            const T *A = (const T *)a; const T y = *(const T *)b; T *O = (T *)o;             \
            for (int64_t i = 0; i < n; i++) { T x = A[i]; O[i] = (T)(EXPR); }                 \
        } else if (sa == 0 && sb == (int64_t)sizeof(T) && so == (int64_t)sizeof(T)) {        \
            const T x = *(const T *)a; const T *B = (const T *)b; T *O = (T *)o;             \
            for (int64_t i = 0; i < n; i++) { T y = B[i]; O[i] = (T)(EXPR); }                 \
        } else {                                                                             \
            for (int64_t i = 0; i < n; i++) {                                                \
                T x = *(const T *)(a + i * sa); T y = *(const T *)(b + i * sb);             \
                *(T *)(o + i * so) = (T)(EXPR);                                              \
            }                                                                                \
        }                                                                                    \
    } while (0)

#define CMP(T, EXPR)                                                                         \
    do {                                                                                     \
        if (sa == (int64_t)sizeof(T) && sb == (int64_t)sizeof(T) && so == 1) {               \
            const T *A = (const T *)a; const T *B = (const T *)b; uint8_t *O = (uint8_t *)o; \
            for (int64_t i = 0; i < n; i++) { T x = A[i]; T y = B[i]; O[i] = (uint8_t)(EXPR); } \
        } else if (sa == (int64_t)sizeof(T) && sb == 0 && so == 1) {                         \
            const T *A = (const T *)a; const T y = *(const T *)b; uint8_t *O = (uint8_t *)o; \
            for (int64_t i = 0; i < n; i++) { T x = A[i]; O[i] = (uint8_t)(EXPR); }           \
        } else {                                                                             \
            for (int64_t i = 0; i < n; i++) {                                                \
                T x = *(const T *)(a + i * sa); T y = *(const T *)(b + i * sb);             \
                *(uint8_t *)(o + i * so) = (uint8_t)(EXPR);                                  \
            }                                                                                \
        }                                                                                    \
    } while (0)

#define COMPARISONS(T)                                   \
    case OP_EQ: CMP(T, x == y); return TSR_OK;           \
    case OP_NE: CMP(T, x != y); return TSR_OK;           \
    case OP_LT: CMP(T, x < y); return TSR_OK;            \
    case OP_LE: CMP(T, x <= y); return TSR_OK;           \
    case OP_GT: CMP(T, x > y); return TSR_OK;            \
    case OP_GE: CMP(T, x >= y); return TSR_OK;

/*
 * NumPy's float floor_divide and remainder (npy_divmod in npy_math_internal.h.src), so x == (x // y) * y + x % y
 * holds the way it does in NumPy. floor(x / y) is wrong whenever x / y rounds up to an integer: 1.0 // 0.1 is 9
 * (0.1 is slightly above 1/10), while floor(1.0 / 0.1) = floor(10.0) = 10. y == 0 gives x / y (inf or nan) and
 * the fmod NaN remainder; a zero remainder takes the sign of y, a zero quotient the sign of x / y.
 *
 * Fast path (bit-identical, the C tests and a 400k-case NumPy comparison check it): npy_divmod returns the
 * exact floor of the real quotient whenever |x / y| < 2^(p-3) (p = 53 or 24: the two roundings in
 * (x - fmod) / y stay below 1/4, so the snap lands on trunc or trunc - 1, which is exactly the floor). When
 * the rounded quotient q = x / y is not an integer it lies strictly between the same two integers as the real
 * quotient, so floor(q) is that exact floor too. Only quotients that round to an integer (or are huge, zero,
 * inf or nan) need the fmod-based path, which keeps the common case at one divide.
 */
#define DIVMOD_FNS(T, S, LIM)                                                                    \
    static inline T fmod_py##S(T a, T b)                                                         \
    {                                                                                            \
        T mod = fmod##S(a, b);                                                                   \
        if (b == 0) return mod;                                                                  \
        if (mod != 0) { if ((b < 0) != (mod < 0)) mod += b; }                                    \
        else mod = copysign##S((T)0, b);                                                         \
        return mod;                                                                              \
    }                                                                                            \
    static inline T floordiv_py##S(T a, T b)                                                     \
    {                                                                                            \
        const T q = a / b, fq = floor##S(q);                                                     \
        if (q - fq > (T)0 && fabs##S(q) < (LIM)) return fq;  /* not an integer: the exact floor */ \
        if (b == 0) return q;                                                                    \
        const T mod = fmod##S(a, b);                                                             \
        T div = (a - mod) / b;                                                                   \
        if (mod != 0 && ((b < 0) != (mod < 0))) div -= (T)1;                                     \
        if (div == 0) return copysign##S((T)0, q);                                               \
        T fl = floor##S(div);                                                                    \
        if (div - fl > (T)0.5) fl += (T)1;                                                       \
        return fl;                                                                               \
    }
DIVMOD_FNS(double, , 0x1p50)
DIVMOD_FNS(float, f, 0x1p21f)

/* floor_divide over a run, in blocks: a branch-free (vectorisable) pass writes floor(x / y) and notes whether
   any element of the block needs the exact path; only such blocks are redone with floordiv_py. */
#define FDIV_BLOCK 256
#define FDIV_RUN(T, S, LIM)                                                                          \
    do {                                                                                             \
        for (int64_t i0_ = 0; i0_ < n; i0_ += FDIV_BLOCK) {                                          \
            const int64_t m_ = n - i0_ < FDIV_BLOCK ? n - i0_ : FDIV_BLOCK;                          \
            const char *a_ = a + i0_ * sa, *b_ = b + i0_ * sb;                                       \
            char *o_ = o + i0_ * so;                                                                 \
            int slow_ = 0;                                                                           \
            if (sa == (int64_t)sizeof(T) && sb == (int64_t)sizeof(T) && so == (int64_t)sizeof(T)) {  \
                const T *A = (const T *)a_, *B = (const T *)b_; T *O = (T *)o_;                      \
                for (int64_t k = 0; k < m_; k++) {                                                   \
                    const T q = A[k] / B[k], f = floor##S(q);                                        \
                    O[k] = f; slow_ |= !(q - f > (T)0 && fabs##S(q) < (LIM));                        \
                }                                                                                    \
            } else if (sa == (int64_t)sizeof(T) && sb == 0 && so == (int64_t)sizeof(T)) {           \
                const T *A = (const T *)a_, y = *(const T *)b_; T *O = (T *)o_;                      \
                for (int64_t k = 0; k < m_; k++) {                                                   \
                    const T q = A[k] / y, f = floor##S(q);                                           \
                    O[k] = f; slow_ |= !(q - f > (T)0 && fabs##S(q) < (LIM));                        \
                }                                                                                    \
            } else slow_ = 1;                                                                        \
            if (slow_)                                                                               \
                for (int64_t k = 0; k < m_; k++)                                                     \
                    *(T *)(o_ + k * so) = floordiv_py##S(*(const T *)(a_ + k * sa), *(const T *)(b_ + k * sb)); \
        }                                                                                            \
    } while (0)

#define FLOAT_BINARY(NAME, T, POWF, ATAN2F, HYPOTF, FMODF, S, LIM)                                      \
    TSR_CLONES static int NAME(int op, int64_t n, const char *a, int64_t sa, const char *b, int64_t sb, \
                               char *o, int64_t so)                                                     \
    {                                                                                                   \
        switch (op) {                                                                                   \
        case OP_ADD: RUN(T, x + y); return TSR_OK;                                                      \
        case OP_SUB: RUN(T, x - y); return TSR_OK;                                                      \
        case OP_MUL: RUN(T, x * y); return TSR_OK;                                                      \
        case OP_DIV: RUN(T, x / y); return TSR_OK;                                                      \
        case OP_POW: RUN(T, POWF(x, y)); return TSR_OK;                                                 \
        case OP_MAX: RUN(T, FMAX(x, y)); return TSR_OK;                                                 \
        case OP_MIN: RUN(T, FMIN(x, y)); return TSR_OK;                                                 \
        case OP_ATAN2: RUN(T, ATAN2F(x, y)); return TSR_OK;                                             \
        case OP_HYPOT: RUN(T, HYPOTF(x, y)); return TSR_OK;                                             \
        case OP_MOD: RUN(T, FMODF(x, y)); return TSR_OK;                                                \
        case OP_FLOORDIV: FDIV_RUN(T, S, LIM); return TSR_OK;                                           \
        COMPARISONS(T)                                                                                  \
        default: return TSR_ETYPE;                                                                      \
        }                                                                                               \
    }

FLOAT_BINARY(bin_f64, double, pow, atan2, hypot, fmod_py, , 0x1p50)
FLOAT_BINARY(bin_f32, float, powf, atan2f, hypotf, fmod_pyf, f, 0x1p21f)

/*
 * Python semantics for integer floor division and modulo; division by zero gives 0 (NumPy does too, with a warning).
 * y == -1 is answered without dividing: MIN / -1 overflows, which is undefined behaviour and raises SIGFPE on x86
 * (killing the PHP process). NumPy gives the wrapped negation for floor_divide and 0 for remainder.
 */
#define IFLOORDIV(x, y) ((y) == 0 ? 0 : (y) == -1 ? TSR_WSUB(__typeof__(x), 0, x) : (((x) / (y)) - ((((x) % (y)) != 0) && (((x) < 0) != ((y) < 0)))))
#define IMOD(x, y) ((y) == 0 || (y) == -1 ? 0 : ((((x) % (y)) != 0 && (((x) % (y)) < 0) != ((y) < 0)) ? ((x) % (y)) + (y) : ((x) % (y))))

#define INT_BINARY(NAME, T)                                                                             \
    TSR_CLONES static int NAME(int op, int64_t n, const char *a, int64_t sa, const char *b, int64_t sb, \
                               char *o, int64_t so)                                                     \
    {                                                                                                   \
        switch (op) {                                                                                   \
        case OP_ADD: RUN(T, TSR_WADD(T, x, y)); return TSR_OK;                                                      \
        case OP_SUB: RUN(T, TSR_WSUB(T, x, y)); return TSR_OK;                                                      \
        case OP_MUL: RUN(T, TSR_WMUL(T, x, y)); return TSR_OK;                                                      \
        case OP_MAX: RUN(T, x > y ? x : y); return TSR_OK;                                              \
        case OP_MIN: RUN(T, x < y ? x : y); return TSR_OK;                                              \
        case OP_AND: RUN(T, x & y); return TSR_OK;                                                      \
        case OP_OR: RUN(T, x | y); return TSR_OK;                                                       \
        case OP_XOR: RUN(T, x ^ y); return TSR_OK;                                                      \
        case OP_FLOORDIV: RUN(T, IFLOORDIV(x, y)); return TSR_OK;                                       \
        case OP_MOD: RUN(T, IMOD(x, y)); return TSR_OK;                                                 \
        COMPARISONS(T)                                                                                  \
        default: return TSR_ETYPE; /* true division and pow: promote to float first */                  \
        }                                                                                               \
    }

INT_BINARY(bin_i64, int64_t)
INT_BINARY(bin_i32, int32_t)
INT_BINARY(bin_u8, uint8_t)

static int bin_bool(int op, int64_t n, const char *a, int64_t sa, const char *b, int64_t sb, char *o, int64_t so)
{
    switch (op) {
    case OP_AND: case OP_MUL: case OP_MIN: RUN(uint8_t, x & y); return TSR_OK;
    case OP_OR: case OP_ADD: case OP_MAX: RUN(uint8_t, x | y); return TSR_OK;
    case OP_XOR: case OP_NE: RUN(uint8_t, x ^ y); return TSR_OK;
    case OP_EQ: RUN(uint8_t, !(x ^ y)); return TSR_OK;
    case OP_LT: RUN(uint8_t, (!x) & y); return TSR_OK;
    case OP_LE: RUN(uint8_t, (!x) | y); return TSR_OK;
    case OP_GT: RUN(uint8_t, x & (!y)); return TSR_OK;
    case OP_GE: RUN(uint8_t, x | (!y)); return TSR_OK;
    default: return TSR_ETYPE;
    }
}

/* NumPy's complex divide (Smith's method): scale by the larger component of y so |y|^2 is never formed; the
   naive y0^2 + y1^2 overflows for |y| > ~1e154 and underflows for tiny y. NumPy's two branches (|y0| >= |y1|,
   else) are evaluated as selects with each expression's operand order kept, so the bits are NumPy's. y == 0
   gives NumPy's x / 0, written x * inf (identical for every x, including 0 and nan). Branch-free so the
   contiguous loop vectorises. Not multiversioned on purpose: an FMA clone would contract p + q * rat and
   the numerators into fused operations and change the bits. */
#define CDIV_BODY(x0, x1, y0, y1, re, im)                                                            \
    do {                                                                                             \
        const double ar_ = fabs(y0), ai_ = fabs(y1);                                                 \
        const int big_ = ar_ >= ai_, zero_ = (ar_ == 0) & (ai_ == 0);                                \
        const double p_ = big_ ? (y0) : (y1), q_ = big_ ? (y1) : (y0);                               \
        const double rat_ = q_ / p_, scl_ = 1.0 / (p_ + q_ * rat_);                                  \
        const double rb_ = (x0) + (x1) * rat_, rs_ = (x0) * rat_ + (x1);                             \
        const double ib_ = (x1) - (x0) * rat_, is_ = (x1) * rat_ - (x0);                             \
        const double r_ = (big_ ? rb_ : rs_) * scl_, i_ = (big_ ? ib_ : is_) * scl_;                 \
        re = zero_ ? (x0) * (double)INFINITY : r_;                                                   \
        im = zero_ ? (x1) * (double)INFINITY : i_;                                                   \
    } while (0)
static void div_c128(int64_t n, const char *a, int64_t sa, const char *b, int64_t sb, char *o, int64_t so)
{
    if (sa == 16 && sb == 16 && so == 16) {
        const double *x = (const double *)a, *y = (const double *)b;
        double *out = (double *)o;
        for (int64_t i = 0; i < n; i++) {
            const double x0 = x[2 * i], x1 = x[2 * i + 1], y0 = y[2 * i], y1 = y[2 * i + 1];
            double re, im;
            CDIV_BODY(x0, x1, y0, y1, re, im);
            out[2 * i] = re; out[2 * i + 1] = im;
        }
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        const double *x = (const double *)(a + i * sa), *y = (const double *)(b + i * sb);
        const double x0 = x[0], x1 = x[1], y0 = y[0], y1 = y[1];
        double re, im;
        CDIV_BODY(x0, x1, y0, y1, re, im);
        double *out = (double *)(o + i * so);
        out[0] = re; out[1] = im;
    }
}

static int bin_c128(int op, int64_t n, const char *a, int64_t sa, const char *b, int64_t sb, char *o, int64_t so)
{
    if (op == OP_DIV) { div_c128(n, a, sa, b, sb, o, so); return TSR_OK; }
    for (int64_t i = 0; i < n; i++) {
        const double *x = (const double *)(a + i * sa);
        const double *y = (const double *)(b + i * sb);
        double re, im;
        switch (op) {
        case OP_ADD: re = x[0] + y[0]; im = x[1] + y[1]; break;
        case OP_SUB: re = x[0] - y[0]; im = x[1] - y[1]; break;
        case OP_MUL: re = x[0] * y[0] - x[1] * y[1]; im = x[0] * y[1] + x[1] * y[0]; break;
        case OP_EQ: *(uint8_t *)(o + i * so) = x[0] == y[0] && x[1] == y[1]; continue;
        case OP_NE: *(uint8_t *)(o + i * so) = x[0] != y[0] || x[1] != y[1]; continue;
        default: return TSR_ETYPE;
        }
        double *out = (double *)(o + i * so);
        out[0] = re;
        out[1] = im;
    }
    return TSR_OK;
}

typedef int (*bin_fn)(int, int64_t, const char *, int64_t, const char *, int64_t, char *, int64_t);
static const bin_fn BIN[TSR_NDTYPES] = {bin_f64, bin_f32, bin_i64, bin_i32, bin_u8, bin_bool, bin_c128};

int tsr_binary(int op, int dtype, int32_t ndim, const int64_t *shape, const void *a, const int64_t *sa,
               const void *b, const int64_t *sb, void *out, const int64_t *sout)
{
    if (dtype < 0 || dtype >= TSR_NDTYPES || op < 0 || op > OP_FLOORDIV) return TSR_EARG;
    const int64_t *st[3] = {sa, sb, sout};
    tsr_iter it;
    int rc = tsr_iter_init(&it, 3, ndim, shape, st);
    if (rc != TSR_OK || it.size == 0) return rc;
    const void *base[3] = {a, b, out};
    int nd = it.nd;
    int err = TSR_OK;
#ifdef _OPENMP
    int threads = tsr_get_threads();
    if (threads > 1 && it.nd == 1 && it.size >= TSR_PAR_MIN) {
        /* one contiguous-or-strided run split into equal chunks: results are identical to the serial loop */
        const int64_t n_ = it.size, chunk = (n_ + threads - 1) / threads;
        const int64_t s0 = it.st[0][0], s1 = it.st[1][0], s2 = it.st[2][0];
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int t = 0; t < threads; t++) {
            int64_t lo = t * chunk, hi = lo + chunk < n_ ? lo + chunk : n_;
            if (lo < hi) {
                int r = BIN[dtype](op, hi - lo, (const char *)a + lo * s0, s0, (const char *)b + lo * s1, s1, (char *)out + lo * s2, s2);
                if (r != TSR_OK) {
#pragma omp atomic write
                    err = r;
                }
            }
        }
        return err;
    }
#endif
    TSR_ITER_OUTER(it, base, {
        int r = BIN[dtype](op, n, p[0], it.st[0][nd - 1], p[1], it.st[1][nd - 1], p[2], it.st[2][nd - 1]);
        if (r != TSR_OK) { err = r; break; }
    });
    return err;
}

/* ------------------------------------------------------------------ unary */

#define URUN(T, TO, EXPR)                                                                    \
    do {                                                                                     \
        if (sa == (int64_t)sizeof(T) && so == (int64_t)sizeof(TO)) {                         \
            const T *A = (const T *)a; TO *O = (TO *)o;                                      \
            for (int64_t i = 0; i < n; i++) { T x = A[i]; (void)x; O[i] = (TO)(EXPR); }                \
        } else {                                                                             \
            for (int64_t i = 0; i < n; i++) { T x = *(const T *)(a + i * sa); (void)x; *(TO *)(o + i * so) = (TO)(EXPR); } \
        }                                                                                    \
    } while (0)

#define FLOAT_UNARY(NAME, T, S)                                                                         \
    TSR_CLONES static int NAME(int op, int64_t n, const char *a, int64_t sa, char *o, int64_t so)       \
    {                                                                                                   \
        switch (op) {                                                                                   \
        case U_NEG: URUN(T, T, -x); return TSR_OK;                                                      \
        case U_ABS: URUN(T, T, fabs##S(x)); return TSR_OK;                                              \
        case U_SQUARE: URUN(T, T, x * x); return TSR_OK;                                                \
        case U_SIGN: URUN(T, T, x > 0 ? (T)1 : (x < 0 ? (T)-1 : x)); return TSR_OK;                    \
        case U_SQRT: URUN(T, T, sqrt##S(x)); return TSR_OK;                                             \
        case U_EXP: URUN(T, T, tsr_exp##S(x)); return TSR_OK;                                           \
        case U_LOG: URUN(T, T, tsr_log##S(x)); return TSR_OK;                                           \
        case U_LOG10: URUN(T, T, log10##S(x)); return TSR_OK;                                           \
        case U_LOG2: URUN(T, T, log2##S(x)); return TSR_OK;                                             \
        case U_SIN: URUN(T, T, sin##S(x)); return TSR_OK;                                               \
        case U_COS: URUN(T, T, cos##S(x)); return TSR_OK;                                               \
        case U_TAN: URUN(T, T, tan##S(x)); return TSR_OK;                                               \
        case U_ARCSIN: URUN(T, T, asin##S(x)); return TSR_OK;                                           \
        case U_ARCCOS: URUN(T, T, acos##S(x)); return TSR_OK;                                           \
        case U_ARCTAN: URUN(T, T, atan##S(x)); return TSR_OK;                                           \
        case U_SINH: URUN(T, T, sinh##S(x)); return TSR_OK;                                             \
        case U_COSH: URUN(T, T, cosh##S(x)); return TSR_OK;                                             \
        case U_TANH: URUN(T, T, tanh##S(x)); return TSR_OK;                                             \
        case U_FLOOR: URUN(T, T, floor##S(x)); return TSR_OK;                                           \
        case U_CEIL: URUN(T, T, ceil##S(x)); return TSR_OK;                                             \
        case U_RINT: URUN(T, T, rint##S(x)); return TSR_OK;                                             \
        case U_EXPM1: URUN(T, T, expm1##S(x)); return TSR_OK;                                           \
        case U_LOG1P: URUN(T, T, log1p##S(x)); return TSR_OK;                                           \
        case U_RECIPROCAL: URUN(T, T, (T)1 / x); return TSR_OK;                                         \
        case U_ISNAN: URUN(T, uint8_t, isnan(x) != 0); return TSR_OK;                                     \
        case U_ISFINITE: URUN(T, uint8_t, isfinite(x) != 0); return TSR_OK;                               \
        case U_ISINF: URUN(T, uint8_t, isinf(x) != 0); return TSR_OK;                                     \
        default: return TSR_ETYPE;                                                                      \
        }                                                                                               \
    }

FLOAT_UNARY(un_f64, double, )
FLOAT_UNARY(un_f32, float, f)

#define INT_UNARY(NAME, T, ABS)                                                                         \
    TSR_CLONES static int NAME(int op, int64_t n, const char *a, int64_t sa, char *o, int64_t so)       \
    {                                                                                                   \
        switch (op) {                                                                                   \
        case U_NEG: URUN(T, T, TSR_WSUB(T, 0, x)); return TSR_OK;                                       \
        case U_ABS: URUN(T, T, ABS); return TSR_OK;                                                     \
        case U_SQUARE: URUN(T, T, TSR_WMUL(T, x, x)); return TSR_OK;                                    \
        case U_SIGN: URUN(T, T, (x > 0) - (x < 0)); return TSR_OK;                                      \
        case U_NOT: URUN(T, T, ~x); return TSR_OK;                                                      \
        case U_ISNAN: case U_ISINF: URUN(T, uint8_t, 0); return TSR_OK;                                 \
        case U_ISFINITE: URUN(T, uint8_t, 1); return TSR_OK;                                            \
        default: return TSR_ETYPE;                                                                      \
        }                                                                                               \
    }

/* negation and abs wrap like NumPy (abs(MIN) == MIN) instead of overflowing, which is undefined behaviour */
INT_UNARY(un_i64, int64_t, x < 0 ? TSR_WSUB(int64_t, 0, x) : x)
INT_UNARY(un_i32, int32_t, x < 0 ? TSR_WSUB(int32_t, 0, x) : x)
INT_UNARY(un_u8, uint8_t, x)

static int un_bool(int op, int64_t n, const char *a, int64_t sa, char *o, int64_t so)
{
    switch (op) {
    case U_NOT: URUN(uint8_t, uint8_t, !x); return TSR_OK;
    case U_ABS: case U_SQUARE: case U_SIGN: URUN(uint8_t, uint8_t, x); return TSR_OK;
    case U_ISNAN: case U_ISINF: URUN(uint8_t, uint8_t, 0); return TSR_OK;
    case U_ISFINITE: URUN(uint8_t, uint8_t, 1); return TSR_OK;
    default: return TSR_ETYPE;
    }
}

/* complex input; REAL/IMAG/CABS/ANGLE write float64, CONJ/NEG write complex128 */
static int un_c128(int op, int64_t n, const char *a, int64_t sa, char *o, int64_t so)
{
    for (int64_t i = 0; i < n; i++) {
        const double *x = (const double *)(a + i * sa);
        switch (op) {
        case U_REAL: *(double *)(o + i * so) = x[0]; break;
        case U_IMAG: *(double *)(o + i * so) = x[1]; break;
        case U_CABS: case U_ABS: *(double *)(o + i * so) = hypot(x[0], x[1]); break;
        case U_ANGLE: *(double *)(o + i * so) = atan2(x[1], x[0]); break;
        case U_CONJ: ((double *)(o + i * so))[0] = x[0]; ((double *)(o + i * so))[1] = -x[1]; break;
        case U_NEG: ((double *)(o + i * so))[0] = -x[0]; ((double *)(o + i * so))[1] = -x[1]; break;
        case U_ISNAN: *(uint8_t *)(o + i * so) = isnan(x[0]) || isnan(x[1]); break;
        case U_ISFINITE: *(uint8_t *)(o + i * so) = isfinite(x[0]) && isfinite(x[1]); break;
        default: return TSR_ETYPE;
        }
    }
    return TSR_OK;
}

typedef int (*un_fn)(int, int64_t, const char *, int64_t, char *, int64_t);
static const un_fn UN[TSR_NDTYPES] = {un_f64, un_f32, un_i64, un_i32, un_u8, un_bool, un_c128};

int tsr_unary(int op, int dtype, int32_t ndim, const int64_t *shape, const void *in, const int64_t *sin,
              void *out, const int64_t *sout)
{
    if (dtype < 0 || dtype >= TSR_NDTYPES || op < 0 || op > U_CONJ) return TSR_EARG;
    const int64_t *st[2] = {sin, sout};
    tsr_iter it;
    int rc = tsr_iter_init(&it, 2, ndim, shape, st);
    if (rc != TSR_OK || it.size == 0) return rc;
    const void *base[2] = {in, out};
    int nd = it.nd;
    int err = TSR_OK;
#ifdef _OPENMP
    int threads = tsr_get_threads();
    if (threads > 1 && it.nd == 1 && it.size >= TSR_PAR_MIN) {
        const int64_t n_ = it.size, chunk = (n_ + threads - 1) / threads;
        const int64_t s0 = it.st[0][0], s1 = it.st[1][0];
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int t = 0; t < threads; t++) {
            int64_t lo = t * chunk, hi = lo + chunk < n_ ? lo + chunk : n_;
            if (lo < hi) {
                int r = UN[dtype](op, hi - lo, (const char *)in + lo * s0, s0, (char *)out + lo * s1, s1);
                if (r != TSR_OK) {
#pragma omp atomic write
                    err = r;
                }
            }
        }
        return err;
    }
#endif
    TSR_ITER_OUTER(it, base, {
        int r = UN[dtype](op, n, p[0], it.st[0][nd - 1], p[1], it.st[1][nd - 1]);
        if (r != TSR_OK) { err = r; break; }
    });
    return err;
}

/* ------------------------------------------------------------------ copy / cast */

/* Load any real dtype as double or int64, store to any dtype. Contiguous same-dtype runs use memcpy. */
#define LOAD_D(dt, p) ((dt) == TSR_F64 ? *(const double *)(p) : (dt) == TSR_F32 ? (double)*(const float *)(p) : \
                       (dt) == TSR_I64 ? (double)*(const int64_t *)(p) : (dt) == TSR_I32 ? (double)*(const int32_t *)(p) : \
                       (dt) == TSR_C128 ? *(const double *)(p) : (double)*(const uint8_t *)(p))
#define LOAD_I(dt, p) ((dt) == TSR_I64 ? *(const int64_t *)(p) : (dt) == TSR_I32 ? (int64_t)*(const int32_t *)(p) : \
                       (int64_t)*(const uint8_t *)(p))

#define CAST_RUN(TI, TO, EXPR)                                                        \
    do {                                                                              \
        if (sa == (int64_t)sizeof(TI) && so == (int64_t)sizeof(TO)) {                 \
            const TI *A = (const TI *)a; TO *O = (TO *)o;                             \
            for (int64_t i = 0; i < n; i++) { TI x = A[i]; O[i] = (TO)(EXPR); }       \
        } else {                                                                      \
            for (int64_t i = 0; i < n; i++) { TI x = *(const TI *)(a + i * sa); *(TO *)(o + i * so) = (TO)(EXPR); } \
        }                                                                             \
    } while (0)

/* a float source converts through the NumPy-matching helpers (NaN/Inf/out-of-range are defined, see
   internal.h); an integer source keeps C's modular conversion, which is NumPy's wrap-around */
#define CVT_I64(x) _Generic((x), float: tsr_f2i64f(x), double: tsr_f2i64(x), default: (int64_t)(x))
#define CVT_I32(x) _Generic((x), float: tsr_f2i32f(x), double: tsr_f2i32(x), default: (int32_t)(x))
#define CVT_U8(x) _Generic((x), float: tsr_f2u8f(x), double: tsr_f2u8(x), default: (uint8_t)(x))

#define CAST_TO(TI)                                                                 \
    switch (dout) {                                                                   \
    case TSR_F64: CAST_RUN(TI, double, x); return TSR_OK;                             \
    case TSR_F32: CAST_RUN(TI, float, x); return TSR_OK;                              \
    case TSR_I64: CAST_RUN(TI, int64_t, CVT_I64(x)); return TSR_OK;                   \
    case TSR_I32: CAST_RUN(TI, int32_t, CVT_I32(x)); return TSR_OK;                   \
    case TSR_U8: CAST_RUN(TI, uint8_t, CVT_U8(x)); return TSR_OK;                     \
    case TSR_BOOL: CAST_RUN(TI, uint8_t, x != 0); return TSR_OK;                      \
    case TSR_C128:                                                                    \
        for (int64_t i = 0; i < n; i++) {                                             \
            double *O = (double *)(o + i * so);                                       \
            O[0] = (double)*(const TI *)(a + i * sa); O[1] = 0.0;                     \
        }                                                                             \
        return TSR_OK;                                                                \
    default: return TSR_EARG;                                                         \
    }

#if defined(__x86_64__) || defined(_M_X64)
/* Contiguous float -> int32 / uint8 casts with the packed cvtt instructions written out: cvttpd2dq / cvttps2dq
   return exactly the "integer indefinite" INT32_MIN for NaN and out-of-range lanes, i.e. tsr_f2i32 / tsr_f2u8
   (NumPy's x86-64 results), and the explicit form keeps the vector width the plain cast had (the select-based
   helpers made GCC drop to narrower vectors). uint8 keeps the low byte: mask, then two exact packs. */
static inline __m128i cvt4_pd(const double *p)
{
    return _mm_unpacklo_epi64(_mm_cvttpd_epi32(_mm_loadu_pd(p)), _mm_cvttpd_epi32(_mm_loadu_pd(p + 2)));
}
static inline __m128i low_bytes16(__m128i a, __m128i b, __m128i c, __m128i d)
{
    const __m128i m = _mm_set1_epi32(0xFF);
    a = _mm_and_si128(a, m); b = _mm_and_si128(b, m); c = _mm_and_si128(c, m); d = _mm_and_si128(d, m);
    return _mm_packus_epi16(_mm_packs_epi32(a, b), _mm_packs_epi32(c, d));
}
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#include <immintrin.h>
/* float64 -> int32 with 256-bit vcvttpd2dq (4 lanes per instruction, same "integer indefinite" result) */
__attribute__((target("avx2"))) static void cvt_pd_i32_avx2(const double *A, int32_t *O, int64_t n)
{
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm_storeu_si128((__m128i *)(O + i), _mm256_cvttpd_epi32(_mm256_loadu_pd(A + i)));
        _mm_storeu_si128((__m128i *)(O + i + 4), _mm256_cvttpd_epi32(_mm256_loadu_pd(A + i + 4)));
    }
    for (; i < n; i++) O[i] = tsr_f2i32(A[i]);
}
#define TSR_HAVE_AVX2_CVT 1
#endif
TSR_CLONES static int cast_sse(int din, int dout, int64_t n, const char *a, char *o)
{
    int64_t i = 0;
    if (din == TSR_F64 && dout == TSR_I32) {
        const double *A = (const double *)a; int32_t *O = (int32_t *)o;
#ifdef TSR_HAVE_AVX2_CVT
        if (__builtin_cpu_supports("avx2")) { cvt_pd_i32_avx2(A, O, n); return 1; }
#endif
        for (; i + 4 <= n; i += 4) _mm_storeu_si128((__m128i *)(O + i), cvt4_pd(A + i));
        for (; i < n; i++) O[i] = tsr_f2i32(A[i]);
    } else if (din == TSR_F64 && dout == TSR_U8) {
        const double *A = (const double *)a; uint8_t *O = (uint8_t *)o;
        for (; i + 16 <= n; i += 16)
            _mm_storeu_si128((__m128i *)(O + i), low_bytes16(cvt4_pd(A + i), cvt4_pd(A + i + 4), cvt4_pd(A + i + 8), cvt4_pd(A + i + 12)));
        for (; i < n; i++) O[i] = tsr_f2u8(A[i]);
    } else if (din == TSR_F32 && dout == TSR_I32) {
        const float *A = (const float *)a; int32_t *O = (int32_t *)o;
        for (; i + 4 <= n; i += 4) _mm_storeu_si128((__m128i *)(O + i), _mm_cvttps_epi32(_mm_loadu_ps(A + i)));
        for (; i < n; i++) O[i] = tsr_f2i32f(A[i]);
    } else if (din == TSR_F32 && dout == TSR_U8) {
        const float *A = (const float *)a; uint8_t *O = (uint8_t *)o;
        for (; i + 16 <= n; i += 16)
            _mm_storeu_si128((__m128i *)(O + i), low_bytes16(_mm_cvttps_epi32(_mm_loadu_ps(A + i)), _mm_cvttps_epi32(_mm_loadu_ps(A + i + 4)),
                                                             _mm_cvttps_epi32(_mm_loadu_ps(A + i + 8)), _mm_cvttps_epi32(_mm_loadu_ps(A + i + 12))));
        for (; i < n; i++) O[i] = tsr_f2u8f(A[i]);
    } else return 0;
    return 1;
}
#endif

TSR_CLONES static int cast_run(int din, int dout, int64_t n, const char *a, int64_t sa, char *o, int64_t so)
{
    int64_t isz = tsr_itemsize(din);
    if (din == dout) {
        if (sa == isz && so == isz) { memcpy(o, a, (size_t)(n * isz)); return TSR_OK; }
        for (int64_t i = 0; i < n; i++) memcpy(o + i * so, a + i * sa, (size_t)isz);
        return TSR_OK;
    }
#if defined(__x86_64__) || defined(_M_X64)
    if (sa == isz && so == tsr_itemsize(dout) && cast_sse(din, dout, n, a, o)) return TSR_OK;
#endif
    switch (din) {
    case TSR_F64: CAST_TO(double)
    case TSR_F32: CAST_TO(float)
    case TSR_I64: CAST_TO(int64_t)
    case TSR_I32: CAST_TO(int32_t)
    case TSR_U8: CAST_TO(uint8_t)
    case TSR_BOOL: CAST_TO(uint8_t)
    case TSR_C128: /* complex -> real keeps the real part (as NumPy does, with a warning there) */
        for (int64_t i = 0; i < n; i++) {
            double re = *(const double *)(a + i * sa);
            char *q = o + i * so;
            switch (dout) {
            case TSR_F64: *(double *)q = re; break;
            case TSR_F32: *(float *)q = (float)re; break;
            case TSR_I64: *(int64_t *)q = tsr_f2i64(re); break;
            case TSR_I32: *(int32_t *)q = tsr_f2i32(re); break;
            case TSR_U8: *(uint8_t *)q = tsr_f2u8(re); break;
            case TSR_BOOL: *(uint8_t *)q = re != 0 || ((const double *)(a + i * sa))[1] != 0; break;
            default: return TSR_EARG;
            }
        }
        return TSR_OK;
    default: return TSR_EARG;
    }
}

/* out[i][j] = in[j][i] style copy: rows of `out` are columns of `in` (in column stride = isz). */
#define TILE 32
TSR_CLONES static void transpose_tiled(int64_t isz, int64_t rows, int64_t cols, const char *in, int64_t in_col_stride,
                                       char *out, int64_t out_row_stride)
{
    /* element (r, c): in at r * isz + c * in_col_stride, out at r * out_row_stride + c * isz */
    for (int64_t r0 = 0; r0 < rows; r0 += TILE) {
        int64_t r1 = r0 + TILE < rows ? r0 + TILE : rows;
        for (int64_t c0 = 0; c0 < cols; c0 += TILE) {
            int64_t c1 = c0 + TILE < cols ? c0 + TILE : cols;
            if (isz == 8) {
                for (int64_t r = r0; r < r1; r++) {
                    uint64_t *o = (uint64_t *)(out + r * out_row_stride);
                    for (int64_t c = c0; c < c1; c++) o[c] = *(const uint64_t *)(in + r * 8 + c * in_col_stride);
                }
            } else if (isz == 4) {
                for (int64_t r = r0; r < r1; r++) {
                    uint32_t *o = (uint32_t *)(out + r * out_row_stride);
                    for (int64_t c = c0; c < c1; c++) o[c] = *(const uint32_t *)(in + r * 4 + c * in_col_stride);
                }
            } else {
                for (int64_t r = r0; r < r1; r++)
                    for (int64_t c = c0; c < c1; c++)
                        memcpy(out + r * out_row_stride + c * isz, in + r * isz + c * in_col_stride, (size_t)isz);
            }
        }
    }
}
#undef TILE

int tsr_copy(int dtype_in, int dtype_out, int32_t ndim, const int64_t *shape, const void *in, const int64_t *sin,
             void *out, const int64_t *sout)
{
    if (tsr_itemsize(dtype_in) == 0 || tsr_itemsize(dtype_out) == 0) return TSR_EARG;
    const int64_t *st[2] = {sin, sout};
    tsr_iter it;
    int rc = tsr_iter_init(&it, 2, ndim, shape, st);
    if (rc != TSR_OK || it.size == 0) return rc;
    const int64_t isz = tsr_itemsize(dtype_in);
    if (dtype_in == dtype_out && it.nd == 2 && it.st[0][0] == isz && it.st[1][1] == isz && it.st[0][1] != isz
        && (isz == 8 || isz == 4 || isz == 16)) {
        /* transposed copy: walk 32x32 tiles so reads and writes both stay in cache */
        transpose_tiled(isz, it.shape[0], it.shape[1], (const char *)in, it.st[0][1], (char *)out, it.st[1][0]);
        return TSR_OK;
    }
    const void *base[2] = {in, out};
    int nd = it.nd;
    int err = TSR_OK;
    TSR_ITER_OUTER(it, base, {
        int r = cast_run(dtype_in, dtype_out, n, p[0], it.st[0][nd - 1], p[1], it.st[1][nd - 1]);
        if (r != TSR_OK) { err = r; break; }
    });
    return err;
}

/* ------------------------------------------------------------------ fill, arange, where */

int tsr_fill(int dtype, int64_t n, void *out, const void *value)
{
    int64_t sz = tsr_itemsize(dtype);
    if (sz == 0 || n < 0) return TSR_EARG;
    char *o = (char *)out;
    switch (sz) {
    case 1: memset(o, *(const uint8_t *)value, (size_t)n); break;
    case 4: { uint32_t v; memcpy(&v, value, 4); for (int64_t i = 0; i < n; i++) ((uint32_t *)o)[i] = v; break; }
    case 8: { uint64_t v; memcpy(&v, value, 8); for (int64_t i = 0; i < n; i++) ((uint64_t *)o)[i] = v; break; }
    default: for (int64_t i = 0; i < n; i++) memcpy(o + i * sz, value, (size_t)sz);
    }
    return TSR_OK;
}

int tsr_arange(int dtype, int64_t n, double start, double step, void *out)
{
    /* integer starts/steps convert without UB and the index arithmetic wraps like NumPy's int64 */
    const int64_t i0 = tsr_f2i64(start), di = tsr_f2i64(step);
    switch (dtype) {
    case TSR_F64: for (int64_t i = 0; i < n; i++) ((double *)out)[i] = start + (double)i * step; return TSR_OK;
    case TSR_F32: for (int64_t i = 0; i < n; i++) ((float *)out)[i] = (float)(start + (double)i * step); return TSR_OK;
    case TSR_I64: for (int64_t i = 0; i < n; i++) ((int64_t *)out)[i] = TSR_WADD(int64_t, i0, TSR_WMUL(int64_t, i, di)); return TSR_OK;
    case TSR_I32: for (int64_t i = 0; i < n; i++) ((int32_t *)out)[i] = (int32_t)TSR_WADD(int64_t, i0, TSR_WMUL(int64_t, i, di)); return TSR_OK;
    default: return TSR_ETYPE;
    }
}

int tsr_where(int dtype, int32_t ndim, const int64_t *shape, const void *cond, const int64_t *sc, const void *a,
              const int64_t *sa, const void *b, const int64_t *sb, void *out, const int64_t *sout)
{
    int64_t sz = tsr_itemsize(dtype);
    if (sz == 0) return TSR_EARG;
    const int64_t *st[4] = {sc, sa, sb, sout};
    tsr_iter it;
    int rc = tsr_iter_init(&it, 4, ndim, shape, st);
    if (rc != TSR_OK || it.size == 0) return rc;
    const void *base[4] = {cond, a, b, out};
    int nd = it.nd;
    int64_t s0 = it.st[0][nd - 1], s1 = it.st[1][nd - 1], s2 = it.st[2][nd - 1], s3 = it.st[3][nd - 1];
    TSR_ITER_OUTER(it, base, {
        for (int64_t i = 0; i < n; i++) {
            const char *src = *(const uint8_t *)(p[0] + i * s0) ? p[1] + i * s1 : p[2] + i * s2;
            memcpy(p[3] + i * s3, src, (size_t)sz);
        }
    });
    return TSR_OK;
}
