/*
 * NumPy routines with data-dependent output sizes (Tessero\Np): unique and the set operations, bincount,
 * histogram and histogram_bin_edges, cov and corrcoef, correlate and convolve, lexsort.
 *
 * Each follows NumPy 2.4's algorithm (numpy/lib/_arraysetops_impl.py, _histograms_impl.py,
 * _function_base_impl.py, numpy/_core/numeric.py) so the values, their order and the dtypes match.
 * Registry kind: routine (results are allocated here and adopted by the binding).
 */
#include "fn.h"

#include <stdlib.h>
#include <string.h>

void np_sort_doubles(double *x, int64_t n);
void np_argsort_pairs(double *pairs, int64_t n);

static inline int lt_nan(double a, double b) { return a < b || (b != b && a == a); }

/* ---------------------------------------------------------------- argument helpers */

static int arg_bool(const tsr_arg *args, int nargs, int k, int def)
{
    if (k >= nargs || args[k].kind == 0) return def;
    if (args[k].kind == 1 || args[k].kind == 4) return args[k].num != 0;
    return def;
}

static double arg_num(const tsr_arg *args, int nargs, int k, double def)
{
    if (k >= nargs || args[k].kind == 0) return def;
    if (args[k].kind == 1 || args[k].kind == 4) return args[k].num;
    return def;
}

static int is_intlike(int dt) { return dt == TSR_I64 || dt == TSR_I32 || dt == TSR_U8 || dt == TSR_BOOL; }

/* output dtype for values taken from an input: NumPy keeps the input dtype */
static int value_dtype(const tsr_arg *a) { return a->kind == 3 && is_intlike(a->arr.dtype) ? a->arr.dtype : TSR_F64; }

/* the result dtype of a set operation on two arrays: NumPy's promotion of bool, uint8, int32, int64 and float64 */
static int promote_set(int a, int b)
{
    if (a == b) return a;
    if (!is_intlike(a) || !is_intlike(b)) return TSR_F64;
    static const int rank[] = {[TSR_BOOL] = 0, [TSR_U8] = 1, [TSR_I32] = 2, [TSR_I64] = 3};
    return rank[a] >= rank[b] ? a : b;
}

static void store_as(int dtype, void *base, int64_t i, double v)
{
    switch (dtype) {
    case TSR_F64: ((double *)base)[i] = v; break;
    case TSR_F32: ((float *)base)[i] = (float)v; break;
    /* out-of-range floats: INT_MIN, as store_double in fn_core.c (no undefined float -> int cast) */
    case TSR_I64: ((int64_t *)base)[i] = (v > -9223372036854775808.0 && v < 9223372036854775808.0) ? (int64_t)v : INT64_MIN; break;
    case TSR_I32: ((int32_t *)base)[i] = (v > -2147483649.0 && v < 2147483648.0) ? (int32_t)v : INT32_MIN; break;
    case TSR_U8: case TSR_BOOL: ((uint8_t *)base)[i] = (v > -9223372036854775808.0 && v < 9223372036854775808.0) ? (uint8_t)(int64_t)v : 0; break;
    default: break;
    }
}

static int require_array(const tsr_arg *args, int nargs, int k, const char *name)
{
    if (k >= nargs || args[k].kind != 3) { fn_set_error("%s must be an array", name); return 0; }
    if (args[k].arr.dtype == TSR_C128) { fn_set_error("%s: complex input is not supported", name); return 0; }
    return 1;
}

/* a 1-D result from doubles */
static int result_vec(tsr_result *r, int dtype, const double *v, int64_t n)
{
    void *p = fn_result_array(r, dtype, 1, &n);
    if (!p) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++) store_as(dtype, p, i, v[i]);
    return TSR_OK;
}

static int result_vec_i(tsr_result *r, const int64_t *v, int64_t n)
{
    int64_t *p = (int64_t *)fn_result_array(r, TSR_I64, 1, &n);
    if (!p) return TSR_ENOMEM;
    if (n) memcpy(p, v, (size_t)n * sizeof(int64_t));
    return TSR_OK;
}


/* ---------------------------------------------------------------- scalars and exact int64 keys */

/* NumPy converts a scalar argument with np.asanyarray: unique(5) is unique([5]) after ravel. A number or bool
   argument becomes a 0-d array over buf. */
static const tsr_arg *as_array_arg(const tsr_arg *a, tsr_arg *tmp, unsigned char buf[8])
{
    if (a->kind != 1 && a->kind != 4) return a;
    memset(tmp, 0, sizeof *tmp);
    tmp->kind = 3;
    int dt;
    if (a->kind == 4) { dt = TSR_BOOL; buf[0] = a->num != 0; }
    else if (a->flags & 1) { dt = TSR_I64; memcpy(buf, &a->ival, 8); }
    else { dt = TSR_F64; memcpy(buf, &a->num, 8); }
    tsr_array_init(&tmp->arr, dt, 0, NULL, buf);
    return tmp;
}

/* Doubles hold int64 exactly only up to 2^53. When every array is an integer array and one is int64, the set
   routines run on ranks: each value becomes its index in the sorted table of distinct values, which keeps
   order and equality, and results map back through the table. */
typedef struct { int64_t *t; int64_t nt, bytes; } keys;

static int cmp_i64(const void *a, const void *b)
{
    const int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static int64_t load_i64(int dt, const char *p)
{
    switch (dt) {
    case TSR_I64: return *(const int64_t *)p;
    case TSR_I32: return *(const int32_t *)p;
    default: return *(const uint8_t *)p;
    }
}

static int64_t gather_i64(const tsr_array *x, int64_t *out)
{
    const int64_t size = tsr_shape_size(x->ndim, x->shape);
    int64_t idx[TSR_MAXDIM] = {0};
    for (int64_t k = 0; k < size; k++) {
        const char *p = (const char *)x->data + x->offset;
        for (int32_t d = 0; d < x->ndim; d++) p += idx[d] * x->strides[d];
        out[k] = load_i64(x->dtype, p);
        for (int32_t d = x->ndim - 1; d >= 0; d--) { if (++idx[d] < x->shape[d]) break; idx[d] = 0; }
    }
    return size;
}

static int keys_build(keys *ks, const tsr_arg *a0, const tsr_arg *a1)
{
    memset(ks, 0, sizeof *ks);
    const tsr_arg *as[2] = {a0, a1};
    int any64 = 0;
    int64_t total = 0;
    for (int k = 0; k < 2; k++) {
        if (!as[k]) continue;
        if (!is_intlike(as[k]->arr.dtype)) return TSR_OK;
        if (as[k]->arr.dtype == TSR_I64) any64 = 1;
        total += tsr_shape_size(as[k]->arr.ndim, as[k]->arr.shape);
    }
    if (!any64) return TSR_OK;
    ks->bytes = (total > 0 ? total : 1) * 8;
    ks->t = (int64_t *)tsr_alloc(ks->bytes);
    if (!ks->t) return TSR_ENOMEM;
    int64_t m = 0;
    for (int k = 0; k < 2; k++) if (as[k]) m += gather_i64(&as[k]->arr, ks->t + m);
    qsort(ks->t, (size_t)m, 8, cmp_i64);
    int64_t w = 0;
    for (int64_t i = 0; i < m; i++) if (w == 0 || ks->t[i] != ks->t[w - 1]) ks->t[w++] = ks->t[i];
    ks->nt = w;
    return TSR_OK;
}

static void keys_free(keys *ks) { if (ks->t) tsr_free(ks->t, ks->bytes); memset(ks, 0, sizeof *ks); }

/* the argument's values in C order: doubles, or ranks in the key table (freed with fn_free_doubles) */
static double *ks_doubles(const keys *ks, const tsr_arg *a, int64_t *n)
{
    if (!ks->t) return fn_arg_doubles(a, n);
    const int64_t size = tsr_shape_size(a->arr.ndim, a->arr.shape);
    double *out = (double *)tsr_alloc(size > 8 ? size * 8 : 64);
    if (!out) return NULL;
    int64_t *iv = (int64_t *)out;                        /* same width: convert in place */
    gather_i64(&a->arr, iv);
    for (int64_t i = 0; i < size; i++) {
        const int64_t v = iv[i];
        int64_t lo = 0, hi = ks->nt;
        while (lo < hi) { const int64_t mid = lo + (hi - lo) / 2; if (ks->t[mid] < v) lo = mid + 1; else hi = mid; }
        out[i] = (double)lo;
    }
    *n = size;
    return out;
}

/* a 1-D result of values (ranks when the key table is in use) */
static int ks_result(const keys *ks, tsr_result *r, int dtype, const double *v, int64_t n)
{
    if (!ks->t) return result_vec(r, dtype, v, n);
    void *p = fn_result_array(r, dtype, 1, &n);
    if (!p) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++) {
        const int64_t x = ks->t[(int64_t)v[i]];
        if (dtype == TSR_I64) ((int64_t *)p)[i] = x;
        else store_as(dtype, p, i, (double)x);           /* a value of a narrower input: exact */
    }
    return TSR_OK;
}

/* argument k as an array (a scalar becomes 0-d), or NULL with the error set */
static const tsr_arg *set_arg(const tsr_arg *args, int nargs, int k, const char *name, tsr_arg *tmp, unsigned char buf[8])
{
    if (k < nargs && (args[k].kind == 1 || args[k].kind == 4)) return as_array_arg(&args[k], tmp, buf);
    if (!require_array(args, nargs, k, name)) return NULL;
    return &args[k];
}

/* ---------------------------------------------------------------- unique (numpy's _unique1d) */

typedef struct {
    double *vals;       /* unique values */
    int64_t *index;     /* first-occurrence indices (if requested) */
    int64_t *inverse;   /* per input element (if requested) */
    int64_t *counts;    /* per unique value (if requested) */
    int64_t nu, n;
} uniq;

static void uniq_free(uniq *u)
{
    if (u->vals) tsr_free(u->vals, (u->n > 0 ? u->n : 1) * 8);
    if (u->index) tsr_free(u->index, (u->n > 0 ? u->n : 1) * 8);
    if (u->inverse) tsr_free(u->inverse, (u->n > 0 ? u->n : 1) * 8);
    if (u->counts) tsr_free(u->counts, (u->n > 0 ? u->n : 1) * 8);
    memset(u, 0, sizeof *u);
}

static int unique1d(const double *x, int64_t n, int want_index, int want_inverse, int want_counts, int equal_nan, uniq *u)
{
    memset(u, 0, sizeof *u);
    u->n = n;
    const int64_t cap = n > 0 ? n : 1;
    double *pairs = (double *)tsr_alloc(cap * 16);
    unsigned char *mask = (unsigned char *)tsr_alloc(cap);
    u->vals = (double *)tsr_alloc(cap * 8);
    if (want_index) u->index = (int64_t *)tsr_alloc(cap * 8);
    if (want_inverse) u->inverse = (int64_t *)tsr_alloc(cap * 8);
    if (want_counts) u->counts = (int64_t *)tsr_alloc(cap * 8);
    if (!pairs || !mask || !u->vals || (want_index && !u->index) || (want_inverse && !u->inverse) || (want_counts && !u->counts)) {
        if (pairs) tsr_free(pairs, cap * 16);
        if (mask) tsr_free(mask, cap);
        uniq_free(u);
        return TSR_ENOMEM;
    }
    for (int64_t i = 0; i < n; i++) { pairs[2 * i] = x[i]; pairs[2 * i + 1] = (double)i; }
    np_argsort_pairs(pairs, n);                          /* stable: first occurrence first (numpy mergesort) */
    /* mask of first elements of each run; NaNs collapse to one when equal_nan */
    if (n > 0) mask[0] = 1;
    const double last = n > 0 ? pairs[2 * (n - 1)] : 0;
    if (equal_nan && n > 0 && last != last) {
        int64_t firstnan = 0;
        while (firstnan < n && pairs[2 * firstnan] == pairs[2 * firstnan]) firstnan++;
        for (int64_t i = 1; i < firstnan; i++) mask[i] = pairs[2 * i] != pairs[2 * (i - 1)];
        mask[firstnan] = 1;
        for (int64_t i = firstnan + 1; i < n; i++) mask[i] = 0;
    } else {
        for (int64_t i = 1; i < n; i++) mask[i] = pairs[2 * i] != pairs[2 * (i - 1)];
    }
    int64_t k = -1;
    for (int64_t i = 0; i < n; i++) {
        if (mask[i]) {
            k++;
            u->vals[k] = pairs[2 * i];
            if (want_index) u->index[k] = (int64_t)pairs[2 * i + 1];
            if (want_counts) u->counts[k] = 0;
        }
        if (want_inverse) u->inverse[(int64_t)pairs[2 * i + 1]] = k;
        if (want_counts) u->counts[k]++;
    }
    u->nu = k + 1;
    tsr_free(pairs, cap * 16);
    tsr_free(mask, cap);
    return TSR_OK;
}

typedef struct { int variant; } uctx;   /* 0 unique, 1 unique_values, 2 unique_counts, 3 unique_inverse, 4 unique_all */
static const uctx U_UNIQUE = {0}, U_VALUES = {1}, U_COUNTS = {2}, U_INVERSE = {3}, U_ALL = {4};

static int r_unique(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int variant = ((const uctx *)ctx)->variant;
    tsr_arg tmp;
    unsigned char buf[8];
    const tsr_arg *a0 = set_arg(args, nargs, 0, "ar", &tmp, buf);
    if (!a0) return TSR_EARG;
    int want_index = 0, want_inverse = 0, want_counts = 0, equal_nan = 0;
    switch (variant) {
    case 0:
        want_index = arg_bool(args, nargs, 1, 0);
        want_inverse = arg_bool(args, nargs, 2, 0);
        want_counts = arg_bool(args, nargs, 3, 0);
        equal_nan = arg_bool(args, nargs, 4, 1);
        break;
    case 2: want_counts = 1; break;
    case 3: want_inverse = 1; break;
    case 4: want_index = want_inverse = want_counts = 1; break;
    default: break;
    }
    int64_t n = 0;
    keys ks;
    if (keys_build(&ks, a0, NULL) != TSR_OK) return TSR_ENOMEM;
    double *x = ks_doubles(&ks, a0, &n);
    if (!x) { keys_free(&ks); return TSR_ENOMEM; }
    uniq u;
    int rc = unique1d(x, n, want_index, want_inverse, want_counts, equal_nan, &u);
    fn_free_doubles(x, n);
    if (rc != TSR_OK) { keys_free(&ks); return rc; }
    /* result slots follow the declared outputs: unique -> values, indices, inverse, counts */
    const int s_index = variant == 0 || variant == 4 ? 1 : -1;
    const int s_inverse = variant == 0 || variant == 4 ? 2 : (variant == 3 ? 1 : -1);
    const int s_counts = variant == 0 || variant == 4 ? 3 : (variant == 2 ? 1 : -1);
    rc = ks_result(&ks, &res[0], value_dtype(a0), u.vals, u.nu);
    keys_free(&ks);
    if (rc == TSR_OK && want_index) rc = result_vec_i(&res[s_index], u.index, u.nu);
    if (rc == TSR_OK && want_inverse) {
        /* numpy 2.x: the inverse has the input's shape */
        const tsr_array *a = &a0->arr;
        int64_t *p = (int64_t *)fn_result_array(&res[s_inverse], TSR_I64, a->ndim, a->shape);
        if (!p) rc = TSR_ENOMEM;
        else if (n) memcpy(p, u.inverse, (size_t)n * 8);
    }
    if (rc == TSR_OK && want_counts) rc = result_vec_i(&res[s_counts], u.counts, u.nu);
    uniq_free(&u);
    return rc;
}

/* ---------------------------------------------------------------- set operations */

static int unique_of(const keys *ks, const tsr_arg *a, uniq *u, int want_index)
{
    int64_t n = 0;
    double *x = ks_doubles(ks, a, &n);
    if (!x) return TSR_ENOMEM;
    const int rc = unique1d(x, n, want_index, 0, 0, 1, u);
    fn_free_doubles(x, n);
    return rc;
}

static int intersect1d_body(const keys *ks_, const tsr_arg *p0, const tsr_arg *p1, const tsr_arg *args, int nargs, tsr_result *res)
{
    const keys ks = *ks_;
    const int assume_unique = arg_bool(args, nargs, 2, 0), return_indices = arg_bool(args, nargs, 3, 0);
    double *a1, *a2;
    int64_t n1, n2, *i1 = NULL, *i2 = NULL;
    uniq u1 = {0}, u2 = {0};
    double *raw1 = NULL, *raw2 = NULL;
    int rc;
    if (!assume_unique) {
        if ((rc = unique_of(&ks, p0, &u1, return_indices)) != TSR_OK) return rc;
        if ((rc = unique_of(&ks, p1, &u2, return_indices)) != TSR_OK) { uniq_free(&u1); return rc; }
        a1 = u1.vals; n1 = u1.nu; a2 = u2.vals; n2 = u2.nu;
        i1 = u1.index; i2 = u2.index;
    } else {
        raw1 = ks_doubles(&ks, p0, &n1);
        raw2 = ks_doubles(&ks, p1, &n2);
        if (!raw1 || !raw2) { fn_free_doubles(raw1, n1); fn_free_doubles(raw2, n2); return TSR_ENOMEM; }
        a1 = raw1; a2 = raw2;
    }
    const int64_t m = n1 + n2;
    double *pairs = (double *)tsr_alloc((m > 0 ? m : 1) * 16);
    double *vals = (double *)tsr_alloc((m > 0 ? m : 1) * 8);
    int64_t *idx1 = (int64_t *)tsr_alloc((m > 0 ? m : 1) * 8), *idx2 = (int64_t *)tsr_alloc((m > 0 ? m : 1) * 8);
    rc = (!pairs || !vals || !idx1 || !idx2) ? TSR_ENOMEM : TSR_OK;
    int64_t k = 0;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i < n1; i++) { pairs[2 * i] = a1[i]; pairs[2 * i + 1] = (double)i; }
        for (int64_t i = 0; i < n2; i++) { pairs[2 * (n1 + i)] = a2[i]; pairs[2 * (n1 + i) + 1] = (double)(n1 + i); }
        np_argsort_pairs(pairs, m);                      /* numpy: mergesort when indices are wanted, else sort */
        for (int64_t i = 0; i + 1 < m; i++) {
            if (pairs[2 * (i + 1)] == pairs[2 * i]) {
                vals[k] = pairs[2 * i];
                int64_t j1 = (int64_t)pairs[2 * i + 1], j2 = (int64_t)pairs[2 * (i + 1) + 1] - n1;
                if (!assume_unique && return_indices) { j1 = i1[j1]; j2 = i2[j2]; }
                idx1[k] = j1;
                idx2[k] = j2;
                k++;
            }
        }
        const int dt = promote_set(value_dtype(p0), value_dtype(p1));
        rc = ks_result(&ks, &res[0], dt, vals, k);
        if (rc == TSR_OK && return_indices) {
            rc = result_vec_i(&res[1], idx1, k);
            if (rc == TSR_OK) rc = result_vec_i(&res[2], idx2, k);
        }
    }
    if (pairs) tsr_free(pairs, (m > 0 ? m : 1) * 16);
    if (vals) tsr_free(vals, (m > 0 ? m : 1) * 8);
    if (idx1) tsr_free(idx1, (m > 0 ? m : 1) * 8);
    if (idx2) tsr_free(idx2, (m > 0 ? m : 1) * 8);
    uniq_free(&u1);
    uniq_free(&u2);
    fn_free_doubles(raw1, n1);
    fn_free_doubles(raw2, n2);
    return rc;
}

static int r_intersect1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    tsr_arg t0, t1;
    unsigned char b0[8], b1[8];
    const tsr_arg *a0 = set_arg(args, nargs, 0, "ar1", &t0, b0), *a1 = a0 ? set_arg(args, nargs, 1, "ar2", &t1, b1) : NULL;
    if (!a0 || !a1) return TSR_EARG;
    keys ks;
    if (keys_build(&ks, a0, a1) != TSR_OK) return TSR_ENOMEM;
    const int rc = intersect1d_body(&ks, a0, a1, args, nargs, res);
    keys_free(&ks);
    return rc;
}

static int union1d_body(const keys *ks_, const tsr_arg *p0, const tsr_arg *p1, const tsr_arg *args, int nargs, tsr_result *res)
{
    const keys ks = *ks_;
    int64_t n1, n2;
    double *v1 = ks_doubles(&ks, p0, &n1), *v2 = ks_doubles(&ks, p1, &n2);
    if (!v1 || !v2) { fn_free_doubles(v1, n1); fn_free_doubles(v2, n2); return TSR_ENOMEM; }
    const int64_t m = n1 + n2;
    double *cat = (double *)tsr_alloc((m > 0 ? m : 1) * 8);
    int rc = cat ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        memcpy(cat, v1, (size_t)n1 * 8);
        memcpy(cat + n1, v2, (size_t)n2 * 8);
        uniq u;
        rc = unique1d(cat, m, 0, 0, 0, 1, &u);
        if (rc == TSR_OK) {
            const int dt = promote_set(value_dtype(p0), value_dtype(p1));
            rc = ks_result(&ks, &res[0], dt, u.vals, u.nu);
            uniq_free(&u);
        }
        tsr_free(cat, (m > 0 ? m : 1) * 8);
    }
    fn_free_doubles(v1, n1);
    fn_free_doubles(v2, n2);
    return rc;
}

static int r_union1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    tsr_arg t0, t1;
    unsigned char b0[8], b1[8];
    const tsr_arg *a0 = set_arg(args, nargs, 0, "ar1", &t0, b0), *a1 = a0 ? set_arg(args, nargs, 1, "ar2", &t1, b1) : NULL;
    if (!a0 || !a1) return TSR_EARG;
    keys ks;
    if (keys_build(&ks, a0, a1) != TSR_OK) return TSR_ENOMEM;
    const int rc = union1d_body(&ks, a0, a1, args, nargs, res);
    keys_free(&ks);
    return rc;
}

static int contains_sorted(const double *s, int64_t n, double v)
{
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (lt_nan(s[mid], v)) lo = mid + 1;
        else hi = mid;
    }
    return lo < n && s[lo] == v;
}

static int setdiff1d_body(const keys *ks_, const tsr_arg *p0, const tsr_arg *p1, const tsr_arg *args, int nargs, tsr_result *res)
{
    const keys ks = *ks_;
    const int assume_unique = arg_bool(args, nargs, 2, 0);
    int64_t n1, n2;
    double *a1, *a2;
    uniq u1 = {0}, u2 = {0};
    double *raw1 = NULL, *raw2 = NULL;
    int rc;
    if (assume_unique) {
        raw1 = ks_doubles(&ks, p0, &n1);
        if (!raw1) return TSR_ENOMEM;
        a1 = raw1;
    } else {
        if ((rc = unique_of(&ks, p0, &u1, 0)) != TSR_OK) return rc;
        a1 = u1.vals; n1 = u1.nu;
    }
    if (assume_unique) {
        raw2 = ks_doubles(&ks, p1, &n2);
        if (!raw2) { fn_free_doubles(raw1, n1); return TSR_ENOMEM; }
        np_sort_doubles(raw2, n2);
        a2 = raw2;
    } else {
        if ((rc = unique_of(&ks, p1, &u2, 0)) != TSR_OK) { uniq_free(&u1); fn_free_doubles(raw1, n1); return rc; }
        a2 = u2.vals; n2 = u2.nu;
    }
    double *out = (double *)tsr_alloc((n1 > 0 ? n1 : 1) * 8);
    rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        int64_t k = 0;
        for (int64_t i = 0; i < n1; i++) if (!contains_sorted(a2, n2, a1[i])) out[k++] = a1[i];
        rc = ks_result(&ks, &res[0], value_dtype(p0), out, k);
        tsr_free(out, (n1 > 0 ? n1 : 1) * 8);
    }
    uniq_free(&u1);
    uniq_free(&u2);
    fn_free_doubles(raw1, n1);
    fn_free_doubles(raw2, n2);
    return rc;
}

/* setdiff1d of an int64 array and a float array: NumPy keeps ar1's values exactly and tests membership in float64 */
static int setdiff_int_float(const tsr_arg *p0, const tsr_arg *p1, int assume_unique, tsr_result *res)
{
    keys ks;
    if (keys_build(&ks, p0, NULL) != TSR_OK) return TSR_ENOMEM;
    int64_t nr = 0, n1, n2 = 0;
    double *r1 = ks_doubles(&ks, p0, &nr), *a2 = fn_arg_doubles(p1, &n2);
    n1 = nr;
    uniq u = {0};
    int rc = r1 && a2 ? TSR_OK : TSR_ENOMEM;
    const double *v1 = r1;
    if (rc == TSR_OK && !assume_unique && (rc = unique1d(r1, n1, 0, 0, 0, 1, &u)) == TSR_OK) { v1 = u.vals; n1 = u.nu; }
    if (rc == TSR_OK) {
        np_sort_doubles(a2, n2);
        int64_t *p = (int64_t *)tsr_alloc((n1 > 0 ? n1 : 1) * 8);
        if (!p) rc = TSR_ENOMEM;
        else {
            int64_t k = 0;
            for (int64_t i = 0; i < n1; i++) {
                const int64_t v = ks.t[(int64_t)v1[i]];
                if (!contains_sorted(a2, n2, (double)v)) p[k++] = v;
            }
            rc = result_vec_i(&res[0], p, k);
            tsr_free(p, (n1 > 0 ? n1 : 1) * 8);
        }
    }
    uniq_free(&u);
    fn_free_doubles(r1, nr);
    fn_free_doubles(a2, n2);
    keys_free(&ks);
    return rc;
}

static int r_setdiff1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    tsr_arg t0, t1;
    unsigned char b0[8], b1[8];
    const tsr_arg *a0 = set_arg(args, nargs, 0, "ar1", &t0, b0), *a1 = a0 ? set_arg(args, nargs, 1, "ar2", &t1, b1) : NULL;
    if (!a0 || !a1) return TSR_EARG;
    if (a0->arr.dtype == TSR_I64 && !is_intlike(a1->arr.dtype)) return setdiff_int_float(a0, a1, arg_bool(args, nargs, 2, 0), res);
    keys ks;
    if (keys_build(&ks, a0, a1) != TSR_OK) return TSR_ENOMEM;
    const int rc = setdiff1d_body(&ks, a0, a1, args, nargs, res);
    keys_free(&ks);
    return rc;
}

static int setxor1d_body(const keys *ks_, const tsr_arg *p0, const tsr_arg *p1, const tsr_arg *args, int nargs, tsr_result *res)
{
    const keys ks = *ks_;
    const int assume_unique = arg_bool(args, nargs, 2, 0);
    int64_t n1, n2;
    double *a1, *a2;
    uniq u1 = {0}, u2 = {0};
    double *raw1 = NULL, *raw2 = NULL;
    int rc;
    if (assume_unique) {
        raw1 = ks_doubles(&ks, p0, &n1);
        raw2 = ks_doubles(&ks, p1, &n2);
        if (!raw1 || !raw2) { fn_free_doubles(raw1, n1); fn_free_doubles(raw2, n2); return TSR_ENOMEM; }
        a1 = raw1; a2 = raw2;
    } else {
        if ((rc = unique_of(&ks, p0, &u1, 0)) != TSR_OK) return rc;
        if ((rc = unique_of(&ks, p1, &u2, 0)) != TSR_OK) { uniq_free(&u1); return rc; }
        a1 = u1.vals; n1 = u1.nu; a2 = u2.vals; n2 = u2.nu;
    }
    const int64_t m = n1 + n2;
    double *aux = (double *)tsr_alloc((m > 0 ? m : 1) * 8), *out = (double *)tsr_alloc((m > 0 ? m : 1) * 8);
    rc = (aux && out) ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        memcpy(aux, a1, (size_t)n1 * 8);
        memcpy(aux + n1, a2, (size_t)n2 * 8);
        np_sort_doubles(aux, m);
        int64_t k = 0;
        for (int64_t i = 0; i < m; i++) {
            /* flag = [True, aux[1:] != aux[:-1], True]; keep where flag[i] & flag[i+1] */
            const int left = i == 0 || aux[i] != aux[i - 1];
            const int right = i == m - 1 || aux[i + 1] != aux[i];
            if (left && right) out[k++] = aux[i];
        }
        const int dt = promote_set(value_dtype(p0), value_dtype(p1));
        rc = ks_result(&ks, &res[0], dt, out, k);
    }
    if (aux) tsr_free(aux, (m > 0 ? m : 1) * 8);
    if (out) tsr_free(out, (m > 0 ? m : 1) * 8);
    uniq_free(&u1);
    uniq_free(&u2);
    fn_free_doubles(raw1, n1);
    fn_free_doubles(raw2, n2);
    return rc;
}

static int r_setxor1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    tsr_arg t0, t1;
    unsigned char b0[8], b1[8];
    const tsr_arg *a0 = set_arg(args, nargs, 0, "ar1", &t0, b0), *a1 = a0 ? set_arg(args, nargs, 1, "ar2", &t1, b1) : NULL;
    if (!a0 || !a1) return TSR_EARG;
    keys ks;
    if (keys_build(&ks, a0, a1) != TSR_OK) return TSR_ENOMEM;
    const int rc = setxor1d_body(&ks, a0, a1, args, nargs, res);
    keys_free(&ks);
    return rc;
}

/* ---------------------------------------------------------------- bincount */

static int r_bincount(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    if (!require_array(args, nargs, 0, "x")) return TSR_EARG;
    if (args[0].arr.ndim != 1) { fn_set_error("object too deep for desired array"); return TSR_EARG; }
    const double minlength = arg_num(args, nargs, 2, 0);
    if (nargs > 2 && args[2].kind == 1 && !(args[2].flags & 1)) {
        fn_set_error("'float' object cannot be interpreted as an integer");
        return TSR_ETYPE;
    }
    /* NumPy: minlength is an integer (a float is a TypeError), the list an integer array (a float array fails
       the 'safe' cast even when its values are integral; an empty PHP array has no dtype and is accepted) */
    if (minlength != floor(minlength) || !(fabs(minlength) < 9.2e18)) { fn_set_error("'float' object cannot be interpreted as an integer"); return TSR_ETYPE; }
    if (minlength < 0) { fn_set_error("'minlength' must not be negative"); return TSR_EARG; }
    const int from_list = (args[0].flags & 2) != 0;
    if ((args[0].arr.dtype == TSR_F64 || args[0].arr.dtype == TSR_F32) && !from_list) {
        fn_set_error("Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'");
        return TSR_ETYPE;
    }
    int64_t n, nw = 0;
    double *x = fn_arg_doubles(&args[0], &n);
    if (!x) return TSR_ENOMEM;
    if (from_list && (args[0].arr.dtype == TSR_F64 || args[0].arr.dtype == TSR_F32)) {
        /* NumPy 2.4 converts a Python list of floats item by item with int() (deprecated since 2.1) */
        for (int64_t i = 0; i < n; i++) {
            const double v = x[i];
            if (v != v) { fn_free_doubles(x, n); fn_set_error("cannot convert float NaN to integer"); return TSR_EARG; }
            if (isinf(v)) { fn_free_doubles(x, n); fn_set_error("cannot convert float infinity to integer"); return TSR_EARG; }
            if (!(fabs(v) < 9223372036854775808.0)) { fn_free_doubles(x, n); fn_set_error("Python int too large to convert to C long"); return TSR_EARG; }
            x[i] = trunc(v);
        }
    }
    double *w = NULL;
    if (nargs > 1 && args[1].kind == 3) {
        w = fn_arg_doubles(&args[1], &nw);
        if (!w) { fn_free_doubles(x, n); return TSR_ENOMEM; }
        if (nw != n) { fn_free_doubles(x, n); fn_free_doubles(w, nw); fn_set_error("The weights and list don't have the same length."); return TSR_EARG; }
    }
    int64_t mx = -1;
    for (int64_t i = 0; i < n; i++) {
        if (!(x[i] >= 0)) { fn_free_doubles(x, n); fn_free_doubles(w, nw); fn_set_error("'list' argument must have no negative elements"); return TSR_EARG; }
        if (x[i] != floor(x[i]) || !(x[i] < 9.2e18)) { fn_free_doubles(x, n); fn_free_doubles(w, nw); fn_set_error("Cannot cast array data to int64 according to the rule 'safe'"); return TSR_ETYPE; }
        if ((int64_t)x[i] > mx) mx = (int64_t)x[i];
    }
    int64_t len = mx + 1;
    if ((int64_t)minlength > len) len = (int64_t)minlength;
    int rc = TSR_OK;
    if (w && n > 0) {                                     /* numpy: an empty list gives int64 zeros, weights or not */
        double *p = (double *)fn_result_array(&res[0], TSR_F64, 1, &len);
        if (!p) rc = TSR_ENOMEM;
        else {
            memset(p, 0, (size_t)len * 8);
            for (int64_t i = 0; i < n; i++) p[(int64_t)x[i]] += w[i];
        }
    } else {
        int64_t *p = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, &len);
        if (!p) rc = TSR_ENOMEM;
        else {
            memset(p, 0, (size_t)len * 8);
            for (int64_t i = 0; i < n; i++) p[(int64_t)x[i]]++;
        }
    }
    fn_free_doubles(x, n);
    fn_free_doubles(w, nw);
    return rc;
}

/* ---------------------------------------------------------------- histogram */

enum { EST_NONE = -1, EST_AUTO, EST_FD, EST_DOANE, EST_SCOTT, EST_STONE, EST_RICE, EST_STURGES, EST_SQRT };

static int estimator_of(const char *s)
{
    static const char *const names[] = {"auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"};
    for (int k = 0; k < 8; k++) if (strcmp(s, names[k]) == 0) return k;
    return -2;
}

static double ptp_of(const double *x, int64_t n)
{
    double lo = x[0], hi = x[0];
    for (int64_t i = 1; i < n; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
    return hi - lo;
}

/* numpy.std of a 1-D contiguous array (pairwise mean, pairwise sum of squared deviations) */
static double std_of(const double *x, int64_t n, double *tmp)
{
    const double mean = tsr_psum(x, n) / (double)n;
    for (int64_t i = 0; i < n; i++) { const double d = x[i] - mean; tmp[i] = d * d; }
    return sqrt(tsr_psum(tmp, n) / (double)n);
}

double np_percentile_linear(const double *sorted, int64_t n, double q);

static int hist_counts_uniform(const double *a, int64_t n, const double *w, double first, double last, int64_t nb,
                               const double *edges, double *out);

/* bin width of an estimator (numpy's _hist_bin_*) */
static double estimator_width(int est, double *x, int64_t n, double first, double last, double *tmp, int *err)
{
    *err = TSR_OK;
    const double N = (double)n;
    switch (est) {
    case EST_SQRT: return ptp_of(x, n) / sqrt(N);
    case EST_STURGES: return ptp_of(x, n) / (log2(N) + 1.0);
    case EST_RICE: return ptp_of(x, n) / (2.0 * pow(N, 1.0 / 3));
    case EST_SCOTT: return pow(24.0 * pow(M_PI, 0.5) / N, 1.0 / 3.0) * std_of(x, n, tmp);
    case EST_DOANE: {
        if (n > 2) {
            const double sg1 = sqrt(6.0 * (N - 2) / ((N + 1.0) * (N + 3)));
            const double sigma = std_of(x, n, tmp);
            if (sigma > 0.0) {
                const double mean = tsr_psum(x, n) / N;
                for (int64_t i = 0; i < n; i++) { double t = (x[i] - mean) / sigma; tmp[i] = t * t * t; }
                const double g1 = tsr_psum(tmp, n) / N;
                return ptp_of(x, n) / (1.0 + log2(N) + log2(1.0 + fabs(g1) / sg1));
            }
        }
        return 0.0;
    }
    case EST_FD: {
        memcpy(tmp, x, (size_t)n * 8);
        np_sort_doubles(tmp, n);
        const double iqr = np_percentile_linear(tmp, n, 75) - np_percentile_linear(tmp, n, 25);
        return 2.0 * iqr * pow(N, -1.0 / 3.0);
    }
    case EST_AUTO: {
        const double fd = estimator_width(EST_FD, x, n, first, last, tmp, err);
        const double sturges = estimator_width(EST_STURGES, x, n, first, last, tmp, err);
        const double sq = estimator_width(EST_SQRT, x, n, first, last, tmp, err);
        const double fdc = fd > sq / 2 ? fd : sq / 2;
        return fdc < sturges ? fdc : sturges;
    }
    case EST_STONE: {
        const double ptp = ptp_of(x, n);
        if (n <= 1 || ptp == 0) return 0;
        int64_t upper = (int64_t)sqrt(N);
        if (upper < 100) upper = 100;
        double best = 0;
        int64_t best_nb = 1;
        double *counts = (double *)tsr_alloc(upper * 8 + 64), *edges = (double *)tsr_alloc((upper + 1) * 8 + 64);
        if (!counts || !edges) {
            if (counts) tsr_free(counts, upper * 8 + 64);
            if (edges) tsr_free(edges, (upper + 1) * 8 + 64);
            *err = TSR_ENOMEM;
            return 0;
        }
        for (int64_t nb = 1; nb <= upper; nb++) {
            /* np.histogram(x, bins=nb, range=range)[0] / n, jhat = (2 - (n + 1) p.p) / hh */
            const double delta = last - first, step = delta / (double)nb;
            for (int64_t i = 0; i <= nb; i++) edges[i] = (step == 0 ? ((double)i / (double)nb) * delta : (double)i * step) + first;
            edges[nb] = last;
            hist_counts_uniform(x, n, NULL, first, last, nb, edges, counts);
            double dot = 0;
            for (int64_t i = 0; i < nb; i++) { const double pk = counts[i] / N; dot += pk * pk; }
            const double hh = ptp / (double)nb;
            const double j = (2 - (N + 1) * dot) / hh;
            if (nb == 1 || j < best) { best = j; best_nb = nb; }
        }
        tsr_free(counts, upper * 8 + 64);
        tsr_free(edges, (upper + 1) * 8 + 64);
        return ptp / (double)best_nb;
    }
    default: return 0;
    }
}

/* numpy's uniform-bin fast path: counts (or weighted sums, blockwise) */
static int hist_counts_uniform(const double *a, int64_t n, const double *w, double first, double last, int64_t nb,
                               const double *edges, double *out)
{
    const int64_t BLOCK = 65536;
    for (int64_t i = 0; i < nb; i++) out[i] = 0;
    const double norm_denom = last - first, norm_numerator = (double)nb;
    double *blk = w ? (double *)tsr_alloc(nb * 8 + 64) : NULL;
    if (w && !blk) return TSR_ENOMEM;
    for (int64_t b0 = 0; b0 < n; b0 += BLOCK) {
        const int64_t b1 = b0 + BLOCK < n ? b0 + BLOCK : n;
        if (blk) for (int64_t i = 0; i < nb; i++) blk[i] = 0;
        for (int64_t i = b0; i < b1; i++) {
            const double v = a[i];
            if (!(v >= first) || !(v <= last)) continue;
            const double f = ((v - first) / norm_denom) * norm_numerator;
            int64_t idx = (int64_t)f;
            if (idx == nb) idx -= 1;
            if (v < edges[idx]) idx -= 1;
            if (v >= edges[idx + 1] && idx != nb - 1) idx += 1;
            if (blk) blk[idx] += w[i];
            else out[idx] += 1;
        }
        if (blk) for (int64_t i = 0; i < nb; i++) out[i] += blk[i];
    }
    if (blk) tsr_free(blk, nb * 8 + 64);
    return TSR_OK;
}

/* outer edges (numpy's _get_outer_edges) */
static int outer_edges(const double *a, int64_t n, const tsr_arg *range, double *first, double *last)
{
    if (range && range->kind == 3) {
        int64_t k;
        double *r = fn_arg_doubles(range, &k);
        if (!r) return TSR_ENOMEM;
        if (k != 2) { fn_free_doubles(r, k); fn_set_error("range must be a pair (min, max)"); return TSR_EARG; }
        *first = r[0];
        *last = r[1];
        fn_free_doubles(r, k);
        if (*first > *last) { fn_set_error("max must be larger than min in range parameter."); return TSR_EARG; }
        if (!(isfinite(*first) && isfinite(*last))) { fn_set_error("supplied range of [%g, %g] is not finite", *first, *last); return TSR_EARG; }
    } else if (n == 0) {
        *first = 0;
        *last = 1;
    } else {
        double lo = a[0], hi = a[0];
        for (int64_t i = 1; i < n; i++) {
            if (a[i] != a[i]) { lo = hi = a[i]; break; }
            if (a[i] < lo) lo = a[i];
            if (a[i] > hi) hi = a[i];
        }
        *first = lo;
        *last = hi;
        if (!(isfinite(*first) && isfinite(*last))) { fn_set_error("autodetected range of [%g, %g] is not finite", *first, *last); return TSR_EARG; }
    }
    if (*first == *last) { *first -= 0.5; *last += 0.5; }
    return TSR_OK;
}

/* edges for the bins argument; *nb_uniform > 0 when the bins are uniform */
static int hist_edges(double *a, int64_t n, const tsr_arg *bins, const tsr_arg *range, int weighted, int int_input,
                      double **edges_out, int64_t *nedges, int64_t *nb_uniform, double *first_out, double *last_out)
{
    int rc;
    double first = 0, last = 1;
    int64_t nb = 0;
    *nb_uniform = 0;
    if (bins->kind == 2) {
        const int est = estimator_of(bins->str);
        if (est < 0) { fn_set_error("'%s' is not a valid estimator for `bins`", bins->str); return TSR_EARG; }
        if (weighted) { fn_set_error("Automated estimation of the number of bins is not supported for weighted data"); return TSR_ETYPE; }
        if ((rc = outer_edges(a, n, range, &first, &last)) != TSR_OK) return rc;
        int64_t m = n;
        if (range && range->kind == 3) {
            m = 0;
            for (int64_t i = 0; i < n; i++) if (a[i] >= first && a[i] <= last) a[m++] = a[i];
        }
        if (m == 0) nb = 1;
        else {
            double *tmp = (double *)tsr_alloc(m * 8 + 64);
            if (!tmp) return TSR_ENOMEM;
            int err;
            double width = estimator_width(est, a, m, first, last, tmp, &err);
            tsr_free(tmp, m * 8 + 64);
            if (err != TSR_OK) return err;
            if (width != 0 && width == width) {
                if (int_input && width < 1) width = 1;
                nb = (int64_t)ceil((last - first) / width);
            } else {
                nb = 1;
            }
        }
    } else if (bins->kind == 1 || bins->kind == 4) {
        if (bins->num != floor(bins->num) || !(bins->num < 9.2e18)) { fn_set_error("`bins` must be an integer, a string, or an array"); return TSR_ETYPE; }
        nb = (int64_t)bins->num;
        if (nb < 1) { fn_set_error("`bins` must be positive, when an integer"); return TSR_EARG; }
        if ((rc = outer_edges(a, n, range, &first, &last)) != TSR_OK) return rc;
    } else if (bins->kind == 3) {
        if (bins->arr.ndim != 1) { fn_set_error("`bins` must be 1d, when an array"); return TSR_EARG; }
        int64_t k;
        double *e = fn_arg_doubles(bins, &k);
        if (!e) return TSR_ENOMEM;
        for (int64_t i = 0; i + 1 < k; i++)
            if (e[i] > e[i + 1]) { fn_free_doubles(e, k); fn_set_error("`bins` must increase monotonically, when an array"); return TSR_EARG; }
        *edges_out = e;
        *nedges = k;
        return TSR_OK;
    } else {
        fn_set_error("`bins` must be an integer, a string, or an array");
        return TSR_ETYPE;
    }
    /* numpy.linspace(first, last, nb + 1) */
    const int64_t k = nb + 1;
    double *e = (double *)tsr_alloc(k > 8 ? k * 8 : 64);
    if (!e) return TSR_ENOMEM;
    const double delta = last - first, step = delta / (double)nb;
    for (int64_t i = 0; i < k; i++) e[i] = (step == 0 ? ((double)i / (double)nb) * delta : (double)i * step) + first;
    e[k - 1] = last;
    for (int64_t i = 0; i + 1 < k; i++)
        if (!(e[i] < e[i + 1])) { tsr_free(e, k > 8 ? k * 8 : 64); fn_set_error("Too many bins for data range. Cannot create %lld finite-sized bins.", (long long)nb); return TSR_EARG; }
    *edges_out = e;
    *nedges = k;
    *nb_uniform = nb;
    *first_out = first;
    *last_out = last;
    return TSR_OK;
}

typedef struct { int edges_only; } hctx;
static const hctx H_HIST = {0}, H_EDGES = {1};

static int r_histogram(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int edges_only = ((const hctx *)ctx)->edges_only;
    /* histogram(a, bins, range, density, weights) / histogram_bin_edges(a, bins, range, weights) */
    if (!require_array(args, nargs, 0, "a")) return TSR_EARG;
    tsr_arg dflt_bins;
    memset(&dflt_bins, 0, sizeof dflt_bins);
    dflt_bins.kind = 1;
    dflt_bins.num = 10.0;
    const tsr_arg *bins = nargs > 1 && args[1].kind != 0 ? &args[1] : &dflt_bins;
    const tsr_arg *range = nargs > 2 ? &args[2] : NULL;
    const int density = edges_only ? 0 : arg_bool(args, nargs, 3, 0);
    const int wi = edges_only ? 3 : 4;
    const tsr_arg *warg = nargs > wi && args[wi].kind == 3 ? &args[wi] : NULL;
    int64_t n, nw = 0;
    double *a = fn_arg_doubles(&args[0], &n);
    if (!a) return TSR_ENOMEM;
    double *w = NULL;
    if (warg) {
        w = fn_arg_doubles(warg, &nw);
        if (!w) { fn_free_doubles(a, n); return TSR_ENOMEM; }
        if (nw != n || warg->arr.ndim != args[0].arr.ndim) { fn_free_doubles(a, n); fn_free_doubles(w, nw); fn_set_error("weights should have the same shape as a."); return TSR_EARG; }
    }
    double *work = (double *)tsr_alloc(n > 0 ? n * 8 : 64);
    if (!work) { fn_free_doubles(a, n); fn_free_doubles(w, nw); return TSR_ENOMEM; }
    memcpy(work, a, (size_t)n * 8);
    double *edges = NULL, first = 0, last = 0;
    int64_t ne = 0, nb_uniform = 0;
    int rc = hist_edges(work, n, bins, range, w != NULL, is_intlike(args[0].arr.dtype), &edges, &ne, &nb_uniform, &first, &last);
    tsr_free(work, n > 0 ? n * 8 : 64);
    if (rc != TSR_OK) { fn_free_doubles(a, n); fn_free_doubles(w, nw); return rc; }
    if (edges_only) {
        rc = result_vec(&res[0], TSR_F64, edges, ne);
        fn_free_doubles(edges, ne);
        fn_free_doubles(a, n);
        fn_free_doubles(w, nw);
        return rc;
    }
    const int64_t nb = ne - 1;
    double *cnt = (double *)tsr_alloc(nb > 0 ? nb * 8 + 64 : 64);
    if (!cnt) rc = TSR_ENOMEM;
    if (rc == TSR_OK && nb_uniform > 0) {
        rc = hist_counts_uniform(a, n, w, first, last, nb, edges, cnt);
    } else if (rc == TSR_OK) {
        /* explicit edges: cumulative counts at each edge from sorted blocks (searchsorted left, last edge right) */
        const int64_t BLOCK = 65536;
        double *cum = (double *)tsr_alloc(ne * 8 + 64);
        double *pairs = (double *)tsr_alloc((n < BLOCK ? n : BLOCK) * 16 + 64);
        if (!cum || !pairs) rc = TSR_ENOMEM;
        else {
            for (int64_t e = 0; e < ne; e++) cum[e] = 0;
            for (int64_t b0 = 0; b0 < n; b0 += BLOCK) {
                const int64_t m = (b0 + BLOCK < n ? b0 + BLOCK : n) - b0;
                for (int64_t i = 0; i < m; i++) { pairs[2 * i] = a[b0 + i]; pairs[2 * i + 1] = w ? w[b0 + i] : 1.0; }
                np_argsort_pairs(pairs, m);
                /* cw = [0, cumsum(sorted weights)] */
                double *cw = (double *)tsr_alloc((m + 1) * 8 + 64);
                if (!cw) { rc = TSR_ENOMEM; break; }
                cw[0] = 0;
                for (int64_t i = 0; i < m; i++) cw[i + 1] = cw[i] + pairs[2 * i + 1];
                for (int64_t e = 0; e < ne; e++) {
                    const int right = e == ne - 1;
                    int64_t lo = 0, hi = m;
                    while (lo < hi) {
                        const int64_t mid = lo + (hi - lo) / 2;
                        const int go = right ? !lt_nan(edges[e], pairs[2 * mid]) : lt_nan(pairs[2 * mid], edges[e]);
                        if (go) lo = mid + 1;
                        else hi = mid;
                    }
                    cum[e] += w ? cw[lo] : (double)lo;
                }
                tsr_free(cw, (m + 1) * 8 + 64);
            }
            for (int64_t e = 0; e < nb; e++) cnt[e] = cum[e + 1] - cum[e];
        }
        if (cum) tsr_free(cum, ne * 8 + 64);
        if (pairs) tsr_free(pairs, (n < BLOCK ? n : BLOCK) * 16 + 64);
    }
    if (rc == TSR_OK) {
        if (density) {
            double *d = (double *)tsr_alloc(nb * 8 + 64);
            if (!d) rc = TSR_ENOMEM;
            else {
                const double total = tsr_psum(cnt, nb);
                for (int64_t i = 0; i < nb; i++) d[i] = cnt[i] / (edges[i + 1] - edges[i]) / total;
                rc = result_vec(&res[0], TSR_F64, d, nb);
                tsr_free(d, nb * 8 + 64);
            }
        } else {
            rc = result_vec(&res[0], w ? TSR_F64 : TSR_I64, cnt, nb);
        }
        if (rc == TSR_OK) rc = result_vec(&res[1], TSR_F64, edges, ne);
    }
    if (cnt) tsr_free(cnt, nb > 0 ? nb * 8 + 64 : 64);
    fn_free_doubles(edges, ne);
    fn_free_doubles(a, n);
    fn_free_doubles(w, nw);
    return rc;
}

/* numpy.percentile(x, q) with the linear method on a sorted, NaN-free sample (used by the fd estimator) */
double np_percentile_linear(const double *s, int64_t n, double q)
{
    const double qq = q / 100.0;
    const double v = ((double)n - 1) * qq;
    double prev = floor(v), next = prev + 1;
    if (v >= (double)n - 1) { prev = -1; next = -1; }
    if (v < 0) { prev = 0; next = 0; }
    const int64_t ip = (int64_t)prev, in = (int64_t)next;
    const double a = s[ip < 0 ? n + ip : ip], b = s[in < 0 ? n + in : in];
    const double gamma = v - (double)ip;
    const double diff = b - a;
    double r = a + diff * gamma;
    if (gamma >= 0.5) r = b - diff * (1 - gamma);
    return r;
}

/* ---------------------------------------------------------------- cov / corrcoef */

/* the observations matrix X (vars x obs) after numpy's rowvar handling, with y appended */
static int cov_matrix(const tsr_arg *args, int nargs, int rowvar, double **Xout, int64_t *vars, int64_t *obs)
{
    const tsr_arg *m = &args[0];
    const tsr_arg *y = nargs > 1 && args[1].kind == 3 ? &args[1] : NULL;
    if (m->arr.ndim > 2) { fn_set_error("m has more than 2 dimensions"); return TSR_EARG; }
    if (y && y->arr.ndim > 2) { fn_set_error("y has more than 2 dimensions"); return TSR_EARG; }
    int64_t nm, ny = 0;
    double *dm = fn_arg_doubles(m, &nm);
    if (!dm) return TSR_ENOMEM;
    double *dy = y ? fn_arg_doubles(y, &ny) : NULL;
    if (y && !dy) { fn_free_doubles(dm, nm); return TSR_ENOMEM; }
    /* shapes after array(m, ndmin=2) and the optional transpose */
    int64_t r1, c1, r2 = 0, c2 = 0;
    int t1 = 0, t2 = 0;
    if (m->arr.ndim <= 1) { r1 = 1; c1 = nm; }
    else { r1 = m->arr.shape[0]; c1 = m->arr.shape[1]; if (!rowvar) { t1 = 1; int64_t t = r1; r1 = c1; c1 = t; } }
    if (y) {
        if (y->arr.ndim <= 1) { r2 = 1; c2 = ny; }
        else { r2 = y->arr.shape[0]; c2 = y->arr.shape[1]; }
        if (!rowvar && r2 != 1) { t2 = 1; int64_t t = r2; r2 = c2; c2 = t; }
        if (c2 != c1) { fn_free_doubles(dm, nm); fn_free_doubles(dy, ny); fn_set_error("all the input array dimensions except for the concatenation axis must match exactly"); return TSR_ESHAPE; }
    }
    const int64_t V = r1 + r2, O = c1;
    double *X = (double *)tsr_alloc(V * O * 8 + 64);
    if (!X) { fn_free_doubles(dm, nm); fn_free_doubles(dy, ny); return TSR_ENOMEM; }
    for (int64_t i = 0; i < r1; i++)
        for (int64_t j = 0; j < c1; j++) X[i * O + j] = t1 ? dm[j * r1 + i] : dm[i * c1 + j];
    for (int64_t i = 0; i < r2; i++)
        for (int64_t j = 0; j < c2; j++) X[(r1 + i) * O + j] = t2 ? dy[j * r2 + i] : dy[i * c2 + j];
    fn_free_doubles(dm, nm);
    fn_free_doubles(dy, ny);
    *Xout = X;
    *vars = V;
    *obs = O;
    return TSR_OK;
}

typedef struct { int corr; } covctx;
static const covctx C_COV = {0}, C_CORR = {1};

static int r_cov(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int corr = ((const covctx *)ctx)->corr;
    if (!require_array(args, nargs, 0, "m")) return TSR_EARG;
    const int rowvar = arg_bool(args, nargs, 2, 1);
    const int bias = corr ? 0 : arg_bool(args, nargs, 3, 0);
    double ddof = corr ? FN_NAN : arg_num(args, nargs, 4, FN_NAN);
    if (ddof == ddof && ddof != floor(ddof)) { fn_set_error("ddof must be integer"); return TSR_EARG; }
    double *X;
    int64_t V, O;
    int rc = cov_matrix(args, nargs, rowvar, &X, &V, &O);
    if (rc != TSR_OK) return rc;
    if (V == 0) {
        int64_t sh[2] = {0, 0};
        tsr_free(X, V * O * 8 + 64);
        return fn_result_array(&res[0], TSR_F64, 2, sh) ? TSR_OK : TSR_ENOMEM;
    }
    if (ddof != ddof) ddof = bias == 0 ? 1 : 0;
    /* weights: fweights * aweights */
    double *w = NULL, *aw = NULL;
    int64_t nwf = 0, nwa = 0;
    if (!corr && nargs > 5 && args[5].kind == 3) {
        w = fn_arg_doubles(&args[5], &nwf);
        if (!w) { tsr_free(X, V * O * 8 + 64); return TSR_ENOMEM; }
        if (args[5].arr.ndim > 1) { rc = TSR_EARG; fn_set_error("cannot handle multidimensional fweights"); }
        else if (nwf != O) { rc = TSR_EARG; fn_set_error("incompatible numbers of samples and fweights"); }
        else for (int64_t i = 0; i < O && rc == TSR_OK; i++) {
            if (w[i] != nearbyint(w[i])) { rc = TSR_ETYPE; fn_set_error("fweights must be integer"); }
            else if (w[i] < 0) { rc = TSR_EARG; fn_set_error("fweights cannot be negative"); }
        }
    }
    if (rc == TSR_OK && !corr && nargs > 6 && args[6].kind == 3) {
        aw = fn_arg_doubles(&args[6], &nwa);
        if (!aw) rc = TSR_ENOMEM;
        else if (args[6].arr.ndim > 1) { rc = TSR_EARG; fn_set_error("cannot handle multidimensional aweights"); }
        else if (nwa != O) { rc = TSR_EARG; fn_set_error("incompatible numbers of samples and aweights"); }
        else for (int64_t i = 0; i < O && rc == TSR_OK; i++) if (aw[i] < 0) { rc = TSR_EARG; fn_set_error("aweights cannot be negative"); }
    }
    double *wt = NULL;
    if (rc == TSR_OK && (w || aw)) {
        wt = (double *)tsr_alloc(O * 8 + 64);
        if (!wt) rc = TSR_ENOMEM;
        else for (int64_t i = 0; i < O; i++) wt[i] = w ? (aw ? w[i] * aw[i] : w[i]) : aw[i];
    }
    double *C = NULL;
    if (rc == TSR_OK) {
        /* avg, w_sum = average(X, axis=1, weights=w, returned=True) */
        double *tmp = (double *)tsr_alloc(O * 8 + 64), *avg = (double *)tsr_alloc(V * 8 + 64);
        C = (double *)tsr_alloc(V * V * 8 + 64);
        if (!tmp || !avg || !C) rc = TSR_ENOMEM;
        double w_sum = (double)O, fact = 0;
        if (rc == TSR_OK) {
            if (wt) w_sum = tsr_psum(wt, O);
            if (wt && w_sum == 0.0) { rc = TSR_EARG; fn_set_error("Weights sum to zero, can't be normalized"); }
        }
        if (rc == TSR_OK) {
            for (int64_t i = 0; i < V; i++) {
                const double *row = X + i * O;
                if (wt) { for (int64_t j = 0; j < O; j++) tmp[j] = row[j] * wt[j]; avg[i] = tsr_psum(tmp, O) / w_sum; }
                else avg[i] = tsr_psum(row, O) / (double)O;
            }
            if (!wt) fact = (double)O - ddof;
            else if (ddof == 0) fact = w_sum;
            else if (!aw) fact = w_sum - ddof;
            else {
                double s = 0;
                for (int64_t j = 0; j < O; j++) s += wt[j] * aw[j];    /* Python sum(): left to right */
                fact = w_sum - ddof * s / w_sum;
            }
            if (fact <= 0) fact = 0.0;
            for (int64_t i = 0; i < V; i++) for (int64_t j = 0; j < O; j++) X[i * O + j] -= avg[i];
            const double inv = 1.0 / fact;
            for (int64_t i = 0; i < V; i++) {
                for (int64_t k = 0; k < V; k++) {
                    double s = 0;
                    for (int64_t j = 0; j < O; j++) s += X[i * O + j] * (wt ? X[k * O + j] * wt[j] : X[k * O + j]);
                    C[i * V + k] = s * inv;
                }
            }
            if (corr) {
                double *sd = tmp;                             /* O >= V is not guaranteed: reuse avg */
                (void)sd;
                for (int64_t i = 0; i < V; i++) avg[i] = sqrt(C[i * V + i]);
                for (int64_t i = 0; i < V; i++) for (int64_t k = 0; k < V; k++) {
                    double c = C[i * V + k] / avg[i];
                    c = c / avg[k];
                    if (c > 1) c = 1;
                    if (c < -1) c = -1;
                    C[i * V + k] = c;
                }
            }
        }
        if (tmp) tsr_free(tmp, O * 8 + 64);
        if (avg) tsr_free(avg, V * 8 + 64);
    }
    if (rc == TSR_OK) {
        /* c.squeeze(): a single variable gives a 0-d result */
        if (V == 1) fn_result_num(&res[0], C[0]);
        else {
            int64_t sh[2] = {V, V};
            double *p = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
            if (!p) rc = TSR_ENOMEM;
            else memcpy(p, C, (size_t)(V * V) * 8);
        }
    }
    if (C) tsr_free(C, V * V * 8 + 64);
    if (wt) tsr_free(wt, O * 8 + 64);
    fn_free_doubles(w, nwf);
    fn_free_doubles(aw, nwa);
    tsr_free(X, V * O * 8 + 64);
    return rc;
}

/* ---------------------------------------------------------------- correlate / convolve */

typedef struct { int convolve; } corrctx;
static const corrctx K_CORR = {0}, K_CONV = {1};

static int is_int_kind(int dtype) { return dtype == TSR_I64 || dtype == TSR_I32 || dtype == TSR_U8; }

/* a 1-D integer argument as int64 values (exact, unlike fn_arg_doubles above 2^53) */
static int64_t *arg_ints_1d(const tsr_array *a, int64_t *n)
{
    *n = a->shape[0];
    int64_t *v = (int64_t *)tsr_alloc(*n * 8 + 64);
    if (!v) return NULL;
    const char *base = (const char *)a->data + a->offset;
    for (int64_t i = 0; i < *n; i++) {
        const char *p = base + i * a->strides[0];
        v[i] = a->dtype == TSR_I64 ? *(const int64_t *)p : a->dtype == TSR_I32 ? (int64_t)*(const int32_t *)p : (int64_t)*(const uint8_t *)p;
    }
    return v;
}

/* correlate / convolve of two integer sequences: NumPy computes in the promoted integer type and wraps around
   on overflow (int64 for any int64 operand, int32 for int32, uint8 for two uint8 arrays) */
static int correlate_int(int conv, int mo, const tsr_array *A, const tsr_array *V, tsr_result *res)
{
    const int dt = (A->dtype == TSR_I64 || V->dtype == TSR_I64) ? TSR_I64
                 : (A->dtype == TSR_I32 || V->dtype == TSR_I32) ? TSR_I32 : TSR_U8;
    int64_t na, nv;
    int64_t *a = arg_ints_1d(A, &na), *v = arg_ints_1d(V, &nv);
    if (!a || !v) { if (a) tsr_free(a, na * 8 + 64); if (v) tsr_free(v, nv * 8 + 64); return TSR_ENOMEM; }
    int rc = TSR_OK;
    int64_t *yr = NULL, *out = NULL, len = 0;
    if (na == 0 || nv == 0) { fn_set_error(conv ? "v cannot be empty" : "zero-size array"); rc = TSR_EARG; goto done; }
    {
        const int64_t *x = a, *y = v;
        int64_t n1 = na, n2 = nv;
        if (conv) {
            if (nv > na) { x = v; y = a; n1 = nv; n2 = na; }
            yr = (int64_t *)tsr_alloc(n2 * 8 + 64);
            if (!yr) { rc = TSR_ENOMEM; goto done; }
            for (int64_t i = 0; i < n2; i++) yr[i] = y[n2 - 1 - i];
            y = yr;
        }
        int inverse = 0;
        if (n1 < n2) { const int64_t *t = x; x = y; y = t; int64_t tn = n1; n1 = n2; n2 = tn; inverse = 1; }
        int64_t left, right;
        if (mo == 0) { left = 0; right = 0; len = n1 - n2 + 1; }
        else if (mo == 1) { left = n2 / 2; right = n2 - left - 1; len = n1; }
        else { left = n2 - 1; right = n2 - 1; len = n1 + n2 - 1; }
        out = (int64_t *)tsr_alloc(len * 8 + 64);
        if (!out) { rc = TSR_ENOMEM; goto done; }
        int64_t k = 0;
        for (int64_t i = left; i > 0; i--) {
            uint64_t acc = 0;
            for (int64_t j = 0; j < n2 - i; j++) acc += (uint64_t)x[j] * (uint64_t)y[i + j];
            out[k++] = (int64_t)acc;
        }
        for (int64_t i = 0; i < n1 - n2 + 1; i++) {
            uint64_t acc = 0;
            for (int64_t j = 0; j < n2; j++) acc += (uint64_t)x[i + j] * (uint64_t)y[j];
            out[k++] = (int64_t)acc;
        }
        for (int64_t i = 1; i <= right; i++) {
            const int64_t m = n2 - i;
            uint64_t acc = 0;
            for (int64_t j = 0; j < m; j++) acc += (uint64_t)x[n1 - m + j] * (uint64_t)y[j];
            out[k++] = (int64_t)acc;
        }
        if (inverse) for (int64_t i = 0; i < len / 2; i++) { int64_t t = out[i]; out[i] = out[len - 1 - i]; out[len - 1 - i] = t; }
        void *p = fn_result_array(&res[0], dt, 1, &len);
        if (!p) { rc = TSR_ENOMEM; goto done; }
        for (int64_t i = 0; i < len; i++) {
            if (dt == TSR_I64) ((int64_t *)p)[i] = out[i];
            else if (dt == TSR_I32) ((int32_t *)p)[i] = (int32_t)(uint32_t)(uint64_t)out[i];   /* wraps modulo 2^32 */
            else ((uint8_t *)p)[i] = (uint8_t)(uint64_t)out[i];
        }
    }
done:
    if (out) tsr_free(out, len * 8 + 64);
    if (yr) tsr_free(yr, (conv ? (nv > na ? na : nv) : 0) * 8 + 64);
    tsr_free(a, na * 8 + 64);
    tsr_free(v, nv * 8 + 64);
    return rc;
}

static int r_correlate(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int conv = ((const corrctx *)ctx)->convolve;
    if (!require_array(args, nargs, 0, "a") || !require_array(args, nargs, 1, "v")) return TSR_EARG;
    const char *mode = nargs > 2 && args[2].kind == 2 ? args[2].str : (conv ? "full" : "valid");
    int mo;
    if (strcmp(mode, "valid") == 0 || strcmp(mode, "v") == 0) mo = 0;
    else if (strcmp(mode, "same") == 0 || strcmp(mode, "s") == 0) mo = 1;
    else if (strcmp(mode, "full") == 0 || strcmp(mode, "f") == 0) mo = 2;
    else { fn_set_error("mode must be one of 'valid', 'same', or 'full' (got '%s')", mode); return TSR_EARG; }
    if (is_int_kind(args[0].arr.dtype) && is_int_kind(args[1].arr.dtype) && args[0].arr.ndim == 1 && args[1].arr.ndim == 1)
        return correlate_int(conv, mo, &args[0].arr, &args[1].arr, res);
    int64_t na, nv;
    double *a = fn_arg_doubles(&args[0], &na), *v = fn_arg_doubles(&args[1], &nv);
    if (!a || !v) { fn_free_doubles(a, na); fn_free_doubles(v, nv); return TSR_ENOMEM; }
    if (na == 0 || nv == 0) { fn_free_doubles(a, na); fn_free_doubles(v, nv); fn_set_error(conv ? "v cannot be empty" : "zero-size array"); return TSR_EARG; }
    /* convolve(a, v) = correlate(a, v[::-1]); numpy's convolve swaps so the longer array comes first */
    const double *x = a, *y = v;
    int64_t n1 = na, n2 = nv;
    double *yr = NULL;
    if (conv) {
        if (nv > na) { x = v; y = a; n1 = nv; n2 = na; }
        yr = (double *)tsr_alloc(n2 * 8 + 64);
        if (!yr) { fn_free_doubles(a, na); fn_free_doubles(v, nv); return TSR_ENOMEM; }
        for (int64_t i = 0; i < n2; i++) yr[i] = y[n2 - 1 - i];
        y = yr;
    }
    /* numpy's _pyarray_correlate: if n1 < n2 swap and reverse the output */
    int inverse = 0;
    if (n1 < n2) { const double *t = x; x = y; y = t; int64_t tn = n1; n1 = n2; n2 = tn; inverse = 1; }
    int64_t left, right, len;
    if (mo == 0) { left = 0; right = 0; len = n1 - n2 + 1; }
    else if (mo == 1) { left = n2 / 2; right = n2 - left - 1; len = n1; }
    else { left = n2 - 1; right = n2 - 1; len = n1 + n2 - 1; }
    double *out = (double *)tsr_alloc(len * 8 + 64);
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        int64_t k = 0;
        /* left part: partial overlaps */
        for (int64_t i = left; i > 0; i--) {
            const int64_t m = n2 - i;
            double s = 0;
            for (int64_t j = 0; j < m; j++) s += x[j] * y[i + j];
            out[k++] = s;
        }
        for (int64_t i = 0; i < n1 - n2 + 1; i++) {
            double s = 0;
            for (int64_t j = 0; j < n2; j++) s += x[i + j] * y[j];
            out[k++] = s;
        }
        for (int64_t i = 1; i <= right; i++) {
            const int64_t m = n2 - i;
            double s = 0;
            for (int64_t j = 0; j < m; j++) s += x[n1 - m + j] * y[j];
            out[k++] = s;
        }
        if (inverse) for (int64_t i = 0; i < len / 2; i++) { double t = out[i]; out[i] = out[len - 1 - i]; out[len - 1 - i] = t; }
        rc = result_vec(&res[0], TSR_F64, out, len);
        tsr_free(out, len * 8 + 64);
    }
    if (yr) tsr_free(yr, n2 * 8 + 64);
    fn_free_doubles(a, na);
    fn_free_doubles(v, nv);
    return rc;
}

/* ---------------------------------------------------------------- lexsort */

static int r_lexsort(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    if (!require_array(args, nargs, 0, "keys")) return TSR_EARG;
    const tsr_array *k = &args[0].arr;
    int64_t nk, N;
    if (k->ndim == 1) { nk = 1; N = k->shape[0]; }
    else if (k->ndim == 2) { nk = k->shape[0]; N = k->shape[1]; }
    else { fn_set_error("lexsort supports 1-D keys or a 2-D stack of keys"); return TSR_EARG; }
    int64_t total;
    double *d = fn_arg_doubles(&args[0], &total);
    if (!d) return TSR_ENOMEM;
    double *pairs = (double *)tsr_alloc(N * 16 + 64);
    int64_t *idx = (int64_t *)tsr_alloc(N * 8 + 64), *tmpi = (int64_t *)tsr_alloc(N * 8 + 64);
    int rc = (pairs && idx && tmpi) ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i < N; i++) idx[i] = i;
        /* stable sorts from the first key to the last (the last key is the primary one) */
        for (int64_t key = 0; key < nk; key++) {
            const double *kv = d + key * N;
            for (int64_t i = 0; i < N; i++) { pairs[2 * i] = kv[idx[i]]; pairs[2 * i + 1] = (double)i; }
            np_argsort_pairs(pairs, N);
            for (int64_t i = 0; i < N; i++) tmpi[i] = idx[(int64_t)pairs[2 * i + 1]];
            memcpy(idx, tmpi, (size_t)N * 8);
        }
        rc = result_vec_i(&res[0], idx, N);
    }
    if (pairs) tsr_free(pairs, N * 16 + 64);
    if (idx) tsr_free(idx, N * 8 + 64);
    if (tmpi) tsr_free(tmpi, N * 8 + 64);
    fn_free_doubles(d, total);
    return rc;
}

/* ---------------------------------------------------------------- registry */

const fn_def NP_SETS[] = {
    ROUTINE("np.unique", 4, "ar, return_index=False, return_inverse=False, return_counts=False, equal_nan=True",
            "values, indices, inverse, counts", r_unique, &U_UNIQUE, "Sorted unique elements, optionally with indices, inverse and counts (numpy.unique)."),
    ROUTINE("np.unique_values", 1, "x", "values", r_unique, &U_VALUES, "Unique values; NaNs are not merged (numpy.unique_values)."),
    ROUTINE("np.unique_counts", 2, "x", "values, counts", r_unique, &U_COUNTS, "Unique values and their counts (numpy.unique_counts)."),
    ROUTINE("np.unique_inverse", 2, "x", "values, inverse_indices", r_unique, &U_INVERSE, "Unique values and the inverse indices (numpy.unique_inverse)."),
    ROUTINE("np.unique_all", 4, "x", "values, indices, inverse_indices, counts", r_unique, &U_ALL, "Unique values, first indices, inverse indices and counts (numpy.unique_all)."),
    ROUTINE("np.intersect1d", 3, "ar1, ar2, assume_unique=False, return_indices=False", "intersect1d, comm1, comm2", r_intersect1d, NULL,
            "Sorted unique values in both arrays (numpy.intersect1d)."),
    ROUTINE("np.union1d", 1, "ar1, ar2", "union1d", r_union1d, NULL, "Sorted unique values in either array (numpy.union1d)."),
    ROUTINE("np.setdiff1d", 1, "ar1, ar2, assume_unique=False", "setdiff1d", r_setdiff1d, NULL, "Unique values of ar1 that are not in ar2 (numpy.setdiff1d)."),
    ROUTINE("np.setxor1d", 1, "ar1, ar2, assume_unique=False", "setxor1d", r_setxor1d, NULL, "Sorted values in exactly one of the arrays (numpy.setxor1d)."),
    ROUTINE("np.bincount", 1, "x, weights=None, minlength=0", "counts", r_bincount, NULL, "Count (or sum weights of) each non-negative integer (numpy.bincount)."),
    ROUTINE("np.histogram", 2, "a, bins=10, range=None, density=False, weights=None", "hist, bin_edges", r_histogram, &H_HIST,
            "Histogram of the data: counts (or density) and bin edges (numpy.histogram)."),
    ROUTINE("np.histogram_bin_edges", 1, "a, bins=10, range=None, weights=None", "bin_edges", r_histogram, &H_EDGES,
            "Bin edges numpy.histogram would use (numpy.histogram_bin_edges)."),
    ROUTINE("np.cov", 1, "m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None", "cov", r_cov, &C_COV,
            "Covariance matrix (numpy.cov)."),
    ROUTINE("np.corrcoef", 1, "x, y=None, rowvar=True", "corrcoef", r_cov, &C_CORR, "Pearson correlation coefficients (numpy.corrcoef)."),
    ROUTINE("np.correlate", 1, "a, v, mode='valid'", "out", r_correlate, &K_CORR, "Cross-correlation of two 1-D sequences (numpy.correlate)."),
    ROUTINE("np.convolve", 1, "a, v, mode='full'", "out", r_correlate, &K_CONV, "Discrete linear convolution of two 1-D sequences (numpy.convolve)."),
    ROUTINE("np.lexsort", 1, "keys", "indices", r_lexsort, NULL, "Indirect stable sort on several keys, the last key primary (numpy.lexsort)."),
};
const fn_table TSR_NP_SETS_TABLE = {NP_SETS, (int)(sizeof NP_SETS / sizeof NP_SETS[0])};
