/* scipy.ndimage: generate_binary_structure, label (2-D), convolve (2-D). Real float64 input.
 *
 *   generate_binary_structure(rank, connectivity)  boolean structuring element, True where sum|offset| <= conn
 *   label(input, structure=None)                   connected components, 1-based in row-major order -> (labels, num)
 *   convolve(input, weights, mode='reflect', cval=0.0)   2-D convolution (kernel flipped) with boundary handling
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- generate_binary_structure(rank, connectivity) ---- */
static int r_gbs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t rank = 2, conn = 1;
    if (nargs > 0 && args[0].kind == 1) rank = args[0].ival ? args[0].ival : (int64_t)args[0].num;
    if (nargs > 1 && args[1].kind == 1) conn = args[1].ival ? args[1].ival : (int64_t)args[1].num;
    if (rank < 1 || rank > 6) { fn_set_error("generate_binary_structure: rank must be 1..6"); return TSR_EARG; }
    int64_t sh[6]; int64_t total = 1;
    for (int d = 0; d < rank; d++) { sh[d] = 3; total *= 3; }
    uint8_t *out = (uint8_t *)fn_result_array(&res[0], TSR_BOOL, (int32_t)rank, sh);
    if (!out) return TSR_ENOMEM;
    for (int64_t k = 0; k < total; k++) {
        int64_t t = k, s = 0;
        for (int d = (int)rank - 1; d >= 0; d--) { int64_t off = (t % 3) - 1; if (off < 0) off = -off; s += off; t /= 3; }
        out[k] = s <= conn ? 1 : 0;
    }
    return TSR_OK;
}

/* ---- label(input, structure=None): 2-D connected components ---- */
static int r_label(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("label: input must be a 2-D array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1];
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti);
    if (!in) return TSR_ENOMEM;
    /* neighbour offsets from the structure (default: 4-connectivity) */
    int nbr[8][2]; int nn = 0;
    if (nargs > 1 && args[1].kind == 3 && args[1].arr.ndim == 2 && args[1].arr.shape[0] == 3 && args[1].arr.shape[1] == 3) {
        int64_t ts; double *st = fn_arg_doubles(&args[1], &ts);
        if (!st) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
        for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++)
            if (!(di == 0 && dj == 0) && st[(di + 1) * 3 + (dj + 1)] != 0.0) { nbr[nn][0] = di; nbr[nn][1] = dj; nn++; }
        fn_free_doubles(st, ts);
    } else {
        const int def[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int k = 0; k < 4; k++) { nbr[k][0] = def[k][0]; nbr[k][1] = def[k][1]; nn = 4; }
    }
    int32_t *lab = (int32_t *)fn_result_array(&res[0], TSR_I32, 2, args[0].arr.shape);
    if (!lab) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    memset(lab, 0, sizeof(int32_t) * (size_t)(m * n));
    int64_t *stack = (int64_t *)malloc(sizeof(int64_t) * (size_t)(m * n > 0 ? m * n : 1));
    if (!stack) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    int32_t num = 0;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            const int64_t p = i * n + j;
            if (in[p] == 0.0 || lab[p] != 0) continue;
            num++;
            int64_t sp = 0; stack[sp++] = p; lab[p] = num;
            while (sp > 0) {
                const int64_t q = stack[--sp], qr = q / n, qc = q % n;
                for (int k = 0; k < nn; k++) {
                    const int64_t ni = qr + nbr[k][0], nj = qc + nbr[k][1];
                    if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
                    const int64_t np = ni * n + nj;
                    if (in[np] != 0.0 && lab[np] == 0) { lab[np] = num; stack[sp++] = np; }
                }
            }
        }
    free(stack);
    fn_free_doubles(in, ti);
    fn_result_int(&res[1], num);
    return TSR_OK;
}

/* ---- convolve(input, weights, mode='reflect', cval=0.0): 2-D ---- */
enum { ND_REFLECT, ND_CONSTANT, ND_NEAREST, ND_MIRROR, ND_WRAP };

static int nd_mode(const tsr_arg *a, int *mode)
{
    if (!a || a->kind == 0) { *mode = ND_REFLECT; return TSR_OK; }
    if (a->kind != 2 || !a->str) { fn_set_error("convolve: mode must be a string"); return TSR_EARG; }
    const struct { const char *n; int m; } T[] = {
        {"reflect", ND_REFLECT}, {"constant", ND_CONSTANT}, {"nearest", ND_NEAREST}, {"mirror", ND_MIRROR}, {"wrap", ND_WRAP},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) if (strcmp(a->str, T[i].n) == 0) { *mode = T[i].m; return TSR_OK; }
    fn_set_error("convolve: mode '%s' is not supported", a->str);
    return TSR_EARG;
}

static int64_t bmap(int64_t p, int64_t n, int mode)
{
    if (p >= 0 && p < n) return p;
    if (n <= 0) return -1;
    switch (mode) {
    case ND_NEAREST: return p < 0 ? 0 : n - 1;
    case ND_WRAP: { p %= n; if (p < 0) p += n; return p; }
    case ND_MIRROR: if (n == 1) return 0; while (p < 0 || p >= n) { if (p < 0) p = -p; else p = 2 * n - 2 - p; } return p;
    case ND_CONSTANT: return -1;
    default: while (p < 0 || p >= n) { if (p < 0) p = -p - 1; else p = 2 * n - 1 - p; } return p;   /* reflect */
    }
}

static int r_nd_convolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("convolve: input and weights must be 2-D arrays"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1];
    const int64_t kh = args[1].arr.shape[0], kw = args[1].arr.shape[1];
    int mode, rc; if ((rc = nd_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    double cval = 0.0;
    if (nargs > 3 && args[3].kind == 1) cval = (args[3].flags & 1) ? (double)args[3].ival : args[3].num;
    int64_t ti, tw; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *w = fn_arg_doubles(&args[1], &tw); if (!w) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, args[0].arr.shape);
    if (!out) { fn_free_doubles(in, ti); fn_free_doubles(w, tw); return TSR_ENOMEM; }
    const int64_t ch = kh / 2, cw = kw / 2;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t a = 0; a < kh; a++)
                for (int64_t b = 0; b < kw; b++) {
                    const double wv = w[a * kw + b];
                    if (wv == 0.0) continue;
                    const int64_t ii = bmap(i - a + ch, m, mode), jj = bmap(j - b + cw, n, mode);
                    acc += wv * ((ii < 0 || jj < 0) ? cval : in[ii * n + jj]);
                }
            out[i * n + j] = acc;
        }
    fn_free_doubles(in, ti); fn_free_doubles(w, tw);
    return TSR_OK;
}

/* structuring-element offsets (including the centre), from a 3x3 structure or the default 4-connected cross */
static void se_offsets(const tsr_arg *s, int nbr[9][2], int *nn)
{
    if (s && s->kind == 3 && s->arr.ndim == 2 && s->arr.shape[0] == 3 && s->arr.shape[1] == 3) {
        int64_t ts; double *st = fn_arg_doubles(s, &ts); int c = 0;
        if (st) {
            for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++)
                if (st[(di + 1) * 3 + (dj + 1)] != 0.0) { nbr[c][0] = di; nbr[c][1] = dj; c++; }
            fn_free_doubles(st, ts);
        }
        *nn = c; return;
    }
    const int def[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int k = 0; k < 5; k++) { nbr[k][0] = def[k][0]; nbr[k][1] = def[k][1]; }
    *nn = 5;
}

/* binary_dilation / binary_erosion (2-D). dilate=1 -> OR over the SE; dilate=0 -> AND, out-of-range treated as 0. */
static int r_binary_morph(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int dilate = ctx && *(const int *)ctx;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("binary morphology: input must be a 2-D array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1];
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti);
    if (!in) return TSR_ENOMEM;
    int nbr[9][2]; int nn; se_offsets(nargs > 1 ? &args[1] : NULL, nbr, &nn);
    uint8_t *out = (uint8_t *)fn_result_array(&res[0], TSR_BOOL, 2, args[0].arr.shape);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            int val = dilate ? 0 : 1;
            for (int k = 0; k < nn; k++) {
                const int64_t ii = i + nbr[k][0], jj = j + nbr[k][1];
                const int inrange = ii >= 0 && ii < m && jj >= 0 && jj < n;
                const int nz = inrange && in[ii * n + jj] != 0.0;
                if (dilate) { if (nz) { val = 1; break; } }
                else { if (!nz) { val = 0; break; } }
            }
            out[i * n + j] = (uint8_t)val;
        }
    fn_free_doubles(in, ti);
    return TSR_OK;
}

static int64_t arg_int(const tsr_arg *a, int64_t dflt);   /* defined with the order-statistic filters below */

/* a 3x3 structuring element as a row-major 0/1 mask; default = the 4-connected cross (generate_binary_structure(2,1)). */
static void se_mask3(const tsr_arg *s, uint8_t mask[9])
{
    if (s && s->kind == 3 && s->arr.ndim == 2 && s->arr.shape[0] == 3 && s->arr.shape[1] == 3) {
        int64_t ts; double *st = fn_arg_doubles(s, &ts);
        if (st) { for (int k = 0; k < 9; k++) mask[k] = st[k] != 0.0; fn_free_doubles(st, ts); return; }
    }
    const uint8_t cross[9] = {0, 1, 0, 1, 1, 1, 0, 1, 0};
    for (int k = 0; k < 9; k++) mask[k] = cross[k];
}

/* one binary erosion (dilate=0, AND) / dilation (dilate=1, OR) pass with a centred 3x3 mask; out-of-range = border. */
static void bin_pass3(const uint8_t *in, uint8_t *out, int64_t m, int64_t n, const uint8_t mask[9], int dilate, int border)
{
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            int val = dilate ? 0 : 1;
            for (int di = -1; di <= 1 && (dilate ? !val : val); di++)
                for (int dj = -1; dj <= 1 && (dilate ? !val : val); dj++) {
                    if (!mask[(di + 1) * 3 + (dj + 1)]) continue;
                    const int64_t ii = i + di, jj = j + dj;
                    const int inrange = ii >= 0 && ii < m && jj >= 0 && jj < n;
                    const int nz = inrange ? (in[ii * n + jj] != 0) : border;
                    if (dilate) { if (nz) val = 1; } else { if (!nz) val = 0; }
                }
            out[i * n + j] = (uint8_t)val;
        }
}

static const int BIN_OPEN = 1;

/* binary_opening (ctx &BIN_OPEN = erosion then dilation) / binary_closing (dilation then erosion), 2-D. */
static int r_binary_openclose(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int opening = ctx && *(const int *)ctx == BIN_OPEN;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("binary opening/closing: input must be a 2-D array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], total = m * n;
    uint8_t mask[9]; se_mask3(nargs > 1 ? &args[1] : NULL, mask);
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    uint8_t *a = (uint8_t *)malloc((size_t)(total > 0 ? total : 1));
    uint8_t *b = a ? (uint8_t *)malloc((size_t)(total > 0 ? total : 1)) : NULL;
    uint8_t *out = b ? (uint8_t *)fn_result_array(&res[0], TSR_BOOL, 2, args[0].arr.shape) : NULL;
    if (!out) { fn_free_doubles(in, ti); free(a); free(b); return TSR_ENOMEM; }
    for (int64_t i = 0; i < total; i++) a[i] = in[i] != 0.0;
    /* opening = dilation(erosion(input)); closing = erosion(dilation(input)); each pass border_value 0 (scipy default) */
    bin_pass3(a, b, m, n, mask, opening ? 0 : 1, 0);
    bin_pass3(b, out, m, n, mask, opening ? 1 : 0, 0);
    fn_free_doubles(in, ti); free(a); free(b);
    return TSR_OK;
}

/* binary_hit_or_miss(input, structure1=None, structure2=None): erosion(input, s1) AND erosion(~input, s2),
   with s2 defaulting to the complement of s1. 2-D, 3x3 structures. */
static int r_binary_hit_or_miss(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("binary_hit_or_miss: input must be a 2-D array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], total = m * n;
    uint8_t s1[9], s2[9];
    se_mask3(nargs > 1 ? &args[1] : NULL, s1);
    if (nargs > 2 && args[2].kind == 3) se_mask3(&args[2], s2);
    else for (int k = 0; k < 9; k++) s2[k] = !s1[k];          /* default structure2 = logical_not(structure1) */
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    uint8_t *img = (uint8_t *)malloc((size_t)(total > 0 ? total : 1));
    uint8_t *comp = img ? (uint8_t *)malloc((size_t)(total > 0 ? total : 1)) : NULL;
    uint8_t *e1 = comp ? (uint8_t *)malloc((size_t)(total > 0 ? total : 1)) : NULL;
    uint8_t *out = e1 ? (uint8_t *)fn_result_array(&res[0], TSR_BOOL, 2, args[0].arr.shape) : NULL;
    if (!out) { fn_free_doubles(in, ti); free(img); free(comp); free(e1); return TSR_ENOMEM; }
    for (int64_t i = 0; i < total; i++) { img[i] = in[i] != 0.0; comp[i] = !img[i]; }
    bin_pass3(img, e1, m, n, s1, 0, 0);
    bin_pass3(comp, out, m, n, s2, 0, 0);
    for (int64_t i = 0; i < total; i++) out[i] = (uint8_t)(e1[i] && out[i]);
    fn_free_doubles(in, ti); free(img); free(comp); free(e1);
    return TSR_OK;
}

/* iterate_structure(structure, iterations): the structure dilated by itself iterations-1 times. 2-D, 3x3 input. */
static int r_iterate_structure(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != 3 || args[0].arr.shape[1] != 3) {
        fn_set_error("iterate_structure: a 3x3 structure is required"); return TSR_EARG;
    }
    const int64_t iterations = arg_int(nargs > 1 ? &args[1] : NULL, 1);
    uint8_t s[9]; se_mask3(&args[0], s);
    if (iterations < 2) {
        const int64_t sh[2] = {3, 3};
        uint8_t *out = (uint8_t *)fn_result_array(&res[0], TSR_BOOL, 2, sh);
        if (!out) return TSR_ENOMEM;
        for (int k = 0; k < 9; k++) out[k] = s[k];
        return TSR_OK;
    }
    const int64_t ni = iterations - 1, M = 3 + 2 * ni, N = 3 + 2 * ni, total = M * N;
    const int64_t sh[2] = {M, N};
    uint8_t *out = (uint8_t *)fn_result_array(&res[0], TSR_BOOL, 2, sh);
    uint8_t *a = out ? (uint8_t *)calloc((size_t)total, 1) : NULL;
    uint8_t *b = a ? (uint8_t *)malloc((size_t)total) : NULL;
    if (!out || !a || !b) { free(a); free(b); return TSR_ENOMEM; }
    for (int di = 0; di < 3; di++) for (int dj = 0; dj < 3; dj++) a[(ni + di) * N + (ni + dj)] = s[di * 3 + dj];
    for (int64_t t = 0; t < ni; t++) { bin_pass3(a, b, M, N, s, 1, 0); uint8_t *tmp = a; a = b; b = tmp; }
    for (int64_t i = 0; i < total; i++) out[i] = a[i];
    free(a); free(b);
    return TSR_OK;
}

/* reconstruction by dilation: grow `seed` by the 3x3 SE, keeping the dilated value only where `mask` is set (the
   original seed value elsewhere), until stable. `border` is the dilation's out-of-range value. */
static int bin_reconstruct(const uint8_t *seed, const uint8_t *mask, uint8_t *out, int64_t m, int64_t n, const uint8_t smask[9], int border)
{
    const int64_t total = m * n;
    uint8_t *dil = (uint8_t *)malloc((size_t)(total > 0 ? total : 1));
    if (!dil) return TSR_ENOMEM;
    for (int64_t i = 0; i < total; i++) out[i] = seed[i];
    for (;;) {
        bin_pass3(out, dil, m, n, smask, 1, border);
        int changed = 0;
        for (int64_t i = 0; i < total; i++) {
            const uint8_t v = mask[i] ? dil[i] : seed[i];
            if (v != out[i]) { changed = 1; out[i] = v; }
        }
        if (!changed) break;
    }
    free(dil);
    return TSR_OK;
}

/* binary_propagation(input, structure=None, mask=None): dilate input within mask until stable (reconstruction). */
static int r_binary_propagation(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("binary_propagation: input must be a 2-D array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], total = m * n;
    uint8_t smask[9]; se_mask3(nargs > 1 ? &args[1] : NULL, smask);
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    uint8_t *seed = (uint8_t *)malloc((size_t)(total > 0 ? total : 1));
    uint8_t *mask = seed ? (uint8_t *)malloc((size_t)(total > 0 ? total : 1)) : NULL;
    uint8_t *out = mask ? (uint8_t *)fn_result_array(&res[0], TSR_BOOL, 2, args[0].arr.shape) : NULL;
    if (!out) { fn_free_doubles(in, ti); free(seed); free(mask); return TSR_ENOMEM; }
    for (int64_t i = 0; i < total; i++) seed[i] = in[i] != 0.0;
    if (nargs > 2 && args[2].kind == 3) {
        int64_t tm; double *mk = fn_arg_doubles(&args[2], &tm);
        if (!mk) { fn_free_doubles(in, ti); free(seed); free(mask); return TSR_ENOMEM; }
        for (int64_t i = 0; i < total; i++) mask[i] = mk[i] != 0.0;
        fn_free_doubles(mk, tm);
    } else for (int64_t i = 0; i < total; i++) mask[i] = 1;     /* no mask: unconstrained */
    int rc = bin_reconstruct(seed, mask, out, m, n, smask, 0);
    fn_free_doubles(in, ti); free(seed); free(mask);
    return rc;
}

/* binary_fill_holes(input, structure=None): fill holes = complement of the background reachable from the border. */
static int r_binary_fill_holes(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("binary_fill_holes: input must be a 2-D array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], total = m * n;
    uint8_t smask[9]; se_mask3(nargs > 1 ? &args[1] : NULL, smask);
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    uint8_t *seed = (uint8_t *)calloc((size_t)(total > 0 ? total : 1), 1);        /* zero seed */
    uint8_t *mask = seed ? (uint8_t *)malloc((size_t)(total > 0 ? total : 1)) : NULL;
    uint8_t *out = mask ? (uint8_t *)fn_result_array(&res[0], TSR_BOOL, 2, args[0].arr.shape) : NULL;
    if (!out) { fn_free_doubles(in, ti); free(seed); free(mask); return TSR_ENOMEM; }
    for (int64_t i = 0; i < total; i++) mask[i] = (in[i] == 0.0);                 /* mask = complement of input */
    int rc = bin_reconstruct(seed, mask, out, m, n, smask, 1);                    /* border 1: the outside leaks in */
    if (rc == 0) for (int64_t i = 0; i < total; i++) out[i] = (uint8_t)!out[i];   /* holes = the background not reached */
    fn_free_doubles(in, ti); free(seed); free(mask);
    return rc;
}

/* ------------------------------------------- n-D windowed order-statistic filters (box footprint) ----- */
static const int OP_MIN = 0, OP_MAX = 1, OP_MEDIAN = 2;

static int64_t arg_int(const tsr_arg *a, int64_t dflt)
{
    if (!a || a->kind != 1) return dflt;
    return (a->flags & 1) ? a->ival : (int64_t)a->num;
}

static int cmp_double(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* a scalar (all axes) or an nd-length sequence -> one int per axis. */
static int perax_i(const tsr_arg *a, int32_t nd, int64_t dflt, int64_t *out)
{
    if (!a || a->kind == 0) { for (int d = 0; d < nd; d++) out[d] = dflt; return TSR_OK; }
    if (a->kind == 1) { const int64_t v = (a->flags & 1) ? a->ival : (int64_t)a->num; for (int d = 0; d < nd; d++) out[d] = v; return TSR_OK; }
    if (a->kind == 5 && a->count == nd) { for (int d = 0; d < nd; d++) out[d] = (a->items[d].flags & 1) ? a->items[d].ival : (int64_t)a->items[d].num; return TSR_OK; }
    if (a->kind == 3 && a->arr.ndim == 1 && a->arr.shape[0] == nd) {
        int64_t t; double *p = fn_arg_doubles(a, &t); if (!p) return TSR_ENOMEM;
        for (int d = 0; d < nd; d++) out[d] = (int64_t)p[d];
        fn_free_doubles(p, t); return TSR_OK;
    }
    fn_set_error("filter: size must be a scalar or one value per axis"); return TSR_EARG;
}

/* mode@mi, cval@mi+1, origin@mi+2 (scalar origin, applied to every axis). */
static int rank_tail(const tsr_arg *args, int nargs, int32_t nd, int mi, int *mode, double *cval, int64_t *origin)
{
    int rc; if ((rc = nd_mode(nargs > mi ? &args[mi] : NULL, mode)) < 0) return rc;
    *cval = (nargs > mi + 1 && args[mi + 1].kind == 1) ? ((args[mi + 1].flags & 1) ? (double)args[mi + 1].ival : args[mi + 1].num) : 0.0;
    const int64_t o = arg_int(nargs > mi + 2 ? &args[mi + 2] : NULL, 0);
    for (int d = 0; d < nd; d++) origin[d] = o;
    return TSR_OK;
}

/* out[p] = (sorted box window around p)[pick], over raw contiguous buffers (chainable). */
static int windowed_rank_buf(const double *in, double *out, const int64_t *shape, int32_t nd,
                             const int64_t *size, const int64_t *origin, int mode, double cval, int64_t pick)
{
    int64_t stride[8], center[8], total = 1, V = 1;
    for (int d = 0; d < nd; d++) { total *= shape[d]; V *= size[d]; center[d] = size[d] / 2; }
    if (total == 0 || V == 0) return TSR_OK;
    stride[nd - 1] = 1;
    for (int d = nd - 2; d >= 0; d--) stride[d] = stride[d + 1] * shape[d + 1];
    if (pick < 0) pick = 0;
    if (pick >= V) pick = V - 1;
    double *win = (double *)malloc((size_t)V * sizeof(double));
    int64_t *coord = (int64_t *)malloc((size_t)nd * sizeof(int64_t)), *off = (int64_t *)malloc((size_t)nd * sizeof(int64_t));
    if (!win || !coord || !off) { free(win); free(coord); free(off); return TSR_ENOMEM; }
    for (int64_t oi = 0; oi < total; oi++) {
        int64_t t = oi;
        for (int d = 0; d < nd; d++) { coord[d] = t / stride[d]; t %= stride[d]; }
        for (int d = 0; d < nd; d++) off[d] = 0;
        for (int64_t v = 0; v < V; v++) {
            int64_t flat = 0, oob = 0;
            for (int d = 0; d < nd; d++) {
                const int64_t sc = bmap(coord[d] + off[d] - center[d] - origin[d], shape[d], mode);
                if (sc < 0) { oob = 1; break; }
                flat += sc * stride[d];
            }
            win[v] = oob ? cval : in[flat];
            for (int d = nd - 1; d >= 0; d--) { if (++off[d] < size[d]) break; off[d] = 0; }
        }
        qsort(win, (size_t)V, sizeof(double), cmp_double);
        out[oi] = win[pick];
    }
    free(win); free(coord); free(off);
    return TSR_OK;
}

static int windowed_rank(const tsr_arg *input, const int64_t *size, const int64_t *origin, int mode, double cval, int64_t pick, tsr_result *res)
{
    const int32_t nd = input->arr.ndim;
    int64_t ti; double *in = fn_arg_doubles(input, &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(res, TSR_F64, nd, input->arr.shape);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    int rc = windowed_rank_buf(in, out, input->arr.shape, nd, size, origin, mode, cval, pick);
    fn_free_doubles(in, ti);
    return rc;
}

/* minimum/maximum/median_filter: input, size=None, mode, cval, origin. ctx picks the operation. */
static int r_order_filter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int op = ctx ? *(const int *)ctx : OP_MIN;
    if (args[0].kind != 3) { fn_set_error("order filter: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("order filter: at most 8 dimensions"); return TSR_EDIM; }
    int64_t size[8], origin[8]; int rc = perax_i(nargs > 1 ? &args[1] : NULL, nd, 3, size);
    if (rc < 0) return rc;
    int64_t V = 1; for (int d = 0; d < nd; d++) { if (size[d] < 1) { fn_set_error("order filter: size must be >= 1"); return TSR_EARG; } V *= size[d]; }
    int mode; double cval;
    if ((rc = rank_tail(args, nargs, nd, 2, &mode, &cval, origin)) < 0) return rc;
    const int64_t pick = op == OP_MAX ? V - 1 : op == OP_MEDIAN ? V / 2 : 0;
    return windowed_rank(&args[0], size, origin, mode, cval, pick, &res[0]);
}

/* rank_filter: input, rank, size=None, mode, cval, origin (rank is a 0-based order statistic; negative wraps). */
static int r_rank_filter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || nargs < 2) { fn_set_error("rank_filter: an input array and a rank are required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("rank_filter: at most 8 dimensions"); return TSR_EDIM; }
    int64_t rank = arg_int(&args[1], 0);
    int64_t size[8], origin[8]; int rc = perax_i(nargs > 2 ? &args[2] : NULL, nd, 3, size);
    if (rc < 0) return rc;
    int64_t V = 1; for (int d = 0; d < nd; d++) { if (size[d] < 1) { fn_set_error("rank_filter: size must be >= 1"); return TSR_EARG; } V *= size[d]; }
    if (rank < 0) rank += V;
    if (rank < 0 || rank >= V) { fn_set_error("rank_filter: rank is out of range"); return TSR_EARG; }
    int mode; double cval;
    if ((rc = rank_tail(args, nargs, nd, 3, &mode, &cval, origin)) < 0) return rc;
    return windowed_rank(&args[0], size, origin, mode, cval, rank, &res[0]);
}

/* percentile_filter: input, percentile, size=None, mode, cval, origin. */
static int r_percentile_filter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || nargs < 2 || args[1].kind != 1) { fn_set_error("percentile_filter: an input array and a percentile are required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("percentile_filter: at most 8 dimensions"); return TSR_EDIM; }
    double pct = args[1].num;
    if (pct < 0.0) pct += 100.0;
    if (pct < 0.0 || pct > 100.0) { fn_set_error("percentile_filter: percentile must be in [-100, 100]"); return TSR_EARG; }
    int64_t size[8], origin[8]; int rc = perax_i(nargs > 2 ? &args[2] : NULL, nd, 3, size);
    if (rc < 0) return rc;
    int64_t V = 1; for (int d = 0; d < nd; d++) { if (size[d] < 1) { fn_set_error("percentile_filter: size must be >= 1"); return TSR_EARG; } V *= size[d]; }
    const int64_t pick = pct == 100.0 ? V - 1 : (int64_t)((double)V * pct / 100.0);
    int mode; double cval;
    if ((rc = rank_tail(args, nargs, nd, 3, &mode, &cval, origin)) < 0) return rc;
    return windowed_rank(&args[0], size, origin, mode, cval, pick, &res[0]);
}

/* grey_erosion (flat SE = minimum_filter) / grey_dilation (= maximum_filter with the footprint reflected).
   input, size=None, mode='reflect', cval=0.0, origin=0. ctx &OP_MAX selects dilation. */
static int r_grey_morph(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int dilate = ctx && *(const int *)ctx == OP_MAX;
    if (args[0].kind != 3) { fn_set_error("grey morphology: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("grey morphology: at most 8 dimensions"); return TSR_EDIM; }
    int64_t size[8], origin[8]; int rc = perax_i(nargs > 1 ? &args[1] : NULL, nd, 3, size);
    if (rc < 0) return rc;
    int64_t V = 1; for (int d = 0; d < nd; d++) { if (size[d] < 1) { fn_set_error("grey morphology: size must be >= 1"); return TSR_EARG; } V *= size[d]; }
    int mode; double cval;
    if ((rc = rank_tail(args, nargs, nd, 2, &mode, &cval, origin)) < 0) return rc;
    if (dilate) for (int d = 0; d < nd; d++) origin[d] = -origin[d] - (size[d] % 2 == 0 ? 1 : 0);   /* reflect the footprint */
    return windowed_rank(&args[0], size, origin, mode, cval, dilate ? V - 1 : 0, &res[0]);
}

static const int GREY_OPEN = 1;

/* grey_opening (ctx &GREY_OPEN = erosion then dilation) / grey_closing (dilation then erosion). */
static int r_grey_openclose(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int opening = ctx && *(const int *)ctx == GREY_OPEN;
    if (args[0].kind != 3) { fn_set_error("grey opening/closing: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("grey opening/closing: at most 8 dimensions"); return TSR_EDIM; }
    int64_t size[8], origin[8]; int rc = perax_i(nargs > 1 ? &args[1] : NULL, nd, 3, size);
    if (rc < 0) return rc;
    int64_t V = 1, total = 1;
    for (int d = 0; d < nd; d++) { if (size[d] < 1) { fn_set_error("grey opening/closing: size must be >= 1"); return TSR_EARG; } V *= size[d]; total *= args[0].arr.shape[d]; }
    int mode; double cval;
    if ((rc = rank_tail(args, nargs, nd, 2, &mode, &cval, origin)) < 0) return rc;
    int64_t ero_o[8], dil_o[8];
    for (int d = 0; d < nd; d++) { ero_o[d] = origin[d]; dil_o[d] = -origin[d] - (size[d] % 2 == 0 ? 1 : 0); }
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape);
    double *tmp = out ? (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double)) : NULL;
    if (!out || !tmp) { fn_free_doubles(in, ti); free(tmp); return TSR_ENOMEM; }
    const int64_t *shape = args[0].arr.shape;
    if (opening) {
        rc = windowed_rank_buf(in, tmp, shape, nd, size, ero_o, mode, cval, 0);
        if (rc == 0) rc = windowed_rank_buf(tmp, out, shape, nd, size, dil_o, mode, cval, V - 1);
    } else {
        rc = windowed_rank_buf(in, tmp, shape, nd, size, dil_o, mode, cval, V - 1);
        if (rc == 0) rc = windowed_rank_buf(tmp, out, shape, nd, size, ero_o, mode, cval, 0);
    }
    free(tmp); fn_free_doubles(in, ti);
    return rc;
}


/* grey tophats and morphological gradient/laplace, built from the same flat-SE erosion/dilation primitives:
 *   white_tophat = input - opening,  black_tophat = closing - input,
 *   morphological_gradient = dilation - erosion,  morphological_laplace = dilation + erosion - 2*input. */
enum { TOP_WHITE, TOP_BLACK, MORPH_GRAD, MORPH_LAP };
static const int MTH_WHITE = TOP_WHITE, MTH_BLACK = TOP_BLACK, MTH_GRAD = MORPH_GRAD, MTH_LAP = MORPH_LAP;

static int r_grey_tophat(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int op = ctx ? *(const int *)ctx : TOP_WHITE;
    if (args[0].kind != 3) { fn_set_error("tophat/morphology: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("tophat/morphology: at most 8 dimensions"); return TSR_EDIM; }
    int64_t size[8], origin[8]; int rc = perax_i(nargs > 1 ? &args[1] : NULL, nd, 3, size);
    if (rc < 0) return rc;
    int64_t V = 1, total = 1;
    for (int d = 0; d < nd; d++) { if (size[d] < 1) { fn_set_error("tophat/morphology: size must be >= 1"); return TSR_EARG; } V *= size[d]; total *= args[0].arr.shape[d]; }
    int mode; double cval;
    if ((rc = rank_tail(args, nargs, nd, 2, &mode, &cval, origin)) < 0) return rc;
    int64_t ero_o[8], dil_o[8];
    for (int d = 0; d < nd; d++) { ero_o[d] = origin[d]; dil_o[d] = -origin[d] - (size[d] % 2 == 0 ? 1 : 0); }
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape);
    const size_t nb = (size_t)(total > 0 ? total : 1) * sizeof(double);
    double *a = out ? (double *)malloc(nb) : NULL;
    double *b = a ? (double *)malloc(nb) : NULL;
    if (!out || !a || !b) { fn_free_doubles(in, ti); free(a); free(b); return TSR_ENOMEM; }
    const int64_t *shape = args[0].arr.shape;
    if (op == TOP_WHITE) {                                   /* input - opening(input) */
        rc = windowed_rank_buf(in, b, shape, nd, size, ero_o, mode, cval, 0);
        if (rc == 0) rc = windowed_rank_buf(b, a, shape, nd, size, dil_o, mode, cval, V - 1);
        if (rc == 0) for (int64_t i = 0; i < total; i++) out[i] = in[i] - a[i];
    } else if (op == TOP_BLACK) {                            /* closing(input) - input */
        rc = windowed_rank_buf(in, b, shape, nd, size, dil_o, mode, cval, V - 1);
        if (rc == 0) rc = windowed_rank_buf(b, a, shape, nd, size, ero_o, mode, cval, 0);
        if (rc == 0) for (int64_t i = 0; i < total; i++) out[i] = a[i] - in[i];
    } else {                                                 /* grad = dilation - erosion; lap = dil + ero - 2*input */
        rc = windowed_rank_buf(in, a, shape, nd, size, dil_o, mode, cval, V - 1);
        if (rc == 0) rc = windowed_rank_buf(in, b, shape, nd, size, ero_o, mode, cval, 0);
        if (rc == 0) {
            if (op == MORPH_GRAD) for (int64_t i = 0; i < total; i++) out[i] = a[i] - b[i];
            else for (int64_t i = 0; i < total; i++) out[i] = a[i] + b[i] - 2.0 * in[i];
        }
    }
    free(a); free(b); fn_free_doubles(in, ti);
    return rc;
}

static const int MORPH_DILATE = 1, MORPH_ERODE = 0;

/* ---------------------------------------------------------------- 1-D filters along an axis ----- */
static const int ND_CONV1D = 1, ND_MAX1D = 1;

/* boundary-folded sample of a 1-D line; cval when 'constant' maps out of range (bmap returns -1). */
static inline double ext1d(const double *line, int64_t idx, int64_t L, int mode, double cval)
{
    const int64_t k = bmap(idx, L, mode);
    return k < 0 ? cval : line[k];
}

typedef struct { const double *w; int64_t W, center, origin; int mode; double cval; } corr1d_ctx;

/* scipy NI_Correlate1D: out[i] = sum_j w[j] * input_ext[i + j - center - origin], center = W//2. */
static void corr1d_line(const double *line, double *out, int64_t L, const void *vc)
{
    const corr1d_ctx *c = (const corr1d_ctx *)vc;
    for (int64_t i = 0; i < L; i++) {
        double acc = 0.0;
        for (int64_t j = 0; j < c->W; j++)
            if (c->w[j] != 0.0) acc += c->w[j] * ext1d(line, i + j - c->center - c->origin, L, c->mode, c->cval);
        out[i] = acc;
    }
}

typedef struct { int64_t size, origin; int mode, is_max; double cval; } mm1d_ctx;

static void mm1d_line(const double *line, double *out, int64_t L, const void *vc)
{
    const mm1d_ctx *c = (const mm1d_ctx *)vc;
    const int64_t center = c->size / 2;
    for (int64_t i = 0; i < L; i++) {
        double best = c->is_max ? -INFINITY : INFINITY;
        for (int64_t j = 0; j < c->size; j++) {
            const double v = ext1d(line, i + j - center - c->origin, L, c->mode, c->cval);
            if (c->is_max ? (v > best) : (v < best)) best = v;
        }
        out[i] = best;
    }
}

/* run a per-line filter along axis `ax` of a contiguous C-order n-D array. */
static int filter_lines(const double *in, double *out, const int64_t *shape, int32_t nd, int ax,
                        void (*line)(const double *, double *, int64_t, const void *), const void *ctx)
{
    const int64_t L = shape[ax];
    int64_t inner = 1, outer = 1;
    for (int d = ax + 1; d < nd; d++) inner *= shape[d];
    for (int d = 0; d < ax; d++) outer *= shape[d];
    if (L == 0 || inner == 0 || outer == 0) return TSR_OK;
    double *buf = (double *)malloc((size_t)L * sizeof(double));
    double *obuf = (double *)malloc((size_t)L * sizeof(double));
    if (!buf || !obuf) { free(buf); free(obuf); return TSR_ENOMEM; }
    for (int64_t o = 0; o < outer; o++)
        for (int64_t c = 0; c < inner; c++) {
            const int64_t base = o * L * inner + c;
            for (int64_t k = 0; k < L; k++) buf[k] = in[base + k * inner];
            line(buf, obuf, L, ctx);
            for (int64_t k = 0; k < L; k++) out[base + k * inner] = obuf[k];
        }
    free(buf); free(obuf);
    return TSR_OK;
}

/* axis@ai, mode@ai+1, cval@ai+2, origin@ai+3 — the correlate/uniform/min/max filter tail. */
static int filter_tail(const tsr_arg *args, int nargs, int32_t nd, int ai,
                       int *ax, int *mode, double *cval, int64_t *origin)
{
    int64_t a = arg_int(nargs > ai ? &args[ai] : NULL, -1);
    if (a < 0) a += nd;
    if (a < 0 || a >= nd) { fn_set_error("filter: axis is out of range"); return TSR_EARG; }
    *ax = (int)a;
    int rc; if ((rc = nd_mode(nargs > ai + 1 ? &args[ai + 1] : NULL, mode)) < 0) return rc;
    *cval = (nargs > ai + 2 && args[ai + 2].kind == 1) ? ((args[ai + 2].flags & 1) ? (double)args[ai + 2].ival : args[ai + 2].num) : 0.0;
    if (origin) *origin = arg_int(nargs > ai + 3 ? &args[ai + 3] : NULL, 0);
    return TSR_OK;
}

/* correlate1d (ctx NULL) / convolve1d (ctx &ND_CONV1D): input, weights, axis=-1, mode, cval, origin. */
static int r_correlate1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int convolve = ctx && *(const int *)ctx;
    if (args[0].kind != 3 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("correlate1d: an input array and 1-D weights are required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    int64_t tw; double *w = fn_arg_doubles(&args[1], &tw);
    if (!w) return TSR_ENOMEM;
    const int64_t W = tw;
    int ax, mode; double cval; int64_t origin;
    int rc = filter_tail(args, nargs, nd, 2, &ax, &mode, &cval, &origin);
    if (rc < 0) { fn_free_doubles(w, tw); return rc; }
    double *rev = NULL, *wk = w;
    if (convolve) {                                   /* reverse weights, flip origin (and -1 for even length) */
        rev = (double *)malloc((size_t)(W > 0 ? W : 1) * sizeof(double));
        if (!rev) { fn_free_doubles(w, tw); return TSR_ENOMEM; }
        for (int64_t j = 0; j < W; j++) rev[j] = w[W - 1 - j];
        wk = rev; origin = -origin; if ((W & 1) == 0) origin -= 1;
    }
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti);
    double *out = in ? (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape) : NULL;
    if (!in || !out) { fn_free_doubles(in, ti); fn_free_doubles(w, tw); free(rev); return TSR_ENOMEM; }
    corr1d_ctx c = { wk, W, W / 2, origin, mode, cval };
    rc = filter_lines(in, out, args[0].arr.shape, nd, ax, corr1d_line, &c);
    fn_free_doubles(in, ti); fn_free_doubles(w, tw); free(rev);
    return rc;
}

/* a correlation with a given kernel over the input (shared by uniform/gaussian). */
static int run_corr1d(const tsr_arg *input, const double *w, int64_t W, int ax, int mode, double cval, int64_t origin, tsr_result *res)
{
    const int32_t nd = input->arr.ndim;
    int64_t ti; double *in = fn_arg_doubles(input, &ti);
    double *out = in ? (double *)fn_result_array(res, TSR_F64, nd, input->arr.shape) : NULL;
    if (!in || !out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    corr1d_ctx c = { w, W, W / 2, origin, mode, cval };
    int rc = filter_lines(in, out, input->arr.shape, nd, ax, corr1d_line, &c);
    fn_free_doubles(in, ti);
    return rc;
}

/* uniform_filter1d: input, size, axis=-1, mode, cval, origin. */
static int r_uniform_filter1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("uniform_filter1d: input must be an array"); return TSR_EARG; }
    const int64_t size = arg_int(nargs > 1 ? &args[1] : NULL, 0);
    if (size < 1) { fn_set_error("uniform_filter1d: size must be >= 1"); return TSR_EARG; }
    int ax, mode; double cval; int64_t origin;
    int rc = filter_tail(args, nargs, args[0].arr.ndim, 2, &ax, &mode, &cval, &origin);
    if (rc < 0) return rc;
    double *w = (double *)malloc((size_t)size * sizeof(double));
    if (!w) return TSR_ENOMEM;
    for (int64_t j = 0; j < size; j++) w[j] = 1.0 / (double)size;
    rc = run_corr1d(&args[0], w, size, ax, mode, cval, origin, &res[0]);
    free(w);
    return rc;
}

/* gaussian_filter1d: input, sigma, axis=-1, order=0, mode, cval, truncate=4.0, radius=None. (order 0 only) */
static int r_gaussian_filter1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || nargs < 2 || args[1].kind != 1) { fn_set_error("gaussian_filter1d: input array and sigma required"); return TSR_EARG; }
    const double sigma = args[1].num;
    const int64_t order = arg_int(nargs > 3 ? &args[3] : NULL, 0);
    if (order != 0) { fn_set_error("gaussian_filter1d: only order=0 is supported"); return TSR_EARG; }
    const double truncate = (nargs > 6 && args[6].kind == 1) ? args[6].num : 4.0;
    int64_t lw = (nargs > 7 && args[7].kind == 1) ? arg_int(&args[7], 0) : (int64_t)(truncate * sigma + 0.5);
    if (lw < 0) lw = 0;
    const int64_t W = 2 * lw + 1;
    /* axis@2, order@3, mode@4, cval@5 (no origin argument). */
    int64_t a = arg_int(nargs > 2 ? &args[2] : NULL, -1);
    if (a < 0) a += args[0].arr.ndim;
    if (a < 0 || a >= args[0].arr.ndim) { fn_set_error("gaussian_filter1d: axis out of range"); return TSR_EARG; }
    int mode; int rc = nd_mode(nargs > 4 ? &args[4] : NULL, &mode);
    if (rc < 0) return rc;
    const double cval = (nargs > 5 && args[5].kind == 1) ? args[5].num : 0.0;
    double *w = (double *)malloc((size_t)W * sizeof(double));
    if (!w) return TSR_ENOMEM;
    const double a2 = -0.5 / (sigma * sigma);
    double s = 0.0;
    for (int64_t k = -lw; k <= lw; k++) { const double v = exp(a2 * (double)(k * k)); w[k + lw] = v; s += v; }
    for (int64_t j = 0; j < W; j++) w[j] /= s;         /* normalise; symmetric, so no reversal needed for order 0 */
    rc = run_corr1d(&args[0], w, W, (int)a, mode, cval, 0, &res[0]);
    free(w);
    return rc;
}

/* minimum_filter1d (ctx NULL) / maximum_filter1d (ctx &ND_MAX1D): input, size, axis=-1, mode, cval, origin. */
static int r_minmax_filter1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    if (args[0].kind != 3) { fn_set_error("minimum/maximum_filter1d: input must be an array"); return TSR_EARG; }
    const int64_t size = arg_int(nargs > 1 ? &args[1] : NULL, 0);
    if (size < 1) { fn_set_error("minimum/maximum_filter1d: size must be >= 1"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    int ax, mode; double cval; int64_t origin;
    int rc = filter_tail(args, nargs, nd, 2, &ax, &mode, &cval, &origin);
    if (rc < 0) return rc;
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti);
    double *out = in ? (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape) : NULL;
    if (!in || !out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    mm1d_ctx c = { size, origin, mode, ctx && *(const int *)ctx, cval };
    rc = filter_lines(in, out, args[0].arr.shape, nd, ax, mm1d_line, &c);
    fn_free_doubles(in, ti);
    return rc;
}

/* a scalar (same for every axis) or an nd-length sequence -> one value per axis. */
static int perax_d(const tsr_arg *a, int32_t nd, double *out)
{
    if (a->kind == 1) { const double v = (a->flags & 1) ? (double)a->ival : a->num; for (int d = 0; d < nd; d++) out[d] = v; return TSR_OK; }
    if (a->kind == 3 && a->arr.ndim == 1 && a->arr.shape[0] == nd) {
        int64_t t; double *p = fn_arg_doubles(a, &t); if (!p) return TSR_ENOMEM;
        for (int d = 0; d < nd; d++) out[d] = p[d];
        fn_free_doubles(p, t); return TSR_OK;
    }
    fn_set_error("filter: a scalar or one value per axis is required"); return TSR_EARG;
}

/* Apply a (per-axis) correlation kernel along every axis in turn, ping-ponging buffers (separable filter). */
static int separable_corr(const tsr_arg *input, double *const *kern, const int64_t *W, const int64_t *origin,
                          int mode, double cval, tsr_result *res)
{
    const int32_t nd = input->arr.ndim;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= input->arr.shape[d] > 0 ? input->arr.shape[d] : 0;
    int64_t ti; double *in = fn_arg_doubles(input, &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(res, TSR_F64, nd, input->arr.shape);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    if (total == 0) { fn_free_doubles(in, ti); return TSR_OK; }
    double *a = (double *)malloc((size_t)total * sizeof(double)), *b = (double *)malloc((size_t)total * sizeof(double));
    if (!a || !b) { free(a); free(b); fn_free_doubles(in, ti); return TSR_ENOMEM; }
    memcpy(a, in, (size_t)total * sizeof(double));
    fn_free_doubles(in, ti);
    double *src = a, *dst = b;
    int rc = TSR_OK;
    for (int d = 0; d < nd && rc == 0; d++) {
        corr1d_ctx c = { kern[d], W[d], W[d] / 2, origin ? origin[d] : 0, mode, cval };
        rc = filter_lines(src, dst, input->arr.shape, nd, d, corr1d_line, &c);
        double *t = src; src = dst; dst = t;
    }
    if (rc == 0) memcpy(out, src, (size_t)total * sizeof(double));
    free(a); free(b);
    return rc;
}

/* gaussian_filter: input, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0, radius=None. (order 0) */
static int r_gaussian_filter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || nargs < 2) { fn_set_error("gaussian_filter: an input array and sigma are required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("gaussian_filter: at most 8 dimensions"); return TSR_EDIM; }
    double sigma[8]; int rc = perax_d(&args[1], nd, sigma); if (rc < 0) return rc;
    if (arg_int(nargs > 2 ? &args[2] : NULL, 0) != 0) { fn_set_error("gaussian_filter: only order=0 is supported"); return TSR_EARG; }
    int mode; if ((rc = nd_mode(nargs > 3 ? &args[3] : NULL, &mode)) < 0) return rc;
    const double cval = (nargs > 4 && args[4].kind == 1) ? args[4].num : 0.0;
    const double truncate = (nargs > 5 && args[5].kind == 1) ? args[5].num : 4.0;
    const int has_radius = nargs > 6 && args[6].kind == 1;
    double *kern[8]; int64_t W[8];
    for (int d = 0; d < nd; d++) {
        int64_t lw = has_radius ? arg_int(&args[6], 0) : (int64_t)(truncate * sigma[d] + 0.5);
        if (lw < 0) lw = 0;
        W[d] = 2 * lw + 1;
        kern[d] = (double *)malloc((size_t)W[d] * sizeof(double));
        if (!kern[d]) { for (int e = 0; e < d; e++) free(kern[e]); return TSR_ENOMEM; }
        const double a2 = -0.5 / (sigma[d] * sigma[d]);
        double s = 0.0;
        for (int64_t k = -lw; k <= lw; k++) { const double v = exp(a2 * (double)(k * k)); kern[d][k + lw] = v; s += v; }
        for (int64_t j = 0; j < W[d]; j++) kern[d][j] /= s;
    }
    rc = separable_corr(&args[0], kern, W, NULL, mode, cval, &res[0]);
    for (int d = 0; d < nd; d++) free(kern[d]);
    return rc;
}

/* uniform_filter: input, size=3, mode='reflect', cval=0.0, origin=0. */
static int r_uniform_filter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("uniform_filter: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd > 8) { fn_set_error("uniform_filter: at most 8 dimensions"); return TSR_EDIM; }
    double sized[8]; int rc = perax_d(nargs > 1 ? &args[1] : NULL, nd, sized);
    if (nargs <= 1 || args[1].kind == 0) { for (int d = 0; d < nd; d++) sized[d] = 3; rc = TSR_OK; }
    if (rc < 0) return rc;
    int mode; if ((rc = nd_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    const double cval = (nargs > 3 && args[3].kind == 1) ? args[3].num : 0.0;
    const int64_t origin0 = arg_int(nargs > 4 ? &args[4] : NULL, 0);
    double *kern[8]; int64_t W[8], orig[8];
    for (int d = 0; d < nd; d++) {
        const int64_t sz = (int64_t)sized[d];
        if (sz < 1) { for (int e = 0; e < d; e++) free(kern[e]); fn_set_error("uniform_filter: size must be >= 1"); return TSR_EARG; }
        W[d] = sz; orig[d] = origin0;
        kern[d] = (double *)malloc((size_t)sz * sizeof(double));
        if (!kern[d]) { for (int e = 0; e < d; e++) free(kern[e]); return TSR_ENOMEM; }
        for (int64_t j = 0; j < sz; j++) kern[d][j] = 1.0 / (double)sz;
    }
    rc = separable_corr(&args[0], kern, W, orig, mode, cval, &res[0]);
    for (int d = 0; d < nd; d++) free(kern[d]);
    return rc;
}

/* scipy.ndimage _gaussian_kernel1d(sigma, order, lw), reversed for correlate1d (fills w[0..2*lw]). order 0..3.
 * The derivative kernel is q(x)*phi(x) where phi is the normalised Gaussian and q is the polynomial built by
 * scipy's recurrence q <- (D + P) q (D = derivative of the monomial basis, P = multiply by phi'(x)/phi(x) = -x/sigma^2). */
static void build_gauss_deriv(double sigma, int order, int64_t lw, double *w)
{
    const int64_t W = 2 * lw + 1;
    const double sigma2 = sigma * sigma;
    double s = 0.0;
    for (int64_t k = -lw; k <= lw; k++) { const double v = exp(-0.5 / sigma2 * (double)(k * k)); w[k + lw] = v; s += v; }
    for (int64_t j = 0; j < W; j++) w[j] /= s;
    if (order > 0) {
        double q[4]; for (int i = 0; i <= order; i++) q[i] = 0.0; q[0] = 1.0;
        for (int it = 0; it < order; it++) {
            double nq[4]; for (int i = 0; i <= order; i++) nq[i] = 0.0;
            for (int i = 0; i <= order; i++) {
                if (i + 1 <= order) nq[i] += (double)(i + 1) * q[i + 1];   /* D: superdiagonal = [1,2,..,order] */
                if (i - 1 >= 0)     nq[i] += (-1.0 / sigma2) * q[i - 1];   /* P: subdiagonal   = -1/sigma^2    */
            }
            for (int i = 0; i <= order; i++) q[i] = nq[i];
        }
        for (int64_t idx = 0; idx < W; idx++) {
            const double x = (double)(idx - lw);
            double poly = 0.0, xp = 1.0;
            for (int j = 0; j <= order; j++) { poly += q[j] * xp; xp *= x; }
            w[idx] *= poly;
        }
    }
    for (int64_t i = 0, j = W - 1; i < j; i++, j--) { double t = w[i]; w[i] = w[j]; w[j] = t; }  /* reverse for correlate1d */
}

/* separable_corr working on plain buffers (ping-pong out/tmp); used to accumulate per-axis passes. */
static int separable_corr_buf(const double *in, const int64_t *shape, int32_t nd,
                              double *const *kern, const int64_t *W, const int64_t *origin,
                              int mode, double cval, double *out, double *tmp)
{
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= shape[d] > 0 ? shape[d] : 0;
    if (nd == 0 || total == 0) { if (total > 0) memcpy(out, in, (size_t)total * sizeof(double)); return TSR_OK; }
    const double *src = in;
    double *bufs[2] = { out, tmp };
    int rc = TSR_OK;
    for (int d = 0; d < nd && rc == 0; d++) {
        double *dst = bufs[d & 1];
        corr1d_ctx c = { kern[d], W[d], W[d] / 2, origin ? origin[d] : 0, mode, cval };
        rc = filter_lines(src, dst, shape, nd, d, corr1d_line, &c);
        src = dst;
    }
    if (rc == 0 && src != out) memcpy(out, src, (size_t)total * sizeof(double));
    return rc;
}

/* gaussian_laplace (ctx &GAUSS_LAP): Σ_axis of the order-2 Gaussian derivative along that axis.
 * gaussian_gradient_magnitude (ctx NULL): sqrt(Σ_axis (order-1 Gaussian derivative along that axis)^2). */
static const int GAUSS_LAP = 1;
static int r_gaussian_deriv_combine(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int lap = ctx && *(const int *)ctx == GAUSS_LAP;
    if (args[0].kind != 3 || nargs < 2) { fn_set_error("gaussian_laplace/gradient_magnitude: input array and sigma required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("gaussian_laplace/gradient_magnitude: 1..8 dimensions"); return TSR_EDIM; }
    double sigma[8]; int rc = perax_d(&args[1], nd, sigma); if (rc < 0) return rc;
    int mode; if ((rc = nd_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    const double cval = (nargs > 3 && args[3].kind == 1) ? args[3].num : 0.0;
    const double truncate = (nargs > 4 && args[4].kind == 1) ? args[4].num : 4.0;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= args[0].arr.shape[d] > 0 ? args[0].arr.shape[d] : 0;
    int64_t W[8]; double *k0[8], *kd[8];     /* per-axis order-0 and high-order (1 or 2) kernels */
    for (int d = 0; d < nd; d++) {
        int64_t lw = (int64_t)(truncate * sigma[d] + 0.5); if (lw < 0) lw = 0;
        W[d] = 2 * lw + 1;
        k0[d] = (double *)malloc((size_t)W[d] * sizeof(double));
        kd[d] = (double *)malloc((size_t)W[d] * sizeof(double));
        if (!k0[d] || !kd[d]) { free(k0[d]); free(kd[d]); for (int e = 0; e < d; e++) { free(k0[e]); free(kd[e]); } return TSR_ENOMEM; }
        build_gauss_deriv(sigma[d], 0, lw, k0[d]);
        build_gauss_deriv(sigma[d], lap ? 2 : 1, lw, kd[d]);
    }
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti);
    double *out = in ? (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape) : NULL;
    const size_t nb = (size_t)(total > 0 ? total : 1) * sizeof(double);
    double *acc = out ? (double *)calloc((size_t)(total > 0 ? total : 1), sizeof(double)) : NULL;
    double *b1 = acc ? (double *)malloc(nb) : NULL;
    double *b2 = b1 ? (double *)malloc(nb) : NULL;
    if (!in || !out || !acc || !b1 || !b2) {
        fn_free_doubles(in, ti); free(acc); free(b1); free(b2);
        for (int d = 0; d < nd; d++) { free(k0[d]); free(kd[d]); }
        return TSR_ENOMEM;
    }
    rc = TSR_OK;
    for (int ax = 0; ax < nd && rc == 0; ax++) {
        double *kern[8]; for (int d = 0; d < nd; d++) kern[d] = (d == ax) ? kd[d] : k0[d];
        rc = separable_corr_buf(in, args[0].arr.shape, nd, kern, W, NULL, mode, cval, b1, b2);
        if (rc == 0) {
            if (lap) for (int64_t i = 0; i < total; i++) acc[i] += b1[i];
            else     for (int64_t i = 0; i < total; i++) acc[i] += b1[i] * b1[i];
        }
    }
    if (rc == 0) {
        if (lap) memcpy(out, acc, (size_t)total * sizeof(double));
        else     for (int64_t i = 0; i < total; i++) out[i] = sqrt(acc[i]);
    }
    fn_free_doubles(in, ti); free(acc); free(b1); free(b2);
    for (int d = 0; d < nd; d++) { free(k0[d]); free(kd[d]); }
    return rc;
}

/* sobel/prewitt/laplace use the 1-D correlation primitives (corr1d_line/filter_lines) defined above. */
static const int FILT_SOBEL = 1;

/* sobel (ctx &FILT_SOBEL) / prewitt: derivative [-1,0,1] along `axis`, smoothing along the other axes. */
static int r_sobel_prewitt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int sobel = ctx && *(const int *)ctx == FILT_SOBEL;
    if (args[0].kind != 3) { fn_set_error("sobel/prewitt: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    int64_t a = arg_int(nargs > 1 ? &args[1] : NULL, -1);
    if (a < 0) a += nd;
    if (a < 0 || a >= nd) { fn_set_error("sobel/prewitt: axis is out of range"); return TSR_EARG; }
    int mode; int rc = nd_mode(nargs > 2 ? &args[2] : NULL, &mode);
    if (rc < 0) return rc;
    const double cval = (nargs > 3 && args[3].kind == 1) ? args[3].num : 0.0;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= args[0].arr.shape[d];
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape);
    double *tmp = out ? (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double)) : NULL;
    if (!out || !tmp) { fn_free_doubles(in, ti); free(tmp); return TSR_ENOMEM; }
    static const double deriv[3] = {-1.0, 0.0, 1.0}, smooth_s[3] = {1.0, 2.0, 1.0}, smooth_p[3] = {1.0, 1.0, 1.0};
    const double *smooth = sobel ? smooth_s : smooth_p;
    const int64_t *shape = args[0].arr.shape;
    corr1d_ctx dc = { deriv, 3, 1, 0, mode, cval };
    rc = filter_lines(in, out, shape, nd, (int)a, corr1d_line, &dc);
    double *src = out, *dst = tmp;
    for (int d = 0; d < nd && rc == 0; d++) {
        if (d == (int)a) continue;
        corr1d_ctx sc = { smooth, 3, 1, 0, mode, cval };
        rc = filter_lines(src, dst, shape, nd, d, corr1d_line, &sc);
        double *t = src; src = dst; dst = t;
    }
    if (rc == 0 && src != out) memcpy(out, src, (size_t)total * sizeof(double));
    free(tmp); fn_free_doubles(in, ti);
    return rc;
}

/* laplace: sum over axes of the second-derivative [1,-2,1] correlation. */
static int r_laplace(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("laplace: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    int mode; int rc = nd_mode(nargs > 1 ? &args[1] : NULL, &mode);
    if (rc < 0) return rc;
    const double cval = (nargs > 2 && args[2].kind == 1) ? args[2].num : 0.0;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= args[0].arr.shape[d];
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape);
    double *tmp = out ? (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double)) : NULL;
    if (!out || !tmp) { fn_free_doubles(in, ti); free(tmp); return TSR_ENOMEM; }
    for (int64_t i = 0; i < total; i++) out[i] = 0.0;
    static const double lap[3] = {1.0, -2.0, 1.0};
    const int64_t *shape = args[0].arr.shape;
    for (int d = 0; d < nd && rc == 0; d++) {
        corr1d_ctx c = { lap, 3, 1, 0, mode, cval };
        rc = filter_lines(in, tmp, shape, nd, d, corr1d_line, &c);
        if (rc == 0) for (int64_t i = 0; i < total; i++) out[i] += tmp[i];
    }
    free(tmp); fn_free_doubles(in, ti);
    return rc;
}

/* labeled measurements: input[, labels[, index]] -> per-region reduction.
 * index None + labels  -> single region (labels != 0); index None + no labels -> whole array;
 * index scalar -> region labels==index (scalar result); index array -> per-index (1-D array result).
 * Empty region: sum/max/min -> 0, mean/var/std/median -> NaN (matches scipy; scipy's *missing-label* median is
 * an internal artifact, so the fixtures exercise median over existing labels only). */
enum { M_SUM = 0, M_MEAN, M_VAR, M_STD, M_MAX, M_MIN, M_MEDIAN };
static const int MEAS_SUM = M_SUM, MEAS_MEAN = M_MEAN, MEAS_VAR = M_VAR, MEAS_STD = M_STD,
                 MEAS_MAX = M_MAX, MEAS_MIN = M_MIN, MEAS_MEDIAN = M_MEDIAN;

static double reduce_region(int op, double *vals, int64_t n)   /* vals may be reordered (median sorts in place) */
{
    if (n == 0) return (op == M_SUM || op == M_MAX || op == M_MIN) ? 0.0 : (double)NAN;
    if (op == M_SUM) { double s = 0; for (int64_t i = 0; i < n; i++) s += vals[i]; return s; }
    if (op == M_MAX) { double m = vals[0]; for (int64_t i = 1; i < n; i++) if (vals[i] > m) m = vals[i]; return m; }
    if (op == M_MIN) { double m = vals[0]; for (int64_t i = 1; i < n; i++) if (vals[i] < m) m = vals[i]; return m; }
    double s = 0; for (int64_t i = 0; i < n; i++) s += vals[i];
    const double mean = s / (double)n;
    if (op == M_MEAN) return mean;
    if (op == M_VAR || op == M_STD) {
        double c = 0; for (int64_t i = 0; i < n; i++) { const double d = vals[i] - mean; c += d * d; }
        const double var = c / (double)n;             /* population variance (scipy divides by N, not N-1) */
        return op == M_STD ? sqrt(var) : var;
    }
    qsort(vals, (size_t)n, sizeof(double), cmp_double);
    return (n & 1) ? vals[n / 2] : 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
}

static int r_measure(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int op = ctx ? *(const int *)ctx : M_SUM;
    if (args[0].kind != 3) { fn_set_error("ndimage measurement: input must be an array"); return TSR_EARG; }
    int64_t N = 1; for (int d = 0; d < args[0].arr.ndim; d++) N *= args[0].arr.shape[d] > 0 ? args[0].arr.shape[d] : 0;
    const int has_labels = nargs > 1 && args[1].kind == 3;
    const tsr_arg *ix = (nargs > 2 && args[2].kind != 0) ? &args[2] : NULL;
    if (ix && !has_labels) { fn_set_error("ndimage measurement: index requires a labels array"); return TSR_EARG; }
    int64_t it_in; double *in = fn_arg_doubles(&args[0], &it_in); if (!in) return TSR_ENOMEM;
    int64_t it_lab = 0; double *lab = NULL;
    if (has_labels) { lab = fn_arg_doubles(&args[1], &it_lab); if (!lab) { fn_free_doubles(in, it_in); return TSR_ENOMEM; } }
    double *scratch = (double *)malloc((size_t)(N > 0 ? N : 1) * sizeof(double));
    if (!scratch) { fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (ix && ix->kind == 3) {                                   /* per-index -> 1-D array result */
        int64_t m = 1; for (int d = 0; d < ix->arr.ndim; d++) m *= ix->arr.shape[d];
        int64_t it_ix; double *idx = fn_arg_doubles(ix, &it_ix);
        int64_t osh[1] = { m };
        double *out = idx ? (double *)fn_result_array(&res[0], TSR_F64, 1, osh) : NULL;
        if (!idx || !out) { rc = TSR_ENOMEM; if (idx) fn_free_doubles(idx, it_ix); }
        else {
            for (int64_t k = 0; k < m; k++) {
                const int64_t want = (int64_t)idx[k];
                int64_t n = 0;
                for (int64_t i = 0; i < N; i++) if ((int64_t)lab[i] == want) scratch[n++] = in[i];
                out[k] = reduce_region(op, scratch, n);
            }
            fn_free_doubles(idx, it_ix);
        }
    } else {                                                     /* scalar result */
        int64_t n = 0;
        if (ix) { const int64_t want = arg_int(ix, 0);
            for (int64_t i = 0; i < N; i++) if ((int64_t)lab[i] == want) scratch[n++] = in[i]; }
        else if (has_labels) { for (int64_t i = 0; i < N; i++) if ((int64_t)lab[i] != 0) scratch[n++] = in[i]; }
        else { for (int64_t i = 0; i < N; i++) scratch[n++] = in[i]; }
        fn_result_num(&res[0], reduce_region(op, scratch, n));
    }
    free(scratch); fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab);
    return rc;
}

/* fill coords[0..nd-1] with the C-order multi-index of flat position i. */
static void unravel(int64_t i, const int64_t *shape, int32_t nd, int64_t *coords)
{
    for (int d = nd - 1; d >= 0; d--) { coords[d] = shape[d] > 0 ? i % shape[d] : 0; if (shape[d] > 0) i /= shape[d]; }
}

/* is flat position i in the requested region?  array/scalar index -> labels==want; index None -> nonzero labels
 * (labels given) or every position (no labels). */
static inline int in_region(const double *lab, int has_labels, int have_idx, int64_t want, int64_t i)
{
    return have_idx ? ((int64_t)lab[i] == want) : (has_labels ? ((int64_t)lab[i] != 0) : 1);
}

/* center_of_mass: input-weighted centroid per labeled region. Result (nd,) for a scalar/None index, (m, nd) for
 * an index array. Empty region -> NaN coords (fixtures use existing labels only). */
static int r_center_of_mass(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("center_of_mass: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("center_of_mass: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t *shape = args[0].arr.shape;
    int64_t N = 1; for (int d = 0; d < nd; d++) N *= shape[d] > 0 ? shape[d] : 0;
    const int has_labels = nargs > 1 && args[1].kind == 3;
    const tsr_arg *ix = (nargs > 2 && args[2].kind != 0) ? &args[2] : NULL;
    if (ix && !has_labels) { fn_set_error("center_of_mass: index requires a labels array"); return TSR_EARG; }
    const int array_idx = ix && ix->kind == 3;
    int64_t m = 1, it_ix = 0; double *idx = NULL;
    int64_t it_in; double *in = fn_arg_doubles(&args[0], &it_in); if (!in) return TSR_ENOMEM;
    int64_t it_lab = 0; double *lab = NULL;
    if (has_labels) { lab = fn_arg_doubles(&args[1], &it_lab); if (!lab) { fn_free_doubles(in, it_in); return TSR_ENOMEM; } }
    if (array_idx) { for (int d = 0; d < ix->arr.ndim; d++) m *= ix->arr.shape[d]; idx = fn_arg_doubles(ix, &it_ix);
        if (!idx) { fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab); return TSR_ENOMEM; } }
    const int64_t sh2[2] = { m, nd }, sh1[1] = { nd };
    double *out = array_idx ? (double *)fn_result_array(&res[0], TSR_F64, 2, sh2)
                            : (double *)fn_result_array(&res[0], TSR_F64, 1, sh1);
    if (!out) { fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab); if (idx) fn_free_doubles(idx, it_ix); return TSR_ENOMEM; }
    int64_t coords[8];
    for (int64_t k = 0; k < m; k++) {
        const int64_t want = array_idx ? (int64_t)idx[k] : (ix ? arg_int(ix, 0) : 0);
        double mass = 0, num[8]; for (int d = 0; d < nd; d++) num[d] = 0.0;
        for (int64_t i = 0; i < N; i++) {
            if (!in_region(lab, has_labels, array_idx || ix != NULL, want, i)) continue;
            const double v = in[i];
            unravel(i, shape, nd, coords);
            mass += v; for (int d = 0; d < nd; d++) num[d] += (double)coords[d] * v;
        }
        double *row = out + k * nd;
        for (int d = 0; d < nd; d++) row[d] = num[d] / mass;
    }
    fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab); if (idx) fn_free_doubles(idx, it_ix);
    return TSR_OK;
}

/* maximum_position (ctx &POS_MAX) / minimum_position: C-order position of the first extreme value per region.
 * Coords returned as float64 (integer-valued). Same result shape convention as center_of_mass. */
static const int POS_MAX = 1;
static int r_minmax_position(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int ismax = ctx && *(const int *)ctx == POS_MAX;
    if (args[0].kind != 3) { fn_set_error("maximum/minimum_position: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("maximum/minimum_position: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t *shape = args[0].arr.shape;
    int64_t N = 1; for (int d = 0; d < nd; d++) N *= shape[d] > 0 ? shape[d] : 0;
    const int has_labels = nargs > 1 && args[1].kind == 3;
    const tsr_arg *ix = (nargs > 2 && args[2].kind != 0) ? &args[2] : NULL;
    if (ix && !has_labels) { fn_set_error("maximum/minimum_position: index requires a labels array"); return TSR_EARG; }
    const int array_idx = ix && ix->kind == 3;
    int64_t m = 1, it_ix = 0; double *idx = NULL;
    int64_t it_in; double *in = fn_arg_doubles(&args[0], &it_in); if (!in) return TSR_ENOMEM;
    int64_t it_lab = 0; double *lab = NULL;
    if (has_labels) { lab = fn_arg_doubles(&args[1], &it_lab); if (!lab) { fn_free_doubles(in, it_in); return TSR_ENOMEM; } }
    if (array_idx) { for (int d = 0; d < ix->arr.ndim; d++) m *= ix->arr.shape[d]; idx = fn_arg_doubles(ix, &it_ix);
        if (!idx) { fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab); return TSR_ENOMEM; } }
    const int64_t sh2[2] = { m, nd }, sh1[1] = { nd };
    double *out = array_idx ? (double *)fn_result_array(&res[0], TSR_F64, 2, sh2)
                            : (double *)fn_result_array(&res[0], TSR_F64, 1, sh1);
    if (!out) { fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab); if (idx) fn_free_doubles(idx, it_ix); return TSR_ENOMEM; }
    int64_t coords[8];
    for (int64_t k = 0; k < m; k++) {
        const int64_t want = array_idx ? (int64_t)idx[k] : (ix ? arg_int(ix, 0) : 0);
        int64_t best_i = -1; double best = 0;
        for (int64_t i = 0; i < N; i++) {
            if (!in_region(lab, has_labels, array_idx || ix != NULL, want, i)) continue;
            const double v = in[i];
            if (best_i < 0 || (ismax ? v > best : v < best)) { best = v; best_i = i; }   /* strict -> first occurrence */
        }
        unravel(best_i >= 0 ? best_i : 0, shape, nd, coords);
        double *row = out + k * nd;
        for (int d = 0; d < nd; d++) row[d] = (double)coords[d];
    }
    fn_free_doubles(in, it_in); if (lab) fn_free_doubles(lab, it_lab); if (idx) fn_free_doubles(idx, it_ix);
    return TSR_OK;
}

/* histogram(input, min, max, bins, labels=None, index=None): bin counts over a region (numpy's uniform-bin rule:
   b = int((v-min)/(max-min)*bins), v==max -> bins-1, values outside [min,max] dropped). Scalar/None index only
   (a 1-D count array). */
static int r_histogram(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || nargs < 4) { fn_set_error("histogram: input, min, max and bins are required"); return TSR_EARG; }
    const double lo = (args[1].flags & 1) ? (double)args[1].ival : args[1].num;
    const double hi = (args[2].flags & 1) ? (double)args[2].ival : args[2].num;
    const int64_t bins = arg_int(&args[3], 0);
    if (bins < 1 || !(hi > lo)) { fn_set_error("histogram: need bins >= 1 and max > min"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    int64_t N = 1; for (int d = 0; d < nd; d++) N *= args[0].arr.shape[d] > 0 ? args[0].arr.shape[d] : 0;
    const int has_labels = nargs > 4 && args[4].kind == 3;
    const tsr_arg *ix = (nargs > 5 && args[5].kind != 0) ? &args[5] : NULL;
    if (ix && ix->kind == 3) { fn_set_error("histogram: an array index is not supported"); return TSR_EARG; }
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    int64_t tl = 0; double *lab = NULL;
    if (has_labels) { lab = fn_arg_doubles(&args[4], &tl); if (!lab) { fn_free_doubles(in, ti); return TSR_ENOMEM; } }
    const int64_t want = ix ? arg_int(ix, 0) : 0;
    const int64_t bsh[1] = { bins };
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, bsh);
    if (!out) { fn_free_doubles(in, ti); if (lab) fn_free_doubles(lab, tl); return TSR_ENOMEM; }
    for (int64_t b = 0; b < bins; b++) out[b] = 0.0;
    for (int64_t i = 0; i < N; i++) {
        if (!in_region(lab, has_labels, ix != NULL, want, i)) continue;
        const double v = in[i];
        if (!(v >= lo && v <= hi)) continue;
        int64_t b = (int64_t)((v - lo) / (hi - lo) * (double)bins);
        if (b == bins) b = bins - 1;
        if (b >= 0 && b < bins) out[b] += 1.0;
    }
    fn_free_doubles(in, ti); if (lab) fn_free_doubles(lab, tl);
    return TSR_OK;
}

/* extrema(input, labels=None, index=None): (minimum, maximum, min_position, max_position) over labeled regions.
   Positions returned as float64 (integer-valued). Scalar/None index -> scalar min/max + (nd,) positions; an index
   array -> (m,) min/max + (m, nd) positions. First occurrence on ties. */
static int r_extrema(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("extrema: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("extrema: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t *shape = args[0].arr.shape;
    int64_t N = 1; for (int d = 0; d < nd; d++) N *= shape[d] > 0 ? shape[d] : 0;
    const int has_labels = nargs > 1 && args[1].kind == 3;
    const tsr_arg *ix = (nargs > 2 && args[2].kind != 0) ? &args[2] : NULL;
    if (ix && !has_labels) { fn_set_error("extrema: index requires a labels array"); return TSR_EARG; }
    const int array_idx = ix && ix->kind == 3;
    int64_t m = 1, it_ix = 0; double *idx = NULL;
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    int64_t tl = 0; double *lab = NULL;
    if (has_labels) { lab = fn_arg_doubles(&args[1], &tl); if (!lab) { fn_free_doubles(in, ti); return TSR_ENOMEM; } }
    if (array_idx) { for (int d = 0; d < ix->arr.ndim; d++) m *= ix->arr.shape[d]; idx = fn_arg_doubles(ix, &it_ix);
        if (!idx) { fn_free_doubles(in, ti); if (lab) fn_free_doubles(lab, tl); return TSR_ENOMEM; } }
    const int64_t sh1[1] = { nd }, shm[1] = { m }, sh2[2] = { m, nd };
    double *mn = NULL, *mx = NULL, *mnp, *mxp;
    if (array_idx) {
        mn = (double *)fn_result_array(&res[0], TSR_F64, 1, shm);
        mx = (double *)fn_result_array(&res[1], TSR_F64, 1, shm);
        mnp = (double *)fn_result_array(&res[2], TSR_F64, 2, sh2);
        mxp = (double *)fn_result_array(&res[3], TSR_F64, 2, sh2);
    } else {
        mnp = (double *)fn_result_array(&res[2], TSR_F64, 1, sh1);
        mxp = (double *)fn_result_array(&res[3], TSR_F64, 1, sh1);
    }
    if ((array_idx && (!mn || !mx)) || !mnp || !mxp) { fn_free_doubles(in, ti); if (lab) fn_free_doubles(lab, tl); if (idx) fn_free_doubles(idx, it_ix); return TSR_ENOMEM; }
    int64_t coord[8];
    for (int64_t k = 0; k < m; k++) {
        const int64_t want = array_idx ? (int64_t)idx[k] : (ix ? arg_int(ix, 0) : 0);
        double vmin = 0, vmax = 0; int64_t imin = -1, imax = -1;
        for (int64_t i = 0; i < N; i++) {
            if (!in_region(lab, has_labels, array_idx || ix != NULL, want, i)) continue;
            const double v = in[i];
            if (imin < 0 || v < vmin) { vmin = v; imin = i; }
            if (imax < 0 || v > vmax) { vmax = v; imax = i; }
        }
        unravel(imin >= 0 ? imin : 0, shape, nd, coord);
        for (int d = 0; d < nd; d++) mnp[k * nd + d] = (double)coord[d];
        unravel(imax >= 0 ? imax : 0, shape, nd, coord);
        for (int d = 0; d < nd; d++) mxp[k * nd + d] = (double)coord[d];
        if (array_idx) { mn[k] = vmin; mx[k] = vmax; }
        else { fn_result_num(&res[0], vmin); fn_result_num(&res[1], vmax); }
    }
    fn_free_doubles(in, ti); if (lab) fn_free_doubles(lab, tl); if (idx) fn_free_doubles(idx, it_ix);
    return TSR_OK;
}

/* find_objects(input, max_label=0): the bounding box of each label 1..L as an (L, nd, 2) array of [start, stop).
   Absent labels get an all-zero box (scipy returns None; fixtures use dense labels). */
static int r_find_objects(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("find_objects: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("find_objects: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t *shape = args[0].arr.shape;
    int64_t N = 1; for (int d = 0; d < nd; d++) N *= shape[d] > 0 ? shape[d] : 0;
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    int64_t L = arg_int(nargs > 1 ? &args[1] : NULL, 0);
    if (L <= 0) { L = 0; for (int64_t i = 0; i < N; i++) { const int64_t v = (int64_t)in[i]; if (v > L) L = v; } }
    const int64_t sh[3] = { L, nd, 2 };
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 3, sh);
    int64_t *lo = (L > 0) ? (int64_t *)malloc((size_t)(L * nd) * sizeof(int64_t)) : NULL;
    int64_t *hi = (L > 0) ? (int64_t *)malloc((size_t)(L * nd) * sizeof(int64_t)) : NULL;
    uint8_t *present = (L > 0) ? (uint8_t *)calloc((size_t)(L > 0 ? L : 1), 1) : NULL;
    if (!out || (L > 0 && (!lo || !hi || !present))) { fn_free_doubles(in, ti); free(lo); free(hi); free(present); return TSR_ENOMEM; }
    int64_t coord[8];
    for (int64_t i = 0; i < N; i++) {
        const int64_t l = (int64_t)in[i];
        if (l < 1 || l > L) continue;
        unravel(i, shape, nd, coord);
        const int64_t base = (l - 1) * nd;
        if (!present[l - 1]) { for (int d = 0; d < nd; d++) { lo[base + d] = coord[d]; hi[base + d] = coord[d]; } present[l - 1] = 1; }
        else for (int d = 0; d < nd; d++) { if (coord[d] < lo[base + d]) lo[base + d] = coord[d]; if (coord[d] > hi[base + d]) hi[base + d] = coord[d]; }
    }
    for (int64_t l = 0; l < L; l++) for (int d = 0; d < nd; d++) {
        out[(l * nd + d) * 2 + 0] = present[l] ? (double)lo[l * nd + d] : 0.0;
        out[(l * nd + d) * 2 + 1] = present[l] ? (double)(hi[l * nd + d] + 1) : 0.0;
    }
    fn_free_doubles(in, ti); free(lo); free(hi); free(present);
    return TSR_OK;
}

/* exact Euclidean distance transform, one axis (Felzenszwalb & Huttenlocher lower-envelope of parabolas):
   d[q] = min_p ( f[p] + (w*(q-p))^2 ). f carries the squared distance accumulated along the earlier axes. */
static void edt_line(const double *f, double *d, int64_t n, const void *vc)
{
    const double w = *(const double *)vc, w2 = w * w;
    int64_t *v = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    double *z = (double *)malloc((size_t)(n + 1) * sizeof(double));
    if (!v || !z) { free(v); free(z); for (int64_t i = 0; i < n; i++) d[i] = f[i]; return; }
    int64_t k = 0;
    v[0] = 0; z[0] = -INFINITY; z[1] = INFINITY;
    for (int64_t q = 1; q < n; q++) {
        double s;
        for (;;) {
            s = ((f[q] + w2 * (double)(q * q)) - (f[v[k]] + w2 * (double)(v[k] * v[k]))) / (2.0 * w2 * (double)(q - v[k]));
            if (s <= z[k]) k--; else break;   /* z[0] = -inf stops this at k = 0 */
        }
        k++; v[k] = q; z[k] = s; z[k + 1] = INFINITY;
    }
    k = 0;
    for (int64_t q = 0; q < n; q++) {
        while (z[k + 1] < (double)q) k++;
        const double dq = w * (double)(q - v[k]);
        d[q] = dq * dq + f[v[k]];
    }
    free(v); free(z);
}

/* distance_transform_edt(input, sampling=None): exact Euclidean distance from each nonzero element to the nearest
   zero element. Separable squared-distance transform along every axis, then sqrt. return_distances only. */
static int r_distance_edt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("distance_transform_edt: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("distance_transform_edt: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t *shape = args[0].arr.shape;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= shape[d] > 0 ? shape[d] : 0;
    double samp[8];
    if (nargs > 1 && args[1].kind != 0) { int rc = perax_d(&args[1], nd, samp); if (rc < 0) return rc; }
    else for (int d = 0; d < nd; d++) samp[d] = 1.0;
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *g = (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double));
    double *g2 = g ? (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double)) : NULL;
    double *out = g2 ? (double *)fn_result_array(&res[0], TSR_F64, nd, shape) : NULL;
    if (!out) { fn_free_doubles(in, ti); free(g); free(g2); return TSR_ENOMEM; }
    for (int64_t i = 0; i < total; i++) g[i] = (in[i] != 0.0) ? 1e18 : 0.0;   /* foreground far, background 0 */
    double *cur = g, *nxt = g2; int rc = TSR_OK;
    for (int ax = 0; ax < nd && rc == 0; ax++) {
        rc = filter_lines(cur, nxt, shape, nd, ax, edt_line, &samp[ax]);
        double *t = cur; cur = nxt; nxt = t;
    }
    if (rc == 0) for (int64_t i = 0; i < total; i++) out[i] = sqrt(cur[i]);
    fn_free_doubles(in, ti); free(g); free(g2);
    return rc;
}

/* brute-force distance to the nearest zero under a metric (exact for euclidean/taxicab/chessboard); this matches
   distance_transform_bf and, for the integer metrics, distance_transform_cdt. O(N * #background). */
enum { METRIC_EUCLID, METRIC_TAXI, METRIC_CHESS };
static int metric_code(const tsr_arg *a, int dflt)
{
    if (!a || a->kind != 2 || !a->str) return dflt;
    if (strcmp(a->str, "euclidean") == 0) return METRIC_EUCLID;
    if (strcmp(a->str, "taxicab") == 0 || strcmp(a->str, "cityblock") == 0 || strcmp(a->str, "manhattan") == 0) return METRIC_TAXI;
    if (strcmp(a->str, "chessboard") == 0) return METRIC_CHESS;
    return dflt;
}

static int bf_distance(const tsr_arg *input, int metric, const double *samp, tsr_result *res)
{
    const int32_t nd = input->arr.ndim;
    const int64_t *shape = input->arr.shape;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= shape[d] > 0 ? shape[d] : 0;
    int64_t ti; double *in = fn_arg_doubles(input, &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(res, TSR_F64, nd, shape);
    int64_t *bg = (int64_t *)malloc((size_t)(total > 0 ? total : 1) * sizeof(int64_t));
    if (!out || !bg) { fn_free_doubles(in, ti); free(bg); return TSR_ENOMEM; }
    int64_t nbg = 0;
    for (int64_t i = 0; i < total; i++) if (in[i] == 0.0) bg[nbg++] = i;
    int64_t ci[8], cj[8];
    for (int64_t i = 0; i < total; i++) {
        if (in[i] == 0.0) { out[i] = 0.0; continue; }
        unravel(i, shape, nd, ci);
        double best = INFINITY;
        for (int64_t b = 0; b < nbg; b++) {
            unravel(bg[b], shape, nd, cj);
            double dist;
            if (metric == METRIC_EUCLID) { double s = 0; for (int d = 0; d < nd; d++) { const double dd = samp[d] * (double)(ci[d] - cj[d]); s += dd * dd; } dist = sqrt(s); }
            else if (metric == METRIC_TAXI) { double s = 0; for (int d = 0; d < nd; d++) s += fabs(samp[d] * (double)(ci[d] - cj[d])); dist = s; }
            else { double m = 0; for (int d = 0; d < nd; d++) { const double dd = fabs(samp[d] * (double)(ci[d] - cj[d])); if (dd > m) m = dd; } dist = m; }
            if (dist < best) best = dist;
        }
        out[i] = best;
    }
    fn_free_doubles(in, ti); free(bg);
    return TSR_OK;
}

/* distance_transform_bf(input, metric='euclidean', sampling=None): exact distance to the nearest zero. */
static int r_distance_bf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("distance_transform_bf: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("distance_transform_bf: 1..8 dimensions"); return TSR_EDIM; }
    const int metric = metric_code(nargs > 1 ? &args[1] : NULL, METRIC_EUCLID);
    double samp[8];
    if (nargs > 2 && args[2].kind != 0) { int rc = perax_d(&args[2], nd, samp); if (rc < 0) return rc; }
    else for (int d = 0; d < nd; d++) samp[d] = 1.0;
    return bf_distance(&args[0], metric, samp, &res[0]);
}

/* distance_transform_cdt(input, metric='chessboard'): integer chamfer distance (exact for chessboard/taxicab). */
static int r_distance_cdt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("distance_transform_cdt: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("distance_transform_cdt: 1..8 dimensions"); return TSR_EDIM; }
    const int metric = metric_code(nargs > 1 ? &args[1] : NULL, METRIC_CHESS);
    double samp[8]; for (int d = 0; d < nd; d++) samp[d] = 1.0;
    return bf_distance(&args[0], metric, samp, &res[0]);
}

/* fourier_gaussian/uniform/shift: multiply an already-FFT'd (complex128, or real treated as complex) input by the
   Fourier transform of the respective kernel. The filter is separable (a product of per-axis 1-D factors): gaussian
   exp(-2*pi^2*sigma^2*f^2) and uniform sinc(size*f) are real, shift exp(-2*pi*i*shift*f) is complex. f is numpy's
   fftfreq (n=-1 full complex transform only). Result is complex128. */
enum { FOUR_GAUSS, FOUR_UNIF, FOUR_SHIFT };
static const int FR_GAUSS = FOUR_GAUSS, FR_UNIF = FOUR_UNIF, FR_SHIFT = FOUR_SHIFT;

static int r_fourier(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int kind = ctx ? *(const int *)ctx : FOUR_GAUSS;
    if (args[0].kind != 3) { fn_set_error("fourier filter: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("fourier filter: 1..8 dimensions"); return TSR_EDIM; }
    const int32_t idt = args[0].arr.dtype;
    if (idt != TSR_C128 && idt != TSR_F64) { fn_set_error("fourier filter: a complex or real array is required"); return TSR_EARG; }
    const int64_t nparam = arg_int(nargs > 2 ? &args[2] : NULL, -1);
    if (nparam >= 0) { fn_set_error("fourier filter: only n=-1 (a full complex transform) is supported"); return TSR_EARG; }
    const int64_t *shape = args[0].arr.shape;
    double par[8]; int rc = perax_d(&args[1], nd, par); if (rc < 0) return rc;
    double *fre[8] = {0}, *fim[8] = {0};
    for (int d = 0; d < nd; d++) {
        const int64_t Nn = shape[d];
        fre[d] = (double *)malloc((size_t)(Nn > 0 ? Nn : 1) * sizeof(double));
        fim[d] = (double *)malloc((size_t)(Nn > 0 ? Nn : 1) * sizeof(double));
        if (!fre[d] || !fim[d]) { for (int e = 0; e <= d; e++) { free(fre[e]); free(fim[e]); } return TSR_ENOMEM; }
        for (int64_t k = 0; k < Nn; k++) {
            const double f = (k <= (Nn - 1) / 2) ? (double)k / (double)Nn : (double)(k - Nn) / (double)Nn;
            double re = 1.0, im = 0.0;
            if (kind == FOUR_GAUSS) { re = exp(-2.0 * M_PI * M_PI * par[d] * par[d] * f * f); }
            else if (kind == FOUR_UNIF) { const double t = M_PI * par[d] * f; re = (t == 0.0) ? 1.0 : sin(t) / t; }
            else { const double ph = -2.0 * M_PI * par[d] * f; re = cos(ph); im = sin(ph); }
            fre[d][k] = re; fim[d][k] = im;
        }
    }
    int64_t total = 1, ostride[8]; for (int d = 0; d < nd; d++) total *= shape[d] > 0 ? shape[d] : 0;
    ostride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) ostride[d] = ostride[d + 1] * shape[d + 1];
    double *out = (double *)fn_result_array(&res[0], TSR_C128, nd, shape);   /* interleaved re, im */
    if (!out) { for (int d = 0; d < nd; d++) { free(fre[d]); free(fim[d]); } return TSR_ENOMEM; }
    const char *base = (const char *)args[0].arr.data + args[0].arr.offset;
    int64_t coord[8];
    for (int64_t oi = 0; oi < total; oi++) {
        int64_t t = oi;
        for (int d = 0; d < nd; d++) { coord[d] = t / ostride[d]; t %= ostride[d]; }
        double fr = 1.0, fi = 0.0;
        for (int d = 0; d < nd; d++) { const double a = fre[d][coord[d]], b = fim[d][coord[d]]; const double nr = fr * a - fi * b, ni = fr * b + fi * a; fr = nr; fi = ni; }
        const char *p = base; for (int d = 0; d < nd; d++) p += coord[d] * args[0].arr.strides[d];
        double ir, ii;
        if (idt == TSR_C128) { ir = ((const double *)p)[0]; ii = ((const double *)p)[1]; }
        else { ir = *(const double *)p; ii = 0.0; }
        out[2 * oi + 0] = ir * fr - ii * fi;
        out[2 * oi + 1] = ir * fi + ii * fr;
    }
    for (int d = 0; d < nd; d++) { free(fre[d]); free(fim[d]); }
    return TSR_OK;
}

extern void tsr_special_j1(const void *ctx, const double *in, double *out);   /* scipy's Bessel J1 (xsf) */
extern void tsr_special_cosdg(const void *ctx, const double *in, double *out);  /* cosine of degrees (exact at 90k) */
extern void tsr_special_sindg(const void *ctx, const double *in, double *out);

/* fourier_ellipsoid (rank 1, 2 or 3): multiply an FFT by the transform of an ellipsoid. Radial (not separable):
   per-axis p_d(k) = pi*size_d*fftfreq(shape[d])[k], r = sqrt(sum_d p_d^2); profile 1-D sin(t)/t, 2-D 2*J1(r)/r,
   3-D 3*(sin r - r cos r)/r^3 (all 1 at the origin). The filter is real; result is complex128. */
static int r_fourier_ellipsoid(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("fourier_ellipsoid: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 3) { fn_set_error("fourier_ellipsoid: implemented for rank 1, 2 or 3"); return TSR_EDIM; }
    const int32_t idt = args[0].arr.dtype;
    if (idt != TSR_C128 && idt != TSR_F64) { fn_set_error("fourier_ellipsoid: a complex or real array is required"); return TSR_EARG; }
    const int64_t nparam = arg_int(nargs > 2 ? &args[2] : NULL, -1);
    if (nparam >= 0) { fn_set_error("fourier_ellipsoid: only n=-1 (a full complex transform) is supported"); return TSR_EARG; }
    const int64_t *shape = args[0].arr.shape;
    double sz[8]; int rc = perax_d(&args[1], nd, sz); if (rc < 0) return rc;
    double *p[8] = {0};
    for (int d = 0; d < nd; d++) {
        const int64_t Nn = shape[d];
        p[d] = (double *)malloc((size_t)(Nn > 0 ? Nn : 1) * sizeof(double));
        if (!p[d]) { for (int e = 0; e <= d; e++) free(p[e]); return TSR_ENOMEM; }
        for (int64_t k = 0; k < Nn; k++) {
            const double f = (k <= (Nn - 1) / 2) ? (double)k / (double)Nn : (double)(k - Nn) / (double)Nn;
            p[d][k] = M_PI * sz[d] * f;
        }
    }
    int64_t total = 1, ostride[8]; for (int d = 0; d < nd; d++) total *= shape[d] > 0 ? shape[d] : 0;
    ostride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) ostride[d] = ostride[d + 1] * shape[d + 1];
    double *out = (double *)fn_result_array(&res[0], TSR_C128, nd, shape);
    if (!out) { for (int d = 0; d < nd; d++) free(p[d]); return TSR_ENOMEM; }
    const char *bp = (const char *)args[0].arr.data + args[0].arr.offset;
    int64_t coord[8];
    for (int64_t oi = 0; oi < total; oi++) {
        int64_t t = oi; for (int d = 0; d < nd; d++) { coord[d] = t / ostride[d]; t %= ostride[d]; }
        double filt;
        if (nd == 1) { const double tt = p[0][coord[0]]; filt = (tt != 0.0) ? sin(tt) / tt : 1.0; }
        else {
            double r2 = 0; for (int d = 0; d < nd; d++) { const double v = p[d][coord[d]]; r2 += v * v; }
            const double r = sqrt(r2);
            if (nd == 2) { if (r > 0.0) { double jj; tsr_special_j1(NULL, &r, &jj); filt = 2.0 * jj / r; } else filt = 1.0; }
            else { filt = (r > 0.0) ? 3.0 * (sin(r) - r * cos(r)) / (r * r * r) : 1.0; }
        }
        const char *q = bp; for (int d = 0; d < nd; d++) q += coord[d] * args[0].arr.strides[d];
        double ir, ii;
        if (idt == TSR_C128) { ir = ((const double *)q)[0]; ii = ((const double *)q)[1]; }
        else { ir = *(const double *)q; ii = 0.0; }
        out[2 * oi + 0] = ir * filt;
        out[2 * oi + 1] = ii * filt;
    }
    for (int d = 0; d < nd; d++) free(p[d]);
    return TSR_OK;
}

/* scipy's ni_interpolation.c map_coordinate: fold an out-of-bounds (float) coordinate in-bounds per mode (constant
   returns -1 to signal "out of bounds"). Matches scipy exactly so the interpolation boundaries agree. */
static double map_coord(double in, int64_t len, int mode)
{
    if (len <= 1) return 0.0;
    if (in < 0) {
        switch (mode) {
        case ND_MIRROR: { const double sz2 = 2.0 * len - 2; in = fmod(in, sz2); in = in <= 1 - len ? in + sz2 : -in; break; }
        case ND_WRAP: { const int64_t sz = len - 1; in = fmod(in, (double)sz) + (double)sz; break; }
        case ND_NEAREST: in = 0; break;
        case ND_CONSTANT: in = -1; break;
        default: { const double sz2 = 2.0 * len; if (in < -sz2) in = fmod(in, sz2); in = in < -len ? in + sz2 : (in > -1e-15 ? 1e-15 : -in) - 1; break; }  /* reflect */
        }
    } else if (in > len - 1) {
        switch (mode) {
        case ND_MIRROR: { const double sz2 = 2.0 * len - 2; in = fmod(in, sz2); if (in > len - 1) in = sz2 - in; break; }
        case ND_WRAP: { const int64_t sz = len - 1; in = fmod(in, (double)sz); break; }
        case ND_NEAREST: in = (double)(len - 1); break;
        case ND_CONSTANT: in = -1; break;
        default: { const double sz2 = 2.0 * len; in = fmod(in, sz2); if (in >= len) in = sz2 - in - 1; break; }  /* reflect */
        }
    }
    return in;
}

/* B-spline interpolation weights (scipy ni_splines.c get_spline_interpolation_weights), orders 0..5. Fills
   order+1 weights for the fractional position of `x` (the function reduces x to the delta to the middle knot). */
static void spline_weights(double x, int order, double *w)
{
    if (order == 0) { w[0] = 1.0; return; }
    x -= floor((order & 1) ? x : x + 0.5);
    double y = x, z = 1.0 - x, t;
    switch (order) {
    case 1: w[0] = 1.0 - x; break;
    case 2: w[1] = 0.75 - x * x; y = 0.5 - x; w[0] = 0.5 * y * y; break;
    case 3: w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0; w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0; w[0] = z * z * z / 6.0; break;
    case 4:
        t = x * x; w[2] = t * (t * 0.25 - 0.625) + 115.0 / 192.0;
        y = 1.0 + x; w[1] = y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0;
        w[3] = z * (z * (z * (5.0 - z) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0;
        y = 0.5 - x; t = y * y; w[0] = t * t / 24.0; break;
    case 5:
        t = y * y; w[2] = t * (t * (0.25 - y / 12.0) - 0.5) + 0.55;
        t = z * z; w[3] = t * (t * (0.25 - z / 12.0) - 0.5) + 0.55;
        y += 1.0; w[1] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
        z += 1.0; w[4] = z * (z * (z * (z * (z / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
        y = 1.0 - x; t = y * y; w[0] = y * t * t / 120.0; break;
    }
    w[order] = 1.0; for (int i = 0; i < order; i++) w[order] -= w[i];
}

/* spline prefilter poles (scipy ni_splines.c), returns the pole count. */
static int spline_poles(int order, double *p)
{
    switch (order) {
    case 2: p[0] = -0.171572875253809902396622551580603843; return 1;
    case 3: p[0] = -0.267949192431122706472553658494127633; return 1;
    case 4: p[0] = -0.361341225900220177092212841325675255; p[1] = -0.013725429297339121360331226939128204; return 2;
    case 5: p[0] = -0.430575347099973791851434783493520110; p[1] = -0.043096288203264653822712376822550182; return 2;
    default: return 0;
    }
}

typedef struct { int order; int mode; } spf_ctx;

/* one line of the spline prefilter: apply the gain and the causal/anticausal pole recursions. Boundary init is
   mirror for mirror/constant/wrap, reflect for reflect/nearest (scipy apply_filter's mode map). */
static void spf_line(const double *in, double *c, int64_t n, const void *vctx)
{
    const spf_ctx *x = (const spf_ctx *)vctx;
    for (int64_t i = 0; i < n; i++) c[i] = in[i];
    if (n <= 1) return;
    double poles[2]; const int np = spline_poles(x->order, poles);
    double gain = 1.0; for (int k = 0; k < np; k++) { const double z = poles[k]; gain *= (1.0 - z) * (1.0 - 1.0 / z); }
    for (int64_t i = 0; i < n; i++) c[i] *= gain;
    const int reflect = (x->mode == ND_REFLECT || x->mode == ND_NEAREST);
    for (int k = 0; k < np; k++) {
        const double z = poles[k];
        if (reflect) {
            double z_i = z; const double z_n = pow(z, (double)n); double sum = c[0] + z_n * c[n - 1];
            for (int64_t i = 1; i < n; i++) { sum += z_i * (c[i] + z_n * c[n - 1 - i]); z_i *= z; }
            c[0] += sum * z / (1.0 - z_n * z_n);
        } else {
            double z_i = z; const double z_n_1 = pow(z, (double)(n - 1));
            c[0] = c[0] + z_n_1 * c[n - 1];
            for (int64_t i = 1; i < n - 1; i++) { c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i]); z_i *= z; }
            c[0] /= 1.0 - z_n_1 * z_n_1;
        }
        for (int64_t i = 1; i < n; i++) c[i] += z * c[i - 1];
        if (reflect) c[n - 1] *= z / (z - 1.0);
        else c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);
        for (int64_t i = n - 2; i >= 0; i--) c[i] = z * (c[i + 1] - c[i]);
    }
}

/* spline prefilter over every axis, in place (order >= 2). */
static int spline_prefilter_nd(double *buf, const int64_t *shape, int32_t nd, int order, int mode)
{
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= shape[d] > 0 ? shape[d] : 0;
    if (order < 2 || total == 0) return TSR_OK;
    double *tmp = (double *)malloc((size_t)total * sizeof(double)); if (!tmp) return TSR_ENOMEM;
    spf_ctx ctx = { order, mode };
    double *cur = buf, *nxt = tmp; int rc = TSR_OK;
    for (int ax = 0; ax < nd && rc == 0; ax++) { rc = filter_lines(cur, nxt, shape, nd, ax, spf_line, &ctx); double *t = cur; cur = nxt; nxt = t; }
    if (rc == 0 && cur != buf) memcpy(buf, cur, (size_t)total * sizeof(double));
    free(tmp);
    return rc;
}

/* interpolate `in` (the spline coefficients for order >= 2, else the raw input) at one coordinate vector, orders
   0..5. Non-constant modes fold the coordinate (map_coord) and each tap (bmap); constant returns cval for the
   point if a coordinate leaves [0, len-1] and uses cval for any tap that falls outside. */
static double geom_interp(const double *in, const int64_t *si, int32_t nd, const int64_t *istride,
                          const double *coord, int order, int mode, double cval, int con)
{
    double cc[8];
    for (int d = 0; d < nd; d++) {
        const double c = coord[d];
        if (con) { if (c < 0.0 || c > (double)(si[d] - 1)) return cval; cc[d] = c; }
        else cc[d] = map_coord(c, si[d], mode);
    }
    if (order == 0) {
        int64_t flat = 0;
        for (int d = 0; d < nd; d++) {
            int64_t idx = (int64_t)floor(cc[d] + 0.5);
            idx = con ? idx : bmap(idx, si[d], mode);
            if (con) { if (idx < 0 || idx >= si[d]) return cval; } else if (idx < 0) idx = 0;
            flat += idx * istride[d];
        }
        return in[flat];
    }
    int64_t start[8]; double wt[8][6]; int64_t idxc[8];
    const int nt = order + 1;
    for (int d = 0; d < nd; d++) {
        start[d] = (int64_t)floor((order & 1) ? cc[d] : cc[d] + 0.5) - order / 2;
        spline_weights(cc[d], order, wt[d]);
        idxc[d] = 0;
    }
    double acc = 0.0;
    for (;;) {
        double w = 1.0; int64_t flat = 0, oob = 0;
        for (int d = 0; d < nd; d++) {
            w *= wt[d][idxc[d]];
            int64_t t = start[d] + idxc[d];
            if (con) { if (t < 0 || t >= si[d]) { oob = 1; } }
            else { t = bmap(t, si[d], mode); if (t < 0) t = 0; }
            if (!oob) flat += t * istride[d];
        }
        acc += w * (oob ? cval : in[flat]);
        int d = nd - 1; while (d >= 0) { if (++idxc[d] < nt) break; idxc[d] = 0; d--; }
        if (d < 0) break;
    }
    return acc;
}

/* map_coordinates(input, coordinates, order, mode, cval): interpolate `input` at the given coordinates.
   coordinates has shape (input.ndim, ...output); the output takes the trailing shape. Orders 0 (nearest) and 1
   (linear) for now (higher orders need the B-spline prefilter). Non-constant modes fold the coordinate (map_coord);
   constant uses cval for taps that fall outside. */
static int r_map_coordinates(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("map_coordinates: input and coordinates arrays are required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("map_coordinates: 1..8 dimensions"); return TSR_EDIM; }
    if (args[1].arr.ndim < 1 || args[1].arr.shape[0] != nd) { fn_set_error("map_coordinates: coordinates must have shape (input.ndim, ...)"); return TSR_EARG; }
    const int64_t order = arg_int(nargs > 2 ? &args[2] : NULL, 3);
    if (order < 0 || order > 5) { fn_set_error("map_coordinates: order must be 0..5"); return TSR_EARG; }
    int mode; int rc = nd_mode(nargs > 3 ? &args[3] : NULL, &mode);
    if (rc < 0) return rc;
    const double cval = (nargs > 4 && args[4].kind == 1) ? ((args[4].flags & 1) ? (double)args[4].ival : args[4].num) : 0.0;
    const int64_t *si = args[0].arr.shape;
    int64_t istride[8]; istride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) istride[d] = istride[d + 1] * si[d + 1];
    const int32_t ond = args[1].arr.ndim - 1;
    int64_t osh[8], M = 1; for (int d = 0; d < ond; d++) { osh[d] = args[1].arr.shape[d + 1]; M *= osh[d]; }
    int64_t ti, tc; double *in = fn_arg_doubles(&args[0], &ti);
    double *co = in ? fn_arg_doubles(&args[1], &tc) : NULL;
    double *out = co ? (double *)fn_result_array(&res[0], TSR_F64, ond, ond > 0 ? osh : NULL) : NULL;
    if (!out && !(ond == 0 && co)) { fn_free_doubles(in, ti); if (co) fn_free_doubles(co, tc); return TSR_ENOMEM; }
    double *dst = (ond == 0) ? &res[0].num : out;
    if (ond == 0) res[0].kind = 1;
    const int con = (mode == ND_CONSTANT);
    const double *src = in; double *coeff = NULL;
    if (order >= 2) {
        int64_t itot = 1; for (int d = 0; d < nd; d++) itot *= si[d] > 0 ? si[d] : 0;
        coeff = (double *)malloc((size_t)(itot > 0 ? itot : 1) * sizeof(double));
        if (!coeff) { fn_free_doubles(in, ti); fn_free_doubles(co, tc); return TSR_ENOMEM; }
        memcpy(coeff, in, (size_t)itot * sizeof(double));
        const int prc = spline_prefilter_nd(coeff, si, nd, (int)order, mode);
        if (prc != 0) { free(coeff); fn_free_doubles(in, ti); fn_free_doubles(co, tc); return prc; }
        src = coeff;
    }
    double coord[8];
    for (int64_t m = 0; m < M; m++) {
        for (int d = 0; d < nd; d++) coord[d] = co[(int64_t)d * M + m];
        dst[m] = geom_interp(src, si, nd, istride, coord, (int)order, mode, cval, con);
    }
    free(coeff); fn_free_doubles(in, ti); fn_free_doubles(co, tc);
    return TSR_OK;
}

/* shift(input, shift, order, mode, cval): output[o] = interp(input, o - shift). order 0/1. */
static int r_shift(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("shift: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("shift: 1..8 dimensions"); return TSR_EDIM; }
    double sh[8]; int rc = perax_d(&args[1], nd, sh); if (rc < 0) return rc;
    const int64_t order = arg_int(nargs > 2 ? &args[2] : NULL, 3);
    if (order < 0 || order > 5) { fn_set_error("shift: order must be 0..5"); return TSR_EARG; }
    int mode; if ((rc = nd_mode(nargs > 3 ? &args[3] : NULL, &mode)) < 0) return rc;
    const double cval = (nargs > 4 && args[4].kind == 1) ? ((args[4].flags & 1) ? (double)args[4].ival : args[4].num) : 0.0;
    const int64_t *si = args[0].arr.shape;
    int64_t istride[8], total = 1; for (int d = 0; d < nd; d++) total *= si[d] > 0 ? si[d] : 0;
    istride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) istride[d] = istride[d + 1] * si[d + 1];
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, si);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    const int con = (mode == ND_CONSTANT);
    const double *src = in; double *coeff = NULL;
    if (order >= 2) {
        coeff = (double *)malloc((size_t)(total > 0 ? total : 1) * sizeof(double));
        if (!coeff) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
        memcpy(coeff, in, (size_t)total * sizeof(double));
        const int prc = spline_prefilter_nd(coeff, si, nd, (int)order, mode);
        if (prc != 0) { free(coeff); fn_free_doubles(in, ti); return prc; }
        src = coeff;
    }
    double coord[8];
    for (int64_t o = 0; o < total; o++) {
        int64_t t = o; for (int d = 0; d < nd; d++) { coord[d] = (double)(t / istride[d]) - sh[d]; t %= istride[d]; }
        out[o] = geom_interp(src, si, nd, istride, coord, (int)order, mode, cval, con);
    }
    free(coeff); fn_free_doubles(in, ti);
    return TSR_OK;
}

/* affine_transform(input, matrix, offset=0, output_shape=None, order, mode, cval): output[o] = interp(input,
   matrix @ o + offset). matrix is (nd, nd) full or (nd,) diagonal. order 0/1. */
static int r_affine_transform(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("affine_transform: input and matrix arrays are required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("affine_transform: 1..8 dimensions"); return TSR_EDIM; }
    const int mat2d = args[1].arr.ndim == 2;
    double off[8];
    if (nargs > 2 && args[2].kind == 3) { int rc = perax_d(&args[2], nd, off); if (rc < 0) return rc; }
    else { const double v = (nargs > 2 && args[2].kind == 1) ? ((args[2].flags & 1) ? (double)args[2].ival : args[2].num) : 0.0; for (int d = 0; d < nd; d++) off[d] = v; }
    int64_t osh[8]; int32_t ond = nd;
    if (nargs > 3 && args[3].kind == 3) { int64_t to; double *os = fn_arg_doubles(&args[3], &to); if (!os) return TSR_ENOMEM; for (int d = 0; d < nd; d++) osh[d] = (int64_t)os[d]; fn_free_doubles(os, to); }
    else for (int d = 0; d < nd; d++) osh[d] = args[0].arr.shape[d];
    const int64_t order = arg_int(nargs > 4 ? &args[4] : NULL, 3);
    if (order < 0 || order > 5) { fn_set_error("affine_transform: order must be 0..5"); return TSR_EARG; }
    int mode; int rc = nd_mode(nargs > 5 ? &args[5] : NULL, &mode); if (rc < 0) return rc;
    const double cval = (nargs > 6 && args[6].kind == 1) ? ((args[6].flags & 1) ? (double)args[6].ival : args[6].num) : 0.0;
    const int64_t *si = args[0].arr.shape;
    int64_t istride[8]; istride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) istride[d] = istride[d + 1] * si[d + 1];
    int64_t ostride[8], otot = 1; for (int d = 0; d < nd; d++) otot *= osh[d] > 0 ? osh[d] : 0;
    ostride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) ostride[d] = ostride[d + 1] * osh[d + 1];
    int64_t ti, tm; double *in = fn_arg_doubles(&args[0], &ti);
    double *mat = in ? fn_arg_doubles(&args[1], &tm) : NULL;
    double *out = mat ? (double *)fn_result_array(&res[0], TSR_F64, ond, osh) : NULL;
    if (!out) { fn_free_doubles(in, ti); if (mat) fn_free_doubles(mat, tm); return TSR_ENOMEM; }
    const int con = (mode == ND_CONSTANT);
    const double *insrc = in; double *coeff = NULL;
    if (order >= 2) {
        int64_t itot = 1; for (int d = 0; d < nd; d++) itot *= si[d] > 0 ? si[d] : 0;
        coeff = (double *)malloc((size_t)(itot > 0 ? itot : 1) * sizeof(double));
        if (!coeff) { fn_free_doubles(in, ti); fn_free_doubles(mat, tm); return TSR_ENOMEM; }
        memcpy(coeff, in, (size_t)itot * sizeof(double));
        const int prc = spline_prefilter_nd(coeff, si, nd, (int)order, mode);
        if (prc != 0) { free(coeff); fn_free_doubles(in, ti); fn_free_doubles(mat, tm); return prc; }
        insrc = coeff;
    }
    double oc[8], coord[8];
    for (int64_t o = 0; o < otot; o++) {
        int64_t t = o; for (int d = 0; d < nd; d++) { oc[d] = (double)(t / ostride[d]); t %= ostride[d]; }
        for (int i = 0; i < nd; i++) {
            double c = off[i];
            if (mat2d) for (int j = 0; j < nd; j++) c += mat[(int64_t)i * nd + j] * oc[j];
            else c += mat[i] * oc[i];
            coord[i] = c;
        }
        out[o] = geom_interp(insrc, si, nd, istride, coord, (int)order, mode, cval, con);
    }
    free(coeff); fn_free_doubles(in, ti); fn_free_doubles(mat, tm);
    return TSR_OK;
}

/* zoom(input, zoom, order, mode, cval): resample to round(shape*zoom); coord = o * (in-1)/(out-1) (grid_mode
   False). order 0/1. */
static int r_zoom(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("zoom: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("zoom: 1..8 dimensions"); return TSR_EDIM; }
    double zm[8]; int rc = perax_d(&args[1], nd, zm); if (rc < 0) return rc;
    const int64_t order = arg_int(nargs > 2 ? &args[2] : NULL, 3);
    if (order < 0 || order > 5) { fn_set_error("zoom: order must be 0..5"); return TSR_EARG; }
    int mode; if ((rc = nd_mode(nargs > 3 ? &args[3] : NULL, &mode)) < 0) return rc;
    const double cval = (nargs > 4 && args[4].kind == 1) ? ((args[4].flags & 1) ? (double)args[4].ival : args[4].num) : 0.0;
    const int64_t *si = args[0].arr.shape;
    int64_t osh[8], otot = 1; double scale[8];
    for (int d = 0; d < nd; d++) { osh[d] = (int64_t)floor((double)si[d] * zm[d] + 0.5); if (osh[d] < 1) osh[d] = 1; otot *= osh[d]; scale[d] = osh[d] > 1 ? (double)(si[d] - 1) / (double)(osh[d] - 1) : 0.0; }
    int64_t istride[8], ostride[8]; istride[nd - 1] = 1; ostride[nd - 1] = 1;
    for (int d = nd - 2; d >= 0; d--) { istride[d] = istride[d + 1] * si[d + 1]; ostride[d] = ostride[d + 1] * osh[d + 1]; }
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, osh);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    const int con = (mode == ND_CONSTANT);
    const double *src = in; double *coeff = NULL;
    if (order >= 2) {
        int64_t itot = 1; for (int d = 0; d < nd; d++) itot *= si[d] > 0 ? si[d] : 0;
        coeff = (double *)malloc((size_t)(itot > 0 ? itot : 1) * sizeof(double));
        if (!coeff) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
        memcpy(coeff, in, (size_t)itot * sizeof(double));
        const int prc = spline_prefilter_nd(coeff, si, nd, (int)order, mode);
        if (prc != 0) { free(coeff); fn_free_doubles(in, ti); return prc; }
        src = coeff;
    }
    double coord[8];
    for (int64_t o = 0; o < otot; o++) {
        int64_t t = o; for (int d = 0; d < nd; d++) { coord[d] = (double)(t / ostride[d]) * scale[d]; t %= ostride[d]; }
        out[o] = geom_interp(src, si, nd, istride, coord, (int)order, mode, cval, con);
    }
    free(coeff); fn_free_doubles(in, ti);
    return TSR_OK;
}

/* rotate(input, angle, axes=(1,0), reshape=True, order, mode, cval): rotate in the plane of two axes by `angle`
   degrees. Builds scipy's rotation matrix [[c,s],[-s,c]], the reshaped output plane, and the centre offset, then
   interpolates (order 0/1). Axes outside the plane pass through. */
static int r_rotate(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || nargs < 2 || args[1].kind != 1) { fn_set_error("rotate: input array and angle required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 2 || nd > 8) { fn_set_error("rotate: 2..8 dimensions"); return TSR_EDIM; }
    const double angle = (args[1].flags & 1) ? (double)args[1].ival : args[1].num;
    int a0 = 1, a1 = 0;
    if (nargs > 2 && args[2].kind == 3 && args[2].arr.ndim == 1 && args[2].arr.shape[0] == 2) {
        int64_t ta; double *ax = fn_arg_doubles(&args[2], &ta); if (!ax) return TSR_ENOMEM;
        a0 = (int)ax[0]; a1 = (int)ax[1]; fn_free_doubles(ax, ta);
    }
    if (a0 < 0) a0 += nd; if (a1 < 0) a1 += nd;
    if (a0 > a1) { int t = a0; a0 = a1; a1 = t; }
    if (a0 < 0 || a1 >= nd || a0 == a1) { fn_set_error("rotate: axes are out of range"); return TSR_EARG; }
    const int reshape = !(nargs > 3 && args[3].kind == 4 && args[3].num == 0.0);
    const int64_t order = arg_int(nargs > 4 ? &args[4] : NULL, 3);
    if (order < 0 || order > 5) { fn_set_error("rotate: order must be 0..5"); return TSR_EARG; }
    int mode; int rc = nd_mode(nargs > 5 ? &args[5] : NULL, &mode); if (rc < 0) return rc;
    const double cval = (nargs > 6 && args[6].kind == 1) ? ((args[6].flags & 1) ? (double)args[6].ival : args[6].num) : 0.0;
    double c, s; tsr_special_cosdg(NULL, &angle, &c); tsr_special_sindg(NULL, &angle, &s);
    const int64_t *si = args[0].arr.shape;
    const double iy = (double)si[a0], ix = (double)si[a1];
    int64_t op0, op1;
    if (reshape) {
        const double cy[4] = {0, 0, iy, iy}, cx[4] = {0, ix, 0, ix};
        double y0 = 1e300, y1 = -1e300, x0 = 1e300, x1 = -1e300;
        for (int k = 0; k < 4; k++) {
            const double ry = c * cy[k] + s * cx[k], rx = -s * cy[k] + c * cx[k];
            if (ry < y0) y0 = ry; if (ry > y1) y1 = ry; if (rx < x0) x0 = rx; if (rx > x1) x1 = rx;
        }
        op0 = (int64_t)(y1 - y0 + 0.5); op1 = (int64_t)(x1 - x0 + 0.5);
    } else { op0 = si[a0]; op1 = si[a1]; }
    if (op0 < 1) op0 = 1; if (op1 < 1) op1 = 1;
    const double ocy = ((double)op0 - 1) / 2.0, ocx = ((double)op1 - 1) / 2.0;
    const double off0 = (iy - 1) / 2.0 - (c * ocy + s * ocx);
    const double off1 = (ix - 1) / 2.0 - (-s * ocy + c * ocx);
    int64_t osh[8]; for (int d = 0; d < nd; d++) osh[d] = si[d];
    osh[a0] = op0; osh[a1] = op1;
    int64_t istride[8], ostride[8], otot = 1;
    for (int d = 0; d < nd; d++) otot *= osh[d] > 0 ? osh[d] : 0;
    istride[nd - 1] = 1; ostride[nd - 1] = 1;
    for (int d = nd - 2; d >= 0; d--) { istride[d] = istride[d + 1] * si[d + 1]; ostride[d] = ostride[d + 1] * osh[d + 1]; }
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, osh);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    const int con = (mode == ND_CONSTANT);
    const double *src = in; double *coeff = NULL;
    if (order >= 2) {
        int64_t itot = 1; for (int d = 0; d < nd; d++) itot *= si[d] > 0 ? si[d] : 0;
        coeff = (double *)malloc((size_t)(itot > 0 ? itot : 1) * sizeof(double));
        if (!coeff) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
        memcpy(coeff, in, (size_t)itot * sizeof(double));
        const int prc = spline_prefilter_nd(coeff, si, nd, (int)order, mode);
        if (prc != 0) { free(coeff); fn_free_doubles(in, ti); return prc; }
        src = coeff;
    }
    double oc[8], coord[8];
    for (int64_t o = 0; o < otot; o++) {
        int64_t t = o; for (int d = 0; d < nd; d++) { oc[d] = (double)(t / ostride[d]); t %= ostride[d]; }
        for (int d = 0; d < nd; d++) coord[d] = oc[d];
        coord[a0] = c * oc[a0] + s * oc[a1] + off0;
        coord[a1] = -s * oc[a0] + c * oc[a1] + off1;
        out[o] = geom_interp(src, si, nd, istride, coord, (int)order, mode, cval, con);
    }
    free(coeff); fn_free_doubles(in, ti);
    return TSR_OK;
}

/* correlate: full n-D correlation with a weights array. out[i] = sum_j w[j] * ext(i + j - center - origin),
   center[d] = weights.shape[d]//2; boundaries folded by `mode` (constant -> cval). */
static int r_correlate_nd(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("correlate: an input array and a weights array are required"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (args[1].arr.ndim != nd) { fn_set_error("correlate: weights must have the same rank as the input"); return TSR_EARG; }
    if (nd < 1 || nd > 8) { fn_set_error("correlate: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t *shape = args[0].arr.shape, *wshape = args[1].arr.shape;
    int mode; int rc = nd_mode(nargs > 2 ? &args[2] : NULL, &mode);
    if (rc < 0) return rc;
    const double cval = (nargs > 3 && args[3].kind == 1) ? ((args[3].flags & 1) ? (double)args[3].ival : args[3].num) : 0.0;
    const int64_t origin0 = arg_int(nargs > 4 ? &args[4] : NULL, 0);
    int64_t istride[8], wstride[8], center[8], total = 1, W = 1;
    for (int d = 0; d < nd; d++) { total *= shape[d]; W *= wshape[d]; center[d] = wshape[d] / 2; }
    istride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) istride[d] = istride[d + 1] * shape[d + 1];
    wstride[nd - 1] = 1; for (int d = nd - 2; d >= 0; d--) wstride[d] = wstride[d + 1] * wshape[d + 1];
    int64_t ti, tw; double *in = fn_arg_doubles(&args[0], &ti);
    double *w = in ? fn_arg_doubles(&args[1], &tw) : NULL;
    double *out = w ? (double *)fn_result_array(&res[0], TSR_F64, nd, shape) : NULL;
    if (!out) { fn_free_doubles(in, ti); if (w) fn_free_doubles(w, tw); return TSR_ENOMEM; }
    int64_t coord[8], woff[8];
    for (int64_t oi = 0; oi < total; oi++) {
        int64_t t = oi;
        for (int d = 0; d < nd; d++) { coord[d] = t / istride[d]; t %= istride[d]; }
        double acc = 0.0;
        for (int64_t wj = 0; wj < W; wj++) {
            int64_t u = wj;
            for (int d = 0; d < nd; d++) { woff[d] = u / wstride[d]; u %= wstride[d]; }
            int64_t flat = 0, oob = 0;
            for (int d = 0; d < nd; d++) {
                const int64_t sc = bmap(coord[d] + woff[d] - center[d] - origin0, shape[d], mode);
                if (sc < 0) { oob = 1; break; }
                flat += sc * istride[d];
            }
            acc += w[wj] * (oob ? cval : in[flat]);
        }
        out[oi] = acc;
    }
    fn_free_doubles(in, ti); fn_free_doubles(w, tw);
    return rc;
}

/* spline_filter1d(input, order, axis, mode): the 1-D B-spline prefilter along one axis (order < 2: a copy). */
static int r_spline_filter1d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("spline_filter1d: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("spline_filter1d: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t order = arg_int(nargs > 1 ? &args[1] : NULL, 3);
    if (order < 0 || order > 5) { fn_set_error("spline_filter1d: order must be 0..5"); return TSR_EARG; }
    int64_t ax = arg_int(nargs > 2 ? &args[2] : NULL, -1); if (ax < 0) ax += nd;
    if (ax < 0 || ax >= nd) { fn_set_error("spline_filter1d: axis out of range"); return TSR_EARG; }
    int mode; int rc = nd_mode(nargs > 3 ? &args[3] : NULL, &mode); if (rc < 0) return rc;
    const int64_t *si = args[0].arr.shape;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= si[d] > 0 ? si[d] : 0;
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, si);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    if (order < 2) { for (int64_t i = 0; i < total; i++) out[i] = in[i]; }
    else { spf_ctx c = { (int)order, mode }; rc = filter_lines(in, out, si, nd, (int)ax, spf_line, &c); }
    fn_free_doubles(in, ti);
    return rc;
}

/* spline_filter(input, order, mode): the B-spline prefilter along every axis (order < 2: a copy). */
static int r_spline_filter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("spline_filter: input must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > 8) { fn_set_error("spline_filter: 1..8 dimensions"); return TSR_EDIM; }
    const int64_t order = arg_int(nargs > 1 ? &args[1] : NULL, 3);
    if (order < 0 || order > 5) { fn_set_error("spline_filter: order must be 0..5"); return TSR_EARG; }
    int mode; int rc = nd_mode(nargs > 2 ? &args[2] : NULL, &mode); if (rc < 0) return rc;
    const int64_t *si = args[0].arr.shape;
    int64_t total = 1; for (int d = 0; d < nd; d++) total *= si[d] > 0 ? si[d] : 0;
    int64_t ti; double *in = fn_arg_doubles(&args[0], &ti); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, si);
    if (!out) { fn_free_doubles(in, ti); return TSR_ENOMEM; }
    for (int64_t i = 0; i < total; i++) out[i] = in[i];
    rc = spline_prefilter_nd(out, si, nd, (int)order, mode);
    fn_free_doubles(in, ti);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("ndimage.generate_binary_structure", 1, "rank, connectivity", "out", r_gbs, NULL, "Boolean structuring element for binary morphology (scipy.ndimage.generate_binary_structure)."),
    ROUTINE("ndimage.gaussian_filter", 1, "input, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0, radius=None", "out", r_gaussian_filter, NULL, "Multidimensional Gaussian filter, order 0 (scipy.ndimage.gaussian_filter)."),
    ROUTINE("ndimage.uniform_filter", 1, "input, size=3, mode='reflect', cval=0.0, origin=0", "out", r_uniform_filter, NULL, "Multidimensional uniform (box) filter (scipy.ndimage.uniform_filter)."),
    ROUTINE("ndimage.correlate1d", 1, "input, weights, axis=-1, mode='reflect', cval=0.0, origin=0", "out", r_correlate1d, NULL, "1-D correlation along an axis with boundary handling (scipy.ndimage.correlate1d)."),
    ROUTINE("ndimage.convolve1d", 1, "input, weights, axis=-1, mode='reflect', cval=0.0, origin=0", "out", r_correlate1d, &ND_CONV1D, "1-D convolution along an axis (scipy.ndimage.convolve1d)."),
    ROUTINE("ndimage.uniform_filter1d", 1, "input, size, axis=-1, mode='reflect', cval=0.0, origin=0", "out", r_uniform_filter1d, NULL, "1-D uniform (box) filter along an axis (scipy.ndimage.uniform_filter1d)."),
    ROUTINE("ndimage.gaussian_filter1d", 1, "input, sigma, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0, radius=None", "out", r_gaussian_filter1d, NULL, "1-D Gaussian filter along an axis, order 0 (scipy.ndimage.gaussian_filter1d)."),
    ROUTINE("ndimage.minimum_filter1d", 1, "input, size, axis=-1, mode='reflect', cval=0.0, origin=0", "out", r_minmax_filter1d, NULL, "1-D sliding-window minimum along an axis (scipy.ndimage.minimum_filter1d)."),
    ROUTINE("ndimage.maximum_filter1d", 1, "input, size, axis=-1, mode='reflect', cval=0.0, origin=0", "out", r_minmax_filter1d, &ND_MAX1D, "1-D sliding-window maximum along an axis (scipy.ndimage.maximum_filter1d)."),
    ROUTINE("ndimage.binary_dilation", 1, "input, structure=None", "out", r_binary_morph, &MORPH_DILATE, "Binary dilation of a 2-D image (scipy.ndimage.binary_dilation)."),
    ROUTINE("ndimage.binary_erosion", 1, "input, structure=None", "out", r_binary_morph, &MORPH_ERODE, "Binary erosion of a 2-D image (scipy.ndimage.binary_erosion)."),
    ROUTINE("ndimage.binary_opening", 1, "input, structure=None", "out", r_binary_openclose, &BIN_OPEN, "Binary opening: erosion then dilation of a 2-D image (scipy.ndimage.binary_opening)."),
    ROUTINE("ndimage.binary_closing", 1, "input, structure=None", "out", r_binary_openclose, NULL, "Binary closing: dilation then erosion of a 2-D image (scipy.ndimage.binary_closing)."),
    ROUTINE("ndimage.binary_hit_or_miss", 1, "input, structure1=None, structure2=None", "out", r_binary_hit_or_miss, NULL, "Binary hit-or-miss transform of a 2-D image (scipy.ndimage.binary_hit_or_miss)."),
    ROUTINE("ndimage.iterate_structure", 1, "structure, iterations", "out", r_iterate_structure, NULL, "A 3x3 structuring element dilated by itself iterations-1 times (scipy.ndimage.iterate_structure)."),
    ROUTINE("ndimage.binary_propagation", 1, "input, structure=None, mask=None", "out", r_binary_propagation, NULL, "Binary propagation: dilate within a mask until stable (scipy.ndimage.binary_propagation)."),
    ROUTINE("ndimage.binary_fill_holes", 1, "input, structure=None", "out", r_binary_fill_holes, NULL, "Fill the holes in a binary 2-D image (scipy.ndimage.binary_fill_holes)."),
    ROUTINE("ndimage.maximum_filter", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_order_filter, &OP_MAX, "Multidimensional windowed maximum filter, box footprint (scipy.ndimage.maximum_filter)."),
    ROUTINE("ndimage.minimum_filter", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_order_filter, &OP_MIN, "Multidimensional windowed minimum filter, box footprint (scipy.ndimage.minimum_filter)."),
    ROUTINE("ndimage.median_filter", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_order_filter, &OP_MEDIAN, "Multidimensional median filter, box footprint (scipy.ndimage.median_filter)."),
    ROUTINE("ndimage.rank_filter", 1, "input, rank, size=None, mode='reflect', cval=0.0, origin=0", "out", r_rank_filter, NULL, "Multidimensional rank filter ( order statistic) over a box footprint (scipy.ndimage.rank_filter)."),
    ROUTINE("ndimage.percentile_filter", 1, "input, percentile, size=None, mode='reflect', cval=0.0, origin=0", "out", r_percentile_filter, NULL, "Multidimensional percentile filter over a box footprint (scipy.ndimage.percentile_filter)."),
    ROUTINE("ndimage.grey_erosion", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_morph, &OP_MIN, "Greyscale erosion with a flat box structuring element (scipy.ndimage.grey_erosion)."),
    ROUTINE("ndimage.grey_dilation", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_morph, &OP_MAX, "Greyscale dilation with a flat box structuring element (scipy.ndimage.grey_dilation)."),
    ROUTINE("ndimage.grey_opening", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_openclose, &GREY_OPEN, "Greyscale opening: erosion then dilation with a flat box SE (scipy.ndimage.grey_opening)."),
    ROUTINE("ndimage.grey_closing", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_openclose, NULL, "Greyscale closing: dilation then erosion with a flat box SE (scipy.ndimage.grey_closing)."),
    ROUTINE("ndimage.white_tophat", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_tophat, &MTH_WHITE, "White tophat: input minus its greyscale opening (scipy.ndimage.white_tophat)."),
    ROUTINE("ndimage.black_tophat", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_tophat, &MTH_BLACK, "Black tophat: greyscale closing minus the input (scipy.ndimage.black_tophat)."),
    ROUTINE("ndimage.morphological_gradient", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_tophat, &MTH_GRAD, "Morphological gradient: greyscale dilation minus erosion (scipy.ndimage.morphological_gradient)."),
    ROUTINE("ndimage.morphological_laplace", 1, "input, size=None, mode='reflect', cval=0.0, origin=0", "out", r_grey_tophat, &MTH_LAP, "Morphological laplace: dilation + erosion - 2*input (scipy.ndimage.morphological_laplace)."),
    ROUTINE("ndimage.sobel", 1, "input, axis=-1, mode='reflect', cval=0.0", "out", r_sobel_prewitt, &FILT_SOBEL, "Sobel edge filter along an axis (scipy.ndimage.sobel)."),
    ROUTINE("ndimage.prewitt", 1, "input, axis=-1, mode='reflect', cval=0.0", "out", r_sobel_prewitt, NULL, "Prewitt edge filter along an axis (scipy.ndimage.prewitt)."),
    ROUTINE("ndimage.laplace", 1, "input, mode='reflect', cval=0.0", "out", r_laplace, NULL, "Laplace filter: sum of second derivatives over all axes (scipy.ndimage.laplace)."),
    ROUTINE("ndimage.gaussian_laplace", 1, "input, sigma, mode='reflect', cval=0.0, truncate=4.0", "out", r_gaussian_deriv_combine, &GAUSS_LAP, "Laplace filter using Gaussian second derivatives (scipy.ndimage.gaussian_laplace)."),
    ROUTINE("ndimage.gaussian_gradient_magnitude", 1, "input, sigma, mode='reflect', cval=0.0, truncate=4.0", "out", r_gaussian_deriv_combine, NULL, "Gradient magnitude using Gaussian derivatives (scipy.ndimage.gaussian_gradient_magnitude)."),
    ROUTINE("ndimage.sum_labels", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_SUM, "Sum of array values over labeled regions (scipy.ndimage.sum_labels)."),
    ROUTINE("ndimage.sum", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_SUM, "Sum of array values over labeled regions; deprecated alias of sum_labels (scipy.ndimage.sum)."),
    ROUTINE("ndimage.correlate", 1, "input, weights, mode='reflect', cval=0.0, origin=0", "out", r_correlate_nd, NULL, "Multidimensional correlation with a weights array (scipy.ndimage.correlate)."),
    ROUTINE("ndimage.histogram", 1, "input, min, max, bins, labels=None, index=None", "out", r_histogram, NULL, "Histogram of array values over a labeled region (scipy.ndimage.histogram)."),
    ROUTINE("ndimage.extrema", 4, "input, labels=None, index=None", "minimum, maximum, min_position, max_position", r_extrema, NULL, "Min, max and their positions over labeled regions (scipy.ndimage.extrema)."),
    ROUTINE("ndimage.find_objects", 1, "input, max_label=0", "out", r_find_objects, NULL, "Bounding box of each label as an (L, ndim, 2) array of start/stop (scipy.ndimage.find_objects)."),
    ROUTINE("ndimage.fourier_gaussian", 1, "input, sigma, n=-1, axis=-1", "out", r_fourier, &FR_GAUSS, "Multiply an FFT by the transform of a Gaussian (scipy.ndimage.fourier_gaussian)."),
    ROUTINE("ndimage.fourier_uniform", 1, "input, size, n=-1, axis=-1", "out", r_fourier, &FR_UNIF, "Multiply an FFT by the transform of a uniform box (scipy.ndimage.fourier_uniform)."),
    ROUTINE("ndimage.fourier_shift", 1, "input, shift, n=-1, axis=-1", "out", r_fourier, &FR_SHIFT, "Multiply an FFT by a linear phase to shift the image (scipy.ndimage.fourier_shift)."),
    ROUTINE("ndimage.fourier_ellipsoid", 1, "input, size, n=-1, axis=-1", "out", r_fourier_ellipsoid, NULL, "Multiply an FFT by the transform of an ellipsoid, rank <= 3 (scipy.ndimage.fourier_ellipsoid)."),
    ROUTINE("ndimage.distance_transform_edt", 1, "input, sampling=None", "out", r_distance_edt, NULL, "Exact Euclidean distance to the nearest zero (scipy.ndimage.distance_transform_edt)."),
    ROUTINE("ndimage.distance_transform_bf", 1, "input, metric='euclidean', sampling=None", "out", r_distance_bf, NULL, "Brute-force distance to the nearest zero, any metric (scipy.ndimage.distance_transform_bf)."),
    ROUTINE("ndimage.distance_transform_cdt", 1, "input, metric='chessboard'", "out", r_distance_cdt, NULL, "Chamfer (chessboard/taxicab) distance to the nearest zero (scipy.ndimage.distance_transform_cdt)."),
    ROUTINE("ndimage.map_coordinates", 1, "input, coordinates, order=3, mode='constant', cval=0.0, prefilter=True", "out", r_map_coordinates, NULL, "Interpolate an array at given coordinates, order 0/1 (scipy.ndimage.map_coordinates)."),
    ROUTINE("ndimage.shift", 1, "input, shift, order=3, mode='constant', cval=0.0, prefilter=True", "out", r_shift, NULL, "Shift an array by interpolation, order 0/1 (scipy.ndimage.shift)."),
    ROUTINE("ndimage.affine_transform", 1, "input, matrix, offset=0.0, output_shape=None, order=3, mode='constant', cval=0.0, prefilter=True", "out", r_affine_transform, NULL, "Apply an affine transform by interpolation, order 0/1 (scipy.ndimage.affine_transform)."),
    ROUTINE("ndimage.zoom", 1, "input, zoom, order=3, mode='constant', cval=0.0, prefilter=True", "out", r_zoom, NULL, "Resample an array by a zoom factor, order 0/1 (scipy.ndimage.zoom)."),
    ROUTINE("ndimage.rotate", 1, "input, angle, axes=None, reshape=True, order=3, mode='constant', cval=0.0, prefilter=True", "out", r_rotate, NULL, "Rotate an array in a plane by interpolation, order 0/1 (scipy.ndimage.rotate)."),
    ROUTINE("ndimage.spline_filter1d", 1, "input, order=3, axis=-1, mode='mirror'", "out", r_spline_filter1d, NULL, "1-D B-spline prefilter along an axis (scipy.ndimage.spline_filter1d)."),
    ROUTINE("ndimage.spline_filter", 1, "input, order=3, mode='mirror'", "out", r_spline_filter, NULL, "N-D B-spline prefilter over all axes (scipy.ndimage.spline_filter)."),
    ROUTINE("ndimage.mean", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_MEAN, "Mean of array values over labeled regions (scipy.ndimage.mean)."),
    ROUTINE("ndimage.variance", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_VAR, "Variance of array values over labeled regions (scipy.ndimage.variance)."),
    ROUTINE("ndimage.standard_deviation", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_STD, "Standard deviation over labeled regions (scipy.ndimage.standard_deviation)."),
    ROUTINE("ndimage.maximum", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_MAX, "Maximum of array values over labeled regions (scipy.ndimage.maximum)."),
    ROUTINE("ndimage.minimum", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_MIN, "Minimum of array values over labeled regions (scipy.ndimage.minimum)."),
    ROUTINE("ndimage.median", 1, "input, labels=None, index=None", "out", r_measure, &MEAS_MEDIAN, "Median of array values over labeled regions (scipy.ndimage.median)."),
    ROUTINE("ndimage.center_of_mass", 1, "input, labels=None, index=None", "out", r_center_of_mass, NULL, "Center of mass of labeled regions (scipy.ndimage.center_of_mass)."),
    ROUTINE("ndimage.maximum_position", 1, "input, labels=None, index=None", "out", r_minmax_position, &POS_MAX, "Position of the maximum over labeled regions (scipy.ndimage.maximum_position)."),
    ROUTINE("ndimage.minimum_position", 1, "input, labels=None, index=None", "out", r_minmax_position, NULL, "Position of the minimum over labeled regions (scipy.ndimage.minimum_position)."),
    ROUTINE("ndimage.label", 2, "input, structure=None", "labels, num", r_label, NULL, "Label connected components of a 2-D array, row-major (scipy.ndimage.label)."),
    ROUTINE("ndimage.convolve", 1, "input, weights, mode='reflect', cval=0.0", "out", r_nd_convolve, NULL, "2-D convolution with boundary handling (scipy.ndimage.convolve)."),
};

const fn_table TSR_SCIPY_NDIMAGE_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
