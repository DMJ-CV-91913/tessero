/*
 * csrc/repro/determinism_test.c
 *
 * Determinism harness + golden generator for the reproducible f32 primitives
 * (tsr_repro_rsqrt_f32 / exp2 / exp / erf, implemented in csrc/src/repro_f32.c).
 *
 * It links the real ABI implementation. Compile repro_f32.c TWO ways (scalar and
 * AVX2+FMA, both -fno-fast-math -ffp-contract=off), run each mode, and compare:
 * byte-identical output == bit-identical across SIMD == reproducible. The driver
 * csrc/repro/run_determinism.sh does exactly that and also varies thread count.
 *
 * Modes (argv[1]):
 *   rsqrt|exp2|exp|erf              write the raw f32 output bytes of the fixed
 *                                   finite in-domain input vector to stdout (pipe
 *                                   to sha256sum: that hash is the golden)
 *   rsqrt|exp2|exp|erf  edges       print classified results for the edge inputs
 *                                   (0, -0, +-Inf, NaN, denormal, over/underflow)
 *   rsqrt|exp2|exp|erf  accuracy    print the max ULP error vs a double reference
 *   spec                            print the canonical spec: every constant as an
 *                                   exact hex float, plus the evaluation order
 *
 * The golden vector is finite and in-domain on purpose: NaN payloads and the
 * exact subnormal/overflow boundary are not guaranteed identical across vendors,
 * so edges are verified by CLASSIFICATION (edges mode), not folded into the hash.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tessero.h"
#include "../include/tessero_repro_f32.h"   /* header-level composites (backward) */

#define N ((int64_t)1 << 20)   /* 1,048,576 elements, matches the rsqrt golden */

/* ---- fixed, deterministic input generators (pure function of the index) ---- */

/* rsqrt: strictly-positive, magnitude spread ~1e-4 .. 1e4. This MUST match
 * csrc/repro/demo_rsqrt.c byte-for-byte so the wired ABI reproduces the proven
 * rsqrt golden. Do not change it. */
static void gen_rsqrt(float *x) {
    for (int64_t i = 0; i < N; ++i) {
        uint32_t s    = (uint32_t)i * 2654435761u;              /* Knuth hash */
        float    base = (float)((s >> 8) & 0xffffu) / 65535.0f;  /* [0, 1] */
        float    mag  = (i % 7 == 0) ? 1.0e4f : (i % 11 == 0) ? 1.0e-4f : 1.0f;
        x[i] = (base + (1.0f / 1024.0f)) * mag;                  /* strictly > 0 */
    }
}

/* linear spread over [lo, hi] from the same index hash. */
static void gen_span(float *x, float lo, float hi) {
    for (int64_t i = 0; i < N; ++i) {
        uint32_t s    = (uint32_t)i * 2654435761u;
        float    base = (float)((s >> 8) & 0xffffu) / 65535.0f;  /* [0, 1] */
        x[i] = lo + base * (hi - lo);
    }
}

/* recip: the rsqrt magnitude spread with alternating sign (exercises both signs). */
static void gen_recip(float *x) {
    gen_rsqrt(x);
    for (int64_t i = 0; i < N; ++i) if (i & 1) x[i] = -x[i];
}

/* tanh: signed spread over [-10, 10] at three scales so every frozen form is
 * hit: |u| < 2^-12 (identity), <= 0.25 (series), < 9.1 (exp form), >= 9.1 (+-1). */
static void gen_tanh(float *x) {
    for (int64_t i = 0; i < N; ++i) {
        uint32_t s    = (uint32_t)i * 2654435761u;
        float    base = (float)((s >> 8) & 0xffffu) / 65535.0f;  /* [0, 1] */
        float    mag  = (i % 7 == 0) ? 1.0e-5f : (i % 11 == 0) ? 1.0e-2f : 1.0f;
        x[i] = (base * 20.0f - 10.0f) * mag;
    }
}

static void gen_input(const char *fn, float *x) {
    if      (!strcmp(fn, "rsqrt")) gen_rsqrt(x);
    else if (!strcmp(fn, "exp2"))  gen_span(x, -30.0f, 30.0f);
    else if (!strcmp(fn, "exp"))   gen_span(x, -20.0f, 20.0f);
    else if (!strcmp(fn, "erf"))   gen_span(x, -4.0f, 4.0f);
    else if (!strcmp(fn, "recip")) gen_recip(x);
    else if (!strcmp(fn, "tanh"))  gen_tanh(x);
    else if (!strcmp(fn, "sigmoid") || !strcmp(fn, "gelu_tanh")) gen_span(x, -12.0f, 12.0f);
    else if (!strcmp(fn, "softmax") || !strcmp(fn, "rmsnorm") || !strcmp(fn, "layernorm")) gen_span(x, -8.0f, 8.0f);
    else if (!strcmp(fn, "gelu_tanh_bwd") || !strcmp(fn, "erf_bwd") || !strcmp(fn, "sigmoid_bwd") || !strcmp(fn, "tanh_bwd")) gen_span(x, -6.0f, 6.0f);
    else if (!strcmp(fn, "rmsnorm_bwd") || !strcmp(fn, "layernorm_bwd")) gen_span(x, -8.0f, 8.0f);
    else { fprintf(stderr, "unknown fn: %s\n", fn); exit(2); }
}

static int run_fn(const char *fn, int64_t n, const float *x, float *y) {
    if (!strcmp(fn, "rsqrt")) return tsr_repro_rsqrt_f32(n, x, y);
    if (!strcmp(fn, "exp2"))  return tsr_repro_exp2_f32(n, x, y);
    if (!strcmp(fn, "exp"))   return tsr_repro_exp_f32(n, x, y);
    if (!strcmp(fn, "erf"))   return tsr_repro_erf_f32(n, x, y);
    if (!strcmp(fn, "recip")) return tsr_repro_recip_f32(n, x, y);
    if (!strcmp(fn, "tanh"))  return tsr_repro_tanh_f32(n, x, y);
    if (!strcmp(fn, "sigmoid"))   return tsr_repro_sigmoid_f32(n, x, y);
    if (!strcmp(fn, "gelu_tanh")) return tsr_repro_gelu_tanh_f32(n, x, y);
    if (!strcmp(fn, "gelu_tanh_bwd") || !strcmp(fn, "erf_bwd") || !strcmp(fn, "sigmoid_bwd") || !strcmp(fn, "tanh_bwd")) {
        for (int64_t i = 0; i < n; ++i) {
            const float dy = 0.5f + (float)(i % 13) * 0.0625f;
            if      (!strcmp(fn, "gelu_tanh_bwd")) y[i] = tsr__gelu_tanh_bwd_f32(0.0f, dy, x[i]);
            else if (!strcmp(fn, "erf_bwd"))       y[i] = tsr__erf_bwd_f32(0.0f, dy, x[i]);
            else if (!strcmp(fn, "sigmoid_bwd"))   y[i] = tsr__sigmoid_bwd_f32(0.0f, dy, tsr__sigmoid_f32(x[i]));
            else                                   y[i] = tsr__tanh_bwd_f32(0.0f, dy, tsr__tanh_f32(x[i]));
        }
        return 0;
    }
    if (!strcmp(fn, "rmsnorm_bwd") || !strcmp(fn, "layernorm_bwd")) {
        static float g[1024], dyr[1024];
        for (int j = 0; j < 1024; ++j) { g[j] = 0.5f + (float)(j % 17) * 0.0625f; dyr[j] = 0.25f + (float)(j % 11) * 0.125f; }
        int64_t rows = n / 1024, cols = (n >= 1024) ? 1024 : n; if (rows == 0) rows = 1;
        for (int64_t i = 0; i < n; ++i) y[i] = 0.0f;
        for (int64_t r = 0; r < rows; ++r) {
            if (!strcmp(fn, "rmsnorm_bwd")) tsr__rmsnorm_dx_row_f32(dyr, x + r * cols, g, 1.0e-5f, y + r * cols, cols);
            else                            tsr__layernorm_dx_row_f32(dyr, x + r * cols, g, 1.0e-5f, y + r * cols, cols);
        }
        return 0;
    }
    if (!strcmp(fn, "softmax") || !strcmp(fn, "rmsnorm") || !strcmp(fn, "layernorm")) {
        /* fixed 1024 x 1024 row-major matrix; gamma/beta pure functions of the column */
        static float g[1024], b[1024];
        for (int j = 0; j < 1024; ++j) { g[j] = 0.5f + (float)(j % 17) * 0.0625f; b[j] = (float)(j % 5) - 2.0f; }
        int64_t rows = n / 1024, cols = (n >= 1024) ? 1024 : n; if (rows == 0) rows = 1;
        if (!strcmp(fn, "softmax")) return tsr_repro_softmax_f32(rows, cols, x, y);
        if (!strcmp(fn, "rmsnorm")) return tsr_repro_rmsnorm_f32(rows, cols, x, g, 1.0e-5f, y);
        return tsr_repro_layernorm_f32(rows, cols, x, g, b, 1.0e-5f, y);
    }
    fprintf(stderr, "unknown fn: %s\n", fn); exit(2);
}

/* double reference for the accuracy report. */
static double ref_fn(const char *fn, double v) {
    if (!strcmp(fn, "rsqrt")) return 1.0 / sqrt(v);
    if (!strcmp(fn, "exp2"))  return exp2(v);
    if (!strcmp(fn, "exp"))   return exp(v);
    if (!strcmp(fn, "erf"))   return erf(v);
    if (!strcmp(fn, "recip")) return 1.0 / v;
    if (!strcmp(fn, "tanh"))  return tanh(v);
    exit(2);
}

/* ---- modes ---- */

static int mode_hash(const char *fn) {
    float *x = malloc((size_t)N * sizeof(float));
    float *y = malloc((size_t)N * sizeof(float));
    if (!x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }
    gen_input(fn, x);
    if (run_fn(fn, N, x, y) != 0) { fprintf(stderr, "run failed\n"); return 1; }
    if (fwrite(y, sizeof(float), (size_t)N, stdout) != (size_t)N) {
        fprintf(stderr, "write failed\n"); return 1;
    }
    free(x); free(y);
    return 0;
}

/* ULP distance between two f32 with the same sign (ordered integer encoding). */
static uint32_t ulp_gap(float a, float b) {
    uint32_t ia, ib;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    if ((ia >> 31) != (ib >> 31)) return 0xffffffffu;   /* sign mismatch */
    return ia > ib ? ia - ib : ib - ia;
}

static int mode_accuracy(const char *fn) {
    float *x = malloc((size_t)N * sizeof(float));
    float *y = malloc((size_t)N * sizeof(float));
    if (!x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }
    gen_input(fn, x);
    run_fn(fn, N, x, y);
    uint32_t worst = 0;                 /* worst ULP gap (away from ~0) */
    double worst_abs = 0.0;             /* worst absolute error */
    float worst_x = 0.0f, worst_y = 0.0f, worst_ref = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
        double r = ref_fn(fn, (double)x[i]);
        float  rf = (float)r;
        if (!isfinite(rf) || !isfinite(y[i])) continue;      /* skip over/underflow */
        double a = fabs((double)y[i] - r);
        if (a > worst_abs) worst_abs = a;
        /* ULP gap is only meaningful where |ref| is not tiny (near 0 a sub-ULP
           absolute error spans a huge integer gap); gate it on |ref| >= 2^-24. */
        if (fabs(r) >= 5.96e-8) {
            uint32_t g = ulp_gap(y[i], rf);
            if (g > worst) { worst = g; worst_x = x[i]; worst_y = y[i]; worst_ref = rf; }
        }
    }
    printf("%-6s max abs err=%.3e   max ULP (|ref|>=2^-24)=%u  "
           "(at x=%.9g, got=%.9g, ref=%.9g)\n",
           fn, worst_abs, worst, (double)worst_x, (double)worst_y, (double)worst_ref);
    free(x); free(y);
    return 0;
}

static const char *classify(float v) {
    if (isnan(v)) return "NaN";
    if (isinf(v)) return signbit(v) ? "-Inf" : "+Inf";
    if (v == 0.0f) return signbit(v) ? "-0" : "+0";
    return signbit(v) ? "-finite" : "+finite";
}

static int mode_edges(const char *fn) {
    float denorm = 1.401298464e-45f;            /* FLT_TRUE_MIN */
    float inf = INFINITY, nan = NAN;
    float probes[] = { 0.0f, -0.0f, 1.0f, -1.0f, 4.0f, 0.5f, 10.0f, -10.0f,
                       100.0f, -100.0f, 128.0f, -150.0f, 1000.0f, -1000.0f,
                       denorm, inf, -inf, nan };
    int np = (int)(sizeof probes / sizeof probes[0]);
    printf("== %s edge inputs ==\n", fn);
    for (int i = 0; i < np; ++i) {
        float xi = probes[i], yi = 0.0f;
        run_fn(fn, 1, &xi, &yi);
        printf("  f(%-12.6g) = %-16.9g  [%s]\n", (double)xi, (double)yi, classify(yi));
    }
    return 0;
}

static void spec_line(const char *name, float v) {
    printf("  %-14s = %-16a  (%.9g)\n", name, (double)v, (double)v);
}

static int mode_spec(void) {
    puts("CANONICAL SPEC for the reproducible f32 primitives (csrc/src/repro_f32.c).");
    puts("Constants are exact f32 bit patterns (hex float), decimal shown for reference.");
    puts("A conformant GPU shader must use these exact values and evaluation order.\n");

    puts("scale2(y, n) = (y * 2^n1) * 2^n2, n1 = n/2 (trunc), n2 = n - n1;");
    puts("  2^m built as bits_to_f32((m + 127) << 23). Left-associated, fixed order.\n");

    puts("exp2(x): x>=128 -> +Inf; x<=-150 -> 0; else k=floorf(x+0.5), r=x-k,");
    puts("  2^r = fmaf(...fmaf(fmaf(C0,r,C1),r,C2)...,r,1), then scale2(2^r, (int)k).");
    spec_line("exp2.C0", 1.535336188319500E-4f);
    spec_line("exp2.C1", 1.339887440266574E-3f);
    spec_line("exp2.C2", 9.618437357674640E-3f);
    spec_line("exp2.C3", 5.550332471162809E-2f);
    spec_line("exp2.C4", 2.402264791363012E-1f);
    spec_line("exp2.C5", 6.931472028550421E-1f);

    puts("\nexp(x): x>=88.72283905 -> +Inf; x<=-103.972084 -> 0; else");
    puts("  k=floorf(fmaf(x,LOG2E,0.5)); r=fmaf(-k,LN2_HI,x); r=fmaf(-k,LN2_LO,r);");
    puts("  e^r = fmaf(P(r), r*r, r) + 1; then scale2(e^r, (int)k). P Horner in fmaf.");
    spec_line("exp.LOG2E",  1.44269504088896341f);
    spec_line("exp.LN2_HI", 6.93359375E-1f);
    spec_line("exp.LN2_LO", -2.12194440E-4f);
    spec_line("exp.Q0", 1.9875691500E-4f);
    spec_line("exp.Q1", 1.3981999507E-3f);
    spec_line("exp.Q2", 8.3334519073E-3f);
    spec_line("exp.Q3", 4.1665795894E-2f);
    spec_line("exp.Q4", 1.6666665459E-1f);
    spec_line("exp.Q5", 5.0000001201E-1f);

    puts("\nerf(x): ax=|x|. For ax<=0.5 Maclaurin series w=x*x,");
    puts("  erf = x * fmaf(...fmaf(fmaf(B5,w,B4),w,B3)...,w,B0).");
    spec_line("erf.B0", 1.1283791671f);
    spec_line("erf.B1", -3.7612638903E-1f);
    spec_line("erf.B2", 1.1283791671E-1f);
    spec_line("erf.B3", -2.6866170645E-2f);
    spec_line("erf.B4", 5.2239776255E-3f);
    spec_line("erf.B5", -8.5483270689E-4f);
    puts("  For ax>0.5: s=sign(x); ax>=4 -> s; else t=1/fmaf(P,ax,1);");
    puts("  poly=fmaf(...fmaf(fmaf(A5,t,A4),t,A3)...,t,A1)*t; e=exp(-(ax*ax));");
    puts("  erf = s * fmaf(-poly, e, 1).  (Abramowitz & Stegun 7.1.26.)");
    spec_line("erf.P",  3.275911E-1f);
    spec_line("erf.A1", 2.54829592E-1f);
    spec_line("erf.A2", -2.84496736E-1f);
    spec_line("erf.A3", 1.421413741f);
    spec_line("erf.A4", -1.453152027f);
    spec_line("erf.A5", 1.061405429f);
    puts("");
    puts("recip(b): canon; NaN -> canonical NaN; +-0 -> +-Inf; +-Inf -> +-0;");
    puts("  t = bits(0x7EF127EA - bits(|b|)); 5x t = t*fmaf(-|b|,t,2);");
    puts("  |b| >= 2^64: rrecip(|b|*2^-64)*2^-64; result canon'd, sign restored.");
    puts("");
    puts("tanh(u): canon; |u|<2^-12 -> u; |u|>=9.1 -> +-1; |u|<=0.25 -> w=u*u,");
    puts("  p=fmaf(fmaf(fmaf(fmaf(C11,w,C9),w,C7),w,C5),w,C3); fmaf(u*w,p,u);");
    puts("  else a=|u|, e2=exp(2a), t=(e2-1)*rrecip(e2+1), sign of u restored by bit.");
    spec_line("tanh.C3",  -3.3333333333E-1f);
    spec_line("tanh.C5",  1.3333333333E-1f);
    spec_line("tanh.C7",  -5.3968253968E-2f);
    spec_line("tanh.C9",  2.1869488536E-2f);
    spec_line("tanh.C11", -8.8632355299E-3f);
    spec_line("tanh.SAT", 9.1f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <rsqrt|exp2|exp|erf|recip|tanh> [edges|accuracy] | spec\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "spec")) return mode_spec();
    if (argc >= 3 && !strcmp(argv[2], "edges"))    return mode_edges(argv[1]);
    if (argc >= 3 && !strcmp(argv[2], "accuracy")) return mode_accuracy(argv[1]);
    return mode_hash(argv[1]);
}
