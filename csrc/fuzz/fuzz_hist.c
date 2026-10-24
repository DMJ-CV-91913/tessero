/*
 * Fuzz target: np.histogram, np.histogram_bin_edges, np.histogram2d and np.histogramdd, whose bins argument
 * becomes an allocation size (an int, a float such as NaN/Inf/1e18, an estimator name, or an edge list up
 * to 32 long) and whose range/weights are cross-checked against the sample. Input: a selector byte, then
 * the sample (small shapes, values from the edge-value palette), then bins, range, density and weights,
 * each with a kind byte. The kernel budget (fuzz_common.h) turns huge bin counts into clean refusals.
 * Oracle: ASan/UBSan clean, well-formed results, no leaked tsr allocations.
 */
#include "fuzz_common.h"

static const char *const EST[] = {"auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt", ""};

/* a bins argument for one axis: int, double, estimator name, or an edge array of 0..32 values */
static tsr_arg bins_arg(fz_in *in, fz_pool *pool, char *sbuf)
{
    const uint8_t k = fz_u8(in);
#ifdef FUZZ_SKIP_KNOWN
    /* open finding F7: an integer bins >= 2^60 makes hist_edges compute (bins + 1) * 8 in int64, which wraps
       to a small allocation that the edge loop then overruns. Keep integer bins below 2^40. */
    if (k % 6 == 1) { const int64_t v = (int64_t)fz_u64(in); return fz_arg_int(v > ((int64_t)1 << 40) ? (int64_t)1 << 40 : v); }
    if (k % 6 == 2) { const double v = fz_double(in); return fz_arg_num(v > 1099511627776.0 && v < 9.2e18 ? 1099511627776.0 : v); }
#endif
    switch (k % 6) {
    case 0: return fz_arg_int((int64_t)(fz_u8(in) % 40) - 2);
    case 1: return fz_arg_int((int64_t)fz_u64(in));
    case 2: return fz_arg_num(fz_double(in));
    case 3: {
        const uint8_t e = fz_u8(in);
        if (e < 240) return fz_arg_str(EST[e % (sizeof EST / sizeof EST[0])]);
        return fz_arg_str(fz_str(in, sbuf, 16, 12));
    }
    default: {
        int64_t n = fz_u8(in) % 33;
        tsr_arg a = fz_arg_array(in, pool, 0, 1, &n);
        if (a.kind == 3 && (k & 0x80)) {                /* sometimes sorted edges, to get past the monotonic check */
            double *v = (double *)a.arr.data;
            for (int64_t i = 1; i < n; i++)
                for (int64_t j = i; j > 0 && v[j - 1] > v[j]; j--) { double t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
        }
        if (a.kind == 3) a.flags = (k & 0x40) ? 2 : 0;
        return a;
    }
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_init();
    static int ids[4] = {-2};
    if (ids[0] == -2) {
        ids[0] = tsr_fn_find("np.histogram");
        ids[1] = tsr_fn_find("np.histogram_bin_edges");
        ids[2] = tsr_fn_find("np.histogram2d");
        ids[3] = tsr_fn_find("np.histogramdd");
        for (int i = 0; i < 4; i++) if (ids[i] < 0) abort();
    }
    fz_in in = {data, size};
    fz_pool pool = {{0}, 0};
    char sb[8][16];
    const uint8_t sel = fz_u8(&in);
    const int which = sel % 4;
    const int dtype = (sel >> 2) % 4 == 3 ? 2 : 0;       /* int64 samples take the integer-input estimator path */
    tsr_arg args[6];
    int nargs = 0;

    if (which <= 1) {                                     /* histogram / histogram_bin_edges */
        tsr_arg a = fz_arg_small(&in, &pool, dtype, 2, 12);
        if (a.kind != 3) goto done;
        args[nargs++] = a;
        args[nargs++] = (fz_u8(&in) & 7) == 0 ? fz_arg_none() : bins_arg(&in, &pool, sb[0]);
        const uint8_t rk = fz_u8(&in) % 4;                /* range: none, [lo, hi], wrong length */
        if (rk == 0) args[nargs++] = fz_arg_none();
        else { int64_t n = rk == 1 ? 2 : (rk == 2 ? 2 : fz_u8(&in) % 4); args[nargs++] = fz_arg_array(&in, &pool, 0, 1, &n); }
        if (which == 0) args[nargs++] = fz_arg_bool(fz_u8(&in) & 1);
        if (fz_u8(&in) & 1) {                             /* weights: same shape, or a fuzzed one */
            args[nargs++] = (fz_u8(&in) & 3) ? fz_arg_array(&in, &pool, 0, a.arr.ndim, a.arr.shape)
                                             : fz_arg_small(&in, &pool, 0, 2, 12);
        }
    } else if (which == 2) {                              /* histogram2d(x, y, bins, range, density, weights) */
        int64_t n = fz_u8(&in) % 12;
        int64_t ny = (fz_u8(&in) & 7) ? n : fz_u8(&in) % 12;
        args[nargs++] = fz_arg_array(&in, &pool, dtype, 1, &n);
        args[nargs++] = fz_arg_array(&in, &pool, 0, 1, &ny);
        const uint8_t bk = fz_u8(&in) % 3;               /* bins: one spec, or [bx, by] as a sequence */
        tsr_arg *pair = (tsr_arg *)fz_take(&pool, 2 * sizeof(tsr_arg));
        if (!pair) goto done;
        if (bk == 0) args[nargs++] = fz_arg_none();
        else if (bk == 1) args[nargs++] = bins_arg(&in, &pool, sb[0]);
        else {
            pair[0] = bins_arg(&in, &pool, sb[0]);
            pair[1] = bins_arg(&in, &pool, sb[1]);
            tsr_arg s = fz_arg_none(); s.kind = 5; s.items = pair; s.count = 1 + (fz_u8(&in) & 1);
            args[nargs++] = s;
        }
        const uint8_t rk = fz_u8(&in) % 3;
        if (rk == 0) args[nargs++] = fz_arg_none();
        else { int64_t sh[2] = {rk == 1 ? 2 : fz_u8(&in) % 4, 2}; args[nargs++] = fz_arg_array(&in, &pool, 0, 2, sh); }
        args[nargs++] = fz_arg_bool(fz_u8(&in) & 1);
        if (fz_u8(&in) & 1) { int64_t nw = (fz_u8(&in) & 3) ? n : fz_u8(&in) % 12; args[nargs++] = fz_arg_array(&in, &pool, 0, 1, &nw); }
    } else {                                              /* histogramdd(sample, bins, range, density, weights) */
        int64_t sh[2];
        sh[0] = fz_u8(&in) % 10;
        sh[1] = 1 + fz_u8(&in) % 5;
        const int32_t snd = (fz_u8(&in) & 3) ? 2 : 1;
        tsr_arg s = fz_arg_array(&in, &pool, dtype, snd, sh);
        if (s.kind != 3) goto done;
        args[nargs++] = s;
        const int64_t D = snd == 2 ? sh[1] : 1;
        const uint8_t bk = fz_u8(&in) % 3;
        if (bk == 0) args[nargs++] = fz_arg_none();
        else if (bk == 1) args[nargs++] = bins_arg(&in, &pool, sb[0]);
        else {                                            /* one spec per dimension, the count sometimes off */
            const int64_t cnt = (fz_u8(&in) & 3) ? D : fz_u8(&in) % 8;
            tsr_arg *per = (tsr_arg *)fz_take(&pool, (size_t)(cnt ? cnt : 1) * sizeof(tsr_arg));
            if (!per) goto done;
            for (int64_t d = 0; d < cnt; d++) per[d] = bins_arg(&in, &pool, sb[d % 8]);
            tsr_arg q = fz_arg_none(); q.kind = 5; q.items = per; q.count = cnt;
            args[nargs++] = q;
        }
        const uint8_t rk = fz_u8(&in) % 3;
        if (rk == 0) args[nargs++] = fz_arg_none();
        else { int64_t rs[2] = {rk == 1 ? D : fz_u8(&in) % 6, 2}; args[nargs++] = fz_arg_array(&in, &pool, 0, 2, rs); }
        args[nargs++] = fz_arg_bool(fz_u8(&in) & 1);
        if (fz_u8(&in) & 1) { int64_t nw = (fz_u8(&in) & 3) ? sh[0] : fz_u8(&in) % 10; args[nargs++] = fz_arg_array(&in, &pool, 0, 1, &nw); }
    }
    /* sometimes drop trailing arguments: the dispatcher pads them as absent */
    if (nargs > 1 && (sel & 0x80)) nargs = 1 + fz_u8(&in) % nargs;
#ifdef FUZZ_SKIP_KNOWN
    /* open finding F9: with an estimator name as bins, hist_edges casts ceil((last - first) / width) to int64
       unchecked, and a huge or infinite quotient then yields a bogus edge count. Keep estimator samples at 0 or
       with magnitude in [1e-6, 1e6] (subnormal spreads also give a huge quotient). */
    if (which <= 1 && nargs > 1 && args[1].kind == 2 && args[0].kind == 3) {
        if (nargs > 2) args[2] = fz_arg_none();          /* a huge range gives the same quotient */
        const int64_t cnt = tsr_array_size(&args[0].arr);
        for (int64_t i = 0; i < cnt; i++) {
            if (args[0].arr.dtype == 0) {
                double *v = (double *)args[0].arr.data + i;
                if (!(fabs(*v) >= 1e-6 && fabs(*v) <= 1e6)) *v = 0.0;
            } else {
                int64_t *v = (int64_t *)args[0].arr.data + i;
                if (*v < -1000000 || *v > 1000000) *v = 0;
            }
        }
    }
#endif
    fz_call(ids[which], nargs, args);
done:
    fz_release(&pool);
    return 0;
}
