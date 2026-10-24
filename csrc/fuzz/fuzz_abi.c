/*
 * Fuzz target: the registry's input gate (fn_gate.c, ADR 0016). The input describes ABI structures with
 * arbitrary metadata: tsr_arg kinds and flag bits, integer/double pairs, NULL strings, nested sequences, and
 * arrays with any dtype, ndim, dimensions, strides and offset. The harness classifies every structure itself,
 * with 128-bit arithmetic independent of the gate:
 *   invalid  -> it is sent to a RANDOM registered routine (or to the ufunc/gufunc/random/rvs entry points),
 *               and the call must be refused: the gate, not the routine, has to catch it;
 *   valid    -> its data block is allocated with exactly the bytes the elements span, so ASan sees any read
 *               outside the described extent, and it goes to a routine that accepts any array;
 *   valid but larger than the harness allocates -> not called (the gate cannot know a block's size).
 * The harness never dereferences a pointer it made up: invalid structures carry NULL or a real 64-byte block.
 */
#include "fuzz_common.h"

void fn_results_free(tsr_result *res, int n);   /* fn.h */

#define FZA_MAXBLOCK ((__int128)1 << 16)

enum { FZ_INVALID = 0, FZ_VALID = 1, FZ_TOOBIG = 2 };

/* add one stride reach (|r| <= 2^126) to an extent sum, saturating at +-2^110 */
static void fza_acc(__int128 *acc, __int128 r)
{
    const __int128 cap = (__int128)1 << 110;
    *acc += r > cap ? cap : (r < -cap ? -cap : r);
    if (*acc > cap) *acc = cap;
    if (*acc < -cap) *acc = -cap;
}

static int64_t fza_elsize(int32_t dt)
{
    switch (dt) {
    case TSR_F64: case TSR_I64: return 8;
    case TSR_F32: case TSR_I32: return 4;
    case TSR_U8: case TSR_BOOL: return 1;
    case TSR_C128: return 16;
    default: return 0;
    }
}

static int64_t fza_dim(fz_in *in)
{
    const uint8_t s = fz_u8(in);
    if (s < 200) return s % 6;
    switch (s % 6) {
    case 0: return -1;
    case 1: return -(int64_t)fz_u8(in);
    case 2: return (int64_t)1 << (20 + fz_u8(in) % 43);
    case 3: return INT64_MAX;
    case 4: return INT64_MIN;
    default: return (int64_t)fz_u64(in);
    }
}

static int64_t fza_stride(fz_in *in, int64_t el, int64_t natural)
{
    const uint8_t s = fz_u8(in);
    if (s < 160) return natural;
    switch (s % 8) {
    case 0: return 0;
    case 1: return -natural;
    case 2: return el * (int64_t)(fz_u8(in) % 9) - el * 4;
    case 3: return (int64_t)(fz_u8(in) % 7) - 3;              /* misaligned */
    case 4: return INT64_MAX / (2 + fz_u8(in) % 7);
    case 5: return INT64_MIN + fz_u8(in);
    case 6: return (int64_t)fz_u64(in);
    default: return natural * 2;
    }
}

/* fill a with fuzzed metadata; returns the harness's classification and allocates a valid array's block */
static int fza_array(fz_in *in, fz_pool *pool, tsr_array *a)
{
    memset(a, 0, sizeof *a);
    const uint8_t h = fz_u8(in);
    a->dtype = h < 230 ? (int32_t)(h % TSR_NDTYPES) : (int32_t)(h % 3 == 0 ? -1 : 7 + h % 50);
    const uint8_t n = fz_u8(in);
    a->ndim = n < 240 ? (int32_t)(n % 4) : (int32_t)(n % 2 ? 33 + n % 8 : -(int32_t)(1 + n % 4));
    const int64_t el = fza_elsize(a->dtype);
    int valid = el != 0 && a->ndim >= 0 && a->ndim <= TSR_MAXDIM;
    const int32_t nd = a->ndim < 0 ? 0 : (a->ndim > TSR_MAXDIM ? TSR_MAXDIM : a->ndim);
    __int128 count = 1;
    int64_t natural = el ? el : 8;
    for (int32_t d = nd - 1; d >= 0; d--) {
        a->shape[d] = fza_dim(in);
        if (a->shape[d] < 0) valid = 0;
    }
    count = 1;
    for (int32_t d = 0; d < nd; d++) {
        if (a->shape[d] < 0) break;
        if (a->shape[d] > 0 && count > ((__int128)1 << 80) / a->shape[d]) count = (__int128)1 << 80;   /* saturate first */
        else count *= a->shape[d];
    }
    if (count > (__int128)(INT64_MAX / 16)) valid = 0;
    for (int32_t d = nd - 1; d >= 0; d--) {
        a->strides[d] = fza_stride(in, el ? el : 8, natural);
        if (a->shape[d] > 0 && natural < INT64_MAX / 8) natural *= a->shape[d] < (INT64_MAX / 8) / natural ? a->shape[d] : 1;
    }
    const uint8_t o = fz_u8(in);
    __int128 lo = 0, hi = 0;
    for (int32_t d = 0; d < nd; d++) {
        if (a->shape[d] <= 1) continue;
        const __int128 r = (__int128)(a->shape[d] - 1) * a->strides[d];
        fza_acc(r < 0 ? &lo : &hi, r);
    }
    hi += el;
    /* offset: usually what makes a reversed view valid, sometimes off by a few bytes, sometimes wild */
    {
        const __int128 back = lo < 0 ? -lo : 0;
        __int128 off = o < 200 ? back : (o < 230 ? (__int128)(o % 9) - 4 + back : (__int128)(int64_t)fz_u64(in));
        if (off > (__int128)INT64_MAX) off = INT64_MAX;
        if (off < (__int128)INT64_MIN) off = INT64_MIN;
        a->offset = (int64_t)off;
    }
    lo += a->offset;
    hi += a->offset;
    if (count == 0) {
        a->data = (fz_u8(in) & 1) ? NULL : fz_take(pool, 64);
        return valid ? FZ_VALID : FZ_INVALID;
    }
    if (lo < 0 || hi > (__int128)INT64_MAX || lo > (__int128)INT64_MAX) valid = 0;
    const int64_t al = el > 8 ? 8 : (el ? el : 1);
    if (valid && al > 1) {
        if (a->offset % al) valid = 0;
        for (int32_t d = 0; d < nd; d++) if (a->shape[d] > 1 && a->strides[d] % al) valid = 0;
    }
    const uint8_t np = fz_u8(in);
    if (!valid) {
        a->data = np & 1 ? NULL : fz_take(pool, 64);
        return FZ_INVALID;
    }
    if (np >= 250) { a->data = NULL; return FZ_INVALID; }       /* NULL data with elements */
    if (hi > FZA_MAXBLOCK) { a->data = fz_take(pool, 64); return FZ_TOOBIG; }
    /* fz_take blocks are malloc'ed (16-aligned), so element alignment holds */
    a->data = fz_take(pool, (size_t)hi);
    if (a->data) for (int64_t b = 0; b < (int64_t)hi; b++) ((uint8_t *)a->data)[b] = fz_u8(in);
    return a->data ? FZ_VALID : FZ_TOOBIG;
}

static int fza_arg(fz_in *in, fz_pool *pool, tsr_arg *a, int depth)
{
    memset(a, 0, sizeof *a);
    const uint8_t k = fz_u8(in);
    a->kind = k < 248 ? (int32_t)(k % 6) : (int32_t)(6 + k % 4);
    const uint8_t f = fz_u8(in);
    a->flags = f < 224 ? 0 : (int32_t)(f % 8);
    int cls = FZ_VALID;
    switch (a->kind) {
    case 0: case 4:
        if (a->flags) cls = FZ_INVALID;
        a->num = a->kind == 4 ? (double)(fz_u8(in) & 1) : 0.0;
        break;
    case 1:
        a->num = fz_double(in);
        if (a->flags & 1) {
            a->ival = (int64_t)fz_u64(in) >> (fz_u8(in) % 64);
            if (fz_u8(in) < 230) a->num = (double)a->ival;
            if (a->num != (double)a->ival) cls = FZ_INVALID;
        }
        if (a->flags & ~1) cls = FZ_INVALID;
        break;
    case 2: {
        if (a->flags) cls = FZ_INVALID;
        char *s = (char *)fz_take(pool, 17);
        if (!s || fz_u8(in) >= 240) { a->str = NULL; cls = FZ_INVALID; break; }
        const size_t n = fz_u8(in) % 17;
        for (size_t i = 0; i < n; i++) s[i] = (char)('a' + fz_u8(in) % 26);
        s[n] = '\0';
        a->str = s;
        break;
    }
    case 3:
        cls = fza_array(in, pool, &a->arr);
        if (a->flags & ~2) cls = FZ_INVALID;
        break;
    case 5: {
        if (a->flags) cls = FZ_INVALID;
        const uint8_t c = fz_u8(in);
        a->count = c < 240 ? c % 4 : (c % 2 ? -1 : (int64_t)1 << 30);
        if (a->count < 0 || a->count > ((int64_t)1 << 24)) { a->items = fz_take(pool, sizeof(tsr_arg)); cls = FZ_INVALID; break; }
        if (depth >= 4) { a->items = NULL; cls = FZ_INVALID; break; }   /* nested too deep (also depth 4 itself) */
        if (a->count > 0 && fz_u8(in) >= 245) { a->items = NULL; cls = FZ_INVALID; break; }
        tsr_arg *items = a->count ? (tsr_arg *)fz_take(pool, (size_t)a->count * sizeof(tsr_arg)) : NULL;
        if (a->count && !items) { a->count = 0; break; }
        a->items = items;
        for (int64_t j = 0; j < a->count; j++) {
            const int c2 = fza_arg(in, pool, &items[j], depth + 1);
            if (c2 == FZ_INVALID) cls = FZ_INVALID;
            else if (c2 == FZ_TOOBIG && cls == FZ_VALID) cls = FZ_TOOBIG;
        }
        if (depth + 1 > 4) cls = FZ_INVALID;
        break;
    }
    default:
        cls = FZ_INVALID;
    }
    return cls;
}

static void fza_must_refuse(int rc, const char *what)
{
    if (rc >= 0) {
        fprintf(stderr, "fuzz_abi: %s accepted a structure the harness classified invalid (rc %d)\n", what, rc);
        abort();
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_init();
    static int uq = -2, nfn, erf, med, sn, norm;
    if (uq == -2) {
        uq = tsr_fn_find("np.unique");
        nfn = tsr_fn_count();
        erf = tsr_fn_find("special.erf");
        med = tsr_fn_find("np.median");
        sn = tsr_fn_find("random.standard_normal");
        norm = tsr_fn_find("stats.norm");
        if (uq < 0 || erf < 0 || med < 0 || sn < 0 || norm < 0) abort();
    }
    fz_in in = {data, size};
    fz_pool pool = {{0}, 0};
    const int64_t before = tsr_allocated();
    const uint8_t mode = fz_u8(&in) % 4;
    if (mode == 0) {
        /* routines: one fuzzed argument; invalid ones go to any registered routine */
        tsr_arg arg;
        const int cls = fza_arg(&in, &pool, &arg, 0);
        const uint32_t pick = fz_u32(&in);
        tsr_result res[8];
        if (cls == FZ_INVALID) {
            const int id = (int)(pick % (uint32_t)nfn);
            char info[64];
            if (tsr_fn_info(id, info, sizeof info) > 0 && strstr(info, "\"routine\"")) {
                const int rc = tsr_fn_routine(id, 1, &arg, 8, res);
                if (rc >= 0) fn_results_free(res, 8);
                fza_must_refuse(rc, "tsr_fn_routine");
            }
            const int rc = tsr_fn_routine(uq, 1, &arg, 8, res);
            if (rc >= 0) fn_results_free(res, 8);
            fza_must_refuse(rc, "tsr_fn_routine(np.unique)");
        } else if (cls == FZ_VALID && arg.kind == 3) {
            const int rc = tsr_fn_routine(uq, 1, &arg, 8, res);
            if (rc >= 0) fn_results_free(res, 8);
        }
    } else if (mode == 1) {
        /* gufunc (np.median over all axes): input and output arrays */
        tsr_array a, o;
        const int ca = fza_array(&in, &pool, &a), co = fza_array(&in, &pool, &o);
        uint64_t axm = a.ndim > 0 && a.ndim <= TSR_MAXDIM ? (((uint64_t)1 << a.ndim) - 1) : 0;
        if (ca == FZ_INVALID || co == FZ_INVALID) {
            fza_must_refuse(tsr_fn_gufunc(med, 1, &a, &axm, 0, NULL, 1, &o), "tsr_fn_gufunc");
        } else if (ca == FZ_VALID && co == FZ_VALID) {
            (void)tsr_fn_gufunc(med, 1, &a, &axm, 0, NULL, 1, &o);
        }
        int32_t ond[1];
        int64_t osh[32];
        const int shape_ok = a.dtype >= 0 && a.dtype < TSR_NDTYPES && a.ndim >= 0 && a.ndim <= TSR_MAXDIM;
        const int rcs = tsr_fn_gufunc_shape(med, 1, &a, &axm, 0, NULL, ond, osh);
        if (!shape_ok) fza_must_refuse(rcs, "tsr_fn_gufunc_shape");
    } else {
        /* ufunc / random / rvs: a strided operand set over a shared fuzzed shape */
        int32_t nd = (int32_t)(fz_u8(&in) % 4);
        if (fz_u8(&in) >= 250) nd = fz_u8(&in) & 1 ? -1 : 33;
        int64_t shape[4] = {0}, st[3 * 4];
        int valid = nd >= 0 && nd <= TSR_MAXDIM;
        __int128 count = 1;
        for (int32_t d = 0; d < (nd > 0 && nd <= 4 ? nd : 0); d++) {
            shape[d] = fza_dim(&in);
            if (shape[d] < 0) valid = 0;
            else if (shape[d] > 0 && count > ((__int128)1 << 80) / shape[d]) count = (__int128)1 << 80;   /* saturate first */
            else count *= shape[d];
        }
        if (count > (__int128)(INT64_MAX / 16)) valid = 0;
        const int nop = mode == 2 ? 2 : 3;
        void *ptr[3];
        int opok[3] = {1, 1, 1}, opbig[3] = {0, 0, 0};       /* per operand: random sees only operand 2 */
        for (int op = 0; op < nop; op++) {
            __int128 hi = 8, lo = 0;
            for (int32_t d = 0; d < (nd > 0 && nd <= 4 ? nd : 0); d++) {
                st[op * (nd > 0 ? nd : 1) + d] = fza_stride(&in, 8, 8);
                if (shape[d] > 1) {
                    const __int128 r = (__int128)(shape[d] - 1) * st[op * nd + d];
                    fza_acc(r < 0 ? &lo : &hi, r);
                    if (count != 0 && st[op * nd + d] % 8 != 0) opok[op] = 0;   /* misaligned float64 step */
                    if (count != 0 && op == nop - 1 && st[op * nd + d] == 0) opok[op] = 0;   /* aliasing output */
                }
            }
            if (count != 0 && (hi > (__int128)INT64_MAX || lo < (__int128)INT64_MIN)) opok[op] = 0;   /* empty: no element is visited */
            if (count == 0 || nd == 0) { ptr[op] = fz_take(&pool, 64); continue; }
            if (lo < 0 || hi > FZA_MAXBLOCK) { opbig[op] = 1; ptr[op] = fz_take(&pool, 64); continue; }
            ptr[op] = fz_take(&pool, (size_t)hi);
            if (!ptr[op]) opbig[op] = 1;
        }
        if (fz_u8(&in) >= 248) { const int k = fz_u8(&in) % nop; ptr[k] = NULL; opok[k] = 0; }
        const int vall = valid && opok[0] && opok[1] && (nop < 3 || opok[2]);
        const int ball = opbig[0] || opbig[1] || (nop == 3 && opbig[2]);
        int rc;
        uint64_t state[6];
        tsr_pcg64_seed(state, 0, 1, 0, 3);
        if (mode == 2) {
            if (vall && ball) goto done;
            rc = tsr_fn_ufunc(erf, 0, TSR_F64, nd, nd > 0 ? shape : NULL, 2, ptr, nd > 0 ? st : NULL);
            if (!vall) fza_must_refuse(rc, "tsr_fn_ufunc");
        } else {
            if (!(vall && ball)) {
                /* stats.norm rvs: loc, scale, out (zero-filled scale is fine: rvs gives NaN or refuses) */
                rc = tsr_fn_rvs(norm, state, nd, nd > 0 ? shape : NULL, 3, ptr, nd > 0 ? st : NULL);
                if (!vall) fza_must_refuse(rc, "tsr_fn_rvs");
            }
            const int vr = valid && opok[2];
            if (!(vr && opbig[2])) {
                void *rp[1] = {ptr[2]};
                int64_t rst[4];
                for (int32_t d = 0; d < (nd > 0 && nd <= 4 ? nd : 0); d++) rst[d] = st[2 * nd + d];
                rc = tsr_fn_random(sn, state, NULL, nd, nd > 0 ? shape : NULL, 1, rp, nd > 0 ? rst : NULL, TSR_F64);
                if (!vr) fza_must_refuse(rc, "tsr_fn_random");
            }
        }
    }
done:
    fz_release(&pool);
    if (tsr_allocated() != before) { fprintf(stderr, "fuzz_abi: leaked %lld bytes\n", (long long)(tsr_allocated() - before)); abort(); }
    return 0;
}
