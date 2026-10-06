/*
 * Array metadata in C: a tsr_array describes a strided n-d view of a block
 * of memory (it never owns memory). Used by the native Zend extension, and
 * available to FFI callers, for:
 *
 *   - NumPy broadcasting (shape resolution, zero strides for stretched axes)
 *   - views: slicing from a string spec, transposes, reshapes, all without copies
 *   - dtype-polymorphic element-wise dispatch on two arrays
 *   - direct-to-text JSON serialisation of any strided view
 */
#include "internal.h"
#include "ops.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int tsr_array_init(tsr_array *a, int dtype, int32_t ndim, const int64_t *shape, void *data)
{
    if (!a || tsr_itemsize(dtype) == 0 || ndim < 0) return TSR_EARG;
    if (ndim > TSR_MAXDIM) return TSR_EDIM;
    if (ndim > 0 && (!shape || tsr_shape_size(ndim, shape) < 0)) return TSR_EARG;   /* negative or overflowing shape */
    memset(a, 0, sizeof(*a));
    a->data = data;
    a->offset = 0;
    a->ndim = ndim;
    a->dtype = dtype;
    int64_t stride = tsr_itemsize(dtype);
    for (int d = ndim - 1; d >= 0; d--) {
        if (shape[d] < 0) return TSR_EARG;
        a->shape[d] = shape[d];
        a->strides[d] = stride;
        /* wrapping: only a zero-size array can get here with a product beyond int64, and its strides are unused */
        if (d > 0) stride = (int64_t)((uint64_t)stride * (uint64_t)(shape[d] > 0 ? shape[d] : 1));
    }
    return TSR_OK;
}

int64_t tsr_array_size(const tsr_array *a)
{
    int64_t n = 1;
    for (int d = 0; d < a->ndim; d++) n *= a->shape[d];
    return n;
}

int tsr_array_is_contiguous(const tsr_array *a)
{
    int64_t expected = tsr_itemsize(a->dtype);
    if (tsr_array_size(a) <= 1) return 1;
    for (int d = a->ndim - 1; d >= 0; d--) {
        if (a->shape[d] == 1) continue;
        if (a->strides[d] != expected) return 0;
        expected *= a->shape[d];
    }
    return 1;
}

/* NumPy rule: align trailing axes; sizes must match or one of them be 1. */
int tsr_broadcast_shape(int32_t nda, const int64_t *sa, int32_t ndb, const int64_t *sb, int32_t *out_nd, int64_t *out_shape)
{
    int32_t nd = nda > ndb ? nda : ndb;
    if (nd > TSR_MAXDIM) return TSR_EDIM;
    for (int32_t i = 0; i < nd; i++) {
        int32_t ia = nda - nd + i, ib = ndb - nd + i;
        int64_t da = ia >= 0 ? sa[ia] : 1, db = ib >= 0 ? sb[ib] : 1;
        if (da == db || db == 1) out_shape[i] = da;
        else if (da == 1) out_shape[i] = db;
        else return TSR_ESHAPE;
    }
    if (tsr_shape_size(nd, out_shape) < 0) return TSR_ESHAPE;   /* broadcast result too large to address */
    *out_nd = nd;
    return TSR_OK;
}

/* Strides that make `a` read as if it had `shape` (stretched axes get stride 0; no data is copied). */
int tsr_broadcast_strides(const tsr_array *a, int32_t nd, const int64_t *shape, int64_t *out_strides)
{
    if (nd < a->ndim) return TSR_ESHAPE;
    int32_t pad = nd - a->ndim;
    for (int32_t i = 0; i < nd; i++) {
        if (i < pad) { out_strides[i] = 0; continue; }
        int64_t len = a->shape[i - pad];
        if (len == shape[i]) out_strides[i] = a->strides[i - pad];
        else if (len == 1) out_strides[i] = 0;
        else return TSR_ESHAPE;
    }
    return TSR_OK;
}

/* ------------------------------------------------------------------ views */

int tsr_array_transpose(const tsr_array *src, const int32_t *axes, tsr_array *out)
{
    tsr_array t = *src;
    int32_t n = src->ndim;
    int seen[TSR_MAXDIM] = {0};
    for (int32_t i = 0; i < n; i++) {
        int32_t ax = axes ? axes[i] : n - 1 - i;
        if (ax < 0) ax += n;
        if (ax < 0 || ax >= n || seen[ax]) return TSR_EARG;
        seen[ax] = 1;
        t.shape[i] = src->shape[ax];
        t.strides[i] = src->strides[ax];
    }
    *out = t;
    return TSR_OK;
}

/* Reshape as a view; returns TSR_ECONTIG when the source is not contiguous (the caller copies first). */
int tsr_array_reshape(const tsr_array *src, int32_t nd, const int64_t *shape, tsr_array *out)
{
    if (nd < 0 || nd > TSR_MAXDIM) return nd > TSR_MAXDIM ? TSR_EDIM : TSR_EARG;
    int64_t size = tsr_array_size(src), known = 1;
    int32_t unknown = -1;
    int64_t sh[TSR_MAXDIM];
    for (int32_t i = 0; i < nd; i++) {
        sh[i] = shape[i];
        if (shape[i] == -1) {
            if (unknown >= 0) return TSR_EARG;
            unknown = i;
        } else if (shape[i] < 0) {
            return TSR_EARG;
        } else {
            if (shape[i] != 0 && known > (INT64_MAX / 16) / shape[i]) return TSR_ESHAPE;   /* would wrap around */
            known *= shape[i];
        }
    }
    if (unknown >= 0) {
        if (known == 0 || size % known != 0) return TSR_ESHAPE;
        sh[unknown] = size / known;
    } else if (known != size) {
        return TSR_ESHAPE;
    }
    if (!tsr_array_is_contiguous(src)) return TSR_ECONTIG;
    tsr_array r;
    const int rc = tsr_array_init(&r, src->dtype, nd, sh, src->data);
    if (rc != TSR_OK) return rc;
    r.offset = src->offset;
    *out = r;
    return TSR_OK;
}

/* Skip blanks, never past `end` (items are delimited by commas, not by the terminator). */
static const char *skip_ws(const char *p, const char *end)
{
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    return p;
}

/* Parse an optional signed integer; returns 1 if present. */
static int parse_int(const char **pp, const char *end, int64_t *v)
{
    const char *p = skip_ws(*pp, end);
    const char *q = p;
    int neg = 0;
    if (q < end && (*q == '-' || *q == '+')) { neg = *q == '-'; q++; }
    if (q >= end || !isdigit((unsigned char)*q)) {
        if (q != p) return -1; /* a lone sign */
        *pp = p;
        return 0;
    }
    int64_t x = 0;
    while (q < end && isdigit((unsigned char)*q)) {
        if (x > (INT64_MAX - 9) / 10) return -1;
        x = x * 10 + (*q - '0');
        q++;
    }
    *v = neg ? -x : x;
    *pp = skip_ws(q, end);
    return 1;
}

typedef struct { int kind; int64_t start, stop, step; int has_start, has_stop; } spec_item; /* kind 0 int, 1 slice, 2 newaxis, 3 ellipsis */

static int parse_item(const char *p, const char *end, spec_item *it)
{
    p = skip_ws(p, end);                                   /* bounded: a blank item ("1, ,2") must not run past its comma */
    while (end > p && (end[-1] == ' ' || end[-1] == '\t')) end--;
    size_t len = (size_t)(end - p);
    memset(it, 0, sizeof(*it));
    if ((len == 4 && strncmp(p, "None", 4) == 0) || (len == 7 && strncmp(p, "newaxis", 7) == 0)) { it->kind = 2; return TSR_OK; }
    if (len == 3 && strncmp(p, "...", 3) == 0) { it->kind = 3; return TSR_OK; }
    if (memchr(p, ':', len) == NULL) {
        int64_t v;
        const char *q = p;
        if (parse_int(&q, end, &v) != 1 || q != end) return TSR_EARG;
        it->kind = 0;
        it->start = v;
        return TSR_OK;
    }
    it->kind = 1;
    it->step = 1;
    const char *q = p;
    int r = parse_int(&q, end, &it->start);
    if (r < 0) return TSR_EARG;
    it->has_start = r;
    if (q >= end || *q != ':') return TSR_EARG;
    q++;
    r = parse_int(&q, end, &it->stop);
    if (r < 0) return TSR_EARG;
    it->has_stop = r;
    if (q < end && *q == ':') {
        q++;
        int64_t st;
        r = parse_int(&q, end, &st);
        if (r < 0) return TSR_EARG;
        if (r == 1) {
            if (st == 0) return TSR_EARG;
            it->step = st;
        }
    }
    return q == end ? TSR_OK : TSR_EARG;
}

/*
 * Basic indexing from a NumPy-style spec: "1:10:2, ::-1", "..., 0", "2, None, :".
 * Returns a view in *out (same data pointer, new offset/shape/strides).
 * TSR_EARG on a syntax error, TSR_EINDEX when an integer index is out of range.
 */
int tsr_array_slice(const tsr_array *src, const char *spec, tsr_array *out)
{
    if (!src || !spec || !out) return TSR_EARG;
    spec_item items[TSR_MAXDIM * 2];
    int n = 0, consumes = 0, ellipsis = 0;
    const char *p = spec;
    const char *end = spec + strlen(spec);
    if (skip_ws(p, end) == end) return TSR_EARG;
    while (p <= end) {
        const char *comma = memchr(p, ',', (size_t)(end - p));
        const char *stop = comma ? comma : end;
        if (n >= TSR_MAXDIM * 2) return TSR_EDIM;
        int rc = parse_item(p, stop, &items[n]);
        if (rc != TSR_OK) return rc;
        if (items[n].kind == 3) { if (ellipsis++) return TSR_EARG; }
        else if (items[n].kind != 2) consumes++;
        n++;
        if (!comma) break;
        p = comma + 1;
    }
    if (consumes > src->ndim) return TSR_EINDEX;

    tsr_array r;
    memset(&r, 0, sizeof(r));
    r.data = src->data;
    r.dtype = src->dtype;
    r.offset = src->offset;
    int32_t axis = 0, od = 0;
    for (int i = 0; i < n; i++) {
        spec_item *it = &items[i];
        if (it->kind == 3) {
            for (int k = 0; k < src->ndim - consumes; k++) {
                if (od >= TSR_MAXDIM) return TSR_EDIM;
                r.shape[od] = src->shape[axis];
                r.strides[od] = src->strides[axis];
                od++;
                axis++;
            }
            continue;
        }
        if (it->kind == 2) {
            if (od >= TSR_MAXDIM) return TSR_EDIM;
            r.shape[od] = 1;
            r.strides[od] = 0;
            od++;
            continue;
        }
        int64_t len = src->shape[axis], stride = src->strides[axis];
        if (it->kind == 0) {
            int64_t j = it->start < 0 ? it->start + len : it->start;
            if (j < 0 || j >= len) return TSR_EINDEX;
            r.offset += j * stride;
            axis++;
            continue;
        }
        int64_t start, stop, step = it->step, count;
        if (step > 0) {
            start = it->has_start ? it->start : 0;
            stop = it->has_stop ? it->stop : len;
            start = start < 0 ? (start + len < 0 ? 0 : start + len) : (start > len ? len : start);
            stop = stop < 0 ? (stop + len < 0 ? 0 : stop + len) : (stop > len ? len : stop);
            count = stop > start ? 1 + (stop - start - 1) / step : 0;       /* no overflow for huge steps */
        } else {
            start = it->has_start ? it->start : len - 1;
            stop = it->has_stop ? it->stop : -1;
            start = start < 0 ? (start + len < -1 ? -1 : start + len) : (start > len - 1 ? len - 1 : start);
            if (it->has_stop) stop = stop < 0 ? (stop + len < -1 ? -1 : stop + len) : (stop > len - 1 ? len - 1 : stop);
            count = start > stop ? 1 + (start - stop - 1) / (-step) : 0;
        }
        if (od >= TSR_MAXDIM) return TSR_EDIM;
        if (count > 0) r.offset += start * stride;
        r.shape[od] = count;
        r.strides[od] = count > 1 ? stride * step : stride;             /* |step| < len here, so no overflow */
        od++;
        axis++;
    }
    for (; axis < src->ndim; axis++) {
        if (od >= TSR_MAXDIM) return TSR_EDIM;
        r.shape[od] = src->shape[axis];
        r.strides[od] = src->strides[axis];
        od++;
    }
    r.ndim = od;
    *out = r;
    return TSR_OK;
}

/* Element pointer for a full index tuple (negative indices count from the end). */
void *tsr_array_at(const tsr_array *a, const int64_t *index)
{
    int64_t off = a->offset;
    for (int d = 0; d < a->ndim; d++) {
        int64_t i = index[d] < 0 ? index[d] + a->shape[d] : index[d];
        if (i < 0 || i >= a->shape[d]) return NULL;
        off += i * a->strides[d];
    }
    return (char *)a->data + off;
}

/* out = a (op) b with broadcasting; `out` must already have the broadcast shape. */
int tsr_array_binary(int op, const tsr_array *a, const tsr_array *b, tsr_array *out)
{
    if (a->dtype != b->dtype) return TSR_ETYPE;
    int64_t sa[TSR_MAXDIM], sb[TSR_MAXDIM];
    int rc = tsr_broadcast_strides(a, out->ndim, out->shape, sa);
    if (rc == TSR_OK) rc = tsr_broadcast_strides(b, out->ndim, out->shape, sb);
    if (rc != TSR_OK) return rc;
    return tsr_binary(op, a->dtype, out->ndim, out->shape, (const char *)a->data + a->offset, sa,
                      (const char *)b->data + b->offset, sb, (char *)out->data + out->offset, out->strides);
}

/* ------------------------------------------------------------------ JSON */

typedef struct { char *buf; int64_t cap, len; } jbuf;

static void jput(jbuf *j, const char *s, int64_t n)
{
    if (j->len + n <= j->cap) memcpy(j->buf + j->len, s, (size_t)n);
    j->len += n;
}

/* Shortest decimal that round-trips (what PHP's json_encode prints with serialize_precision=-1). */
static int fmt_double(char *out, double v)
{
    if (!isfinite(v)) { memcpy(out, "null", 4); return 4; }
    if (v == 0.0) {
        if (signbit(v)) { memcpy(out, "-0.0", 4); return 4; }
        memcpy(out, "0.0", 3);
        return 3;
    }
    /* fast path: integral values print as digits + ".0" without printf */
    if (fabs(v) < 1e15 && v == (double)(int64_t)v) {
        int64_t iv = (int64_t)v;
        char tmp[24];
        int k = 0, n = 0;
        uint64_t u = iv < 0 ? (uint64_t)(-iv) : (uint64_t)iv;
        do { tmp[k++] = (char)('0' + u % 10); u /= 10; } while (u);
        if (iv < 0) out[n++] = '-';
        while (k) out[n++] = tmp[--k];
        out[n++] = '.';
        out[n++] = '0';
        return n;
    }
    int n = 0;
    for (int prec = 15; prec <= 17; prec++) {
        n = snprintf(out, 40, "%.*g", prec, v);
        if (strtod(out, NULL) == v) break;
    }
    /* keep a float recognisable as a float: 2 -> 2.0 (as json_encode with PRESERVE_ZERO_FRACTION) */
    if (!memchr(out, '.', (size_t)n) && !memchr(out, 'e', (size_t)n) && !memchr(out, 'n', (size_t)n)) {
        out[n++] = '.';
        out[n++] = '0';
    }
    return n;
}

static int fmt_float(char *out, float v)
{
    if (!isfinite(v)) { memcpy(out, "null", 4); return 4; }
    int n = 0;
    for (int prec = 6; prec <= 9; prec++) {
        n = snprintf(out, 40, "%.*g", prec, (double)v);
        if (strtof(out, NULL) == v) break;
    }
    if (!memchr(out, '.', (size_t)n) && !memchr(out, 'e', (size_t)n)) {
        out[n++] = '.';
        out[n++] = '0';
    }
    return n;
}

static void json_elem(jbuf *j, int dtype, const char *p)
{
    char tmp[96];
    int n = 0;
    switch (dtype) {
    case TSR_F64: n = fmt_double(tmp, *(const double *)p); break;
    case TSR_F32: n = fmt_float(tmp, *(const float *)p); break;
    case TSR_I64: n = snprintf(tmp, sizeof(tmp), "%lld", (long long)*(const int64_t *)p); break;
    case TSR_I32: n = snprintf(tmp, sizeof(tmp), "%d", *(const int32_t *)p); break;
    case TSR_U8: n = snprintf(tmp, sizeof(tmp), "%u", (unsigned)*(const uint8_t *)p); break;
    case TSR_BOOL:
        if (*(const uint8_t *)p) { memcpy(tmp, "true", 4); n = 4; }
        else { memcpy(tmp, "false", 5); n = 5; }
        break;
    case TSR_C128:
        tmp[0] = '[';
        n = 1 + fmt_double(tmp + 1, ((const double *)p)[0]);
        tmp[n++] = ',';
        n += fmt_double(tmp + n, ((const double *)p)[1]);
        tmp[n++] = ']';
        break;
    }
    jput(j, tmp, n);
}

static void json_dim(jbuf *j, const tsr_array *a, int d, const char *p)
{
    if (d == a->ndim) {
        json_elem(j, a->dtype, p);
        return;
    }
    jput(j, "[", 1);
    for (int64_t i = 0; i < a->shape[d]; i++) {
        if (i) jput(j, ",", 1);
        json_dim(j, a, d + 1, p + i * a->strides[d]);
    }
    jput(j, "]", 1);
}

/*
 * Write `a` as nested JSON arrays into buf (at most cap bytes, no terminator).
 * Returns the full length required; call with cap = 0 to size the buffer.
 * NaN and +/-Inf become null (JSON has no representation for them).
 */
int64_t tsr_array_json(const tsr_array *a, char *buf, int64_t cap)
{
    jbuf j = {buf, buf ? cap : 0, 0};
    json_dim(&j, a, 0, (const char *)a->data + a->offset);
    return j.len;
}
