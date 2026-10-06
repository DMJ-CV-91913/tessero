/*
 * Fuzz target: the JSON writer (tsr_array_json), which formats arbitrary bit
 * patterns (NaN payloads, subnormals, -0.0) straight into a caller buffer.
 * Oracle: the size query equals the written length, a short buffer is never
 * overrun (guard bytes), the output is valid JSON with the array's shape, and
 * the output is deterministic.
 */
#include "../src/internal.h"
#include <stdlib.h>

static const char *json_value(const char *p, const char *e, int depth, const tsr_array *a);

static const char *json_scalar(const char *p, const char *e)
{
    if (e - p >= 4 && !memcmp(p, "null", 4)) return p + 4;
    if (e - p >= 4 && !memcmp(p, "true", 4)) return p + 4;
    if (e - p >= 5 && !memcmp(p, "false", 5)) return p + 5;
    const char *s = p;
    if (p < e && *p == '-') p++;
    if (p >= e || *p < '0' || *p > '9') return NULL;
    while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) p++;
    char tmp[64];
    if (p - s >= (long)sizeof(tmp)) return NULL;
    memcpy(tmp, s, (size_t)(p - s));
    tmp[p - s] = '\0';
    char *end;
    strtod(tmp, &end);
    return *end == '\0' ? p : NULL;
}

static const char *json_value(const char *p, const char *e, int depth, const tsr_array *a)
{
    if (depth == a->ndim) {
        if (a->dtype == TSR_C128) {                     /* [re, im] */
            if (p >= e || *p++ != '[') return NULL;
            if (!(p = json_scalar(p, e)) || p >= e || *p++ != ',') return NULL;
            if (!(p = json_scalar(p, e)) || p >= e || *p++ != ']') return NULL;
            return p;
        }
        return json_scalar(p, e);
    }
    if (p >= e || *p++ != '[') return NULL;
    for (int64_t i = 0; i < a->shape[depth]; i++) {
        if (i && (p >= e || *p++ != ',')) return NULL;
        if (!(p = json_value(p, e, depth + 1, a))) return NULL;
    }
    return p < e && *p == ']' ? p + 1 : NULL;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 4) return 0;
    const int dt = data[0] % TSR_NDTYPES;
    const int nd = data[1] % 4;
    int64_t shape[4] = {1, 1, 1, 1}, n = 1;
    for (int d = 0; d < nd; d++) { shape[d] = 1 + (data[2 + d % 2] >> (d * 2)) % 5; n *= shape[d]; }
    const int64_t item = tsr_itemsize(dt);
    char *buf = calloc((size_t)(n * item), 1);
    const size_t avail = size - 4 < (size_t)(n * item) ? size - 4 : (size_t)(n * item);
    memcpy(buf, data + 4, avail);
    tsr_array a;
    tsr_array_init(&a, dt, nd, shape, buf);
    if (dt == TSR_BOOL) for (int64_t i = 0; i < n; i++) buf[i] &= 1;

    const int64_t need = tsr_array_json(&a, NULL, 0);
    if (need <= 0) abort();
    char *out = malloc((size_t)need + 16);
    memset(out, 0x5a, (size_t)need + 16);
    if (tsr_array_json(&a, out, need) != need) abort();
    for (int i = 0; i < 16; i++) if ((unsigned char)out[need + i] != 0x5a) abort();
    const char *end = json_value(out, out + need, 0, &a);
    if (end != out + need) abort();
    char *again = malloc((size_t)need);
    tsr_array_json(&a, again, need);
    if (memcmp(out, again, (size_t)need) != 0) abort();
    char *shorter = malloc((size_t)(need / 2 + 1));
    tsr_array_json(&a, shorter, need / 2);             /* must stop at the cap (ASan checks) */
    free(shorter);
    free(again);
    free(out);
    free(buf);
    return 0;
}
