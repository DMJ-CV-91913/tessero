/* Gather/scatter and boolean masks on contiguous data (the PHP side makes inputs contiguous first). */
#include "internal.h"

int tsr_take(int64_t itemsize, int64_t outer, int64_t axis_len, int64_t inner, const void *src,
             const int64_t *idx, int64_t nidx, void *dst)
{
    if (itemsize <= 0) return TSR_EARG;
    int64_t chunk = inner * itemsize;
    const char *s = (const char *)src;
    char *d = (char *)dst;
    for (int64_t k = 0; k < nidx; k++) {
        int64_t j = idx[k] < 0 ? idx[k] + axis_len : idx[k];
        if (j < 0 || j >= axis_len) return TSR_EINDEX;
    }
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t k = 0; k < nidx; k++) {
            int64_t j = idx[k] < 0 ? idx[k] + axis_len : idx[k];
            memcpy(d + (o * nidx + k) * chunk, s + (o * axis_len + j) * chunk, (size_t)chunk);
        }
    }
    return TSR_OK;
}

int tsr_put(int64_t itemsize, int64_t outer, int64_t axis_len, int64_t inner, void *dst, const int64_t *idx,
            int64_t nidx, const void *values)
{
    if (itemsize <= 0) return TSR_EARG;
    int64_t chunk = inner * itemsize;
    for (int64_t k = 0; k < nidx; k++) {
        int64_t j = idx[k] < 0 ? idx[k] + axis_len : idx[k];
        if (j < 0 || j >= axis_len) return TSR_EINDEX;
    }
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t k = 0; k < nidx; k++) {
            int64_t j = idx[k] < 0 ? idx[k] + axis_len : idx[k];
            memcpy((char *)dst + (o * axis_len + j) * chunk, (const char *)values + (o * nidx + k) * chunk, (size_t)chunk);
        }
    }
    return TSR_OK;
}

int64_t tsr_count_true(int64_t n, const uint8_t *mask)
{
    int64_t c = 0;
    for (int64_t i = 0; i < n; i++) c += mask[i] != 0;
    return c;
}

int64_t tsr_compress(int64_t itemsize, int64_t n, const void *src, const uint8_t *mask, void *dst)
{
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        if (mask[i]) {
            memcpy((char *)dst + k * itemsize, (const char *)src + i * itemsize, (size_t)itemsize);
            k++;
        }
    }
    return k;
}

/* dst[mask] = values; values_n == 1 broadcasts a scalar, otherwise it must equal the number of true entries. */
int tsr_mask_assign(int64_t itemsize, int64_t n, void *dst, const uint8_t *mask, const void *values, int64_t values_n)
{
    if (values_n != 1 && values_n != tsr_count_true(n, mask)) return TSR_EARG;
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        if (mask[i]) {
            memcpy((char *)dst + i * itemsize, (const char *)values + (values_n == 1 ? 0 : k) * itemsize, (size_t)itemsize);
            k++;
        }
    }
    return TSR_OK;
}
