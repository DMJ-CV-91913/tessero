/*
 * Fuzz target: the slice parser (tsr_array_slice), which reads untrusted text
 * when an application passes request input to $a['...'] or slice().
 *
 * Input: byte 0 = ndim (1..6), bytes 1..6 = dimension lengths (0..15),
 * byte 7 = dtype, the rest = the spec string. Oracle: on success every
 * element of the view lies inside the source buffer, the view has at most 32
 * dimensions, and non-negative lengths.
 */
#include "../src/internal.h"
#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 8) return 0;
    const int nd = 1 + data[0] % 6;
    int64_t shape[6];
    int64_t n = 1;
    for (int d = 0; d < nd; d++) { shape[d] = data[1 + d] % 16; n *= shape[d]; }
    const int dt = data[7] % TSR_NDTYPES;
    const int64_t item = tsr_itemsize(dt);
    const int64_t bytes = n * item > 0 ? n * item : 1;
    char *buf = malloc((size_t)bytes);
    char *spec = malloc(size - 8 + 1);
    memcpy(spec, data + 8, size - 8);
    spec[size - 8] = '\0';

    tsr_array a, v;
    tsr_array_init(&a, dt, nd, shape, buf);
    if (tsr_array_slice(&a, spec, &v) == TSR_OK) {
        if (v.ndim < 0 || v.ndim > TSR_MAXDIM) abort();
        int64_t lo = v.offset, hi = v.offset, count = 1;
        for (int d = 0; d < v.ndim; d++) {
            if (v.shape[d] < 0) abort();
            count *= v.shape[d];
            if (v.shape[d] > 0) {
                const int64_t span = (v.shape[d] - 1) * v.strides[d];
                if (span < 0) lo += span; else hi += span;
            }
        }
        if (count > 0 && (lo < 0 || hi + item > n * item)) abort();   /* a view element outside the buffer */
        if (count > 0) {
            volatile char sink = *((char *)v.data + lo);             /* ASan checks the extremes too */
            sink = *((char *)v.data + hi + item - 1);
            (void)sink;
        }
    }
    free(spec);
    free(buf);
    return 0;
}
