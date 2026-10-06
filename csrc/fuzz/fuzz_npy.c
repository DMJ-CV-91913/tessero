/*
 * Fuzz target: the .npy header parser (tsr_npy_header), which reads files
 * from anywhere (uploads, shared storage). Oracle: on success the data offset
 * is inside the input and at least the header, the shape fits the element
 * count, and writing the parsed header back and parsing it again round-trips.
 */
#include "../src/internal.h"
#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    int dt = -1, fortran = -1;
    int32_t nd = -1;
    int64_t shape[32], off = -1;
    if (tsr_npy_header((const char *)data, (int64_t)size, &dt, &nd, shape, &fortran, &off) != TSR_OK) return 0;
    if (dt < 0 || dt >= TSR_NDTYPES || nd < 0 || nd > 32 || (fortran != 0 && fortran != 1)) abort();
    if (off < 10 || off > (int64_t)size) abort();
    int64_t count = 1;
    for (int d = 0; d < nd; d++) {
        if (shape[d] < 0) abort();
        if (shape[d] && count > INT64_MAX / shape[d]) abort();
        count *= shape[d];
    }
    char buf[4096];
    const int64_t hl = tsr_npy_write_header(dt, nd, shape, fortran, buf, sizeof(buf));
    if (hl <= 0 || hl > (int64_t)sizeof(buf) || hl % 64 != 0) abort();
    int dt2, fo2;
    int32_t nd2;
    int64_t sh2[32], off2;
    if (tsr_npy_header(buf, hl, &dt2, &nd2, sh2, &fo2, &off2) != TSR_OK) abort();
    if (dt2 != dt || nd2 != nd || fo2 != fortran || off2 != hl || memcmp(sh2, shape, sizeof(int64_t) * (size_t)nd) != 0) abort();
    return 0;
}
