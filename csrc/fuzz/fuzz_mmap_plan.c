/*
 * Fuzz target: memory-map planning (tsr_mmap_plan), the arithmetic that turns
 * user-supplied shape, dtype and offset plus a file size into the window to
 * map. Overflow here would map too little and let an array read past the
 * mapping. Oracle: on success the window is aligned, contains exactly the
 * data, and never extends a read-only or copy-on-write file.
 */
#include "../src/internal.h"
#include <stdlib.h>

static int64_t rd64(const uint8_t **p, size_t *n)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) { v = (v << 8) | (*n ? (*p)++[0] : 0); if (*n) (*n)--; }
    return (int64_t)v;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 4) return 0;
    const int dt = data[0] % (TSR_NDTYPES + 1);          /* includes one invalid dtype */
    const int mode = data[1] % 5;                       /* includes one invalid mode */
    int32_t nd = (int32_t)(data[2] % 36) - 1;          /* -1 (derive), 0..34 */
    const int64_t gran = (int64_t)1 << (data[3] % 20);
    const uint8_t *p = data + 4;
    size_t n = size - 4;
    int64_t shape[40];
    for (int d = 0; d < 34; d++) shape[d] = d < (nd > 0 ? nd : 0) ? rd64(&p, &n) >> (data[3] & 0x3f) : 0;
    const int64_t offset = rd64(&p, &n) >> 20;
    const int64_t fsize = rd64(&p, &n) >> 16;
    int64_t bytes = -1, start = -1, len = -1, ns = -2;
    const int32_t nd_in = nd;
    if (tsr_mmap_plan(dt, mode, &nd, shape, offset, fsize, gran, &bytes, &start, &len, &ns) != TSR_OK) return 0;
    if (dt >= TSR_NDTYPES || mode > 3 || nd < 0 || nd > 32 || (nd_in >= 0 && nd != nd_in)) abort();
    const int64_t item = tsr_itemsize(dt);
    int64_t count = 1;
    for (int d = 0; d < nd; d++) count *= shape[d];      /* validated not to overflow */
    if (count <= 0 || bytes != count * item) abort();
    if (start < 0 || start % gran != 0 || start > offset || offset - start >= gran) abort();
    if (start + len != offset + bytes) abort();
    const int64_t have = mode == 2 ? 0 : fsize;
    if (ns == -1 && have < offset + bytes) abort();
    if (ns != -1 && (mode == 0 || mode == 3 || ns != offset + bytes)) abort();
    return 0;
}
