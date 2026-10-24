/*
 * Fuzz target: the .npy header writer (tsr_npy_write_header) with fuzzed dtype, rank, shape (incl. negative
 * and 19-digit dimensions), fortran flag and a short output capacity. Oracle: the size query (buf NULL)
 * equals the length written; a too-small buffer is never written; a sufficient buffer is written only up
 * to the returned length (guard bytes); the header is a multiple of 64 and always parses back to the same
 * values; a shape the parser would refuse (byte count beyond int64) is refused by the writer too.
 */
#include "../src/internal.h"
#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    size_t i = 0;
#define BYTE() (i < size ? data[i++] : 0)
    const int dtype = (int)(int8_t)BYTE() % 9;            /* -8..8: includes invalid dtypes */
    const int32_t nd = (int32_t)(int8_t)BYTE() % 36;      /* -35..35: includes invalid ranks */
    const int fortran = BYTE() & 3;                       /* any nonzero means True */
    const int cap_lo = BYTE(), cap_hi = BYTE();
    const int64_t cap = (int64_t)(cap_lo | cap_hi << 8) % 1200;
    int64_t shape[40];
    for (int d = 0; d < 40; d++) {
        const uint8_t k = BYTE();
        if (k < 64) shape[d] = k % 8;
        else if (k < 128) shape[d] = INT64_MAX - (k % 4);
        else if (k < 160) shape[d] = -(int64_t)(k % 3) - 1;
        else { uint64_t v = 0; for (int b = 0; b < 8; b++) v |= (uint64_t)BYTE() << (8 * b); shape[d] = (int64_t)v; }
    }
#undef BYTE
    const int64_t need = tsr_npy_write_header(dtype, nd, shape, fortran, NULL, 0);
    const int valid = dtype >= 0 && dtype < TSR_NDTYPES && nd >= 0 && nd <= TSR_MAXDIM;
    int negative = 0, overflow = 0;
    for (int d = 0; valid && d < nd; d++) negative |= shape[d] < 0;
    if (valid && !negative) {
        /* the parser refuses shapes whose byte count overflows int64, and (a conservative digit check) any
           dimension >= 9223372036854775800 even when another dimension is 0: the writer must refuse them too */
        int64_t count = 1, isz = tsr_itemsize(dtype);
        for (int d = 0; d < nd; d++) if (shape[d] >= INT64_MAX - 7) overflow = 1;
        for (int d = 0; d < nd && !overflow; d++) {
            if (shape[d] && count > INT64_MAX / shape[d]) { overflow = 1; break; }
            count *= shape[d];
        }
        if (!overflow && count > INT64_MAX / isz) overflow = 1;
    }
    if (!valid || negative || overflow) {
        if (need >= 0) abort();                           /* invalid input must be refused */
        char big[2048];
        if (tsr_npy_write_header(dtype, nd, shape, fortran, big, sizeof big) != need) abort();
        return 0;
    }
    if (need <= 0 || need % 64 != 0 || need > 65536) abort();
    /* exact-size heap block plus a guard tail, so ASan and the guard both see an overrun */
    char *buf = (char *)malloc((size_t)(cap > 0 ? cap : 0) + 64);
    memset(buf, 0xA5, (size_t)(cap > 0 ? cap : 0) + 64);
    const int64_t got = tsr_npy_write_header(dtype, nd, shape, fortran, buf, cap);
    if (got != need) abort();                             /* size query == bytes (to be) written */
    for (int64_t k = (cap >= need ? need : 0); k < cap + 64; k++)
        if ((unsigned char)buf[k] != 0xA5) abort();       /* nothing written past the header / into a short buffer */
    if (cap >= need) {
        if (buf[need - 1] != '\n') abort();
        int dt2, fo2;
        int32_t nd2;
        int64_t sh2[32], off2;
        if (tsr_npy_header(buf, need, &dt2, &nd2, sh2, &fo2, &off2) != TSR_OK) abort();   /* written => readable */
        if (dt2 != dtype || nd2 != nd || fo2 != (fortran != 0) || off2 != need) abort();
        if (nd && memcmp(sh2, shape, sizeof(int64_t) * (size_t)nd) != 0) abort();
    }
    free(buf);
    return 0;
}
