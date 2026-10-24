/*
 * Fuzz target: the dtype-string parsers over arbitrary short strings:
 *   - the .npy 'descr' parser (descr_dtype in mmap.c, static), reached through tsr_npy_header with a
 *     well-formed header around the fuzzed descr;
 *   - io_dtype (np_io.c), reached through np.fromfile(dtype=...);
 *   - the dtype-name tables of np.promote_types / np.can_cast / np.result_type (np_numeric.c, np_shape.c).
 * Input: a selector byte, then up to three strings (a length byte then bytes; a few "names" bytes pick
 * common spellings so mutations explore near-valid names). Oracle: ASan/UBSan clean, an accepted descr
 * yields a valid dtype, results well formed, no leaked tsr allocations.
 */
#include "fuzz_common.h"
#include <unistd.h>

static char g_path[64];

static void cleanup(void) { if (g_path[0]) unlink(g_path); }

static void ensure_file(void)
{
    if (g_path[0]) return;
    snprintf(g_path, sizeof g_path, "/tmp/tsr_fuzz_dtype_XXXXXX");
    const int fd = mkstemp(g_path);
    if (fd < 0) { perror("mkstemp"); abort(); }
    const unsigned char bytes[40] = {0, 0, 0, 0, 0, 0, 0xf0, 0x3f, 1, 2, 3, 4, 5, 6, 7, 8, 0xff, 0xff, 0xff, 0xff};
    if (write(fd, bytes, sizeof bytes) != (ssize_t)sizeof bytes) abort();
    close(fd);
    atexit(cleanup);
}

static const char *pick(fz_in *in, char *buf)
{
    static const char *const N[] = {"float64", "f8", "<f8", "|u1", "=i4", ">f8", "complex128", "c16", "<c16",
                                    "bool", "?", "|b1", "int", "intp", "uint8", "B", "D", "d", "same_kind", "safe"};
    const uint8_t k = fz_u8(in);
    if (k < 96) return N[k % (sizeof N / sizeof N[0])];
    return fz_str(in, buf, 24, 20);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_init();
    ensure_file();
    static int ids[4] = {-2};
    if (ids[0] == -2) {
        ids[0] = tsr_fn_find("np.fromfile");
        ids[1] = tsr_fn_find("np.promote_types");
        ids[2] = tsr_fn_find("np.can_cast");
        ids[3] = tsr_fn_find("np.result_type");
        for (int i = 0; i < 4; i++) if (ids[i] < 0) abort();
    }
    fz_in in = {data, size};
    char s[3][24];
    const uint8_t sel = fz_u8(&in);
    const char *a = pick(&in, s[0]), *b = pick(&in, s[1]), *c = pick(&in, s[2]);

    switch (sel % 5) {
    case 0: {                                           /* .npy descr via a well-formed v1 header */
        char hdr[256];
        int n = snprintf(hdr + 10, sizeof hdr - 10, "{'descr': '%s', 'fortran_order': False, 'shape': (%d,), }",
                         a, (int)(fz_u8(&in) % 4));
        if (n < 0 || n > 200) break;
        const int total = 10 + n + 1, pad = (64 - total % 64) % 64;
        memcpy(hdr, "\x93NUMPY\x01\x00", 8);
        const int hl = n + pad + 1;
        hdr[8] = (char)(hl & 0xff);
        hdr[9] = (char)(hl >> 8);
        memset(hdr + 10 + n, ' ', (size_t)pad);
        hdr[10 + n + pad] = '\n';
        const int64_t len = 10 + hl;
        char *exact = (char *)malloc((size_t)len);       /* exact size so ASan sees overreads */
        memcpy(exact, hdr, (size_t)len);
        int dt = -1, fo = -1;
        int32_t nd = -1;
        int64_t sh[32], off = -1;
        if (tsr_npy_header(exact, len, &dt, &nd, sh, &fo, &off) == TSR_OK) {
            if (dt < 0 || dt >= TSR_NDTYPES || nd != 1 || fo != 0 || off != len) abort();
        }
        free(exact);
        break;
    }
    case 1: {                                           /* np.fromfile(fname, dtype, count, sep) */
        tsr_arg args[4] = {fz_arg_str(g_path), fz_arg_str(a), fz_arg_int(fz_u8(&in) % 4 - 1),
                           (sel & 0x20) ? fz_arg_str(" ") : fz_arg_none()};
        fz_call(ids[0], 4, args);
        break;
    }
    case 2: {
        tsr_arg args[2] = {fz_arg_str(a), fz_arg_str(b)};
        fz_call(ids[1], 2, args);
        break;
    }
    case 3: {
        tsr_arg args[3] = {fz_arg_str(a), fz_arg_str(b), (sel & 0x20) ? fz_arg_str(c) : fz_arg_none()};
        fz_call(ids[2], 3, args);
        break;
    }
    default: {
        tsr_arg items[3] = {fz_arg_str(a), fz_arg_str(b), fz_arg_str(c)};
        tsr_arg seq = fz_arg_none();
        seq.kind = 5;
        seq.items = items;
        seq.count = 1 + (sel >> 5) % 3;
        fz_call(ids[3], 1, &seq);
        break;
    }
    }
    return 0;
}
