/*
 * Fuzz target: the text readers np.loadtxt, np.genfromtxt and np.fromfile (text with sep=..., and binary
 * with offset/count). These routines take a file NAME and fopen it, so each iteration writes the file
 * content to one private temporary file and passes its path. Input: a selector byte, then the options
 * (delimiter, comments, skiprows / skip_header / skip_footer, dtype, count, sep, offset; numeric options
 * are small ints, extreme int64s or palette doubles incl. NaN), then the file content (the rest).
 * Oracle: ASan/UBSan clean, well-formed results, no leaked tsr allocations.
 */
#include "fuzz_common.h"
#include <unistd.h>

static char g_path[64];

static void cleanup(void) { if (g_path[0]) unlink(g_path); }

static void ensure_file(void)
{
    if (g_path[0]) return;
    snprintf(g_path, sizeof g_path, "/tmp/tsr_fuzz_textio_XXXXXX");
    const int fd = mkstemp(g_path);
    if (fd < 0) { perror("mkstemp"); abort(); }
    close(fd);
    atexit(cleanup);
}

/* a count-like option: absent, a small int, an extreme int64, or a double (NaN, Inf, 1e18, -3.5 ...) */
static tsr_arg count_arg(fz_in *in)
{
    const uint8_t k = fz_u8(in);
    switch (k % 5) {
    case 0: return fz_arg_none();
    case 1: case 2: return fz_arg_int((int64_t)(fz_u8(in) % 8) - 2);
    case 3: {
        static const int64_t X[] = {INT64_MIN, INT64_MAX, INT64_MIN + 1, -1, (int64_t)1 << 62, 4096, 1 << 20};
        return fz_arg_int(X[fz_u8(in) % (sizeof X / sizeof X[0])]);
    }
    default:
#ifdef FUZZ_SKIP_KNOWN
        /* open finding F10: skiprows / skip_header / skip_footer / count / offset given as a non-integer number
           are cast to int64 unchecked (NaN, Inf, 1e300). Pass integers only. */
        return fz_arg_int((int64_t)(fz_u8(in) % 8) - 2);
#else
        return fz_arg_num(fz_double(in));
#endif
    }
}

/* a short option string: absent, a common choice, or fuzzed bytes */
static tsr_arg str_arg(fz_in *in, char *buf, const char *const *common, int ncommon)
{
    const uint8_t k = fz_u8(in);
    if (k < 48) return fz_arg_none();
    if (k < 200) return fz_arg_str(common[k % ncommon]);
    return fz_arg_str(fz_str(in, buf, 8, 4));
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_init();
    ensure_file();
    static int ids[3] = {-2};
    if (ids[0] == -2) {
        ids[0] = tsr_fn_find("np.loadtxt");
        ids[1] = tsr_fn_find("np.genfromtxt");
        ids[2] = tsr_fn_find("np.fromfile");
        for (int i = 0; i < 3; i++) if (ids[i] < 0) abort();
    }
    static const char *const DELIM[] = {",", ";", " ", "\t", "|", ",,", ""};
    static const char *const COMM[] = {"#", "//", "%", "", "1", ","};
    static const char *const SEP[] = {",", " ", "\n", ";", "", " ,", "\t"};
    static const char *const DT[] = {"float64", "float32", "int64", "int32", "uint8", "complex128", "f8", "i4", "u1", "d", "bool"};
    fz_in in = {data, size};
    char b1[8], b2[8], b3[8];
    tsr_arg args[6];
    int nargs = 0;
    const uint8_t sel = fz_u8(&in);
    const int which = sel % 3;
    args[nargs++] = fz_arg_str(g_path);
    if (which == 0) {                                     /* loadtxt(fname, delimiter, skiprows, comments) */
        args[nargs++] = str_arg(&in, b1, DELIM, 7);
        args[nargs++] = count_arg(&in);
        args[nargs++] = str_arg(&in, b2, COMM, 6);
    } else if (which == 1) {                              /* genfromtxt(fname, delimiter, skip_header, skip_footer, comments) */
        args[nargs++] = str_arg(&in, b1, DELIM, 7);
        args[nargs++] = count_arg(&in);
        args[nargs++] = count_arg(&in);
        args[nargs++] = str_arg(&in, b2, COMM, 6);
    } else {                                              /* fromfile(fname, dtype, count, sep, offset) */
        args[nargs++] = str_arg(&in, b1, DT, 11);
        args[nargs++] = count_arg(&in);
        args[nargs++] = str_arg(&in, b3, SEP, 7);
        args[nargs++] = count_arg(&in);
    }
    if (sel & 0x80) nargs = 1 + fz_u8(&in) % nargs;     /* trailing arguments absent */
#ifdef FUZZ_SKIP_KNOWN
    /* open finding F10: fromfile computes file_size - offset in int64 unchecked; keep offset small */
    if (which == 2 && nargs > 4 && args[4].kind == 1 && (args[4].ival < -4096 || args[4].ival > 4096))
        args[4] = fz_arg_int(args[4].ival % 4096);
#endif

    FILE *f = fopen(g_path, "wb");
    if (!f) abort();
    if (in.n && fwrite(in.p, 1, in.n, f) != in.n) abort();
    fclose(f);
    fz_call(ids[which], nargs, args);
    return 0;
}
