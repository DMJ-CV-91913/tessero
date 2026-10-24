/*
 * Fuzz target: every registry routine (kind "routine", tsr_fn_routine) with fuzzed arguments. The first two
 * bytes pick the routine (an index into the sorted list of routine ids built at start-up), the next byte the
 * argument count (around the declared count: fewer is padded by the dispatcher, more must be ignored), then
 * each argument gets a kind byte: null, a small int, an extreme int64, a palette double (NaN, Inf, 1e308,
 * subnormal), a bool, an option string (common spellings or raw bytes), a small array of any dtype incl.
 * 0-d and empty, or a sequence of up to 4 arrays/numbers.
 * Oracle (fuzz_common.h): ASan/UBSan clean, well-formed results, no leaked tsr allocations. The kernel
 * budget (256 MiB) turns size-like arguments into clean refusals for tsr_alloc'd memory.
 *
 * Determinism: the routine table depends only on the registry contents, so an input replays the same call
 * on the same build. Replaying a saved input with TSR_FUZZ_VERBOSE=1 prints the routine and argument count
 * before the call; TSR_FUZZ_LIST=1 prints the slot table.
 */
#include "fuzz_common.h"

/* Excluded routines, each for a stated reason. Name prefixes match too (entry ending in '*'). */
static const char *const DENY[] = {
    /* file system: these open, create or overwrite a path taken from an argument (fuzz_textio covers the
       text readers with a private temporary file) */
    "np.loadtxt", "np.genfromtxt", "np.savetxt", "np.fromfile", "np.save", "np.load", "np.tofile",
    "scipy.io.*", "scipyio.*",
};

static int denied(const char *name, const char *info)
{
    for (size_t i = 0; i < sizeof DENY / sizeof DENY[0]; i++) {
        const size_t l = strlen(DENY[i]);
        if (DENY[i][l - 1] == '*' ? strncmp(name, DENY[i], l - 1) == 0 : strcmp(name, DENY[i]) == 0) return 1;
    }
    /* any other routine whose arguments name a file */
    return strstr(info, "\"fname\"") || strstr(info, "\"file\"") || strstr(info, "\"filename\"") ||
           strstr(info, "\"path\"") || strstr(info, "\"filepath\"");
}

#ifdef FUZZ_SKIP_KNOWN
/* open findings: routines skipped at call time (not removed from the table, so slot numbers and therefore
   corpus files keep their meaning). Remove an entry once its fix lands. */
static const char *const KNOWN[] = {
    "signal.stft",                                                          /* F3: free() of a tsr_alloc block */
    "signal.lp2bp_zpk", "signal.lp2hp_zpk",                                 /* F12/F14: output store out of bounds */
    /* F16/F17/F19/F25/F28/F29/F32: the whole npoly.* module (chebint, chebmulx, chebmul, chebdiv, lag2poly,
       herm2poly, hermeint, ...) shares one class of bug -- a coefficient-array length or integer order drives an
       allocation or index with no bounds check. Skipped by prefix below; the maintainer audits the module. */
    "signal.lp2lp",                                                         /* F21: sig_normalize heap overflow */
    "signal.lp2bs_zpk", "signal.bilinear_zpk",                              /* F22/F37: complex store out of bounds in the zpk pipeline */
    "windows.tukey",                                                        /* F24: leaked tsr allocation */
    "signal.zpk2sos", "signal.iirdesign", "signal.dbode",                   /* F30/F35/F38: heap overflow in the filter routines */
    "signal.welch", "signal.csd",                                           /* F39: calloc count*size overflow */
    "signal.peak_widths", "signal.peak_prominences",                        /* F26: wild read in peak_prom_core */
    "windows.chebwin",                                                      /* F27: runs past the timeout */
};

static int known(const char *name)
{
    if (strncmp(name, "npoly.", 6) == 0) return 1;      /* F32: npoly.* argument-validation class (see note above) */
    for (size_t i = 0; i < sizeof KNOWN / sizeof KNOWN[0]; i++) if (strcmp(name, KNOWN[i]) == 0) return 1;
    return 0;
}
#endif

static int g_ids[4096], g_nargs[4096], g_n = -1;

static void build_table(void)
{
    char info[8192];
    g_n = 0;
    const int count = tsr_fn_count();
    for (int id = 0; id < count && g_n < 4096; id++) {
        const int64_t len = tsr_fn_info(id, info, sizeof info);
        if (len <= 0 || len >= (int64_t)sizeof info) continue;
        if (!strstr(info, "\"kind\":\"routine\"")) continue;
        const char *name = tsr_fn_name(id);
        if (!name || denied(name, info)) continue;
        int declared = 0;                                /* count the strings in "args":[...] */
        const char *a = strstr(info, "\"args\":[");
        if (a) {
            a += 8;
            int q = 0;
            for (; *a && !(*a == ']' && !q); a++) if (*a == '"' && a[-1] != '\\') { q = !q; if (q) declared++; }
        }
        g_ids[g_n] = id;
        g_nargs[g_n++] = declared;
    }
    if (getenv("TSR_FUZZ_LIST")) {
        for (int i = 0; i < g_n; i++) fprintf(stderr, "%d %s %d\n", i, tsr_fn_name(g_ids[i]), g_nargs[i]);
    }
}

static const char *const OPTS[] = {
    "auto", "linear", "nearest", "constant", "reflect", "wrap", "mirror", "edge", "full", "same", "valid", "C", "F",
    "left", "right", "both", "mean", "sum", "max", "min", "ortho", "forward", "backward", "raise", "omit", "propagate",
    "two-sided", "less", "greater", "hann", "hamming", "exact", "fft", "direct", "fro", "nuc", "inf", "-inf", "ij", "xy",
    "float64", "int64", "complex128", "bool", "lower", "upper", "gesdd", "gelsd", "cubic", "quadratic", "zero", "",
    "a", "ii", "i,i", "...", "->", "BFGS", "Nelder-Mead", "highs", "lm", "LU", "SM", "LM", "kaiser", "boxcar",
};

/* one argument; depth bounds nested sequences */
static tsr_arg make_arg(fz_in *in, fz_pool *pool, char strbuf[][16], int *nstr, int depth)
{
    const uint8_t k = fz_u8(in);
    switch (k % 12) {
    case 0: return fz_arg_none();
    case 1: case 2: return fz_arg_int((int64_t)(fz_u8(in) % 40) - 3);
    case 3: {
        /* "larger" integers, but capped to +-65536: an uncapped size-like argument (a shape, order or count) makes
           one routine allocate or loop unboundedly, which masks every other routine behind a timeout. Real
           size-overflow handling is covered by the parser targets (npy_write, hist) and the direct C tests. */
        static const int64_t X[] = {-1, 1, -65, 65, 256, -257, 4096, -4097, 65536, 33};
        return fz_arg_int(X[fz_u8(in) % (sizeof X / sizeof X[0])]);
    }
    case 4: return fz_arg_num(fz_double(in));
    case 5: return fz_arg_bool(fz_u8(in) & 1);
    case 6: {
        const uint8_t s = fz_u8(in);
        if (s < 224) return fz_arg_str(OPTS[s % (sizeof OPTS / sizeof OPTS[0])]);
        if (*nstr >= 8) return fz_arg_str("x");
        return fz_arg_str(fz_str(in, strbuf[(*nstr)++], 16, 12));
    }
    case 7: case 8: case 9: return fz_arg_small(in, pool, 0, 3, 5);       /* float64: the common case */
    case 10: {                                                            /* any dtype, rank 0..4 */
        tsr_arg a = fz_arg_small(in, pool, fz_u8(in) % 7, 4, 4);
        if (a.kind == 3 && (k & 0x80)) a.flags = 2;                      /* "built from a PHP list" */
        return a;
    }
    default: {
        if (depth >= 2) return fz_arg_none();
        const int64_t cnt = fz_u8(in) % 5;
        tsr_arg *items = (tsr_arg *)fz_take(pool, (size_t)(cnt ? cnt : 1) * sizeof(tsr_arg));
        if (!items) return fz_arg_none();
        for (int64_t i = 0; i < cnt; i++) items[i] = make_arg(in, pool, strbuf, nstr, depth + 1);
        tsr_arg q = fz_arg_none();
        q.kind = 5;
        q.items = items;
        q.count = cnt;
        return q;
    }
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_init();
    if (g_n < 0) build_table();
    if (g_n == 0) abort();
    fz_in in = {data, size};
    fz_pool pool = {{0}, 0};
    const unsigned slot_lo = fz_u8(&in), slot_hi = fz_u8(&in);
    const int slot = (int)((slot_lo | slot_hi << 8) % (unsigned)g_n);
    const uint8_t nb = fz_u8(&in);
    int nargs = g_nargs[slot] + (int)(nb % 4) - 1;      /* declared - 1 .. declared + 2 */
    if (nb & 0x80) nargs = (int)(nb % 7);               /* or anything 0..6 */
    if (nargs < 0) nargs = 0;
    if (nargs > 8) nargs = 8;
    char strbuf[8][16];
    int nstr = 0;
    tsr_arg args[8];
    for (int i = 0; i < nargs; i++) args[i] = make_arg(&in, &pool, strbuf, &nstr, 0);
    static int verbose = -1;
    if (verbose < 0) verbose = getenv("TSR_FUZZ_VERBOSE") != NULL;
    if (verbose) fprintf(stderr, "routine: %s nargs %d\n", tsr_fn_name(g_ids[slot]), nargs);
#ifdef FUZZ_SKIP_KNOWN
    if (known(tsr_fn_name(g_ids[slot]))) { fz_release(&pool); return 0; }
#endif
    fz_call(g_ids[slot], nargs, nargs ? args : NULL);
    fz_release(&pool);
    return 0;
}
