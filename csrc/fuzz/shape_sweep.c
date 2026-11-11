/*
 * shape_sweep: every registry routine and generalised ufunc called with zero-length axes in the leading, middle and
 * trailing positions, and with 0-d / 1-d arrays whose shape and strides past ndim hold junk (a C caller may leave
 * them; the PHP bindings zero them). Built with ASan + UBSan + leak checks (`make shape-sweep`), each call in a
 * forked child with a time limit, so a read or write past a block, a leak on an error path, a crash or a hang is
 * reported as one line and the sweep goes on. An error code from the routine is fine: refusing is the point.
 *
 *   build/shape_sweep                    # the whole sweep; exit 1 if any call failed
 *   build/shape_sweep NAME LAYOUT SHAPE  # one call, in process, with the sanitizer's report (from a failure line)
 *
 * Found the solvers' right-hand-side overreads (2026-11: matmul_toeplitz, lstsq, ...), which the random fuzzer
 * reached once. Shapes use lengths of 9 so an overread passes the 64-byte minimum block, where ASan sees it.
 */
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../src/internal.h"

void fn_results_free(tsr_result *res, int n);   /* fn.h */

#define NSHAPE 9
#define NLAYOUT 5
static const int32_t SH_ND[NSHAPE] = {1, 2, 2, 3, 3, 3, 3, 0, 1};
static const int64_t SH[NSHAPE][3] = {{0}, {0, 9}, {9, 0}, {0, 9, 9}, {9, 0, 9}, {9, 9, 0}, {9, 9, 0}, {0}, {9}};
static const char *SH_NAME[NSHAPE] = {"(0)", "(0,9)", "(9,0)", "(0,9,9)", "(9,0,9)", "(9,9,0)", "(9,9,0) int64",
                                      "0-d junk", "(9) junk"};
/* layouts: 0 every array argument the shape; 1 a valid 9x9 first, the rest the shape; 2 a valid 9-vector first;
   3 the shape first, valid 9-vectors after; 4 sequences (seq arguments) of two arrays of the shape */
static const char *LAYOUT_NAME[NLAYOUT] = {"all", "9x9 first", "9 first", "shape first", "sequences"};

/* each array's data is a heap block of exactly its own size (none for an empty one), so a read past it -- of the
   input itself or of a copy -- is one ASan sees; freed after the call, so the leak check sees only the library's */
static int g_single;                   /* one call, in process: report what it produced */
static void *g_blocks[64];
static int g_nblocks;

static void *block(int64_t n, int dtype)
{
    const size_t sz = (size_t)n * (dtype == TSR_I64 ? sizeof(int64_t) : sizeof(double));
    void *p = malloc(sz ? sz : 1);
    if (g_nblocks < 64) g_blocks[g_nblocks++] = p;
    for (int64_t i = 0; i < n; i++) {
        if (dtype == TSR_I64) ((int64_t *)p)[i] = i % 7;
        else ((double *)p)[i] = 1.0 + 0.25 * (double)i + (i % 10 == 0 ? 9.0 : 0.0);   /* 9x9: a dominant diagonal */
    }
    return p;
}

static void mk(tsr_arg *a, int shape, int slot)
{
    (void)slot;
    memset(a, 0, sizeof *a);
    a->kind = 3;
    const int dt = shape == 6 ? TSR_I64 : TSR_F64;
    int64_t n = 1;
    for (int d = 0; d < SH_ND[shape]; d++) n *= SH[shape][d];
    tsr_array_init(&a->arr, dt, SH_ND[shape], SH[shape], block(n, dt));
    if (shape >= 7) for (int k = SH_ND[shape]; k < 4; k++) { a->arr.shape[k] = 9; a->arr.strides[k] = 8; }
}

/* the valid array of layouts 1-3: 9 x 9 or 9 values (SWEEP_N=n for n x n, to scale a call's work: meshgrid of a
   9 x 9 array is 9 grids of 9^9 values each) */
static void mk_valid(tsr_arg *a, int nd, int slot)
{
    (void)slot;
    const int64_t v = getenv("SWEEP_N") ? atoi(getenv("SWEEP_N")) : 9, s[2] = {v, v};
    memset(a, 0, sizeof *a);
    a->kind = 3;
    tsr_array_init(&a->arr, TSR_F64, nd, s, block(nd == 2 ? v * v : v, TSR_F64));
}

static void free_blocks(void)
{
    for (int i = 0; i < g_nblocks; i++) free(g_blocks[i]);
    g_nblocks = 0;
}

/* number of strings in the JSON list after "key":[ (an empty list gives 0) */
static int list_len(const char *js, const char *key)
{
    const char *p = strstr(js, key);
    if (!p) return 0;
    p = strchr(p, '[');
    int n = 0, depth = 0;
    for (; *p; p++) {
        if (*p == '[' || *p == '{') depth++;
        else if (*p == ']' || *p == '}') { if (--depth == 0) break; }
        else if (depth == 1 && *p == '"') { n++; p = strchr(p + 1, '"'); if (!p) break; }
        else if (depth == 1 && n == 0 && *p >= '0' && *p <= '9') n = -1;   /* a list of numbers: not counted here */
    }
    return n < 0 ? 0 : n;
}

static int in_seq(const char *js, int k)
{
    const char *p = strstr(js, "\"seq\":[");
    if (!p) return 0;
    for (p += 7; *p && *p != ']'; ) { char *e; long v = strtol(p, &e, 10); if (e == p) { p++; continue; } if (v == k) return 1; p = e; }
    return 0;
}

static int json_int(const char *js, const char *key)
{
    const char *p = strstr(js, key);
    return p ? atoi(p + strlen(key)) : 0;
}

/* one call; 0 when the call returned (success or an error code), else the sanitizer aborts or the signal kills */
static void call_routine(int id, const char *js, int layout, int shape)
{
    tsr_arg args[16], items[4][2];
    int nargs = list_len(js, "\"args\":[");
    if (nargs > 16) nargs = 16;
    int arr = 0;
    for (int k = 0; k < nargs; k++) {
        if (layout == 4 && in_seq(js, k)) {
            memset(&args[k], 0, sizeof args[k]);
            mk(&items[k & 3][0], shape, 2); mk(&items[k & 3][1], shape, 3);
            args[k].kind = 5; args[k].count = 2; args[k].items = items[k & 3];
            continue;
        }
        if (layout == 4) { mk(&args[k], shape, arr++ & 1); continue; }
        if (arr == 0 && layout == 1) mk_valid(&args[k], 2, 0);
        else if (arr == 0 && layout == 2) mk_valid(&args[k], 1, 0);
        else if (arr > 0 && layout == 3) mk_valid(&args[k], 1, 1);
        else mk(&args[k], shape, arr ? 1 : 0);
        arr++;
    }
    tsr_result res[16];
    int nout = json_int(js, "\"nout\":");
    if (nout < 1) nout = 1;
    if (nout > 16) nout = 16;
    if (tsr_fn_routine(id, nargs, args, nout, res) >= 0) {
        if (g_single)                                      /* what the call produced: arrays and their element counts */
            for (int r = 0; r < nout; r++) {
                if (res[r].kind == 3) printf("result %d: array of %lld values\n", r, (long long)tsr_shape_size(res[r].arr.ndim, res[r].arr.shape));
                if (res[r].kind == 6) {
                    const tsr_result *it = (const tsr_result *)res[r].arr.data;
                    long long total = 0;
                    for (int64_t j = 0; j < res[r].arr.shape[0]; j++) if (it[j].kind == 3) total += tsr_shape_size(it[j].arr.ndim, it[j].arr.shape);
                    printf("result %d: %lld arrays, %lld values in all\n", r, (long long)res[r].arr.shape[0], total);
                }
            }
        fn_results_free(res, nout);
    }
}

static void call_gufunc(int id, const char *js, int shape)
{
    int nin = json_int(js, "\"nin\":"), nout = json_int(js, "\"nout\":");
    if (nin < 1 || nin > 4 || nout < 1 || nout > 4) return;
    tsr_array in[4];
    uint64_t mask[4];
    for (int i = 0; i < nin; i++) { tsr_arg a; mk(&a, shape, i & 3); in[i] = a.arr; mask[i] = ((uint64_t)1 << (in[i].ndim < 63 ? in[i].ndim : 63)) - 1; }
    double params[32];
    for (int i = 0; i < 32; i++) params[i] = 0.0;
    int32_t ond[4];
    int64_t osh[4 * 32];
    if (tsr_fn_gufunc_shape(id, nin, in, mask, 0, params, ond, osh) < 0) return;
    tsr_array out[4];
    void *blocks[4] = {0};
    int64_t bytes[4] = {0};
    for (int o = 0; o < nout; o++) {
        int64_t n = 1;
        for (int d = 0; d < ond[o]; d++) n *= osh[o * 32 + d];
        bytes[o] = (n > 0 ? n : 1) * 16;
        blocks[o] = tsr_alloc(bytes[o]);
        if (!blocks[o]) { for (int j = 0; j < o; j++) tsr_free(blocks[j], bytes[j]); return; }
        tsr_array_init(&out[o], TSR_F64, ond[o], osh + o * 32, blocks[o]);
    }
    tsr_fn_gufunc(id, nin, in, mask, 0, params, nout, out);
    for (int o = 0; o < nout; o++) tsr_free(blocks[o], bytes[o]);
}

static void one(int id, const char *js, int layout, int shape)
{
    if (strstr(js, "\"kind\":\"gufunc\"")) call_gufunc(id, js, shape);
    else call_routine(id, js, layout, shape);
    free_blocks();
}

int main(int argc, char **argv)
{
    static char js[1 << 16];
    if (argc == 4) {                                   /* one call, in process */
        const int id = tsr_fn_find(argv[1]);
        if (id < 0) { fprintf(stderr, "no function %s\n", argv[1]); return 2; }
        tsr_fn_info(id, js, sizeof js);
        g_single = 1;
        one(id, js, atoi(argv[2]), atoi(argv[3]));
        printf("returned: %s\n", tsr_fn_error());
        return 0;
    }
    const int n = tsr_fn_count();
    int calls = 0, failed = 0;
    for (int id = 0; id < n; id++) {
        if (tsr_fn_info(id, js, sizeof js) <= 0) continue;
        const int routine = strstr(js, "\"kind\":\"routine\"") != NULL, gufunc = strstr(js, "\"kind\":\"gufunc\"") != NULL;
        if (!routine && !gufunc) continue;
        const char *name = tsr_fn_name(id);
        if (getenv("SWEEP_ONLY") && !strstr(name, getenv("SWEEP_ONLY"))) continue;
        for (int layout = 0; layout < (routine ? NLAYOUT : 1); layout++)
            for (int shape = 0; shape < NSHAPE; shape++) {
                if (layout == 4 && !strstr(js, "\"seq\":[0") && !strstr(js, "\"seq\":[1") && !strstr(js, "\"seq\":[2")) continue;
                fflush(stdout);
                const pid_t pid = fork();
                if (pid == 0) {
                    if (!getenv("SWEEP_VERBOSE")) { FILE *f = freopen("/dev/null", "w", stderr); (void)f; }
                    alarm(20);
                    one(id, js, layout, shape);
                    exit(0);                           /* exit, not _exit: the leak check runs */
                }
                int st = 0;
                waitpid(pid, &st, 0);
                calls++;
                if (WIFSIGNALED(st) || WEXITSTATUS(st) != 0) {
                    failed++;
                    printf("FAIL %s layout %d (%s) shape %d %s: %s %d\n", name, layout, LAYOUT_NAME[layout], shape,
                           SH_NAME[shape], WIFSIGNALED(st) ? (WTERMSIG(st) == SIGALRM ? "timeout, signal" : "signal") : "exit",
                           WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
                }
            }
    }
    printf("shape_sweep: %d calls, %d failed\n", calls, failed);
    return failed ? 1 : 0;
}
