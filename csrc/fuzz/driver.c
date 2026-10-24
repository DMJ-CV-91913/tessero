/*
 * Standalone driver for the fuzz targets, for machines without libFuzzer
 * (gcc, or clang without compiler-rt). It replays corpus files and then runs
 * a seeded mutation loop over them (bit flips, byte sets, inserts, deletes,
 * splices and dictionary tokens). It is not coverage guided: CI runs the same
 * targets under libFuzzer (make fuzz) for that; this driver makes them usable
 * everywhere under AddressSanitizer + UBSan (make fuzz-standalone).
 *
 *   ./fuzz_npy [-runs=N] [-seed=S] [-max_len=L] [-timeout=SEC] [-artifact_prefix=P] corpus_dir_or_files...
 *
 * Deterministic: corpus files are loaded in sorted order and every mutation comes from the seeded
 * generator, so the same seed, run count and corpus replay the same inputs. When an input crashes
 * (sanitizer report, abort, signal) or exceeds -timeout seconds, it is written to P + "crash-<run>" /
 * "timeout-<run>" before the process dies; `./fuzz_x that_file` replays exactly that input. At the end
 * the driver prints the run count, elapsed time and execs/s.
 */
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

typedef struct { uint8_t *d; size_t n; } input;
static input *corpus;
static size_t ncorpus, capcorpus;
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

/* the input being executed, saved by the crash paths below */
static const uint8_t *volatile cur_d;
static volatile size_t cur_n;
static volatile long long cur_run = -1;
static const char *prefix = "";

static void save_current(const char *kind)
{
    static volatile sig_atomic_t saved;
    if (saved || !cur_d) return;
    saved = 1;
    char path[4200], num[24];
    int len = 0, k = 0;
    long long r = cur_run < 0 ? 0 : cur_run;
    do { num[k++] = (char)('0' + r % 10); r /= 10; } while (r && k < 23);
    for (const char *p = prefix; *p && len < 4000; p++) path[len++] = *p;
    for (const char *p = kind; *p; p++) path[len++] = *p;
    if (cur_run < 0) { const char *c = "corpus"; while (*c) path[len++] = *c++; }
    else while (k) path[len++] = num[--k];
    path[len] = 0;
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        size_t off = 0;
        while (off < cur_n) { ssize_t w = write(fd, (const uint8_t *)cur_d + off, cur_n - off); if (w <= 0) break; off += (size_t)w; }
        close(fd);
    }
    const char *msg = "\n==fuzz driver== input saved to ";
    if (write(2, msg, strlen(msg)) < 0 || write(2, path, (size_t)len) < 0 || write(2, "\n", 1) < 0) { /* best effort */ }
}

static void on_death(void) { save_current("crash-"); }

static void on_signal(int sig)
{
    save_current(sig == SIGALRM ? "timeout-" : "crash-");
    signal(sig, SIG_DFL);
    if (sig == SIGALRM) _exit(70);
    raise(sig);
}

/* sanitizer runtimes call this before exiting with a report (weak: absent without a sanitizer) */
void __sanitizer_set_death_callback(void (*cb)(void)) __attribute__((weak));

static uint64_t rnd(void)
{
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void add(const uint8_t *d, size_t n)
{
    if (ncorpus == capcorpus) { capcorpus = capcorpus ? capcorpus * 2 : 64; corpus = realloc(corpus, capcorpus * sizeof(input)); }
    corpus[ncorpus].d = malloc(n ? n : 1);
    memcpy(corpus[ncorpus].d, d, n);
    corpus[ncorpus++].n = n;
}

static int cmp_name(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static void load(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return;
    if (S_ISDIR(st.st_mode)) {
        DIR *dir = opendir(path);                           /* sorted, so the corpus order is reproducible */
        struct dirent *e;
        char **names = NULL;
        size_t nn = 0, cn = 0;
        while (dir && (e = readdir(dir))) {
            if (e->d_name[0] == '.') continue;
            if (nn == cn) { cn = cn ? cn * 2 : 64; names = realloc(names, cn * sizeof(char *)); }
            names[nn++] = strdup(e->d_name);
        }
        if (dir) closedir(dir);
        if (nn) qsort(names, nn, sizeof(char *), cmp_name);
        for (size_t i = 0; i < nn; i++) {
            char p[4096];
            snprintf(p, sizeof(p), "%s/%s", path, names[i]);
            load(p);
            free(names[i]);
        }
        free(names);
        return;
    }
    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint8_t *buf = malloc((size_t)st.st_size + 1);
    size_t n = fread(buf, 1, (size_t)st.st_size, f);
    fclose(f);
    add(buf, n);
    free(buf);
}

static const char *DICT[] = {
    ":", "::", "::-1", "...", "None", ",", "-1", "0", "1:", ":-1", "9223372036854775807", "-9223372036854775808",
    "\x93NUMPY", "'descr'", "'<f8'", "'>f8'", "'|b1'", "'<c16'", "'fortran_order'", "True", "False", "'shape'", "(", ")", "{", "}",
    "\xff\xff\xff\x7f", "\x00\x00\x00\x80", "\xff\xff\xff\xff\xff\xff\xff\x7f",
};

static size_t mutate(uint8_t *d, size_t n, size_t max)
{
    const int k = 1 + (int)(rnd() % 4);
    for (int m = 0; m < k; m++) {
        switch (rnd() % 7) {
        case 0: if (n) d[rnd() % n] ^= (uint8_t)(1u << (rnd() % 8)); break;
        case 1: if (n) d[rnd() % n] = (uint8_t)rnd(); break;
        case 2: if (n < max) { size_t at = rnd() % (n + 1); memmove(d + at + 1, d + at, n - at); d[at] = (uint8_t)rnd(); n++; } break;
        case 3: if (n) { size_t at = rnd() % n, len = 1 + rnd() % (n - at); memmove(d + at, d + at + len, n - at - len); n -= len; } break;
        case 4: {
            const char *t = DICT[rnd() % (sizeof(DICT) / sizeof(DICT[0]))];
            size_t tl = strlen(t);
            if (n + tl <= max) { size_t at = rnd() % (n + 1); memmove(d + at + tl, d + at, n - at); memcpy(d + at, t, tl); n += tl; }
            break;
        }
        case 5: if (ncorpus) {                                  /* splice from another input */
            const input *o = &corpus[rnd() % ncorpus];
            if (o->n) {
                size_t from = rnd() % o->n, len = 1 + rnd() % (o->n - from), at = n ? rnd() % n : 0;
                if (at + len > max) len = max - at;
                memcpy(d + at, o->d + from, len);
                if (at + len > n) n = at + len;
            }
            }
            break;
        case 6: if (n >= 8) { size_t at = rnd() % (n - 7); uint64_t v = rnd() % 3 == 0 ? UINT64_MAX >> (rnd() % 64) : rnd(); memcpy(d + at, &v, 8); } break;
        }
    }
    return n;
}

int main(int argc, char **argv)
{
    long long runs = 100000;
    size_t max = 4096;
    unsigned timeout = 0;
    unsigned long long seed_arg = 0;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "-runs=", 6)) runs = atoll(argv[i] + 6);
        else if (!strncmp(argv[i], "-seed=", 6)) { seed_arg = strtoull(argv[i] + 6, NULL, 10); rng_state = seed_arg * 0x9E3779B97F4A7C15ull + 1; }
        else if (!strncmp(argv[i], "-timeout=", 9)) timeout = (unsigned)atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "-artifact_prefix=", 17)) prefix = argv[i] + 17;
        else if (!strncmp(argv[i], "-max_len=", 9)) max = (size_t)atoll(argv[i] + 9);
        else if (argv[i][0] != '-') load(argv[i]);
    }
    /* with a sanitizer runtime, leave SIGSEGV/SIGBUS/SIGFPE to it (it prints the report, then the death
       callback saves the input); without one, save the input from the signal handler */
    if (__sanitizer_set_death_callback) __sanitizer_set_death_callback(on_death);
    else {
        signal(SIGSEGV, on_signal);
        signal(SIGBUS, on_signal);
        signal(SIGFPE, on_signal);
        signal(SIGILL, on_signal);
    }
    signal(SIGABRT, on_signal);
    signal(SIGALRM, on_signal);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < ncorpus; i++) {
        cur_d = corpus[i].d; cur_n = corpus[i].n;
        if (timeout) alarm(timeout);
        LLVMFuzzerTestOneInput(corpus[i].d, corpus[i].n);
    }
    const size_t nreplayed = ncorpus;
    if (ncorpus == 0) { uint8_t z[16] = {0}; add(z, sizeof(z)); }
    uint8_t *buf = malloc(max + 64);
    for (long long r = 0; r < runs; r++) {
        const input *seed = &corpus[rnd() % ncorpus];
        size_t n = seed->n < max ? seed->n : max;
        memcpy(buf, seed->d, n);
        n = mutate(buf, n, max);
        uint8_t *exact = malloc(n ? n : 1);                     /* exact-size copy so ASan sees overreads */
        memcpy(exact, buf, n);
        cur_d = exact; cur_n = n; cur_run = r;
        if (timeout) alarm(timeout);
        LLVMFuzzerTestOneInput(exact, n);
        if (r % 97 == 0 && ncorpus < 4096) add(exact, n);      /* keep some mutants as new seeds */
        free(exact);
    }
    if (timeout) alarm(0);
    cur_d = NULL;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("%s: seed %llu, %zu corpus inputs replayed, %lld mutated runs, %.1f s, %.0f execs/s, no failures\n",
           argv[0], seed_arg, nreplayed, runs, secs, (double)((long long)nreplayed + runs) / (secs > 0 ? secs : 1e-9));
    free(buf);
    return 0;
}
