/*
 * Standalone driver for the fuzz targets, for machines without libFuzzer
 * (gcc, or clang without compiler-rt). It replays corpus files and then runs
 * a seeded mutation loop over them (bit flips, byte sets, inserts, deletes,
 * splices and dictionary tokens). It is not coverage guided: CI runs the same
 * targets under libFuzzer (make fuzz) for that; this driver makes them usable
 * everywhere under AddressSanitizer + UBSan (make fuzz-standalone).
 *
 *   ./fuzz_npy [-runs=N] [-seed=S] [-max_len=L] corpus_dir_or_files...
 */
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

typedef struct { uint8_t *d; size_t n; } input;
static input *corpus;
static size_t ncorpus, capcorpus;
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

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

static void load(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return;
    if (S_ISDIR(st.st_mode)) {
        DIR *dir = opendir(path);
        struct dirent *e;
        while (dir && (e = readdir(dir))) {
            if (e->d_name[0] == '.') continue;
            char p[4096];
            snprintf(p, sizeof(p), "%s/%s", path, e->d_name);
            load(p);
        }
        if (dir) closedir(dir);
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
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "-runs=", 6)) runs = atoll(argv[i] + 6);
        else if (!strncmp(argv[i], "-seed=", 6)) rng_state = strtoull(argv[i] + 6, NULL, 10) * 0x9E3779B97F4A7C15ull + 1;
        else if (!strncmp(argv[i], "-max_len=", 9)) max = (size_t)atoll(argv[i] + 9);
        else if (argv[i][0] != '-') load(argv[i]);
    }
    for (size_t i = 0; i < ncorpus; i++) LLVMFuzzerTestOneInput(corpus[i].d, corpus[i].n);
    if (ncorpus == 0) { uint8_t z[16] = {0}; add(z, sizeof(z)); }
    uint8_t *buf = malloc(max + 64);
    for (long long r = 0; r < runs; r++) {
        const input *seed = &corpus[rnd() % ncorpus];
        size_t n = seed->n < max ? seed->n : max;
        memcpy(buf, seed->d, n);
        n = mutate(buf, n, max);
        uint8_t *exact = malloc(n ? n : 1);                     /* exact-size copy so ASan sees overreads */
        memcpy(exact, buf, n);
        LLVMFuzzerTestOneInput(exact, n);
        if (r % 97 == 0 && ncorpus < 4096) add(exact, n);      /* keep some mutants as new seeds */
        free(exact);
    }
    printf("%s: %zu corpus inputs, %lld mutated runs, no failures\n", argv[0], ncorpus, runs);
    free(buf);
    return 0;
}
