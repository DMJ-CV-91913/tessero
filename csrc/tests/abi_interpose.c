/* Two libtessero copies in one process (ADR 0014): `abi_interpose <first.so> <second.so>`.
 * The first copy is loaded RTLD_GLOBAL (as Talario's PHP extension does), the second RTLD_LOCAL (as the FFI binding
 * does). The second copy's internal calls to its own exported functions (tsr_calloc -> tsr_alloc, the budget
 * check) must bind inside it: its allocations are counted in its own tsr_allocated(), its budget applies, and the
 * first copy sees nothing. Exit status 0 when that holds. Run by `make abi-check`. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

typedef void *(*alloc_fn)(int64_t);
typedef void (*free_fn)(void *, int64_t);
typedef int64_t (*count_fn)(void);
typedef void (*budget_fn)(int64_t);

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: abi_interpose <first.so> <second.so>\n"); return 2; }
    void *a = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    void *b = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!a || !b) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    count_fn a_allocated = (count_fn)dlsym(a, "tsr_allocated"), b_allocated = (count_fn)dlsym(b, "tsr_allocated");
    alloc_fn b_calloc = (alloc_fn)dlsym(b, "tsr_calloc");
    free_fn b_free = (free_fn)dlsym(b, "tsr_free");
    budget_fn b_budget = (budget_fn)dlsym(b, "tsr_set_budget");
    if (!a_allocated || !b_allocated || !b_calloc || !b_free || !b_budget || a_allocated == b_allocated) {
        fprintf(stderr, "symbols missing, or both handles resolve to one copy\n");
        return 2;
    }
    int fails = 0;
    void *p = b_calloc(4096);                       /* tsr_calloc calls tsr_alloc internally */
    if (!p || b_allocated() != 4096 || a_allocated() != 0) {
        fprintf(stderr, "FAIL accounting: second copy %lld, first copy %lld (want 4096, 0)\n",
                (long long)b_allocated(), (long long)a_allocated());
        fails++;
    }
    b_free(p, 4096);
    b_budget(1 << 20);                              /* the second copy's budget must govern its own allocations */
    void *q = b_calloc((int64_t)2 << 20);
    if (q) { fprintf(stderr, "FAIL budget: a 2 MiB block passed a 1 MiB budget\n"); b_free(q, (int64_t)2 << 20); fails++; }
    b_budget(0);
    if (b_allocated() != 0 || a_allocated() != 0) { fprintf(stderr, "FAIL: counters not back to zero\n"); fails++; }
    printf("abi_interpose: %s\n", fails ? "FAILED (internal calls bound to the other copy)" : "internal calls bind locally");
    return fails ? 1 : 0;
}
