/* selfcheck_run.c -- tiny driver that calls the library's canonical self-check and
 * reports the verdict. Build + run via run_selfcheck.sh. No em dashes. */
#include <stdint.h>
#include <stdio.h>

int tsr_selfcheck(void);
int tsr_fpenv_check(char *buf, int64_t cap);
uint64_t tsr_selfcheck_digest(int variant);
uint64_t tsr_selfcheck_canonical(void);

int main(void)
{
    char why[256];
    int fe = tsr_fpenv_check(why, sizeof why);
    printf("fpenv -> %d  %s\n", fe, why);
    printf("KAT digests: frozen %016llx  reference %016llx  dispatched %016llx\n",
           (unsigned long long)tsr_selfcheck_canonical(), (unsigned long long)tsr_selfcheck_digest(0),
           (unsigned long long)tsr_selfcheck_digest(1));
    printf("  deliberately wrong (must differ): mul+add %016llx  reversed-k %016llx\n",
           (unsigned long long)tsr_selfcheck_digest(2), (unsigned long long)tsr_selfcheck_digest(3));
    int r = tsr_selfcheck();
    const char *m = r == 0  ? "CANONICAL (fast == reference == frozen)"
                  : r == -1 ? "DIVERGED: fast clone != reference -- substitute the reference"
                  : r == -2 ? "DIVERGED: reference != frozen canonical -- bad toolchain / build"
                  : r == -3 ? "DIVERGED: libm fmaf is not correctly rounded"
                  : r == -4 ? "DIVERGED: subnormals are flushed to zero"
                  : r == -5 ? "DIVERGED: signed zero is not IEEE"
                  : r == -6 ? "DIVERGED: FP control register non-canonical (see fpenv above)"
                  :           "DIVERGED";
    printf("tsr_selfcheck -> %d  %s\n", r, m);
    return r != 0;
}
