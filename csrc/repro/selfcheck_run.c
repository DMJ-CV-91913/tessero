/* selfcheck_run.c -- tiny driver that calls the library's canonical self-check and
 * reports the verdict. Build + run via run_selfcheck.sh. No em dashes. */
#include <stdio.h>

int tsr_selfcheck(void);

int main(void)
{
    int r = tsr_selfcheck();
    const char *m = r == 0  ? "CANONICAL (fast == reference == frozen)"
                  : r == -1 ? "DIVERGED: fast clone != reference -- substitute the reference"
                  : r == -2 ? "DIVERGED: reference != frozen canonical -- bad toolchain / build"
                  :           "DIVERGED";
    printf("tsr_selfcheck -> %d  %s\n", r, m);
    return r != 0;
}
