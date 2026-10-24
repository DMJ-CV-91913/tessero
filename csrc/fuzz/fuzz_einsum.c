/*
 * Fuzz target: np.einsum (r_einsum), which parses an attacker-controlled subscripts string into fixed-size
 * label tables and walks operands of matching rank. Input: one byte for the operand count (1-3) and
 * whether to pass them as a sequence, then per operand a rank (0-4) and dims (0-3), then the subscripts
 * string (up to 300 bytes, so the 256-byte limit is crossed), then the element values.
 * Oracle (fuzz_common.h): ASan/UBSan clean, result rank <= 32, results freed, no leaked tsr allocations.
 */
#include "fuzz_common.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_init();
    static int id = -2;
    if (id == -2) id = tsr_fn_find("np.einsum");
    if (id < 0) abort();
    fz_in in = {data, size};
    fz_pool pool = {{0}, 0};
    const uint8_t hdr = fz_u8(&in);
    const int nop = 1 + hdr % 3, as_seq = (hdr >> 2) & 1;
    int32_t nd[3];
    int64_t shape[3][32];
    for (int o = 0; o < nop; o++) {
        nd[o] = fz_u8(&in) % 5;
        for (int d = 0; d < nd[o]; d++) shape[o][d] = fz_u8(&in) % 4;
    }
    /* subscripts: most bytes map onto the einsum alphabet (so mutations stay meaningful), bytes >= 0xF0
       pass through raw (invalid characters, NUL-free high bytes) */
    static const char ALPHA[] = "abcdeijkabcijAZz...,,,->-> ";
    char sub[320];
    const size_t slen = fz_u8(&in) % 301;
    size_t sl = 0;
    for (size_t i = 0; i < slen; i++) {
        const uint8_t b = fz_u8(&in);
        const char c = b >= 0xF0 ? (char)b : ALPHA[b % (sizeof ALPHA - 1)];
        if (c) sub[sl++] = c;
    }
    sub[sl] = '\0';
    tsr_arg args[4], ops[3];
    args[0] = fz_arg_str(sub);
    for (int o = 0; o < nop; o++) {
        ops[o] = fz_arg_array(&in, &pool, 0, nd[o], shape[o]);
        if (ops[o].kind != 3) { fz_release(&pool); return 0; }
    }
#ifdef FUZZ_SKIP_KNOWN
    /* open finding F5: a label bound to a 0-length axis and a longer one is accepted (size 0 means "unset"),
       then the operand data is read out of bounds. Skip operands with a 0-length axis. */
    for (int o = 0; o < nop; o++)
        for (int d = 0; d < nd[o]; d++) if (shape[o][d] == 0) { fz_release(&pool); return 0; }
    /* open finding F8: dots that are not one contiguous "..." per term pass the first parse and are misparsed
       by the second. Skip subscripts with any '.' outside a run of exactly three. */
    for (size_t i = 0; sub[i]; ) {
        if (sub[i] != '.') { i++; continue; }
        size_t run = 0;
        while (sub[i + run] == '.') run++;
        if (run != 3) { fz_release(&pool); return 0; }
        i += run;
    }
#endif
    if (as_seq) {
        tsr_arg seq = fz_arg_none();
        seq.kind = 5;
        seq.items = ops;
        seq.count = nop;
        args[1] = seq;
        fz_call(id, 2, args);
    } else {
        for (int o = 0; o < nop; o++) args[1 + o] = ops[o];
        fz_call(id, 1 + nop, args);
    }
    fz_release(&pool);
    return 0;
}
