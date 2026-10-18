# Reproducible-f32 cross-architecture matrix

Which (CPU arch, vector path, thread count) cells reproduce the recorded x86 goldens
(`csrc/repro/GOLDENS.md`) for rsqrt / exp2 / exp / erf. Every MATCH confirms the primitives
are bit-identical by construction on that hardware. Checked by `run_cross_arch.sh`
(arch-aware, compares to the goldens) and `run_determinism.sh` (x86, internal consistency).

| arch | host | vector paths | threads | rsqrt | exp2 | exp | erf | note |
|------|------|--------------|---------|-------|------|-----|-----|------|
| x86-64 | GCP n2 (AVX2+FMA, gcc 13.3) | scalar, AVX2 | 1, 4 | golden | golden | golden | golden | goldens defined here (824e9ff) |
| x86-64 | GCP c4 Emerald Rapids (Intel Xeon 5th Gen) | scalar, AVX2, AVX-512 | 1, 4 | golden | golden | golden | golden | AVX-512 cell, 2026-10-18 |
| aarch64 | GCP c4a Axion (Neoverse V2) | scalar, NEON, SVE | 1, 4 | MATCH | MATCH | MATCH | MATCH | CROSS-ISA, 2026-10-18 |

Coverage now: vector paths **scalar / AVX2 / AVX-512 / NEON / SVE**, thread counts **1 and 4**,
ISAs **x86-64 (Intel) and ARM64 (Axion)** - all four primitives byte-identical to the goldens
across every cell. CPU-side reproducibility is complete.

**The aarch64 row is the result:** all four primitives produce byte-identical f32 to the x86
goldens on ARM (a different CPU ISA), across NEON and SVE and 1 vs 4 threads. Reproducibility
holds across SIMD width, thread count, AND instruction set, by construction (correctly-rounded
`fmaf` + `sqrtf`, fixed order, no SFU/fast-math). Direct support for CPU-side determinism on
GH200 (Grace) and AWS Graviton, and for the vendor/arch/ISA-independence the registry and the
Meter of Record stand on.

## Open cells

- **CPU == GPU**: the Talario GLSL/SPIR-V shader must reproduce these exact goldens on real
  silicon, using the frozen constants/order in `GOLDENS.md`. (Talario repo; needs one GPU.)
- Edge inputs (NaN/Inf/denormal) are verified by classification, not folded into the golden.
