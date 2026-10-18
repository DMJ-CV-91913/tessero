# Reproducible-f32 golden vectors (WIP)

Golden SHA-256 hashes for the reproducible-f32 transcendental primitives. A hash here is
the single defined f32 bit-result for the primitive on a fixed input. Every conformant
implementation (Tessero at any SIMD level, the Talario GPU shader, any vendor) must
reproduce it. See REPRO_F32_KICKOFF.md and
`talario/docs/papers/fma-only-transcendentals-design.md`.

## rsqrt(x) = 1/sqrt(x)   [status: CPU SIMD-invariance PROVEN]

- Definition: `y = 1.0f / sqrtf(x)` (IEEE correctly-rounded sqrt + divide, no SFU estimate,
  no polynomial). Source: `csrc/repro/demo_rsqrt.c`.
- Input: fixed, index-derived, strictly positive, magnitude spread ~1e-4..1e4, N = 2^20
  (1,048,576) f32 (exact generator in demo_rsqrt.c; no RNG, no time).
- Build: `-fno-fast-math -ffp-contract=off`, compiled both scalar (`-fno-tree-vectorize`)
  and AVX2+FMA (`-mavx2 -mfma`). Reproduce with `csrc/repro/run_demo.sh`.
- GOLDEN (SHA-256 of raw f32 output bytes):
  `fce3ce9bb62423fb56ec0c331d6b6262d8f76be629718572b37d31d80b9a696b`
- First confirmed: 2026-10-17, GCP n2-highcpu (x86-64, AVX2+FMA, no AVX-512). scalar hash
  == avx2 hash, byte-for-byte.

What this PROVES: on x86 with AVX2+FMA, rsqrt is bit-identical across SIMD width (scalar ==
vectorized), so it is reproducible by construction - the sqrt+divide route needs no
polynomial and carries no SFU/driver dependence (the thing that split the transcendental
tier into driver families in the GPU attestation).

What it does NOT yet prove (honest scope / open cells):
- CPU == GPU: the Talario GLSL shader must compute `inversesqrt`/`1.0/sqrt` the same way and
  reproduce this exact golden on a real GPU. Vulkan f32 `OpSqrt`/`OpFDiv` must be correctly
  rounded on the target; confirm per device.
- Cross-CPU-ISA: re-run on ARM/NEON (e.g. GH200) to confirm the golden holds there too.
- Edge inputs: this input is strictly positive finite. NaN/+Inf/-Inf/denormal/zero handling
  must be pinned and tested separately (sqrt of negative -> NaN payload, 1/sqrt(0) -> +Inf).
- Not yet wired into the Tessero ABI (`tsr_repro_rsqrt_f32`) / Makefile; demo is standalone.

## exp2(x), exp(x), erf(x)   [status: TODO]

The real work (minimax polynomial + exact range reduction in explicit `fmaf`). To be added
with their goldens once implemented. See the op-by-op plan in the design note.
