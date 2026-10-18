# Reproducible-f32 golden vectors

Golden SHA-256 hashes and the canonical spec for the reproducible-f32 transcendental
primitives (`tsr_repro_rsqrt_f32` / `tsr_repro_exp2_f32` / `tsr_repro_exp_f32` /
`tsr_repro_erf_f32`, implemented in `csrc/src/repro_f32.c`, declared in
`csrc/include/tessero.h`). A hash here is the single defined f32 bit-result for the
primitive on a fixed input. Every conformant implementation (Tessero at any SIMD level,
any thread count, any compiler, and the Talario GPU shader) must reproduce it. See
`REPRO_F32_KICKOFF.md` and `talario/docs/papers/fma-only-transcendentals-design.md`.

## How to reproduce

```
bash csrc/repro/run_determinism.sh
```

It builds `csrc/src/repro_f32.c` + `csrc/repro/determinism_test.c` three ways, all with
`-fno-fast-math -ffp-contract=off` (never fast-math, which would swap in the
non-reproducible approximations): scalar (no vectorization), AVX2+FMA, and AVX-512 if the
host has it. For each primitive it hashes the raw f32 output and confirms it is
byte-identical across SIMD level AND across 1 vs 4 OpenMP threads. Equal bytes == bit
identical == reproducible, and that hash is the golden.

## Input vectors

- N = 2^20 (1,048,576) f32, generated as a pure function of the index (no RNG, no time),
  so the golden is stable for everyone. Generators are in `determinism_test.c`.
- `rsqrt`: strictly positive, magnitude spread ~1e-4 .. 1e4 (identical to the original
  proof-of-concept `demo_rsqrt.c`, so the wired ABI reproduces the first proven golden).
- `exp2`: linear spread over [-30, 30]. `exp`: [-20, 20]. `erf`: [-4, 4].
- The golden vector is finite and in-domain on purpose. NaN payloads and the exact
  subnormal/overflow boundary are not guaranteed identical across vendors, so edge inputs
  are verified by CLASSIFICATION (the `edges` mode), not folded into the hash.

## Goldens (first confirmed 2026-10-17, GCP n2 x86-64, AVX2+FMA, gcc 13.3, no AVX-512)

| primitive | GOLDEN(f32, N=2^20) SHA-256 | determinism | accuracy vs f64 ref |
|-----------|------------------------------|-------------|---------------------|
| rsqrt | `fce3ce9bb62423fb56ec0c331d6b6262d8f76be629718572b37d31d80b9a696b` | scalar == avx2 == 4-thread | <= 1 ULP |
| exp2  | `d7b57bb03e2c26b20642f0697daf64a366bedd9d7a3ee5edd79d8dae502aabba` | scalar == avx2 == 4-thread | <= 1 ULP |
| exp   | `7c0748f832cf0c6ea64d0382af76615033014602c90e882437dac5108d0aa336` | scalar == avx2 == 4-thread | <= 1 ULP |
| erf   | `e268a1e2103e4ab6e33af6324bc5f01a208f18a0e3152f00309e9740c032a6eb` | scalar == avx2 == 4-thread | <= 5 ULP, max abs err 2.9e-7 |

The rsqrt golden is byte-for-byte the same hash the original proof-of-concept produced,
which confirms the wired ABI function matches it exactly.

## Canonical spec (exact f32 bit patterns)

Constants are exact f32 hex floats (printed by `determinism_test spec`). The evaluation
order is part of the spec: a conformant GPU shader MUST use these exact values and this
exact fmaf order, because CPU/GPU bit-identity depends on both sides running the identical
sequence. Only correctly-rounded f32 ops are used: `+ - * /`, `sqrtf`, and `fmaf` (one
rounding). No libm exp/exp2/erf, no SVML, no reciprocal/rsqrt estimate, no powf, no f64 or
x87 intermediates, no reassociation.

### scale2(y, n) = y * 2^n  (shared by exp2 and exp)
```
n1 = n / 2 (C integer truncation);  n2 = n - n1
f1 = bits_to_f32((uint32)(n1 + 127) << 23)      // exact power of two
f2 = bits_to_f32((uint32)(n2 + 127) << 23)
return (y * f1) * f2                             // left-associated, fixed order
```
Split so each factor is a normal power of two; the first multiply is exact, so the second
is a single correctly-rounded step even into the subnormal range. Valid for n in [-150, 128].

### rsqrt(x) = 1/sqrt(x)
```
y = 1.0f / sqrtf(x)
```
Both `sqrtf` and `/` are IEEE-correctly-rounded, so this is reproducible with no polynomial
and no SFU estimate. Edges (IEEE-natural): x < 0 -> NaN; +0 -> +Inf; -0 -> -Inf;
+Inf -> +0; -Inf -> NaN; NaN -> NaN; subnormal -> large finite.

### exp2(x) = 2^x
```
if x >= 128   -> +Inf        (also +Inf input)
if x <= -150  -> 0           (also -Inf input)
k = floorf(x + 0.5f);  r = x - k;  n = (int)k      // r in [-0.5, 0.5]
2^r = fmaf(fmaf(fmaf(fmaf(fmaf(C0,r,C1),r,C2),r,C3),r,C4),r,C5) then fmaf(.,r,1)
return scale2(2^r, n)
```
| const | hex float | decimal |
|-------|-----------|---------|
| C0 | `0x1.41fbbcp-13` | 0.00015353362 |
| C1 | `0x1.5f3e52p-10` | 0.00133988739 |
| C2 | `0x1.3b2d4cp-7`  | 0.00961843692 |
| C3 | `0x1.c6aee8p-5`  | 0.0555033237 |
| C4 | `0x1.ebfbdcp-3`  | 0.240226477 |
| C5 | `0x1.62e43p-1`   | 0.693147182 |

### exp(x) = e^x
```
if x >= 88.72283905  -> +Inf     (also +Inf input)
if x <= -103.972084  -> 0        (also -Inf input)
k = floorf(fmaf(x, LOG2E, 0.5f))                   // round(x*log2e)
r = fmaf(-k, LN2_HI, x);  r = fmaf(-k, LN2_LO, r)  // r = x - k*ln2, two-part ln2
z = r*r;  P = fmaf(fmaf(fmaf(fmaf(fmaf(Q0,r,Q1),r,Q2),r,Q3),r,Q4),r,Q5)
e^r = fmaf(P, z, r) + 1
return scale2(e^r, (int)k)
```
| const | hex float | decimal |
|-------|-----------|---------|
| LOG2E  | `0x1.715476p+0`   | 1.44269502 |
| LN2_HI | `0x1.63p-1`       | 0.693359375 |
| LN2_LO | `-0x1.bd0106p-13` | -0.000212194442 |
| Q0 | `0x1.a0d2cep-13` | 0.000198756912 |
| Q1 | `0x1.6e879cp-10` | 0.00139819994 |
| Q2 | `0x1.11121p-7`   | 0.00833345205 |
| Q3 | `0x1.555382p-5`  | 0.0416657962 |
| Q4 | `0x1.555554p-3`  | 0.166666657 |
| Q5 | `0x1p-1`         | 0.5 |

### erf(x)
Odd function, two frozen forms by magnitude (the branch is a deterministic function of x,
so it stays bit-reproducible):
```
ax = |x|
if ax <= 0.5:        // Maclaurin series, avoids the 1 - P*exp cancellation near 0
    w = x*x
    erf = x * (fmaf(fmaf(fmaf(fmaf(fmaf(B5,w,B4),w,B3),w,B2),w,B1),w,B0))
else:                // Abramowitz & Stegun 7.1.26
    s = sign(x)
    if ax >= 4 -> s                               // saturate; +-Inf -> +-1
    t = 1.0f / fmaf(P, ax, 1.0f)
    poly = fmaf(fmaf(fmaf(fmaf(A5,t,A4),t,A3),t,A2),t,A1) * t
    e = exp(-(ax*ax))                             // the reproducible exp above
    erf = s * fmaf(-poly, e, 1.0f)
```
| const | hex float | decimal |
|-------|-----------|---------|
| B0 | `0x1.20dd76p+0`   | 1.12837923 |
| B1 | `-0x1.812746p-2`  | -0.376126379 |
| B2 | `0x1.ce2f22p-4`   | 0.112837918 |
| B3 | `-0x1.b82ce4p-6`  | -0.0268661715 |
| B4 | `0x1.565bcep-8`   | 0.00522397785 |
| B5 | `-0x1.c02db4p-11` | -0.000854832702 |
| P  | `0x1.4f740ap-2`   | 0.327591091 |
| A1 | `0x1.04f20cp-2`   | 0.254829586 |
| A2 | `-0x1.23531cp-2`  | -0.284496725 |
| A3 | `0x1.6be1c6p+0`   | 1.42141378 |
| A4 | `-0x1.7401c6p+0`  | -1.45315206 |
| A5 | `0x1.0fb844p+0`   | 1.06140542 |

## What this PROVES

On x86 with AVX2+FMA, all four primitives are bit-identical across SIMD width (scalar ==
vectorized) and across thread count (1 == 4), by construction: each element runs the
identical `fmaf` sequence with no reduction and no SIMD/thread-dependent order. This is the
CPU canonical the Talario GPU shader mirrors bit-for-bit to collapse the transcendental
tier (gelu/softmax/rmsnorm/layernorm) into the single bit-identical family, the same way
the EXACT-op tier (matmul/add/FMA) already is.

## Open cells (honest scope, not yet proven)

- CPU == GPU: the Talario GLSL/SPIR-V shader must compute each primitive with these exact
  constants and fmaf order and reproduce these goldens on real silicon. Vulkan f32
  `OpFMul`/`OpFMA`/`OpSqrt`/`OpFDiv` must be correctly rounded on the target; confirm per
  device. (Talario half, separate repo.)
- Cross-CPU-ISA: re-run on ARM/NEON (e.g. GH200) to confirm the goldens hold there too.
  Expected to hold (same IEEE fmaf), but not yet run.
- AVX-512: this host has no AVX-512, so the scalar == avx512 cell is unconfirmed; the
  runner checks it automatically on a host that has it.
- erf accuracy is ~2.9e-7 abs / <= 5 ULP in the A&S region (ax > 0.5); the near-zero
  region (gelu-critical) is tight via the series. Correctly-rounded erf is a later tier.
