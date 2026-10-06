# Upstream bugs (PHP engine)

Five PHP engine defects found while building Tessero and SciPHP. Each has a
reproducer that runs on stock PHP; each is worked around in Tessero, so none
of them affects users. They are drafted for https://github.com/php/php-src/issues
and have **not been filed yet** — file them from an account you control after
re-running the reproducers on the latest 8.4/8.5 patch release.

| # | Component | Symptom | Reproducer | Tessero workaround |
|---|-----------|---------|------------|--------------------|
| 1 | Opcache tracing JIT | `$arr[$i] -= $arr[$k]` on an array property uses a float's bit pattern as the key; wrong results, "Undefined array key -4616189618054758400" | `laravel-sciphp/tools/jit-repro/repro.php` | explicit read-then-write in hot loops |
| 2 | ext/ffi | passing `$p + n` as a temporary argument breaks later `$p + m` ("Unsupported operand types: FFI\CData + int") | [`ffi-pointer-temporary-repro.php`](upstream/ffi-pointer-temporary-repro.php) | no CData pointer arithmetic; pointers are built from integer addresses (`Library::ptr()`) |
| 3 | ext/ffi | `FFI::cast('uintptr_t', $voidPtr)` returns a different value from `FFI::cast('uintptr_t', FFI::cast('char*', $voidPtr))` | [`ffi-void-pointer-cast-repro.php`](upstream/ffi-void-pointer-cast-repro.php) | `Library::address()` casts through `char*` |
| 4 | Opcache function JIT | a large loop function (Nelder-Mead) prints `ENTRY 293 (...) - live var ...` and its loop counter becomes garbage | [`jit-function-nelder-mead-repro.php`](upstream/jit-function-nelder-mead-repro.php) | the loop counter lives in an object property; plain assignments instead of ternary list-destructuring |
| 5 | Opcache tracing JIT | in one long test sequence, the stacked `Linalg::det` returned `-0` for one matrix of a stack | not isolated (see below) | pivots copied out of `CData` in one call; no list destructuring in the stacked loop |

---

## 1. Tracing JIT: compound assignment on an array property uses a float's bits as the key

**PHP:** 8.4.21 NTS x86-64 (Linux). `opcache.jit=tracing` only; `function` and JIT-off are correct.

**Code shape** (SciPHP `TwoPhaseSimplex::solve`, phase-1 objective set-up):

```php
foreach ($this->basis as $r => $column) {
    if ($column < $artificialStart) { continue; }
    $base = $r * $width;
    for ($j = 0; $j < $width; $j++) {
        $this->tableau[$objective + $j] -= $this->tableau[$base + $j];
    }
}
```

**Observed** after the loop has become hot:

```
first warning: Undefined array key -4616189618054758400
PHP 8.4.21, opcache.jit=tracing: 398 / 400 wrong, 399 warnings
```

`-4616189618054758400` is `0xBFF0000000000000`, the IEEE-754 bits of `-1.0` — a
value stored in the tableau — so the compiled trace uses the *value* register
as the *key* of the compound assignment's write. The corrupted phase-1 row makes
feasible LPs report "infeasible".

**Expected:** identical results with and without JIT (`0 / 400 wrong`).

**Reproduce:**

```bash
cd laravel-sciphp
php tools/jit-repro/repro.php                        # 0 / 400 wrong
php -d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=64M tools/jit-repro/repro.php
```

The reproducer loads the real solver stack with only the one statement switched
back to the compound form. Hand-written minimal versions of the loop (same
shape, synthetic data) did **not** trigger it, so the trace that miscompiles
depends on the surrounding type feedback; the reproducer is therefore not yet
minimal. Next step before filing: bisect `TwoPhaseSimplexCompound.php` down
(remove the pivoting code paths one by one while the failure persists) and run
with `opcache.jit_debug=0x...` to capture the trace.

**Workaround:** `$v = $a[$i] - $a[$k]; $a[$i] = $v;` — used in every hot loop
in SciPHP and Tessero's PHP code.

---

## 2. FFI: pointer-arithmetic temporaries break later arithmetic on the source pointer

**PHP:** 8.4.21. Reproducer: `php -d ffi.enable=1 ffi-pointer-temporary-repro.php`

```
1. FFI::memcpy($p + 8, ...): ok
2. $p + 16 afterwards: FAILED - Unsupported operand types: FFI\CData + int
```

Passing `$p + 8` directly as a function argument (FFI::memcpy, FFI::string or
any FFI call) leaves `$p` in a state where `$p + n` no longer dispatches to
FFI's `do_operation` handler. If `$x = $p + 8;` runs once beforehand, the
failure does not occur — which points at the temporary CData created for the
argument releasing something (the pointer's type?) that `$p` still references.
The same root cause likely explains an earlier observation: `FFI::memcpy` onto
`$ptr + $offset` past 131,072 doubles either threw "Attempt to write over data
boundary" or silently wrote nothing.

**Expected:** `$p + 16` works regardless of earlier temporaries.

**Workaround:** never do arithmetic on CData pointers. Tessero stores integer
addresses and creates each pointer with `FFI::cast('char*', $address)`.

---

## 3. FFI: casting `void*` to an integer type returns the wrong value

**PHP:** 8.4.21. Reproducer: `php -d ffi.enable=1 ffi-void-pointer-cast-repro.php`

```
void* -> uintptr_t:          140192952367904
void* -> char* -> uintptr_t: 94488916751648
INCONSISTENT (expected equal, non-zero addresses)
```

For a `void*` returned by a C function, `FFI::cast('uintptr_t', $p)` does not
return the pointer value (with a pointer returned from libtessero it returned
0; with libc `malloc` it returns an unrelated address). Casting to `char*`
first gives the correct address.

**Expected:** both casts return the address held by the pointer.

**Workaround:** `Library::address()` always casts through `char*`.

---

## 4. Function JIT: register allocation failure corrupts a loop counter

**PHP:** 8.4.21 NTS x86-64, `opcache.jit=function` only (JIT off and tracing are correct).
Found by Tessero's parity suite: `Minimize::nelderMead()` with the whole
simplex loop in one method.

```bash
cd /tmp   # run by absolute path from another directory; see note below
php -d opcache.enable_cli=1 -d opcache.jit=function -d opcache.jit_buffer_size=128M \
    /path/to/tessero/docs/project/upstream/jit-function-nelder-mead-repro.php
```

```
ENTRY 293 (block 1723 start 9858) - live var 9734          <- printed by the JIT to stderr
x[0]=1.3390000000000004 nit=140605720573025 nfev=8 (Maximum number of iterations has been exceeded.)
```

Expected (JIT off, or tracing): `x[0]=0.99910115125894539 nit=141 nfev=243`.

The "ENTRY … live var" line is the JIT's own register-allocator diagnostic;
afterwards `$iterations` holds what looks like a pointer value and the loop exits
at once. The code must be in an included file ([`jit-function-nelder-mead-case.php`](upstream/jit-function-nelder-mead-case.php));
pasted into the main script it compiles correctly. In one session the
reproducer did not trigger when started from inside `docs/project/upstream/` — the
reason is unknown; from other directories it failed every time (5/5).

**Trigger narrowed down:** replacing

```php
[$rho, $chi, $psi, $sigma] = $adaptive
    ? [1.0, 1 + 2 / $n, 0.75 - 1 / (2 * $n), 1 - 1 / $n]
    : [1.0, 2.0, 0.5, 0.5];
```

with four plain assignments makes the defect disappear, so the list
destructuring of a ternary's array result is involved. A 20-line function with
just that statement and a counting loop did *not* reproduce it; reducing the
reproducer further is the next step before filing.

**Workaround:** plain assignments (Tessero's `Minimize::nelderMead()`).

---

## 5. Tracing JIT: wrong stacked determinant after a long warm-up (not isolated)

**PHP:** 8.4.21 NTS x86-64, `opcache.jit=tracing`, `jit_buffer_size=128M`.

**Observed:** running the unit suites and then the parity suite in one
process (`NDArrayTest`, `NumericsTest`, `SolversTest`, `ParityTest`,
`FuzzTest`) made `Linalg::det()` on a stack of 4×4 matrices return `-0` for
the second matrix instead of `12.548574973263474`. This happened in every one
of 4 runs. Removing any one of the three unit suites from the sequence, adding
a debugging `fwrite`, or adding an unused assignment to `slogdet()` all made it
pass, so the failure depends on which traces were compiled earlier. JIT off
and the function JIT were correct.

**Status:** root cause not isolated, and no standalone reproducer yet. The code
involved read LAPACK pivot indices element by element from an `int[]` `CData`
inside a hot loop (PHP 8.4 JIT-compiles FFI element access) and destructured
the recursive call's result with `[$s, $l] = …`.

**Mitigation in Tessero:** pivots are copied out with one `FFI::string` +
`unpack` call (`Linalg::ints()`), and loops read the PHP copy. The stacked loop
uses plain indexing instead of list destructuring. After the change, the full
sequence passes in all three JIT modes, in both suite orders and from two
working directories (12 of 12 runs). CI runs every suite in all three JIT modes
so that a recurrence is caught. Treat the JIT as a risk for numerical PHP code
generally: results with `opcache.jit=disable` are the reference.

**Next step before filing:** capture the failing trace with
`opcache.jit_debug=0x1000000` in the failing sequence, then reduce.
