# Laravel

`tessero/laravel` connects Tessero to a Laravel 10, 11 or 12 application.

```bash
composer require tessero/laravel
php artisan vendor:publish --tag=tessero-config     # optional
php artisan tessero:doctor
```

## What you get

| Piece | Purpose |
|---|---|
| `TesseroServiceProvider` (auto-discovered) | config, `TesseroManager` singleton, runtime settings, response macro, `tessero:doctor` |
| `Tessero` facade | `array`, `zeros`, `fromBytes`, `memmap`, `load`, `openMemmap`, `linprog`, `milp`, `solveMdp`, `normal`, `random`, `info`, `backend` |
| `Casts\AsNDArray` (alias `Casts\TensorCast`) | Eloquent column ⇄ array |
| `Rules\NumericArray` | validates request input before it becomes an array |
| `response()->ndarray($a)` | JSON response written from native memory |

## Backends

`TESSERO_BACKEND=auto` (default) uses the native extension when it is loaded
and the FFI package otherwise. Arrays come back as `Tessero\Ext\NDArray` or
`Tessero\NDArray`. Both have the same core API, and solver results from the
facade are plain PHP arrays with the same keys on either backend, so
controllers and jobs do not depend on which one is installed.

Set `ext` or `ffi` to require one. Boot then fails loudly if it is missing,
which is useful in production images.

## Configuration

`config/tessero.php`. Every key can be set from the environment:

| Key | Env | Default | Meaning |
|---|---|---|---|
| `backend` | `TESSERO_BACKEND` | `auto` | `auto`, `ext`, `ffi` |
| `threads` | `TESSERO_THREADS` | 1 | OpenMP threads for large element-wise kernels and MDP sweeps |
| `memory_budget` | `TESSERO_MEMORY_BUDGET` | 268435456 (256 MiB) | native memory cap per worker, bytes; 0 = unlimited |
| `blas_threads` | `TESSERO_BLAS_THREADS` | 1 | OpenBLAS threads (FFI linear algebra) |
| `epsilon` | `TESSERO_EPSILON` | 1e-6 | default MDP tolerance |
| `cast_format` | `TESSERO_CAST_FORMAT` | `json` | default storage format for casts |
| `max_elements` | `TESSERO_MAX_ELEMENTS` | 1 000 000 | default cap for validation |

`threads`, `memory_budget`, `blas_threads` and `epsilon` are applied at boot
**and re-applied at the start of every Octane request and task, and before
every queued job**. Long-lived workers therefore never carry one request's
settings into the next. See [Memory and threads](../operations/memory-and-threads.md)
before raising `threads` under FPM.

## Eloquent casts

```php
use Tessero\Laravel\Casts\AsNDArray;

class Forecast extends Model
{
    protected $casts = [
        'quantiles'  => AsNDArray::class,                   // JSON nested arrays
        'covariance' => AsNDArray::class . ':binary',       // exact, compact
        'features'   => AsNDArray::class . ':binary,float32',
    ];
}

$f = Forecast::find(1);
$f->covariance->shape();          // an NDArray
$f->quantiles = $q->mul(1.1);     // assign an NDArray, a PHP array, or null
$f->save();
```

| Format | Column contents | Size (float64) | Exact |
|---|---|---|---|
| `json` | `[[1.5, 2.0], …]` | ~20 bytes/value | to 17 significant digits (round-trips) |
| `binary` | `{"dtype":"float64","shape":[24],"b64":"…"}` | ~10.7 bytes/value | bit-exact, including NaN payloads and -0.0 |

Either format is read back whatever the setting, so you can switch
`cast_format` without migrating data. Use a `json` or `text`/`longtext` column.
`toArray()`/`toJson()` on the model serialise the attribute as nested lists.

## Validating input

```php
use Tessero\Laravel\Rules\NumericArray;

$data = $request->validate([
    'prices'  => ['required', new NumericArray(ndim: 1, maxElements: 8784)],
    'weights' => ['required', new NumericArray(shape: [null, 3])],        // n × 3
    'mask'    => ['sometimes', new NumericArray(allowBool: true)],
]);
$prices = Tessero::array($data['prices']);
```

The rule checks list arrays (no string keys), a rectangular shape, the
dimension count or exact shape, the element cap, and that every leaf is an int
or finite float. It rejects numeric strings, so use it on JSON request bodies.
Form-encoded input arrives as strings and needs casting first.
`NumericArray::check($value, …)` returns the error message (or null) outside
Laravel.

**Always cap `maxElements` on public endpoints.** A request body is cheap to
send, and native memory is not counted by `memory_limit`
([Security](../operations/security.md)).

## Responses

```php
return response()->ndarray($forecast->mean(0));                 // 200, application/json
return response()->ndarray($matrix, 201, ['X-Model' => 'v3']);
```

The body is written by `toJson()` straight from native memory.

## Solvers from controllers and jobs

```php
use Tessero\Laravel\Facades\Tessero;

$plan = Tessero::linprog(c: $c, A_ub: $A, b_ub: $b, bounds: $bounds);
if (! $plan['success']) {
    abort(422, $plan['message']);
}

$policy = Tessero::solveMdp($P, $R, gamma: 0.97)['policy'];   // 'policy_iteration' or 'value_iteration'
$draws  = Tessero::normal(seed: $scenarioId, shape: [1000, 24]);
```

Long solves belong in queued jobs. The provider re-applies thread and memory
settings before each job.

## Octane

Tessero holds no per-request global state: arrays are ordinary PHP objects
freed when they go out of scope. Two things to know:

- Do not keep large arrays in static properties or singletons unless you mean
  to. They stay resident for the worker's lifetime and count against its
  memory budget.
- Settings are process-wide (see above); the provider resets them for each
  request.

## Diagnostics

```bash
php artisan tessero:doctor          # backend, versions, SIMD, OpenMP, threads, memory
php artisan tessero:doctor --json   # for health checks
```

## Testing your application

The FFI package works in any PHPUnit run with `ffi.enable=1`. Add it to
`phpunit.xml`:

```xml
<php><ini name="ffi.enable" value="1"/></php>
```

To force a backend in tests set `TESSERO_BACKEND` in `phpunit.xml` `<env>`.
