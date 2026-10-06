# Monitoring

## Health checks

| Check | Command | Healthy |
|---|---|---|
| Installation | `vendor/bin/tessero doctor` | exit code 0 |
| Machine-readable facts | `vendor/bin/tessero info --json` | JSON with `libtessero`, `simd`, `blas`, `openmp` |
| Laravel | `php artisan tessero:doctor --json` | `backend` is what you deployed |
| Extension | `php -r 'echo Tessero\Ext\Engine::version();'` | the version you deployed |

Run the check in the **same SAPI and php.ini** as production. For FPM, expose
a small internal endpoint rather than relying on the CLI:

```php
Route::get('/internal/tessero', fn (Tessero\Laravel\TesseroManager $t) => response()->json($t->info()))
    ->middleware('internal-only');
```

Add a functional probe that computes something small and checks the answer,
so a broken BLAS or a wrong binary is caught as well as a missing one:

```php
$ok = Tessero::linprog([-1, -1], [[1, 2], [3, 1]], [4, 6])['success']
   && abs(Tessero::array([1.0, 2.0, 3.0])->sum() - 6.0) < 1e-12;
```

## Metrics worth exporting

| Metric | Source | Why |
|---|---|---|
| native bytes in use | `Engine::memoryInUse()` / `Tessero::memoryInUse()` at request/job end | should return to a baseline. A rising floor is a leak in application code (arrays held in statics) |
| native peak | `peakMemory()` | budget headroom. Alert when the peak approaches the budget |
| mapped bytes | `Engine::info()['memory_mapped']` | file mappings held by the process; should return to 0 when memmaps go out of scope |
| `MemoryError` / `MemoryException` count | exception handler | requests hitting the budget: either raise the budget or tighten input validation |
| solver status counts | `status` field of LP/MILP/MDP results | a rise in `1` (limit) or `4` (numerical) means models have grown or become ill-conditioned |
| solve duration | wrap calls in your timer | capacity planning for queue workers |
| `nodes` / `iterations` | solver results | early warning before limits are hit |

Example: a Laravel job middleware that records memory and duration.

```php
final class TesseroMetrics
{
    public function handle(object $job, Closure $next): void
    {
        $t = hrtime(true);
        try {
            $next($job);
        } finally {
            Log::info('tessero.job', [
                'job'        => $job::class,
                'ms'         => (hrtime(true) - $t) / 1e6,
                'native_mb'  => Tessero\Ext\Engine::memoryInUse() / 1048576,
                'native_peak_mb' => Tessero\Ext\Engine::peakMemory() / 1048576,
            ]);
        }
    }
}
```

## Logs

Tessero writes no logs of its own. Everything it has to report arrives as an
exception or a result status. Log:

- exception class and message. Messages include shapes, sizes and the offending state/action or constraint.
- solver `message` and `status` for non-optimal results.
- the backend and version once at worker start (`TesseroManager::info()`).

## What not to alert on

- The first request after an FPM restart being slower: preload and OpenMP
  pool start-up. It is a one-off per worker.
- `peakMemory()` staying high: it is a high-water mark for the process
  lifetime and never decreases.
