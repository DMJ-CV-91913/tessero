# tessero/laravel

Laravel integration for Tessero.

```bash
composer require tessero/laravel          # pulls tessero/tessero
php artisan vendor:publish --tag=tessero-config
php artisan tessero:doctor
```

| Piece | Use |
|---|---|
| `TesseroServiceProvider` (auto-discovered) | config, `TesseroManager` singleton, runtime settings applied at boot and re-applied per Octane request / queued job, `response()->ndarray()`, `tessero:doctor` |
| `Tessero` facade | `Tessero::array()`, `linprog()`, `milp()`, `solveMdp()`, `normal()`, `random()`, `info()` — same results on the native and FFI backends |
| `Casts\AsNDArray` (alias `Casts\TensorCast`) | Eloquent column ⇄ NDArray; `:json` (default) or `:binary` (exact, compact), optional dtype |
| `Rules\NumericArray` | validates request input before it becomes an array: numeric, rectangular, dimensions, size cap |

```php
use Tessero\Laravel\Casts\AsNDArray;
use Tessero\Laravel\Facades\Tessero;
use Tessero\Laravel\Rules\NumericArray;

class Portfolio extends Model
{
    protected $casts = [
        'covariance' => AsNDArray::class . ':binary',
        'weights'    => AsNDArray::class,
    ];
}

public function store(Request $request)
{
    $data = $request->validate(['returns' => ['required', new NumericArray(ndim: 2, maxElements: 250_000)]]);
    $r = Tessero::array($data['returns']);
    return response()->ndarray($r->mean(0));        // JSON straight from native memory
}
```

Configuration lives in `config/tessero.php` (backend, threads, memory budget,
BLAS threads, MDP tolerance, cast format). See the platform manuals:
`docs/guide/laravel.md` and `docs/operations/`.

Tested: casts, the validation rule and the manager (on both backends) run in
the package's own suite; the service provider is exercised in CI inside a
fresh Laravel application.
