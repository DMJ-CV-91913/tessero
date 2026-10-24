# NumericArray

`Tessero\Laravel\Rules\NumericArray` implements `Illuminate\Contracts\Validation\ValidationRule`

Validates request input that will become an NDArray: a (nested) list of
numbers with a rectangular shape, optional exact dimensions and a size cap.

  $request->validate(['prices' => ['required', new NumericArray(ndim: 1, maxElements: 8784)]]);
  $request->validate(['matrix' => ['required', new NumericArray(shape: [null, 3])]]);   // n x 3

Without $maxElements the cap is config('tessero.max_elements') (TESSERO_MAX_ELEMENTS, default 1,000,000).

## Constants

| Name | Value |
|---|---|
| `MAX_NDIM` | `32` |
| `DEFAULT_MAX_ELEMENTS` | `1000000` |

## Methods

### check

```php
static check(mixed $value, ?int $ndim = null, ?array $shape = null, ?int $maxElements = null, bool $allowBool = false): ?string
```

### configuredMaxElements

```php
static configuredMaxElements(): int
```

config('tessero.max_elements') when running inside Laravel, else DEFAULT_MAX_ELEMENTS.

### __construct

```php
__construct(?int $ndim = null, ?array $shape = null, ?int $maxElements = null, bool $allowBool = false)
```

### validate

```php
validate(string $attribute, mixed $value, Closure $fail): void
```
