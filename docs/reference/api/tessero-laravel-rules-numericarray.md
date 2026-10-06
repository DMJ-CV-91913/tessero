# NumericArray

`Tessero\Laravel\Rules\NumericArray` implements `Illuminate\Contracts\Validation\ValidationRule`

Validates request input that will become an NDArray: a (nested) list of
numbers with a rectangular shape, optional exact dimensions and a size cap.

  $request->validate(['prices' => ['required', new NumericArray(ndim: 1, maxElements: 8784)]]);
  $request->validate(['matrix' => ['required', new NumericArray(shape: [null, 3])]]);   // n x 3

## Methods

### check

```php
static check(mixed $value, ?int $ndim = null, ?array $shape = null, int $maxElements = 1000000, bool $allowBool = false): ?string
```

### __construct

```php
__construct(?int $ndim = null, ?array $shape = null, int $maxElements = 1000000, bool $allowBool = false)
```

### validate

```php
validate(string $attribute, mixed $value, Closure $fail): void
```
