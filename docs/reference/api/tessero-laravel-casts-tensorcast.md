# TensorCast

`Tessero\Laravel\Casts\TensorCast` implements `Illuminate\Contracts\Database\Eloquent\Castable`

Alias of AsNDArray under the name used in the original design notes:
  protected $casts = ['transition_matrix' => TensorCast::class];

## Methods

### castUsing

```php
static castUsing(array $arguments): Illuminate\Contracts\Database\Eloquent\CastsAttributes
```
