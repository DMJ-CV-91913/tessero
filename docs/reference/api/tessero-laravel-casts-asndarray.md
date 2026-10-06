# AsNDArray

`Tessero\Laravel\Casts\AsNDArray` implements `Illuminate\Contracts\Database\Eloquent\Castable`

Eloquent cast between a column and an NDArray.

  protected $casts = [
      'weights'  => AsNDArray::class,                  // JSON nested arrays (default)
      'profile'  => AsNDArray::class.':binary',        // exact bytes, base64, compact
      'features' => AsNDArray::class.':binary,float32',
  ];

Binary storage is {"dtype":"float64","shape":[24],"b64":"..."}: exact to
the bit, about 2.7x smaller than JSON for float64 and decodes without any
float parsing. Either format can be read back regardless of the setting.

## Methods

### castUsing

```php
static castUsing(array $arguments): Illuminate\Contracts\Database\Eloquent\CastsAttributes
```
