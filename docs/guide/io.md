# Input and output

## Choosing a format

| Need | Use | Why |
|---|---|---|
| Exchange with Python/NumPy | `.npy` / `.npz` | NumPy's own format; exact; memory layout preserved |
| Store in a database column | binary cast `{"dtype","shape","b64"}` | exact to the bit, ~2.7× smaller than JSON for float64, no float parsing |
| Send to a browser or API client | `toJson()` / `response()->ndarray()` | written from native memory in one pass |
| Cache, queue payload, session | `serialize()` | dtype + shape + raw bytes; exact |
| Another process or language | `toBytes()` + dtype + shape | raw little-endian, C order |
| Spreadsheet or CSV | `toArray()` then your CSV writer | Tessero has no CSV parser; PHP's `fgetcsv`/`fputcsv` are fine |

## NumPy files

```php
use Tessero\Io\Npy;

Npy::save('/data/weights.npy', $w);
$w = Npy::load('/data/weights.npy');          // any file written by numpy.save

Npy::saveZ('/data/model.npz', ['coef' => $coef, 'cov' => $cov]);             // compressed
Npy::saveZ('/data/model.npz', ['coef' => $coef], compress: false);
['coef' => $coef, 'cov' => $cov] = Npy::loadZ('/data/model.npz');

$bytes = Npy::encode($a); $a = Npy::decode($bytes);   // in memory (e.g. object storage)
```

Format versions 1.0, 2.0 and 3.0 are read (1.0 or 2.0 written), as are C and Fortran order
(Fortran-order files load as C-order copies; map them with `NDArray::load($path, 'r')` to keep the file layout) and every
Tessero dtype. The parity suite reads files written by NumPy and checks every
byte.

`Npy::load` (and `NDArray::load()`, on either backend) reads the header with
libtessero's strict, fuzzed parser: it never evaluates the header
dictionary and rejects object arrays (`dtype=object`) and pickles. Loading
an untrusted `.npy` file is therefore safe in the way `numpy.load(...,
allow_pickle=False)` is. Size limits are still your job (a header can declare
a huge shape, and the whole file is read into PHP memory first); see
[Security](../operations/security.md).

## Raw bytes

```php
$bytes = $a->toBytes();                               // C order, little-endian
$b = NDArray::fromBytes($bytes, 'float32', [480, 640]);
```

## JSON

```php
$a->toJson();                // "[[1.0,2.5],[3.0,4.0]]": nested by shape, floats keep ".0"
json_encode($a);             // same data through JsonSerializable
json_encode(['series' => $a, 'meta' => $meta]);   // arrays nest inside larger payloads
```

For large arrays prefer `toJson()` over `json_encode($a->toList())`, because
it skips building the PHP array. On 1M float64 values it takes 61 ms (FFI,
integer-valued data) or 148 ms (extension, general floats), against 216 ms for
`json_encode($a->toList())`.

NaN and ±INF have no JSON representation. `toJson()` writes them as `null`,
while `json_encode($a)` fails on them as it does for plain PHP floats. If
`null` is not what the client expects, replace them first, e.g.
`NDArray::where($a->isfinite(), $a, 0)`.

## serialize

Both array classes implement `__serialize`/`__unserialize`. Views are
serialized as compact copies. Unserializing checks that the byte length
matches dtype × shape and throws otherwise, so a truncated cache entry cannot
produce an array that reads past its buffer.
