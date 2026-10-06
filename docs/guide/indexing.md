# Indexing and slicing

Tessero uses NumPy's indexing rules and Python's slice syntax, written as a
string.

## Basic indexing returns views

```php
$a = arange(24)->reshape(4, 6);

$a[1];                  // row 1, a view of shape [6]
$a['1, 2'];             // scalar 8
$a['1:3'];              // rows 1-2
$a[':, ::2'];           // every other column
$a['::-1'];             // rows reversed (negative stride, no copy)
$a['..., -1'];          // last column; `...` fills the remaining axes
$a['None, :, 0'];       // new axis at the front: shape [1, 4]
$a->slice('1:3', 2);    // the same as $a['1:3, 2']: one argument per axis (null = new axis)
```

Rules:

- `start:stop:step`. Omitted parts take Python's defaults. Negative values
  count from the end. Out-of-range bounds are clipped, as in Python.
- A plain integer removes the axis. Out of range throws `IndexError`.
- `...` stands for as many `:` as needed (at most one per index).
- `None` (or `newaxis`) inserts an axis of length 1.
- The index string is parsed in C on both backends. The same parser is used for
  `sliceNative()` and the extension's `$a['…']`, and it is fuzz-tested against
  NumPy on random shapes.

Assigning to a basic index writes into the array:

```php
$a['1:3, ::2'] = 0;                 // scalar broadcast
$a[0] = arange(6)->mul(10);         // array broadcast to the target shape
```

## Boolean masks return copies

```php
$prices = arr([42.0, 39.5, 88.0, 41.0, 120.0]);
$mask = $prices->gt(80);            // bool array
$prices[$mask];                     // [88.0, 120.0]  (a copy)
$prices->filter($mask);             // the same
$prices[$mask] = 80;                // cap in place
$prices->setWhere($mask, 80);       // the same, as a method
$mask->flatNonzero();               // indices of true elements
```

The mask must have the array's shape (or the shape of its leading axes).

## Integer arrays (fancy indexing) return copies

```php
$a->take([3, 0, 0]);                // elements of the flattened array
$a->take([2, 0], axis: 0);          // rows 2 and 0
$a->put([0, 5], [-1, -1]);          // write by flat index
```

With the extension, an integer array or PHP list inside `[]` selects rows:
`$x[[2, 0]]`.

## Conditional selection

```php
NDArray::where($prices->gt(80), 80, $prices);   // like numpy.where(cond, a, b)
$prices->clip(40, 80);
```

## Broadcasting

Binary operations, `where` and assignment broadcast their operands the way
NumPy does. Shapes are aligned from the right, and axes of length 1 stretch to
match:

```php
$m = arange(6)->reshape(2, 3);      // [2, 3]
$m->add(arr([10, 20, 30]));         // [2, 3] + [3]    -> [2, 3]
$m->add(arr([[100], [200]]));       // [2, 3] + [2, 1] -> [2, 3]
$m->broadcastTo([4, 2, 3]);         // view with zero strides; do not write through it
```

Incompatible shapes throw `ShapeError` with both shapes in the message.
Broadcasting never copies: the kernel reads broadcast axes with stride 0.
