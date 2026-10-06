# Coverage of real-world NumPy and SciPy usage

Symbol coverage counts every public NumPy and SciPy function equally: `numpy.zeros` weighs as much as
`numpy.polynomial.hermite_e.hermeint`. Real code does not use them equally. This page weights each function by
how often widely used scientific Python libraries call it, so it answers a different question: **if you port a
typical piece of NumPy/SciPy code to PHP, how much of what it calls does Tessero already provide, verified
against NumPy and SciPy on both backends?**

## How it is measured

- **Corpus.** `tools/parity/usage.py` clones the source of widely used scientific packages at pinned release
  tags (scikit-learn, pandas, statsmodels, scikit-image, astropy, networkx, matplotlib, xarray, seaborn,
  librosa, arviz-stats, MNE, nilearn, pingouin, lifelines, pvlib, PyMC, scanpy). NumPy and SciPy themselves are
  left out. Tests, docs, examples, benchmarks and vendored code are skipped, so the counts reflect what library
  code calls in production.
- **What counts.** A call site whose callee resolves through the file's imports to a symbol of the census
  (`np.linalg.norm(x)`, `stats.norm.cdf(x)`, `gammaln(x)`). Aliases count for their canonical function. Method
  calls on arrays (`x.sum()`) cannot be attributed statically and are not counted.
- **What "verified" means.** The same as everywhere else in the program: the function's NumPy/SciPy fixture test
  (or, for functions older than the function registry, its PHPUnit suite) passed on that backend in the recorded
  run. Nothing on this page is typed by hand. `tools/parity/report.py` generates it from `usage.json`, the census,
  the exclusions and `results.json`.
- **Exclusions.** Calls to symbols excluded by category E1–E7 (`numpy.dtype`, `numpy.errstate`,
  `scipy.sparse.issparse`, ...: Python-runtime helpers with no numerical counterpart in PHP) are shown
  separately and are not part of the in-scope figures.

## Results

--8<-- "docs/project/_generated/usage.md"

## Reading the figures

The usage-weighted figure and the symbol coverage on the [coverage page](numpy-scipy-coverage.md) measure
different things, and both are reported. Symbol coverage is the finish line of the parity program: every
in-scope symbol verified on both backends. The usage-weighted figure shows how much of everyday code that
coverage already serves. The list of the most-called symbols not yet verified on both backends is the order in
which closing gaps helps the most users.
