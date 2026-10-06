---
name: Bug report
about: A reproducible problem in the kernel, the FFI package or the extension
title: ""
labels: bug
---

**What happened**

<!-- What you did, what you expected, and what actually happened. -->

**`bin/tessero doctor` output**

<!-- Run the deployment check and paste its output. It reports ffi.enable, preload,
     the located binary and the BLAS/LAPACK it links, which answers most environment questions. -->

```
$ vendor/bin/tessero doctor

```

**Which backend**

- [ ] FFI package (`Tessero\...`)
- [ ] Native extension (`Tessero\Ext\...`)
- [ ] Both / not sure

**Minimal reproducer**

```php
<?php
// the smallest script that shows the problem
```

**Environment**

- Tessero version:
- PHP version (`php -v`, note NTS/ZTS and JIT if relevant):
- OS and architecture:
