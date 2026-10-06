<?php
/*
 * PHP 8.4.21 function-level JIT miscompile (Tessero's Nelder-Mead before the workaround).
 *   php jit-function-nelder-mead-repro.php                      -> x[0]=0.99910115125894539 nit=141 nfev=243
 *   php -d opcache.enable_cli=1 -d opcache.jit=function -d opcache.jit_buffer_size=128M jit-function-nelder-mead-repro.php
 *                                                               -> "ENTRY 293 (...) - live var ..." on stderr,
 *                                                                  nit=<garbage>, x unchanged, 8 evaluations
 * The code must live in an included file; pasted into the main script it compiles correctly.
 */
require __DIR__ . '/jit-function-nelder-mead-case.php';
