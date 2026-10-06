<?php

return [

    /*
    |--------------------------------------------------------------------------
    | Backend
    |--------------------------------------------------------------------------
    | "auto" - the native extension (ext-tessero) when loaded, else FFI.
    | "ext"  - require ext-tessero.
    | "ffi"  - require the FFI package (libtessero + ext-ffi).
    | Both give identical numbers; ext has far lower per-call overhead and
    | needs no FFI/preload configuration.
    */
    'backend' => env('TESSERO_BACKEND', 'auto'),

    /*
    |--------------------------------------------------------------------------
    | Runtime settings (re-applied at every request / job under Octane)
    |--------------------------------------------------------------------------
    | threads        OpenMP threads for large element-wise kernels and MDP
    |                sweeps. Keep 1 under PHP-FPM (a worker per core already).
    | memory_budget  Native memory cap per worker in bytes (0 = unlimited).
    |                Native buffers are invisible to memory_limit.
    | blas_threads   OpenBLAS threads for linear algebra (FFI backend).
    | epsilon        Default tolerance for the MDP solvers.
    */
    'threads' => (int) env('TESSERO_THREADS', 1),
    'memory_budget' => (int) env('TESSERO_MEMORY_BUDGET', 256 * 1024 * 1024),
    'blas_threads' => (int) env('TESSERO_BLAS_THREADS', 1),
    'epsilon' => (float) env('TESSERO_EPSILON', 1e-6),

    /*
    |--------------------------------------------------------------------------
    | Eloquent casts
    |--------------------------------------------------------------------------
    | Default storage format for AsNDArray / TensorCast columns:
    | "json"   - human-readable nested arrays (JSON column or text)
    | "binary" - {"dtype","shape","b64"} with base64 little-endian bytes:
    |            exact, ~2.7x smaller than JSON for float64, and much faster.
    */
    'cast_format' => env('TESSERO_CAST_FORMAT', 'json'),

    /* Largest array (elements) a request may send through the validation rule. */
    'max_elements' => (int) env('TESSERO_MAX_ELEMENTS', 1_000_000),
];
