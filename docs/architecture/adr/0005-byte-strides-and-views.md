# 0005. NumPy memory layout: byte strides and shared-memory views

- Status: Accepted
- Date: 2026-09-26

## Context

Slicing, transposing and broadcasting are the most common operations in
numerical code. Copying on each would make them O(n) and double memory use.

## Decision

An array is (data block, byte offset, shape, byte strides, dtype), exactly as
in NumPy. Strides may be zero (broadcast) or negative (reversed). Basic
slicing, transposes, reshapes of contiguous arrays and broadcasts produce
views that share the block. Every computation writes a new C-contiguous array.
Views keep the block alive: in the FFI package through the shared `Buffer`,
in the extension through a reference to the root owner object.

## Consequences

- Slicing is O(ndim), and broadcasting never copies.
- Kernels must handle arbitrary strides. The iterator coalesces dimensions so
  the common contiguous case still runs as one flat loop.
- Writes through views are visible in the parent, as in NumPy. This is
  documented, because PHP developers expect value semantics from arrays.
- `clone` in the extension deliberately makes a compact copy (value semantics
  for the explicit clone operation).
