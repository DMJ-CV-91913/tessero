"""
.npy files written by NumPy, for the load / memmap / save tests of both
backends. Each entry records the array as C-order bytes (base64), so readers
are checked byte for byte, and the exact file NumPy writes, so save() is too.

    python tests/fixtures/generate_npy.py   ->  tests/fixtures/npy/*.npy + npy.json
"""
import base64
import io
import json
import pathlib

import numpy as np

rng = np.random.default_rng(20260928)
out = pathlib.Path(__file__).parent / "npy"
out.mkdir(exist_ok=True)

cases = {
    "f8_2d": rng.standard_normal((5, 7)),
    "f4_3d": rng.standard_normal((2, 3, 4)).astype(np.float32),
    "i8_1d": rng.integers(-2**62, 2**62, 11, dtype=np.int64),
    "i4_2d": rng.integers(-2**31, 2**31 - 1, (4, 3), dtype=np.int32),
    "u1_1d": rng.integers(0, 256, 9, dtype=np.uint8),
    "b1_2d": rng.random((3, 5)) > 0.5,
    "c16_1d": rng.standard_normal(6) + 1j * rng.standard_normal(6),
    "f8_0d": np.array(3.25),
    "f8_empty": np.zeros((0, 3)),
    "f8_nan": np.array([np.nan, np.inf, -np.inf, -0.0, 5e-324]),
    "f8_fortran": np.asfortranarray(rng.standard_normal((4, 6))),
    "i4_fortran3": np.asfortranarray(rng.integers(-99, 99, (2, 3, 4), dtype=np.int32)),
}
manifest = {"numpy": np.__version__, "files": {}}
for name, a in cases.items():
    np.save(out / f"{name}.npy", a)
    manifest["files"][name] = {
        "dtype": {"float64": "float64", "float32": "float32", "int64": "int64", "int32": "int32",
                  "uint8": "uint8", "bool": "bool", "complex128": "complex128"}[str(a.dtype)],
        "shape": list(a.shape),
        "fortran": bool(a.flags.f_contiguous and not a.flags.c_contiguous),
        "c_bytes": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode(),
    }

# version 2.0 header (NumPy writes it only for huge headers; force it)
a = rng.standard_normal(4)
with open(out / "f8_v2.npy", "wb") as f:
    np.lib.format.write_array(f, a, version=(2, 0))
manifest["files"]["f8_v2"] = {"dtype": "float64", "shape": [4], "fortran": False,
                              "c_bytes": base64.b64encode(a.tobytes()).decode()}

# unsupported: big-endian, structured
np.save(out / "bad_bigendian.npy", np.arange(3, dtype=">f8"))
np.save(out / "bad_structured.npy", np.zeros(2, dtype=[("x", "<f8"), ("y", "<i4")]))
(out / "bad_truncated.npy").write_bytes((out / "f8_2d.npy").read_bytes()[:-8])
(out / "bad_magic.npy").write_bytes(b"NOTNUMPY" + (out / "f8_2d.npy").read_bytes()[8:])

(pathlib.Path(__file__).parent / "npy.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(f"wrote {len(manifest['files'])} files + 4 invalid ones")
