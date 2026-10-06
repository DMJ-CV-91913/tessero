#!/usr/bin/env python3
"""numpy.loadtxt fixtures: the fixture carries the file CONTENT; run-tests.php writes it to a temp file and calls
loadtxt. Pure text parsing (no BLAS), host-independent; no pinned environment required."""
import base64, io, json, os, tempfile
import numpy as np
import fixtures_np as F  # enc / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'io.json')


def case(content, kwargs=None):
    kwargs = kwargs or {}
    r = np.loadtxt(io.StringIO(content), **kwargs)
    return {'fn': 'loadtxt', 'content': content,
            'kwargs': {k: F.enc(v) for k, v in kwargs.items()}, 'expect': F.enc_result(r, [])}


cases = [
    case("1 2 3\n4 5 6\n7 8 9\n"),                 # grid -> 2-D
    case("1 2 3\n"),                               # single row -> 1-D
    case("1\n2\n3\n"),                             # single column -> 1-D
    case("5\n"),                                   # single value -> scalar
    case("1,2,3\n4,5,6\n", {'delimiter': ','}),
    case("# header line\n1 2\n3 4\n"),             # comment-only line skipped
    case("1 2 # inline comment\n3 4\n"),           # inline comment stripped
    case("skip this\n1 2\n3 4\n", {'skiprows': 1}),
    case("1.5 -2.5 3.0\n-4.25 5.0 -6.5\n"),
    case("\n1 2\n\n3 4\n"),                        # blank lines skipped
    case("10,20\n30,40\n50,60\n", {'delimiter': ',', 'skiprows': 1}),
]

def ff_bin(content_bytes, read_dtype, kwargs=None):
    kwargs = kwargs or {}
    f = tempfile.mktemp()
    with open(f, 'wb') as fh:
        fh.write(content_bytes)
    expect = np.fromfile(f, dtype=read_dtype, **kwargs)
    os.remove(f)
    enc_kw = {'dtype': F.enc(np.dtype(read_dtype).name)}
    for k, v in kwargs.items():
        enc_kw[k] = F.enc(v)
    return {'fn': 'fromfile', 'content_b64': base64.b64encode(content_bytes).decode('ascii'),
            'kwargs': enc_kw, 'expect': F.enc_result(expect, [])}


def ff_text(text, read_dtype, sep, kwargs=None):
    kwargs = kwargs or {}
    f = tempfile.mktemp()
    with open(f, 'w') as fh:
        fh.write(text)
    expect = np.fromfile(f, dtype=read_dtype, sep=sep, **kwargs)
    os.remove(f)
    enc_kw = {'dtype': F.enc(np.dtype(read_dtype).name), 'sep': F.enc(sep)}
    for k, v in kwargs.items():
        enc_kw[k] = F.enc(v)
    return {'fn': 'fromfile', 'content': text, 'kwargs': enc_kw, 'expect': F.enc_result(expect, [])}


f64 = np.array([1.5, -2.5, 3.0, 4.25, 5.0], dtype=np.float64).tobytes()
cases += [
    ff_bin(f64, 'float64'),
    ff_bin(f64, 'float64', {'count': 3}),
    ff_bin(f64, 'float64', {'offset': 16}),
    ff_bin(np.array([10, 20, 30, 40], dtype=np.int32).tobytes(), 'int32'),
    ff_bin(np.array([1.5, 2.5, 3.5], dtype=np.float32).tobytes(), 'float32'),
    ff_bin(np.arange(6, dtype=np.int64).tobytes(), 'int64', {'count': 4}),
    ff_bin(np.array([0, 1, 255, 128], dtype=np.uint8).tobytes(), 'uint8'),
    ff_text("1.5 2.5 3.5 4.5", 'float64', ' '),
    ff_text("1,2,3,4,5", 'float64', ','),
    ff_text("1,2,3,4,5", 'float64', ',', {'count': 3}),
    ff_text("10 20 30 40", 'int32', ' '),
]

out = {'module': 'io', 'numpy': np.__version__, 'io_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'io: {len(cases)} cases (loadtxt + fromfile)')
