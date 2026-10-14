#!/usr/bin/env python3
"""scipy.io file-format fixtures (Matrix Market, WAV, FortranFile) for Tessero's pure-PHP readers/writers.

The fixture carries scipy-written file CONTENT (base64); run-tests.php's scipyio_cases branch decodes it,
parses it with Tessero\\Io\\{MatrixMarket,Wav,FortranFile} and compares the flat data + metadata. For writers
the fixture carries the input array; the facade writes bytes, re-reads them with the Tessero reader and the
flat values must match. Pure byte/text parsing (no BLAS), host-independent; no pinned environment needed.
"""
import base64, io, os, tempfile, json
import numpy as np
from scipy.io import mmread, mmwrite, mminfo, FortranFile
from scipy.io import wavfile
from scipy.sparse import coo_matrix

OUT = os.environ.get('SCIPYIO_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'scipyio.json')

cases = []


def b64(b):
    return base64.b64encode(b).decode('ascii')


def flat_real(a):
    return [float(x) for x in np.asarray(a).ravel(order='C')]


def flat_int(a):
    return [int(x) for x in np.asarray(a).ravel(order='C')]


def flat_complex(a):
    out = []
    for x in np.asarray(a).ravel(order='C'):
        out += [float(x.real), float(x.imag)]
    return out


# ---------------- Matrix Market reads ----------------
def mm_write_bytes(obj, **kw):
    b = io.BytesIO(); mmwrite(b, obj, **kw); return b.getvalue()


def mm_read_case(name, obj, field):
    data = mm_write_bytes(obj)
    dense = np.asarray(mmread(io.BytesIO(data)).todense() if hasattr(mmread(io.BytesIO(data)), 'todense') else mmread(io.BytesIO(data)))
    r = mmread(io.BytesIO(data))
    dense = r.toarray() if hasattr(r, 'toarray') else np.asarray(r)
    nfo = mminfo(io.BytesIO(data))
    if field == 'complex':
        flat = flat_complex(dense)
    elif field == 'integer':
        flat = flat_int(dense)
    else:
        flat = flat_real(dense)
    cases.append({'kind': 'mmread', 'name': name, 'content_b64': b64(data),
                  'field': field,
                  'rows': int(dense.shape[0]), 'cols': int(dense.shape[1]),
                  'flat': flat,
                  'info': [int(nfo[0]), int(nfo[1]), int(nfo[2]), nfo[3], nfo[4], nfo[5]]})


mm_read_case('array_real', np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 'real')
mm_read_case('array_real_1x1', np.array([[2.5]]), 'real')
mm_read_case('array_integer', np.array([[1, 2], [3, 4]], dtype=np.int64), 'integer')
mm_read_case('array_complex', np.array([[1 + 2j, 3 - 1j], [0 + 0j, 4 + 0.5j]]), 'complex')
mm_read_case('coord_real', coo_matrix(np.array([[0, 0, 3.0], [4.0, 0, 0], [0, 5.0, 0]])), 'real')
mm_read_case('coord_integer', coo_matrix(np.array([[0, 2], [3, 0]], dtype=np.int64)), 'integer')
mm_read_case('coord_complex', coo_matrix(np.array([[0, 1 + 1j], [2 - 3j, 0]])), 'complex')


# a hand-written file with comments / blank handling that scipy also reads the same way
def mm_read_text_case(name, text, field, rows, cols):
    r = mmread(io.StringIO(text))
    dense = r.toarray() if hasattr(r, 'toarray') else np.asarray(r)
    nfo = mminfo(io.StringIO(text))
    flat = {'complex': flat_complex, 'integer': flat_int}.get(field, flat_real)(dense)
    cases.append({'kind': 'mmread', 'name': name, 'content_b64': b64(text.encode()),
                  'field': field, 'rows': rows, 'cols': cols, 'flat': flat,
                  'info': [int(nfo[0]), int(nfo[1]), int(nfo[2]), nfo[3], nfo[4], nfo[5]]})


mm_read_text_case('array_comments',
                  "%%MatrixMarket matrix array real general\n% a comment\n%another\n2 2\n1\n2\n3\n4\n",
                  'real', 2, 2)


# ---------------- Matrix Market writes (facade writes -> Tessero re-reads -> compare) ----------------
def mm_write_case(name, rows_data, fmt, field):
    cases.append({'kind': 'mmwrite', 'name': name, 'data': rows_data,
                  'format': fmt, 'field': field,
                  'rows': len(rows_data), 'cols': len(rows_data[0])})


mm_write_case('w_array_real', [[1.5, -2.25, 3.0], [4.0, 5.0, -6.5]], 'array', 'real')
mm_write_case('w_array_integer', [[1, 2], [3, 4]], 'array', 'integer')
mm_write_case('w_array_complex', [[[1.0, 2.0], [3.0, -1.0]]], 'array', 'complex')
mm_write_case('w_coord_real', [[0.0, 3.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 5.0]], 'coordinate', 'real')
mm_write_case('w_coord_integer', [[0, 2], [3, 0]], 'coordinate', 'integer')


# ---------------- WAV reads ----------------
def wav_read_case(name, rate, arr, dtype, channels):
    b = io.BytesIO(); wavfile.write(b, rate, arr); data = b.getvalue()
    if dtype.startswith('float'):
        flat = flat_real(arr)
    else:
        flat = flat_int(arr)
    cases.append({'kind': 'wavread', 'name': name, 'content_b64': b64(data),
                  'rate': rate, 'dtype': dtype, 'channels': channels, 'flat': flat})


wav_read_case('int16_mono', 8000, np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16), 'int16', 1)
wav_read_case('int16_stereo', 44100, np.array([[0, 1], [100, -100], [200, -200]], dtype=np.int16), 'int16', 2)
wav_read_case('float32', 16000, np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32), 'float32', 1)
wav_read_case('int32', 22050, np.array([0, 1000000, -1000000], dtype=np.int32), 'int32', 1)
wav_read_case('uint8', 8000, np.array([0, 128, 255], dtype=np.uint8), 'uint8', 1)


# ---------------- WAV writes (facade writes -> Tessero re-reads) ----------------
def wav_write_case(name, rate, flat, dtype, channels):
    cases.append({'kind': 'wavwrite', 'name': name, 'rate': rate,
                  'flat': flat, 'dtype': dtype, 'channels': channels})


wav_write_case('w_int16', 8000, [0, 100, -100, 32767, -32768], 'int16', 1)
wav_write_case('w_int32', 22050, [0, 1000000, -1000000], 'int32', 1)
wav_write_case('w_uint8', 8000, [0, 128, 255], 'uint8', 1)
wav_write_case('w_float32', 16000, [0.0, 0.5, -0.5, 1.0], 'float32', 1)
wav_write_case('w_int16_stereo', 44100, [0, 1, 100, -100, 200, -200], 'int16', 2)


# ---------------- FortranFile reads ----------------
def ff_write_bytes(records, header_dtype='<u4'):
    p = tempfile.mktemp()
    f = FortranFile(p, 'w', header_dtype=np.dtype(header_dtype))
    for r in records:
        f.write_record(r)
    f.close()
    with open(p, 'rb') as fh:
        b = fh.read()
    os.remove(p)
    return b


def ff_read_case(name, records, dtype, header='uint32'):
    hd = '<u8' if header == 'uint64' else '<u4'
    data = ff_write_bytes(records, hd)
    if dtype.startswith('float'):
        flat = [flat_real(r) for r in records]
    else:
        flat = [flat_int(r) for r in records]
    cases.append({'kind': 'fortranread', 'name': name, 'content_b64': b64(data),
                  'dtype': dtype, 'header_dtype': header, 'records': flat})


ff_read_case('int32_single', [np.array([1, 2, 3, 4, 5], dtype=np.int32)], 'int32')
ff_read_case('float64_single', [np.array([1.5, -2.5, 3.25], dtype=np.float64)], 'float64')
ff_read_case('int32_multi', [np.array([1, 2], dtype=np.int32), np.array([3, 4, 5], dtype=np.int32)], 'int32')
ff_read_case('int64_single', [np.array([10, 20, 30], dtype=np.int64)], 'int64')
ff_read_case('float32_single', [np.array([1.5, 2.5, 3.5], dtype=np.float32)], 'float32')


# ---------------- FortranFile writes (facade writes -> Tessero re-reads) ----------------
def ff_write_case(name, records, dtype):
    cases.append({'kind': 'fortranwrite', 'name': name, 'records': records, 'dtype': dtype})


ff_write_case('w_int32', [[1, 2, 3, 4, 5]], 'int32')
ff_write_case('w_float64', [[1.5, -2.5, 3.25]], 'float64')
ff_write_case('w_int32_multi', [[1, 2], [3, 4, 5]], 'int32')
ff_write_case('w_int64', [[10, 20, 30]], 'int64')


out = {'module': 'scipyio', 'numpy': np.__version__, 'scipyio_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'scipyio: {len(cases)} cases (mmread/mmwrite/mminfo + wavfile + FortranFile)')
