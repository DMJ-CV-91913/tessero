#!/usr/bin/env python3
"""numpy datetime64 / timedelta64 / business-day fixtures. Values are compared as NumPy's int64 counts (the raw
datetime64/timedelta64 storage), bools, or ISO strings. Pure calendar logic, host-independent."""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'datetime.json')
NAT = -9223372036854775808
cases = []


def c(fn, args, kwargs, expect):
    cases.append({'fn': fn, 'args': args, 'kwargs': kwargs, 'expect': expect})


def i64(x):
    return int(np.asarray(x).astype('int64'))


# datetime64: parse -> int64 count at a unit
c('datetime64', ['2021-01-15'], {}, i64(np.datetime64('2021-01-15')))
c('datetime64', ['1969-12-20'], {}, i64(np.datetime64('1969-12-20')))                 # before epoch (negative)
c('datetime64', ['2021-01-15', 's'], {}, i64(np.datetime64('2021-01-15', 's')))
c('datetime64', ['2021-01-15T12:30:45', 's'], {}, i64(np.datetime64('2021-01-15T12:30:45', 's')))
c('datetime64', ['2021-01-15T12:30:45.250', 'ms'], {}, i64(np.datetime64('2021-01-15T12:30:45.250', 'ms')))
c('datetime64', ['2021-07-01', 'M'], {}, i64(np.datetime64('2021-07-01', 'M')))
c('datetime64', ['2021-07-01', 'Y'], {}, i64(np.datetime64('2021-07-01', 'Y')))
c('datetime64', [['2021-01-01', '2021-06-15', 'NaT'], 'D'], {},
  [int(x) for x in np.array(['2021-01-01', '2021-06-15', 'NaT'], dtype='datetime64[D]').astype('int64')])

# timedelta64
c('timedelta64', [5, 'D'], {}, int(np.timedelta64(5, 'D').astype('int64')))
c('timedelta64', [90, 'm'], {}, int(np.timedelta64(90, 'm').astype('int64')))

# datetime_data
c('datetime_data', ['s'], {}, list(np.datetime_data(np.dtype('datetime64[s]'))))
c('datetime_data', ['10m'], {}, list(np.datetime_data(np.dtype('datetime64[10m]'))))
c('datetime_data', ['D'], {}, list(np.datetime_data(np.dtype('datetime64[D]'))))

# isnat (on int64 counts)
c('isnat', [[i64(np.datetime64('2021-01-01')), NAT, i64(np.datetime64('2020-02-29'))]], {},
  np.isnat(np.array(['2021-01-01', 'NaT', '2020-02-29'], dtype='datetime64[D]')).tolist())

# is_busday
c('is_busday', [['2021-01-02', '2021-01-04', '2021-01-08']], {},
  np.is_busday(['2021-01-02', '2021-01-04', '2021-01-08']).tolist())
c('is_busday', ['2021-12-25'], {}, bool(np.is_busday('2021-12-25')))

# busday_count
c('busday_count', ['2021-01-01', '2021-02-01'], {}, int(np.busday_count('2021-01-01', '2021-02-01')))
c('busday_count', ['2021-02-01', '2021-01-01'], {}, int(np.busday_count('2021-02-01', '2021-01-01')))
c('busday_count', ['2021-01-01', '2021-02-01'], {'holidays': ['2021-01-18']},
  int(np.busday_count('2021-01-01', '2021-02-01', holidays=['2021-01-18'])))

# busday_offset
c('busday_offset', ['2021-01-01', 5], {'roll': 'forward'}, i64(np.busday_offset('2021-01-01', 5, roll='forward')))
c('busday_offset', ['2021-01-02', 0], {'roll': 'forward'}, i64(np.busday_offset('2021-01-02', 0, roll='forward')))
c('busday_offset', ['2021-03-15', -3], {'roll': 'backward'}, i64(np.busday_offset('2021-03-15', -3, roll='backward')))

# datetime_as_string
c('datetime_as_string', [i64(np.datetime64('2021-01-15'))], {}, str(np.datetime_as_string(np.datetime64('2021-01-15'))))
c('datetime_as_string', [i64(np.datetime64('2021-01-15T12:30:45', 's')), 's'], {},
  str(np.datetime_as_string(np.datetime64('2021-01-15T12:30:45', 's'))))

out = {'module': 'datetime', 'numpy': np.__version__, 'datetime_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'datetime: {len(cases)} cases')
