#!/usr/bin/env python3
"""Run harness.php per function under ASan/UBSan, resume after crashes/hangs, collect attributed reports.
Environment: TESSERO_LIB (the sanitizer build of libtessero, required), LOGDIR (default $TMPDIR/tessero-asan-logs).
usage: driver.py jobs.txt results.jsonl [parallel]
jobs.txt: lines "kind method"
"""
import json, os, select, subprocess, sys, time, threading, queue, re, signal

W = os.path.dirname(os.path.abspath(__file__))
HANG = float(os.environ.get('HANG', '20'))
JOBCAP = float(os.environ.get('JOBCAP', '1500'))
gcc = lambda n: subprocess.check_output(['gcc', f'-print-file-name={n}']).decode().strip()
ENV = dict(os.environ)
ENV.update({
    'TESSERO_LIB': os.environ['TESSERO_LIB'],
    'LD_PRELOAD': gcc('libasan.so') + ':' + gcc('libubsan.so'),
    'USE_ZEND_ALLOC': '0',
    'ASAN_OPTIONS': os.environ.get('ASAN_OPTS', 'halt_on_error=0:detect_leaks=0:malloc_fill_byte=190:max_malloc_fill_size=67108864:allocator_may_return_null=1:detect_odr_violation=0:symbolize=1'),
    'UBSAN_OPTIONS': 'print_stacktrace=1:halt_on_error=0',
    'OMP_NUM_THREADS': '1',
})
os.makedirs(os.environ.get('LOGDIR', os.path.join(os.environ.get('TMPDIR', '/tmp'), 'tessero-asan-logs')), exist_ok=True)
lock = threading.Lock()


def run_job(kind, method, out):
    start = 0
    t_job = time.time()
    part = 0
    ncases = None
    while True:
        part += 1
        logp = os.environ.get('LOGDIR', os.path.join(os.environ.get('TMPDIR', '/tmp'), 'tessero-asan-logs')) + f'/{kind}.{method}.{part}.log'
        cmd = ['php', '-d', 'ffi.enable=1', '-d', 'opcache.enable_cli=0', f'{W}/harness.php', kind, method, str(start)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV, cwd=W)
        cur, cur_desc, t_case = None, '', time.time()
        done = False
        reports = []
        buf = b''
        log = open(logp, 'wb')
        in_report = None
        status = None
        while True:
            r, _, _ = select.select([p.stdout], [], [], 1.0)
            if r:
                chunk = os.read(p.stdout.fileno(), 65536)
                if not chunk:
                    break
                log.write(chunk)
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    s = line.decode('utf-8', 'replace')
                    if s.startswith('@@NCASES'):
                        ncases = int(s.split()[1])
                    elif s.startswith('@@CASE'):
                        parts = s.split(' ', 2)
                        cur, cur_desc, t_case = int(parts[1]), parts[2] if len(parts) > 2 else '', time.time()
                        in_report = None
                    elif s.startswith('@@END'):
                        parts = s.split(' ', 3)
                        if float(parts[2]) > 5:
                            reports.append({'type': 'SLOW', 'case': cur, 'desc': cur_desc, 'secs': float(parts[2])})
                        in_report = None
                    elif s.startswith('@@POISON') or s.startswith('@@NONDET'):
                        reports.append({'type': s[2:8], 'case': cur, 'desc': cur_desc})
                    elif s.startswith('@@DONE'):
                        done = True
                    elif 'ERROR: AddressSanitizer' in s or 'runtime error:' in s or 'ERROR: LeakSanitizer' in s or 'AddressSanitizer:DEADLYSIGNAL' in s:
                        in_report = {'type': 'SAN', 'case': cur, 'desc': cur_desc, 'head': s.strip(), 'frames': []}
                        reports.append(in_report)
                    elif in_report is not None and re.match(r'\s+#\d+ ', s) and len(in_report['frames']) < 12:
                        in_report['frames'].append(s.strip())
                    elif in_report is not None and s.startswith('SUMMARY:'):
                        in_report['summary'] = s.strip()
            if cur is not None and not done and time.time() - t_case > HANG:
                p.kill()
                status = 'HANG'
                reports.append({'type': 'HANG', 'case': cur, 'desc': cur_desc})
                break
            if time.time() - t_job > JOBCAP:
                p.kill()
                status = 'JOBCAP'
                reports.append({'type': 'JOBCAP', 'case': cur, 'desc': cur_desc})
                break
            if p.poll() is not None and not r:
                break
        # drain
        try:
            rest = p.stdout.read()
            if rest:
                log.write(rest)
        except Exception:
            pass
        rc = p.wait()
        log.close()
        if not done and status is None:
            reports.append({'type': 'CRASH', 'case': cur, 'desc': cur_desc, 'rc': rc})
        with lock:
            for rep in reports:
                rep.update({'kind': kind, 'method': method, 'log': logp})
                out.write(json.dumps(rep) + '\n')
            out.write(json.dumps({'type': 'PART', 'kind': kind, 'method': method, 'start': start, 'last': cur, 'ncases': ncases, 'done': done, 'rc': rc, 'status': status}) + '\n')
            out.flush()
        if done or status == 'JOBCAP' or cur is None:
            return
        start = cur + 1
        if ncases is not None and start >= ncases:
            return


def main():
    jobs = [l.split() for l in open(sys.argv[1]) if l.strip()]
    out = open(sys.argv[2], 'a')
    par = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    q = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker():
        while True:
            try:
                k, m = q.get_nowait()
            except queue.Empty:
                return
            t = time.time()
            run_job(k, m, out)
            with lock:
                print(f'{k}.{m} {time.time() - t:.1f}s', flush=True)

    ts = [threading.Thread(target=worker) for _ in range(par)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


main()
