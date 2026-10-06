#!/usr/bin/env python3
"""Reap stuck Tessero parity-tooling processes on the shared build box — safely.

Scope — this only ever touches Tessero's own parity tooling, nothing else on the box:
  1. the command must be fixtures.py / run-tests.php / gen-fixtures.sh (the only processes these runs spawn), AND
  2. the process must be running inside the Tessero repo (its /proc/<pid>/cwd is under --root, default this repo).
Both conditions must hold, so production and other apps are never candidates: php-fpm, horizon, reverb, artisan,
supervisord, the betting-data scrapers (curl-impersonate), any other python/php — none match, none are touched.
If a process's cwd cannot be read (gone, or a different user), it is skipped, never killed.

A fixture generation or test run finishes in minutes; when an ssh session drops, a run is superseded, or a
generator wedges, the process can linger for hours and starve the box. This kills matches that are BOTH older
than --max-age minutes AND not making progress — a genuinely busy generator (the scipy.special prolate
spheroidal functions peg a core for 10+ min) keeps running (CPU time from /proc is sampled over --window).

    python3 tools/reap.py                 # reap matches older than 30 min that are idle (in this repo)
    python3 tools/reap.py --max-age 60    # only long-lived zombies (good for an unattended cron)
    python3 tools/reap.py --list          # just show every Tessero tooling process (age, busy/idle); kill nothing
    python3 tools/reap.py --dry-run       # show what WOULD be reaped at the current thresholds
    python3 tools/reap.py --stop          # deploy "stop": reap EVERY Tessero tooling process now (busy or not)
    python3 tools/reap.py --root /path    # scope to a different repo checkout

Cron (reap genuine zombies, never a legitimate run):
    */15 * * * * python3 $HOME/tessero/tools/reap.py --max-age 60 >> $HOME/reap.log 2>&1

There is no "start": the tooling is spawned on demand by gate/fixture runs, not a daemon. --stop is the clean
"halt all Tessero background work" primitive a deploy can call; it still cannot touch any non-Tessero process.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time

# Only our parity tooling — deliberately narrow so production is never a candidate.
TARGET = re.compile(r'(fixtures\.py|run-tests\.php|gen-fixtures\.sh)')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def processes():
    """Yield (pid, elapsed_seconds, command) for every process, via ps."""
    out = subprocess.run(['ps', '-eo', 'pid=,etimes=,args='], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, etimes, cmd = parts
        if pid.isdigit() and etimes.isdigit():
            yield int(pid), int(etimes), cmd


def proc_cwd(pid):
    """The process's working directory, or None if it can't be read (gone / different user)."""
    try:
        return os.path.realpath(os.readlink(f'/proc/{pid}/cwd'))
    except OSError:
        return None


def under(path, root):
    if path is None:
        return False
    root = os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def cpu_ticks(pid):
    """CPU time (utime+stime) in clock ticks from /proc, or None if the process is gone."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            data = f.read()
    except OSError:
        return None
    # comm (field 2) is parenthesised and may contain spaces/parens; split after the last ')'.
    after = data.rsplit(')', 1)[1].split()
    try:
        return int(after[11]) + int(after[12])   # utime, stime (0-based from 'state')
    except (IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description='Reap stuck (old and idle) Tessero parity-tooling processes.')
    ap.add_argument('--max-age', type=int, default=30, help='minutes; only consider matches at least this old (default 30)')
    ap.add_argument('--all', action='store_true', help='consider every match regardless of age')
    ap.add_argument('--force', action='store_true', help='skip the CPU-progress check; reap every stale match')
    ap.add_argument('--stop', action='store_true', help='deploy stop: reap every Tessero tooling process now (implies --all --force)')
    ap.add_argument('--list', action='store_true', help='list every Tessero tooling process (age, busy/idle); kill nothing')
    ap.add_argument('--window', type=float, default=1.5, help='seconds to sample CPU progress over (default 1.5)')
    ap.add_argument('--min-cpu', type=float, default=0.10, help='fraction of one core over the window to count as "busy" (default 0.10)')
    ap.add_argument('--root', default=REPO_ROOT, help='only touch processes whose cwd is under this repo (default: this checkout)')
    ap.add_argument('--dry-run', action='store_true', help='list only; kill nothing')
    args = ap.parse_args()
    if args.stop:
        args.all = args.force = True
    consider_all = args.all or args.list

    me = os.getpid()
    candidates = []
    skipped_elsewhere = 0
    for pid, etimes, cmd in processes():
        if pid == me or 'reap.py' in cmd or not TARGET.search(cmd):
            continue
        if not under(proc_cwd(pid), args.root):     # outside this repo -> never ours, never touched
            skipped_elsewhere += 1
            continue
        if consider_all or etimes // 60 >= args.max_age:
            candidates.append((pid, etimes // 60, cmd))

    if not candidates:
        print(f'no Tessero tooling to {"list" if args.list else "reap"}'
              + (f' ({skipped_elsewhere} match(es) outside the repo left alone)' if skipped_elsewhere else ''))
        return 0

    # Sample CPU progress for all candidates at once, unless we are force-killing.
    busy = {}
    if not args.force or args.list:
        clk = os.sysconf('SC_CLK_TCK')
        t0 = {pid: cpu_ticks(pid) for pid, _, _ in candidates}
        time.sleep(args.window)
        t1 = {pid: cpu_ticks(pid) for pid, _, _ in candidates}
        need = args.min_cpu * args.window * clk
        busy = {pid: (t0[pid] is not None and t1[pid] is not None and (t1[pid] - t0[pid]) >= need)
                for pid, _, _ in candidates}

    reaped = spared = denied = 0
    for pid, age, cmd in candidates:
        state = 'busy' if busy.get(pid) else 'idle'
        if args.list:
            print(f'  pid {pid} ({state}, {age}m): {cmd[:100]}')
            continue
        if busy.get(pid) and not args.force:
            spared += 1
            print(f'spare pid {pid} (busy, {age}m): {cmd[:100]}')
            continue
        print(f'{"would reap" if args.dry_run else "reap"} pid {pid} ({state}, {age}m): {cmd[:100]}')
        if args.dry_run:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            reaped += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            denied += 1
            print(f'  no permission to kill {pid} (different user)', file=sys.stderr)

    if args.list:
        print(f'{len(candidates)} Tessero tooling process(es)'
              + (f'; {skipped_elsewhere} other match(es) outside the repo ignored' if skipped_elsewhere else ''))
        return 0
    extra = f', spared {spared} busy' if spared else ''
    extra += f', {denied} denied' if denied else ''
    print(f'{"(dry run) " if args.dry_run else ""}reaped {reaped}{extra}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
