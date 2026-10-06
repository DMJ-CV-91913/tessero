#!/usr/bin/env bash
# Boots `php -S` (a non-CLI SAPI) with ffi.enable=preload + opcache.preload and
# checks that libtessero and LAPACK were bound by the preload script.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PORT:-8765}"
USER_OPT=""
if [ "$(id -u)" = "0" ]; then USER_OPT="-d opcache.preload_user=root"; fi
php -d ffi.enable=preload -d opcache.enable=1 -d opcache.enable_cli=1 \
    -d opcache.preload="$ROOT/resources/preload.php" $USER_OPT \
    -S 127.0.0.1:"$PORT" "$ROOT/tests/e2e/router.php" >/tmp/tessero-preload.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
    curl -sf "http://127.0.0.1:$PORT/" -o /tmp/tessero-preload.json && break
    sleep 0.1
done
php -r '
$r = json_decode(file_get_contents("/tmp/tessero-preload.json"), true);
$ok = $r["sapi"] === "cli-server" && $r["ffi_enable"] === "preload" && $r["cdef_allowed_at_request_time"] === false
    && $r["info"]["load_mode"] === "scope" && $r["residual"] < 1e-9 && $r["sum"] == 499999500000.0;
echo json_encode(["sapi" => $r["sapi"], "load_mode" => $r["info"]["load_mode"], "blas" => $r["info"]["blas"], "cdef_allowed" => $r["cdef_allowed_at_request_time"], "residual" => $r["residual"]]), "\n";
if (! $ok) { fwrite(STDERR, "preload e2e FAILED\n"); exit(1); }
echo "preload e2e OK\n";
'
