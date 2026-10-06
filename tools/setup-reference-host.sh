#!/usr/bin/env bash
# Provision the Tessero reference host on a Debian/Ubuntu glibc x86-64 box (ADR 0012).
# Tested target: WSL2 Ubuntu 24.04. Run ONCE with sudo:
#   sudo bash tools/setup-reference-host.sh
# Idempotent-ish; safe to re-run. Installs the toolchain, PHP 8.4 + FFI,
# OpenBLAS/LAPACKE, and a pinned Python 3.11 venv for the parity fixtures.
set -euo pipefail

# Repo root = parent of the directory holding this script, resolved absolutely.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
VENV="$REPO/.venv-parity"

echo "== [1/6] apt update + base toolchain =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  build-essential make pkg-config git ca-certificates software-properties-common \
  libgomp1 libopenblas0-pthread liblapacke liblapacke-dev

echo "== [2/6] PHP 8.4 (ondrej ppa) with FFI, dev headers, and common exts =="
add-apt-repository -y ppa:ondrej/php
apt-get update -y
apt-get install -y --no-install-recommends \
  php8.4-cli php8.4-dev php8.4-ffi php8.4-mbstring php8.4-xml php8.4-bcmath
# Ensure FFI is enabled for CLI.
PHPINI=$(php -i 2>/dev/null | awk -F'=> ' '/Loaded Configuration File/{print $2}' | tr -d ' ')
if [ -n "${PHPINI:-}" ] && ! php -r 'exit(extension_loaded("FFI")?0:1);'; then
  echo "ffi.enable=1" >> "$PHPINI"
fi
php -v | head -1
php -r 'echo "FFI: ", extension_loaded("FFI")?"enabled":"DISABLED", "\n";'

echo "== [3/6] Python 3.11 (deadsnakes) for the pinned fixture env =="
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev

echo "== [4/6] pinned Python venv (NumPy 2.4.4 / SciPy 1.17.1) =="
# Create the venv as the invoking (non-root) user so the repo stays user-owned.
RUN_AS="${SUDO_USER:-$(id -un)}"
sudo -u "$RUN_AS" python3.11 -m venv "$VENV"
sudo -u "$RUN_AS" "$VENV/bin/python" -m pip install --upgrade pip
sudo -u "$RUN_AS" "$VENV/bin/pip" install -r "$REPO/tools/parity/requirements.txt"
sudo -u "$RUN_AS" "$VENV/bin/pip" install mkdocs-material
"$VENV/bin/python" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'

echo "== [5/6] quick FFI smoke test against the prebuilt .so =="
TSR_LIB="$REPO/lib/linux-x86_64/libtessero.so" \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
php -d ffi.enable=1 -r '
  $lib = getenv("TSR_LIB");
  $h = FFI::cdef("", $lib);
  echo "loaded libtessero.so OK\n";
' || echo "NOTE: FFI load needs the C ABI header; the real check is run-tests.php below."

echo "== [6/6] done. Next steps (run as your normal user, NOT root): =="
cat <<EOF

  cd "$REPO"
  source .venv-parity/bin/activate         # pinned NumPy/SciPy for fixtures
  make -C csrc -j                          # rebuild libtessero from source (optional; a .so is shipped)
  php -d ffi.enable=1 tools/parity/gen-facades.php --check
  QUICK=1 bash tools/parity/gate.sh        # must be green before new work
  python3 tools/parity/report.py           # the measured baseline

EOF
echo "Reference host provisioning complete."
