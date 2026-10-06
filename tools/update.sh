#!/usr/bin/env bash
# One-command update for a Tessero library mirror (e.g. the tessero-ref server):
# force-sync to origin/main, rebuild + test (via tools/pull-build.sh), and
# optionally publish the beta issue list to GitHub.
#
#   bash tools/update.sh                 # sync + build kernel+ext + C tests + FFI parity
#   bash tools/update.sh --publish       # ... then create/update GitHub issues (idempotent)
#   bash tools/update.sh --quick         # pass-through: skip the ~4 min FFI parity run
#   bash tools/update.sh --no-ext        # pass-through: skip the extension build
#   bash tools/update.sh --publish --quick
#
# Force-sync DISCARDS local changes: this is meant for a read-only mirror of
# origin/main, not a dev checkout. --publish needs the GitHub CLI authenticated
# once ("gh auth login"); after that it is silent. Repo defaults to the origin
# slug, override with REPO=owner/name.
set -euo pipefail
cd "$(dirname "$0")/.."

PUBLISH=0
PASS=()
for a in "$@"; do
    case $a in
        --publish)        PUBLISH=1 ;;
        --quick|--no-ext) PASS+=("$a") ;;
        -h|--help)        sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $a (try --help)" >&2; exit 2 ;;
    esac
done

REPO=${REPO:-$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')}
step() { printf '\n== %s\n' "$*"; }

step "sync to origin/main (force; discards local changes)"
git fetch origin
git checkout -f main
before=$(git rev-parse --short HEAD)
git reset --hard origin/main
after=$(git rev-parse --short HEAD)
[ "$before" = "$after" ] && echo "already at $after" || echo "$before -> $after"

step "build and test"
bash tools/pull-build.sh ${PASS[@]+"${PASS[@]}"}

if [ "$PUBLISH" -eq 1 ]; then
    step "publish issues to GitHub ($REPO)"
    if ! gh auth status >/dev/null 2>&1; then
        echo "gh is not authenticated. Run 'gh auth login' once (GitHub.com -> HTTPS -> browser)," >&2
        echo "then re-run with --publish. Nothing was published." >&2
        exit 3
    fi
    php tools/beta-issues.php create --repo="$REPO"
    echo "(run 'php artisan issues:sync' in the web app to surface them on the site)"
fi

step "done"
echo "HEAD $(git rev-parse --short HEAD) ready$([ "$PUBLISH" -eq 1 ] && echo ' (issues published)')"
