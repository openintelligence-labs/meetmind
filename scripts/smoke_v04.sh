#!/usr/bin/env bash
#
# v0.4 smoke test — runs the CLI end-to-end with the mock sidecars and
# greps for an expected transcript marker.

set -euo pipefail

cd "$(dirname "$0")/.."

VENV=".venv"
if [[ ! -x "${VENV}/bin/meetmind" ]]; then
    echo "smoke: venv missing or meetmind not installed; run 'pip install -e .[dev]' first" >&2
    exit 64
fi

OUT=$(mktemp -t meetmind-smoke.XXXXXX)
trap 'rm -f "$OUT"' EXIT

echo "smoke: meetmind --version"
"${VENV}/bin/meetmind" --version

echo "smoke: meetmind status"
"${VENV}/bin/meetmind" status >/dev/null

echo "smoke: meetmind record --mock --duration 1.0"
"${VENV}/bin/meetmind" record --mock --duration 1.0 --stream mic >"$OUT" 2>&1 || {
    echo "smoke: record exited non-zero" >&2
    cat "$OUT" >&2
    exit 1
}

if grep -E '\[partial|\[final|the quick|fox|dog' "$OUT" >/dev/null; then
    echo "smoke: PASS"
else
    echo "smoke: FAIL — no transcript markers in output" >&2
    cat "$OUT" >&2
    exit 1
fi
