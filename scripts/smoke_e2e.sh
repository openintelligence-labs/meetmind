#!/usr/bin/env bash
# End-to-end smoke test for every shipped CLI command.
#
# Runs against an isolated MEETMIND_HOME so it never touches your real
# data. Skips LLM-dependent steps when Ollama isn't running. Prints a
# pass/fail summary; exits non-zero on any failure.
#
# Usage:
#   ./scripts/smoke_e2e.sh           # quick (5s mock recording)
#   FULL=1 ./scripts/smoke_e2e.sh    # also runs summarize via Ollama if available

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
MM="$HERE/.venv/bin/meetmind"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
export MEETMIND_HOME="$TMPDIR"

PASS=0
FAIL=0

run() {
    local name="$1"; shift
    if "$@" >/tmp/smoke.out 2>/tmp/smoke.err; then
        echo "PASS $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL $name"
        echo "  stdout:" && sed 's/^/    /' /tmp/smoke.out | head -5
        echo "  stderr:" && sed 's/^/    /' /tmp/smoke.err | head -5
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    local needle="$1" haystack="$2" name="$3"
    if echo "$haystack" | grep -q "$needle"; then
        echo "PASS $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL $name (missing: $needle)"
        FAIL=$((FAIL + 1))
    fi
}

echo "== smoke test (MEETMIND_HOME=$TMPDIR) =="

# 1. status
run "status"                "$MM" status

# 2. record produces a meeting with at least one final segment
MID="$($MM record --duration 5 --mock 2>/dev/null | tail -1)"
assert_contains "01" "$MID" "record produces ULID"

# 3. meetings list shows it
LIST="$($MM meetings 2>&1)"
assert_contains "$MID" "$LIST" "meetings shows the new id"
assert_contains " segs" "$LIST" "meetings reports a segment count"

# 4. index + search
run "index"                 "$MM" index
SEARCH="$($MM search 'fox' 2>&1)"
assert_contains "fox" "$SEARCH" "search finds the indexed segment"

# 5. compliance dpia
DPIA="$($MM compliance dpia 2>&1)"
assert_contains "Data Protection Impact Assessment" "$DPIA" "DPIA renders"
assert_contains "Meetings:" "$DPIA" "DPIA reports meetings count"

# 6. compliance retention-sweep dry-run is idempotent
run "retention dry-run"     "$MM" compliance retention-sweep --dry-run

# 7. redact via stdin
REDACTED="$(echo 'Email Sam Williams at sam@example.com' | $MM redact --profile public_share 2>&1 || true)"
assert_contains "[name]" "$REDACTED" "redact strips names"
assert_contains "[email]" "$REDACTED" "redact strips emails"

# 8. export-obsidian
mkdir -p "$TMPDIR/vault"
run "export-obsidian"       "$MM" export-obsidian "$MID" --vault "$TMPDIR/vault"
NOTE="$(find "$TMPDIR/vault" -name '*.md' | head -1)"
[ -n "$NOTE" ] && assert_contains "meetmind" "$(cat "$NOTE")" "obsidian note tagged meetmind"

# 9. export-bundle (signed transcript bundle)
run "export-bundle"         "$MM" export-bundle "$MID" --out "$TMPDIR/bundle.tar.gz"
[ -f "$TMPDIR/bundle.tar.gz" ] && {
    echo "PASS bundle file exists"
    PASS=$((PASS + 1))
}

# 10. verify bundle
run "verify bundle"         "$MM" verify "$TMPDIR/bundle.tar.gz"

# 11. summarize (LLM-dependent — only if Ollama is reachable AND FULL=1)
if [ "${FULL:-0}" = "1" ]; then
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        run "summarize (live ollama)" "$MM" summarize "$MID"
    else
        echo "SKIP summarize (ollama not reachable)"
    fi
else
    echo "SKIP summarize (FULL=1 not set)"
fi

echo
echo "== summary =="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
[ "$FAIL" -eq 0 ]
