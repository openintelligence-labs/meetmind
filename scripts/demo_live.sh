#!/usr/bin/env bash
#
# MeetMind end-to-end live demo.
#
# Drives the full pipeline on this Mac, against real local models:
#   1. Synthesize a realistic meeting clip via `say` (background)
#   2. Capture system audio via Core Audio Tap, transcribe with
#      Parakeet TDT 0.6B v3 (FluidAudio), persist to SQLCipher store
#   3. Embed every segment with `nomic-embed-text v2` (Ollama via
#      actants), index in LanceDB
#   4. Run analyze: action-items + decisions + Chain-of-Density
#      summary via local Ollama (model defaults to llama2:latest;
#      override with MEETMIND_LLM_MODEL=...)
#   5. Demo a hybrid search query
#
# Prerequisites:
#   - macOS 14.4+ (Core Audio Tap)
#   - Microphone + Screen Recording permissions granted
#   - Ollama running on localhost:11434 with a chat model (llama2,
#     qwen3, etc.) and `nomic-embed-text:latest` pulled
#   - Native sidecars built: `cd sidecars/macos && swift build -c release`
#   - venv set up: `python -m venv .venv && .venv/bin/pip install -e '.[dev,storage,audio]'`

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/meetmind" ]]; then
    echo "demo: .venv missing. Run: python -m venv .venv && .venv/bin/pip install -e '.[dev,storage,audio]'" >&2
    exit 64
fi

if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "demo: Ollama not reachable at localhost:11434." >&2
    exit 64
fi

LLM_MODEL="${MEETMIND_LLM_MODEL:-llama2:latest}"
DURATION="${DURATION:-22}"
TITLE="${TITLE:-product-weekly-demo-$(date +%H%M%S)}"

SCRIPT="${SCRIPT:-Hi everyone, thanks for joining the product weekly. Sam, can you walk us through the migration plan for next quarter? We need to decide if we are going with LanceDB or staying on Postgres pgvector. Priya pushed back on Postgres because of the vector index latency we saw last sprint. Bob, can you write up the proposal? Send it by Friday so we can review it Monday morning.}"

echo "MeetMind live demo"
echo "  duration:  ${DURATION}s"
echo "  llm model: ${LLM_MODEL}"
echo "  title:     ${TITLE}"
echo

echo "1/5 synthesizing meeting audio via say(1)..."
/usr/bin/say -v Daniel "${SCRIPT}" &
SAY_PID=$!
trap 'kill $SAY_PID 2>/dev/null || true' EXIT
sleep 0.5

echo "2/5 capturing + transcribing via Core Audio Tap -> Parakeet TDT 0.6B v3 -> SQLCipher..."
MID=$(.venv/bin/meetmind record \
    --duration "${DURATION}" \
    --stream loopback \
    --title "${TITLE}" \
    | tail -n 1)
echo "    meeting_id = ${MID}"

wait $SAY_PID 2>/dev/null || true

echo
echo "3/5 embedding + indexing via nomic-embed-text v2 -> LanceDB..."
.venv/bin/meetmind index --meeting-id "${MID}"

echo
echo "4/5 analyzing via actants -> Ollama (${LLM_MODEL})..."
MEETMIND_LLM_MODEL="${LLM_MODEL}" .venv/bin/meetmind summarize "${MID}"

echo
echo "5/5 hybrid search demo: 'database choice for next quarter'"
.venv/bin/meetmind search --meeting-id "${MID}" --limit 3 \
    "database choice for next quarter"

echo
echo "Done."
echo "  Try it interactively:"
echo "    .venv/bin/meetmind meetings"
echo "    .venv/bin/meetmind search 'your query here'"
echo "    .venv/bin/meetmind summarize ${MID}"
