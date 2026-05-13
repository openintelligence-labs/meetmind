"""Speech-to-text routing.

Default everywhere: Parakeet TDT 0.6B v3. Live captions tier:
Moonshine v2 / Parakeet EOU. Polish tier: Canary-Qwen-2.5B / Canary-1B-v2.
CPU + Hinglish fallback: distil-large-v3.5 / Whisper-large-v3-turbo.

Module boundary: this package does not import from `diarize/`. It consumes
audio chunks from `capture/` and emits transcript partials/finals.
"""
