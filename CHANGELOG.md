# Changelog

All notable changes to MeetMind. Format follows [Keep a Changelog](https://keepachangelog.com/);
versioning is [SemVer](https://semver.org/).

## [Unreleased]

Nothing yet.

## [1.0.2] — 2026-07-30

### Security
- Windows: `~/.meetmind/token` is now restricted to the current user via an
  owner-only `icacls` DACL (`/reset`, then `/inheritance:r /grant:r <user>:F`) —
  `os.chmod(0o600)` is a no-op for access control on Windows. If the ACL cannot
  be applied, the token file is deleted and a clear error is raised instead of
  leaving a world-readable bearer token on disk (#3).
- Token-file ACL tests are re-enabled on win32 with `icacls`-based assertions
  (owner-only; no `Everyone`, no `BUILTIN\Users`), including the
  narrow-back-after-widening case.

## [1.0.1] — 2026-07-29

### Changed
- Published to PyPI as **`meetmind-ai`** (the name `meetmind` on PyPI belongs to an
  unrelated package). Install with `pip install meetmind-ai`; the CLI command and import
  name remain `meetmind`.
- CI: Windows-only POSIX test fixtures are now skipped with tracked reasons (#3); smoke
  harness updated to the v1.0 output format.

## [1.0.0] — 2026-07-29

First public release. Everything below is the feature surface v1.0
ships with, verified end-to-end on macOS (Apple Silicon): native
sidecars build from source, `meetmind selftest` is green, and a real
audio file runs the full capture-IPC → Parakeet STT → Ollama
summarize pipeline.

### Fixed
- `meetmind summarize` no longer crashes with `RuntimeError: Event
  loop is closed` on the second extraction pass: the shared
  `actants.LLM` httpx pool is now hosted on one persistent background
  event loop instead of a fresh `asyncio.run()` loop per call.

### Capture & transcription
- macOS Core Audio Tap loopback capture (PID translation, real subtap
  UID, clock-device aggregate).
- macOS AVAudioEngine microphone capture.
- FluidAudio Parakeet TDT 0.6B v3 STT (Swift sidecar).
- Length-prefixed binary IPC over stdio between Python and the Swift
  sidecars.
- 48→16 kHz resample + Silero VAD pipeline.
- Dual-stream capture (`--stream both`) — mic + loopback in parallel.
- Sidecar watchdog: mid-meeting death surfaces a `SidecarEvent` on the
  bus instead of silent hangs.
- Pure-Python mock sidecars for CI / dev / no-permission machines.

### Storage & retrieval
- SQLite store with WAL, `synchronous=NORMAL`, `busy_timeout=5000`,
  `foreign_keys=ON` on every open.
- Optional SQLCipher AES-256 at-rest encryption via the
  `meetmind[encrypted]` extra; per-DB DEK minted on first run and held
  in the OS keychain. `meetmind status` and the dashboard's Compliance
  panel both surface which mode is active.
- Schema migration runner with version stamping (`schema_meta`).
- LanceDB hybrid search — BM25 + dense (nomic-embed-text v2) + RRF.
- Per-meeting summaries table persisted via `Store.upsert_summary`.
- Opt-in raw audio persistence (`--persist-audio` / `MEETMIND_PERSIST_AUDIO=1`)
  → 48 kHz mono WAV per stream under `~/.meetmind/audio/`.

### Analysis (BYO-LLM via `actants`)
- Substring-guarded action item extraction.
- Decision extraction with rationale + dissenters.
- Chain-of-Density summarization with auto-densify pass.
- Six providers via `MEETMIND_LLM_*` env vars: Ollama (default),
  OpenAI, Anthropic, Gemini, Groq, Mistral.
- Auto-select best local Ollama model from a preference ladder.
- One shared `actants.LLM` per `summarize` run (cuts httpx warmup).

### Voice identity (opt-in, GDPR-aware)
- ReDimNet-B3 voiceprint embedder (ONNX) with MelHash fallback when
  the model isn't installed.
- Cosine matcher with EMA centroid + FAR ≈ 0.1% threshold ladder.
- Bayesian calendar-prior fusion.
- Real audio enrollment via `meetmind enroll NAME --audio CLIP.wav`;
  deterministic name-hash stub kept behind `--stub` for tests.
- ConsentEvent audit log with Ed25519 signatures.
- Active-learning enrollment queue.

### Diarization
- Channel-prior gate (mic vs loopback) live in the record path.
- Streaming Sortformer adapter (Swift sidecar) + pure-Python mock.
- Post-hoc `meetmind diarize <meeting_id>` runs the diarizer over
  persisted audio + stitches onto the transcript; optionally matches
  clusters to enrolled voiceprints.

### Local HTTP API (`meetmind[api]`)
- FastAPI on 127.0.0.1 (host is locked — never 0.0.0.0).
- Per-launch bearer token with file mode `0600` (chmod-on-overwrite
  hardened against `O_TRUNC` perms drift).
- Loopback handshake `/v1/auth/handshake` — no copy-paste token flow.
- CORS allow-list narrowed to the bound port when supplied.
- SSE bus `/v1/transcripts/live` with slow-subscriber metrics.
- Archive endpoints: `/v1/meetings`, `/v1/meeting/{id}`, `/v1/search`,
  `/v1/meeting/{id}/audio/{stream}` (with query-param token for
  HTML5 `<audio>`), `PATCH/DELETE /v1/meeting/{id}`,
  `/v1/meeting/{id}/summarize`, `/v1/meeting/{id}/diarize`,
  `/v1/meeting/{id}/export/{target}`.
- Recording control: `POST /v1/recording/{start,stop}`,
  `GET /v1/recording/status`.
- Compliance surface: `GET /v1/compliance/status` for the dashboard.

### UI (Tauri / browser, same-origin)
- Editorial monospace + paper-on-ink aesthetic.
- Live overlay mode — speaker blocks, blinking-caret partials,
  FNV-hashed ink-stain palette, coach-tip sticky-note card.
- Dashboard mode — three-column layout (list / detail / metadata).
- In-UI recorder: title input, stream selector, record/stop with live
  elapsed timer.
- In-UI summarize / diarize buttons.
- Rename + delete with cascade through transcripts, actions,
  decisions, and persisted WAV unlink.
- Search jump-to-segment with fade-in highlight.
- Audio playback widget when audio is persisted.
- Export buttons: Obsidian, GitHub, Slack — with `localStorage`
  remembering vault path / repo / webhook.
- Compliance panel surfacing storage mode + retention + counts live.
- Skeleton loading + responsive layout.

### MCP server
- stdio JSON-RPC 2.0 server with 15+ canonical tools (search,
  get_meeting, who_said, list_action_items, get_decisions,
  find_unanswered_questions, get_speaker_history, summarize_period,
  extract_quotes, get_attendees, attendee_stats, link_to_moment,
  compare_meetings, get_followups, export_to,
  start_recording, stop_recording).
- Resources: `meetmind://meetings`, `meetmind://meeting/{id}/...`,
  `meetmind://people`, `meetmind://person/{id}/profile`.
- Prompts: `daily_digest`, `what_changed`, `prep_for_meeting`,
  `follow_up_draft`, `review_my_talk_time`, `weekly_review`,
  `quarterly_summary`.

### Privacy & compliance
- Audio + transcripts + voiceprints never leave the device.
- HTTP API binds 127.0.0.1 only — architecturally enforced, not a
  parameter.
- Outbound-call refusal CI test
  (`tests/security/test_no_outbound_calls.py`).
- DPIA auto-generator (`meetmind compliance dpia`).
- Retention TTL sweep (defaults 3y meetings, 1y voiceprints;
  env-tunable).
- Right-to-erasure cascade with ConsentEvent tombstones.
- Three redaction profiles: `raw`, `team_internal`, `public_share`.
- Ed25519-signed transcript bundles (`export-bundle` / `verify`).
- i18n scaffolding for EN / DE / FR (DE + FR strings are placeholders
  awaiting native review).

### Integrations
- Obsidian filesystem export (YAML frontmatter, Dataview-compatible).
- GitHub Issues via `gh` CLI shell-out.
- Slack Incoming Webhook (`SLACK_WEBHOOK_URL`) Block Kit digests.

### Operational
- `meetmind selftest` — telemetry-free end-to-end health check with
  JSON output for sharing in GitHub issues.
- `meetmind status` reports backends, hardware, storage mode, sidecar
  discovery, and active LLM config as JSON.
- `meetmind demo` single-process show-everything entrypoint.
- `scripts/smoke_e2e.sh` 16-step end-to-end smoke harness.
- `scripts/bench_latency.py`, `scripts/bench_memory.py`.
- 357 tests pass, ruff clean, 6/6 import-linter contracts kept.

### Documentation
- README with install / five-minute tour / common workflows /
  configuration / troubleshooting.
- `SECURITY.md` with threat model + responsible-disclosure address.
- `CONTRIBUTING.md` enforcing the anti-features contract.
- GitHub issue + PR templates.

### Externally blocked (documented unblocks)
The following ship in the same v1.0 line once their external prereq
is satisfied:

- Notion + Linear OAuth integrations.
- Apple Developer ID + notarization for a signed `.app`.
- Homebrew tap.
- Auto-update infra (Ed25519-signed release manifest).
- Linux + Windows native sidecars (macOS-first; PRs welcome).
