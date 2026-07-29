# MeetMind

MeetMind is a local-first meeting assistant for capturing, transcribing, summarizing, and searching meetings from your own machine. It records system audio and microphone input without adding a bot to the call, stores meeting data locally, and supports local or bring-your-own LLM providers for analysis.

## Key capabilities

- Capture microphone and system audio without a meeting bot
- Transcribe meetings and organize speaker segments
- Generate summaries, decisions, and action items
- Search across meeting transcripts with an optional local index
- Export meeting notes to Obsidian, GitHub Issues, Slack, or a signed transcript bundle
- Keep audio, transcripts, voiceprints, and indexes on the local device by default
- Use a local LLM through Ollama by default, or configure a hosted provider explicitly

## Requirements

- Python 3.12 or 3.13
- macOS for native system-audio capture
- Optional: Ollama for fully local summarization
- Optional: SQLCipher support for encrypted local storage
- Optional: LanceDB support for hybrid transcript search

## Installation

Install from the latest GitHub release (MeetMind is not on PyPI yet):

```bash
pip install https://github.com/openintelligence-labs/meetmind/releases/latest/download/meetmind-1.0.0-py3-none-any.whl
```

Or install from source:

```bash
git clone https://github.com/openintelligence-labs/meetmind.git
cd meetmind
pip install .
```

For the recommended local setup, install the API, dashboard, encrypted storage, search, and audio extras:

```bash
pip install 'meetmind[api,encrypted,storage,audio] @ git+https://github.com/openintelligence-labs/meetmind.git'
```

Available extras:

| Extra | Purpose |
| --- | --- |
| `api` | Local HTTP API, dashboard, and live event stream |
| `encrypted` | SQLCipher-backed encrypted local database |
| `storage` | LanceDB-backed transcript index and search |
| `audio` | Voice activity detection and voiceprint dependencies |
| `dev` | Test, lint, and contributor tooling |

Verify the installation:

```bash
meetmind selftest
```

If the `encrypted` extra is not installed, MeetMind uses plain SQLite and reports that mode in `meetmind status`.

### Native capture sidecars (macOS)

Live capture and on-device transcription use small Swift sidecar binaries (capture, Parakeet STT, diarization). Get them one of two ways:

- Download `meetmind-sidecars-macos-arm64.tar.gz` from the GitHub release, unpack it, and place the binaries on your `PATH` (or point `MEETMIND_CAPTURE_SIDECAR` / `MEETMIND_STT_SIDECAR` at them).
- Build from source (requires Xcode command line tools, macOS 14.4+):

```bash
cd sidecars/macos
swift build -c release
```

A source checkout is auto-discovered at `sidecars/macos/.build/release/`. `meetmind selftest` reports whether the sidecars are found. Without them, `meetmind record --mock` still exercises the full pipeline with synthetic audio. The STT sidecar downloads the Parakeet model (~690 MB, one time) on first use.

## LLM configuration

MeetMind uses a local Ollama provider by default. To use Ollama:

```bash
brew install ollama
ollama serve &
ollama pull gemma3:4b
```

To use a hosted provider, configure it explicitly:

```bash
export MEETMIND_LLM_PROVIDER=openai
export MEETMIND_LLM_MODEL=gpt-4o-mini
export MEETMIND_LLM_API_KEY=sk-...
```

Supported provider configuration is controlled through environment variables. Run the following command to inspect the active provider, model, storage mode, and resolved paths:

```bash
meetmind status
```

## Quick start

### Open the dashboard

```bash
meetmind ui
```

The dashboard runs locally at `http://127.0.0.1:7857`. It provides meeting capture, transcript browsing, summaries, action items, decisions, search, and export controls.

### Record a meeting

From the dashboard, choose a meeting title and capture mode, then start recording.

From the terminal:

```bash
meetmind record --duration 0 --emit-sse --persist-audio --title "Weekly sync"
```

Use `Ctrl+C` to stop an open-ended recording. The command prints the meeting ID on the last line.

For development or CI environments without native audio capture:

```bash
meetmind record --duration 60 --mock
```

### Summarize a meeting

```bash
meetmind summarize "$MID"
```

This generates a meeting summary, action items, and decisions from the transcript.

### Build the search index

```bash
meetmind index
meetmind search "snowflake migration"
```

Search can also be run from the dashboard. Results link back to matching transcript segments.

### Export meeting output

```bash
meetmind export-obsidian "$MID" --vault ~/MyVault
meetmind export-github "$MID" --repo owner/name
meetmind export-slack "$MID"
```

Slack export requires a webhook URL:

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

To create a signed transcript bundle:

```bash
meetmind export-bundle "$MID" --out ./bundle.tar.gz
meetmind verify ./bundle.tar.gz
```

## Common workflows

### Record, summarize, index, and export

```bash
meetmind ui
meetmind record --emit-sse --persist-audio --title "Weekly sync"

MID=<meeting-id>
meetmind summarize "$MID"
meetmind index
meetmind export-slack "$MID"
```

### Manage previous meetings

```bash
meetmind meetings
```

The dashboard also supports viewing transcripts, summaries, action items, decisions, and recorded audio when audio persistence is enabled.

### Search one meeting

```bash
meetmind search "decided to ship in june" --meeting-id "$MID"
```

### Speaker enrollment

Voiceprint identity is optional and should only be used with appropriate consent.

```bash
meetmind enroll "Alice" --audio ~/clips/alice.wav
meetmind speakers
meetmind diarize "$MID"
```

To remove an enrolled speaker:

```bash
meetmind forget "$SPEAKER_ID"
```

## Privacy and security

MeetMind is designed for local-first meeting workflows.

- Audio, transcripts, voiceprints, local archives, and search indexes are stored on the device by default.
- The default LLM provider is local Ollama.
- Hosted LLM providers are opt-in and require explicit environment configuration.
- Local database encryption is available through the `encrypted` extra.
- Voiceprint enrollment is optional and should be consent-based.
- Retention controls are available for meetings and voiceprints.
- Redaction profiles are available before sharing transcripts externally.

Useful commands:

```bash
meetmind compliance dpia > dpia.md
meetmind compliance retention-sweep --dry-run
meetmind compliance retention-sweep

echo "Email Sam at sam@example.com" | meetmind redact --profile public_share
```

Redaction profiles:

| Profile | Behavior |
| --- | --- |
| `raw` | No redaction |
| `team_internal` | Light redaction for internal sharing |
| `public_share` | Aggressive redaction for external sharing |

## Configuration

MeetMind is configured through environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `MEETMIND_HOME` | `~/.meetmind` | Root directory for local data, audio, indexes, and tokens |
| `MEETMIND_LLM_PROVIDER` | `ollama` | LLM provider used for summaries and extraction |
| `MEETMIND_LLM_MODEL` | Auto-detected | Model name for the configured provider |
| `MEETMIND_LLM_BASE_URL` | Provider default | Custom provider endpoint |
| `MEETMIND_LLM_API_KEY` | Unset | API key for hosted providers |
| `MEETMIND_COACH_MODEL` | Falls back to `MEETMIND_LLM_MODEL` | Model used by the live coaching workflow |
| `MEETMIND_PERSIST_AUDIO` | `0` | Set to `1` to persist recorded audio by default |
| `MEETMIND_RETENTION_MEETINGS_DAYS` | `1095` | Meeting retention period |
| `MEETMIND_RETENTION_VOICEPRINT_DAYS` | `365` | Voiceprint retention period |
| `MEETMIND_DISABLE_ENCRYPTION` | Unset | Set to `1` to disable encrypted storage lookup |
| `MEETMIND_DB_PASSPHRASE` | Unset | Explicit database passphrase override |
| `SLACK_WEBHOOK_URL` | Unset | Default Slack webhook for Slack export |

Run `meetmind status` to see the resolved configuration.

## Architecture

```text
macOS audio sidecar / mock capture
        |
        v
Python recording pipeline
        |
        +--> voice activity detection
        +--> speech-to-text
        +--> speaker segmentation and diarization
        +--> local storage
        +--> dashboard event stream
        +--> optional search index
        |
        v
Analysis layer
        |
        +--> summary generation
        +--> action item extraction
        +--> decision extraction
        +--> export integrations
```

The project separates capture, transcription, storage, analysis, and export layers so each part can be developed or replaced independently.

## Integrations

| Integration | Status | Notes |
| --- | --- | --- |
| Obsidian | Available | Filesystem export |
| GitHub Issues | Available | Uses authenticated `gh` CLI |
| Slack | Available | Uses Incoming Webhook |
| Signed transcript bundles | Available | Verifiable archive export |
| Notion | Planned | Requires OAuth setup |
| Linear | Planned | Requires OAuth setup |

## Platform support

| Platform | Status |
| --- | --- |
| macOS | Primary supported platform for native capture |
| Linux | Native capture sidecar planned |
| Windows | Native capture sidecar planned |

Mock recording is available for testing and development on unsupported environments.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| `capture-sidecar` or `stt-sidecar` warning in `selftest` | Build the native sidecars or use `--mock` for development capture |
| Database opens unencrypted | Install `meetmind[encrypted]` or check encryption configuration |
| Dashboard opens with no meetings | Record a test meeting and refresh the dashboard |
| Search returns no results | Run `meetmind index` to build or refresh the transcript index |
| Slack export reports no webhook | Set `SLACK_WEBHOOK_URL` or pass a webhook URL explicitly |
| GitHub export fails because `gh` is missing | Install the GitHub CLI and run `gh auth login` |
| LLM calls do not complete | Run `meetmind status` and verify provider, model, API key, and base URL |

For a diagnostic report:

```bash
meetmind selftest --json
```

## Development

Install development dependencies:

```bash
pip install 'meetmind[dev]'
```

Run checks:

```bash
pytest
ruff check .
```

Security-sensitive changes should preserve the local-first defaults and avoid silent network egress.

## Contributing

Issues and pull requests are welcome. Before proposing a feature, check whether it changes the local-first privacy model or introduces external data transfer. Any external service integration should be explicit, configurable, and documented.

Security disclosures should be sent to `security@openintelligence-labs.dev` rather than filed as public issues.

## License

MIT
