# Capture Sidecar IPC Protocol

> Wire format between the Python core and native capture sidecars
> (Swift on macOS, C++ on Windows, C on Linux). Frozen for v0.4 — backwards-compatible
> additions only past this point.

## Why a sidecar

PyObjC is brittle on modern macOS (issue [pyobjc#647](https://github.com/ronaldoussoren/pyobjc/issues/647)).
WASAPI's `ActivateAudioInterfaceAsync` and PipeWire's `pw_stream` are C-level APIs that don't
fit Python's import model cleanly. A small native binary speaking a documented protocol is the
ergonomic and supportable path.

## Transport

- **stdio:** sidecar reads control messages from stdin (newline-delimited JSON), writes
  output to stdout. Stderr is reserved for human-readable logs (forwarded to Python's
  logger).
- **Framing:** stdout is a stream of length-prefixed binary frames. No JSON-on-stdout —
  audio data is hot path; we don't want to base64-encode every PCM chunk.
- **Control plane:** JSON commands on stdin trigger sidecar lifecycle; responses come back
  as a special control frame on stdout (see frame types below).

## Frame format

Every output frame is:

```
+--------+----------+-------------------+
| 1 byte | 4 bytes  | N bytes           |
| type   | length N | payload           |
| (u8)   | (u32 LE) | (raw)             |
+--------+----------+-------------------+
```

- **type** — frame discriminator (see below)
- **length** — little-endian u32, length of payload in bytes
- **payload** — raw bytes whose interpretation depends on `type`

### Frame types

| Hex  | Name           | Payload                                                              |
|------|----------------|----------------------------------------------------------------------|
| 0x01 | `HELLO`        | UTF-8 JSON: sidecar identity, version, capabilities                  |
| 0x02 | `READY`        | UTF-8 JSON: `{ "stream_id": "mic"|"loopback", "format": {...} }`     |
| 0x10 | `AUDIO`        | header (8 B) + raw PCM                                               |
| 0x20 | `EVENT`        | UTF-8 JSON: lifecycle event (e.g. permission revoked, device change) |
| 0x30 | `LOG`          | UTF-8 string (informational; same as stderr but structured)          |
| 0x40 | `CONTROL_ACK`  | UTF-8 JSON: response to a stdin command                              |
| 0xFF | `BYE`          | UTF-8 JSON: clean shutdown summary                                   |

### `AUDIO` payload header (8 bytes)

```
+--------+--------+----------+
| 1 byte | 1 byte | 6 bytes  |
| stream | flags  | timestamp|
| (u8)   | (u8)   | (u48 LE) |
+--------+--------+----------+
```

- **stream** — `0x00` = mic, `0x01` = loopback. **Mic and loopback always travel as
  separate frames** (architecture §2.1).
- **flags** — bit 0: 1 if first frame of a new utterance/segment; bit 1: 1 if last frame
  before silence; bits 2–7 reserved.
- **timestamp** — little-endian u48 monotonic timestamp in microseconds since session start.
- Followed by **raw interleaved PCM** in the format declared in the matching `READY` frame
  (default: `s16le` mono 48 kHz). One channel per stream — interleaving inside a single
  stream is reserved for future multi-mic arrays.

## Control commands (Python → sidecar, stdin)

Newline-delimited JSON. Each command has an `id` so the sidecar's `CONTROL_ACK` can be
correlated. Schema:

```json
{ "id": "<correlation-id>", "op": "<verb>", "args": { ... } }
```

| `op`        | `args`                                                                 | Effect |
|-------------|------------------------------------------------------------------------|--------|
| `start`     | `{ "streams": ["mic", "loopback"], "format": { "rate": 48000, "encoding": "s16le" } }` | Begin capture; sidecar emits `READY` then `AUDIO` frames. |
| `pause`     | `{}`                                                                   | Stop emitting audio frames (keep handles open). |
| `resume`    | `{}`                                                                   | Resume emitting. |
| `stop`      | `{}`                                                                   | Clean shutdown; sidecar flushes, sends `BYE`, exits 0. |
| `query`     | `{ "what": "devices" \| "permissions" }`                               | Returns inventory in `CONTROL_ACK`. |

Sidecar MUST reply to every command with a `CONTROL_ACK` frame containing
`{ "id": "<same id>", "ok": true | false, "error"?: "..." , "data"?: {...} }`.

## Lifecycle

```
sidecar starts
   │
   │   HELLO (capabilities + permissions snapshot)
   ▼
stdin: { "id":"1", "op":"start", "args": {...} }
   │
   │   CONTROL_ACK { "id":"1", "ok": true }
   │   READY { "stream_id":"mic", ... }
   │   READY { "stream_id":"loopback", ... }
   │   AUDIO ... AUDIO ... AUDIO ...      (continuous)
   │   EVENT { "kind": "device_changed" } (occasional)
   ▼
stdin: { "id":"42", "op":"stop" }
   │
   │   CONTROL_ACK { "id":"42", "ok": true }
   │   BYE { "frames_mic": ..., "frames_loopback": ..., "duration_ms": ... }
   │
sidecar exits 0
```

If the sidecar exits non-zero, Python treats it as a fatal error and surfaces stderr.

## Permissions

Sidecars never prompt the user themselves — they exit `2` with an error message if a
required permission is missing. Python catches the exit code and surfaces a UI prompt. The
sidecar emits `HELLO` before any prompts, so the Python core always knows what
permissions are required for the current platform.

## Versioning

`HELLO` includes `protocol_version` (semver). Python pins a major version range. Breaking
changes require a major bump and parallel old-version support for one minor cycle.

## Mock sidecar (for tests)

`tests/fixtures/mock_sidecar.py` is a pure-Python implementation that emits the protocol
from a WAV file. Used to exercise the IPC layer without OS-specific permissions.

## v0.4 protocol version: `1.0.0`
