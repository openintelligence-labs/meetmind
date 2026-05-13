# meetmind-capture-macos

Native macOS capture sidecar. Speaks the IPC protocol in
[`docs/SPEC_CAPTURE_IPC.md`](../../docs/SPEC_CAPTURE_IPC.md).

## Build

```bash
cd sidecars/macos
swift build -c release
# binary: .build/release/meetmind-capture-macos
```

## Run (manual smoke test)

```bash
# Fake source — silent PCM, no permissions required.
MEETMIND_CAPTURE_FAKE=1 .build/release/meetmind-capture-macos < commands.txt | hexdump -C | head
```

`commands.txt` example:

```
{"id":"1","op":"start","args":{"streams":["mic","loopback"]}}
{"id":"2","op":"stop"}
```

## Permissions

`Info.plist` (added at packaging time) needs:

- `NSMicrophoneUsageDescription` — for mic capture.
- `NSAudioCaptureUsageDescription` — for system audio capture (macOS 14.4+).
- Screen Recording permission — granted by the user via System Settings →
  Privacy & Security → Screen Recording.

If a required permission is missing, the sidecar exits with code `2`. The
Python core surfaces a UI prompt explaining what to grant and how.

## Architecture

- **macOS 14.4+:** `CoreAudioTapSource` opens `AudioHardwareCreateProcessTap`
  against the default output device. Per-process exclusion of self handled
  via `CATapDescription`.
- **macOS 13.x:** falls back to ScreenCaptureKit with
  `excludesCurrentProcessAudio = true` (S1.3).
- **Mic:** `MicSource` opens the default input via `AVAudioEngine`.

S1.2 ships the package skeleton + protocol plumbing + a `SilentSource` so
the IPC + Python integration is testable without OS permissions. The real
mic + tap wiring lands in S1.2b once a permissioned CI machine is ready.

## Codesigning (release builds)

```bash
codesign --force --options runtime --sign "Developer ID Application: ..." \
    .build/release/meetmind-capture-macos
xcrun notarytool submit ...
```

Hardened runtime entitlements live in `Entitlements.plist` (added at
packaging time, not in this directory).
