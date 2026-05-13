# MeetMind overlay UI

A minimal, build-step-free SvelteKit-shaped UI that subscribes to the
local SSE endpoint at `/v1/transcripts/live` and renders captions in
real time. Architecture-locked stack (Tauri 2 + transparent
always-on-top window). The full Tauri shell lives in
`../src-tauri/`; this directory is the WebView frontend.

## Run as a regular browser overlay (right now)

1. Start the Python pipeline:

   ```bash
   meetmind record --emit-sse --port 7857
   # or, to test the UI without capture:
   meetmind serve --port 7857
   ```

2. Read the bearer token (rotated per-launch, written to
   `~/.meetmind/token` mode 0600):

   ```bash
   cat ~/.meetmind/token
   ```

3. Open `tauri/ui/index.html` directly in any browser, paste the token,
   click **Connect**.

`localStorage` remembers the endpoint + token across reloads.

## Run inside Tauri (full shell)

Requires Rust + cargo + the Tauri CLI:

```bash
cargo install tauri-cli --version "^2.0"
cd tauri
cargo tauri dev
```

The shell at `../src-tauri/` is configured for a transparent,
always-on-top, frame-less window matching the architecture spec
(`_shared/architecture/03_MEETMIND.md` §2.7).
