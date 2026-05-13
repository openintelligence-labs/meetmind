# MeetMind — Tauri shell

The MeetMind UI ships as a single HTML page served straight from the
Python API (`tauri/ui/index.html`). It has two modes selected by
``body[data-mode]``:

- **overlay** — transparent, always-on-top caption strip + sticky-note
  coach card. Subscribes to ``/v1/transcripts/live`` via SSE.
- **dashboard** — three-column archive view: meeting list (left),
  transcript + summary + audio playback (centre), tools + actions +
  decisions + voices-heard + compliance (right).

The Rust shell in ``src-tauri/`` is a thin window manager around the
same UI.

## Layout

```
tauri/
├── ui/                  # WebView frontend (HTML + CSS + ES modules)
│   ├── index.html
│   ├── styles.css       # editorial monospace + paper-on-ink aesthetic
│   ├── app.js           # overlay mode: live SSE client + coach card
│   └── dashboard.js     # dashboard mode: meetings/search/export/tools
└── src-tauri/           # Rust shell
    ├── Cargo.toml
    ├── build.rs
    ├── tauri.conf.json  # window: transparent + always-on-top + frameless
    ├── capabilities/default.json
    └── src/
        ├── main.rs
        └── lib.rs       # tray icon + window lifecycle
```

## Run the UI right now (browser, no Rust toolchain needed)

The recommended path is one command:

```bash
meetmind ui                          # serves UI + auto-opens the browser
```

`meetmind ui` starts the FastAPI server on 127.0.0.1:7857, opens
``http://127.0.0.1:7857/`` in your default browser, and the UI
auto-handshakes the bearer token over loopback — no copy-paste.

If you'd rather run the API and open the page yourself:

```bash
meetmind serve --port 7857
open http://127.0.0.1:7857/          # any browser
```

This is what the SSE-based architecture buys us: the overlay is a
thin WebView client, so we don't depend on the Rust build to use it.

## Run inside Tauri (full transparent overlay) — experimental

The Rust shell is not yet signed or notarized. To run it in dev mode:

```bash
# One-time
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2.0"

cd tauri
cargo tauri dev      # development run with hot reload
cargo tauri build    # local release bundle (.app) — not yet signed
```

A signed/notarized release pipeline is blocked on Apple Developer
enrollment ($99/yr).

## What's locked + what's deferred

Locked (current behaviour):

- Tauri 2 + native WebView (no Electron).
- Window: transparent, always-on-top, frameless.
- Tray icon with show/hide overlay + quit.
- macOS-private API enabled (transparent backdrop).
- Vanilla HTML + ES modules — no build step.
- **Dashboard ships live**: meetings list, search with
  jump-to-segment, rename/delete, Obsidian/GitHub/Slack export
  buttons, diarize trigger, compliance panel, audio playback.

Deferred (locked architecturally, blocked on external steps):

- Click-through hit zones (60 fps cursor-poll toggling
  ``setIgnoreCursorEvents``). Lands alongside the first real
  production overlay run.
- Auto-update via Tauri Updater plugin + signed Ed25519 manifest.
- Apple Developer ID notarization pipeline.
- Linux + Windows shells (macOS-first; PRs welcome).
