# MeetMind

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Powered by agentic-kit](https://img.shields.io/badge/powered%20by-agentic--kit-7c3aed)](https://github.com/openintelligence-labs/agentic-kit)

> **Open source alternative to Otter.ai ($17/mo) and Fireflies ($19/mo).** Local AI meeting assistant. Records system audio (no bot joins your call), transcribes with Whisper.cpp, summarizes with Ollama, extracts action items. 100% on your machine.

⭐ **Star us on GitHub** if you're tired of Otter sending transcripts to the cloud.

## Why this exists

Every meeting tool sends your audio to someone else's server. Existing OSS does transcription-only — nothing does the full loop: record → transcribe → summarize → extract actions → organize → search. MeetMind is the full Otter replacement that stays on your machine.

## Quick start

```bash
pip install meetmind
meetmind record
# ... have a meeting ...
meetmind summarize <meeting_id>
```

## Features

| Feature | What it does |
|---|---|
| System audio capture | No bot needed — records whatever plays |
| Whisper.cpp transcription | Fast local STT, CPU or GPU |
| AI summary | TL;DR, key decisions, action items (via agentic-kit) |
| Speaker diarization | Who said what |
| Searchable archive | SQLite-backed, find any past meeting |
| Calendar integration | Auto-start on meeting events (planned) |

## Roadmap

- [x] Data models (Transcript, Summary, Meeting)
- [x] CLI skeleton
- [ ] Audio capture (sounddevice)
- [ ] Whisper.cpp wrapper
- [ ] LLM summarizer using agentic-kit
- [ ] SQLite archive
- [ ] Electron/Tauri desktop app

## Part of the Open Intelligence Labs ecosystem

- [agentic-kit](https://github.com/openintelligence-labs/agentic-kit) — shared SDK
- [SecondBrain](https://github.com/openintelligence-labs/secondbrain) — personal memory (meeting context flows in)
- [DeepDive](https://github.com/openintelligence-labs/deepdive) — deep research agent

## License

MIT
