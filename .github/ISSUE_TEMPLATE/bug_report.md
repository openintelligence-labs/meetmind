---
name: Bug report
about: Something is broken or behaving unexpectedly.
title: ''
labels: bug
assignees: ''
---

**What happened?**

A short description of the behaviour you saw vs. what you expected.

**Steps to reproduce**

1.
2.
3.

**Selftest output**

```
$ meetmind selftest --json
<paste here — copy the WHOLE JSON; we triage from this>
```

**Other context**

- OS:
- `meetmind --version`:
- LLM provider in use (Ollama / OpenAI / etc.):
- Was the recording in mock mode (`--mock`) or against the real sidecar?

**Logs / stderr**

If the issue involves the recording pipeline, paste the last 50 lines
of stderr from `meetmind record` here.
