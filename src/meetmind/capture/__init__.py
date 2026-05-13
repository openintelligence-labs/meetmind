"""OS-native audio capture.

Two streams kept separate end-to-end (mic + loopback) — the channel split
is the single largest accuracy lever for the rest of the pipeline. Native
sidecar binaries (Swift / C++ / C) own platform-specific capture; Python
orchestrates them via newline-JSON stdio.

Module boundary: this package does not import from `stt/` or `analyze/`.
"""
