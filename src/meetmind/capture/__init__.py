"""OS-native audio capture.

Mic and loopback are kept as separate streams end-to-end. Native sidecar
binaries own platform-specific capture; Python orchestrates them over stdio.

Module boundary: does not import from `stt/` or `analyze/`.
"""
