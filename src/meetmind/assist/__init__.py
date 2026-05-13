"""Live Q&A overlay mode (post-1.0 feature).

Detects questions on streaming STT partials, retrieves from a session-
scoped LanceDB collection of user-loaded context, streams answers onto
a transparent overlay. Ephemeral by default; opt-in archival via
`--archive`.
"""
