"""Global pytest configuration.

Two responsibilities:

  1. Disable the OS-keychain DEK path. Tests must never write a key to
     the developer's real keychain — that would persist across runs and
     leak state. `MEETMIND_DISABLE_ENCRYPTION=1` makes `Store.open()`
     skip the keychain lookup and run the stdlib sqlite3 driver.

  2. Block accidental outbound network calls. Most code paths are
     loopback-only by design; this catches regressions where a new
     dependency would silently dial the network during tests.
"""

from __future__ import annotations

import os

# Must be set before any meetmind import. pytest imports conftest first.
os.environ.setdefault("MEETMIND_DISABLE_ENCRYPTION", "1")
