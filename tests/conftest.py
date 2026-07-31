"""Global pytest configuration.

Disables the OS-keychain DEK path so tests never write a key to the
developer's real keychain, where it would persist and leak state across runs.
`MEETMIND_DISABLE_ENCRYPTION=1` makes `Store.open()` skip the keychain lookup
and use the stdlib sqlite3 driver.
"""

from __future__ import annotations

import os

# Must be set before any meetmind import. pytest imports conftest first.
os.environ.setdefault("MEETMIND_DISABLE_ENCRYPTION", "1")
