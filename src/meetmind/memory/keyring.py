"""Per-database encryption-key management via the OS keychain.

Each MeetMind install holds a single Data Encryption Key (DEK) wrapped
by the OS keychain (macOS Keychain, Windows Credential Manager, Linux
Secret Service). The DEK is what SQLCipher consumes as its
`PRAGMA key`; the keychain is the wrapper.

Why not derive from a passphrase? Because there is no passphrase — the
threat model is offline disk theft, not a remote attacker who can prompt
the user. The OS keychain unlocks when the user logs in; that's the
trust root. If you want passphrase-derived keys, set
`MEETMIND_DB_PASSPHRASE` and we'll prefer it.

This module is import-cycle-safe: `memory/keyring.py` imports nothing
from `meetmind.memory.store`. `Store.open()` calls into here.
"""

from __future__ import annotations

import logging
import os
import secrets

log = logging.getLogger(__name__)

_SERVICE = "meetmind"
_USER = "db-dek"
_ENV_OVERRIDE = "MEETMIND_DB_PASSPHRASE"


def get_or_create_dek() -> str | None:
    """Return the DEK as a hex string, or None if encryption is disabled.

    Order of resolution:
      1. `MEETMIND_DB_PASSPHRASE` env var — explicit user override.
      2. macOS Keychain / Win Cred Manager / linux secret-service via
         `keyring`. We mint a 256-bit secret on first run.
      3. Return None — caller opens the DB unencrypted (dev / CI mode).

    Returns None rather than raising so dev environments without a
    keychain (Docker, headless CI) keep working. Callers should surface
    the unencrypted state to the user via `meetmind status`.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return override
    if os.environ.get("MEETMIND_DISABLE_ENCRYPTION") == "1":
        return None
    try:
        import keyring  # noqa: PLC0415
    except ImportError:
        log.debug("keyring not installed; running unencrypted")
        return None
    try:
        existing = keyring.get_password(_SERVICE, _USER)
        if existing:
            return existing
        # Mint a new 256-bit DEK on first run. token_hex(32) = 64 hex chars.
        minted = secrets.token_hex(32)
        keyring.set_password(_SERVICE, _USER, minted)
        log.info("minted new DEK and stored in OS keychain (service=%s)", _SERVICE)
        return minted
    except Exception as e:
        # Headless CI, locked keychain, or DBus not running.
        log.warning("keyring unavailable (%s); opening DB unencrypted", e)
        return None


def forget_dek() -> bool:
    """Delete the DEK from the OS keychain. Returns True if deleted.

    This is a crypto-shred: SQLCipher data on disk becomes unreadable.
    Used by `meetmind compliance retention-sweep --crypto-shred` and by
    the right-to-erasure cascade.
    """
    try:
        import keyring  # noqa: PLC0415
    except ImportError:
        return False
    try:
        keyring.delete_password(_SERVICE, _USER)
        log.warning("DEK deleted from keychain — existing encrypted DBs are now unreadable")
        return True
    except Exception as e:
        log.error("failed to delete DEK: %s", e)
        return False
