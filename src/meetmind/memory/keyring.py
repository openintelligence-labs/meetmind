"""Per-database encryption-key management via the OS keychain.

Each install holds one Data Encryption Key, stored in the OS keychain and
consumed by SQLCipher as its `PRAGMA key`. The threat model is offline disk
theft, so the keychain (unlocked at login) is the trust root rather than a
user passphrase; `MEETMIND_DB_PASSPHRASE` overrides when set.
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

    Resolves `MEETMIND_DB_PASSPHRASE` first, then the OS keychain (minting a
    256-bit secret on first run). Returns None rather than raising when no
    keychain is available, so headless CI and containers keep working; callers
    are expected to surface the resulting unencrypted state.
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
        minted = secrets.token_hex(32)  # 256-bit DEK
        keyring.set_password(_SERVICE, _USER, minted)
        log.info("minted new DEK and stored in OS keychain (service=%s)", _SERVICE)
        return minted
    except Exception as e:
        # Headless CI, locked keychain, or DBus not running.
        log.warning("keyring unavailable (%s); opening DB unencrypted", e)
        return None


def forget_dek() -> bool:
    """Delete the DEK from the OS keychain, returning True if deleted.

    This is a crypto-shred: SQLCipher data on disk becomes unreadable.
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
