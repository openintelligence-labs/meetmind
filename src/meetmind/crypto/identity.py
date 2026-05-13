"""Per-install Ed25519 identity key.

The user's signing key for legal-mode transcript bundles. Generated on
first use and stored in the OS keychain via the `keyring` library on
production paths; tests use an in-memory KeyStore.

Public key is derived deterministically from the private key, so we
don't need to also persist it. We do persist a fingerprint (sha256 over
the public-key bytes, hex) for "show me my key" UX.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

log = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "meetmind"
KEYCHAIN_KEY = "ed25519-identity-v1"


class KeyStore(Protocol):
    """Minimal interface used by `Identity`. Tests pass an in-memory dict."""

    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, value: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


def _system_keystore() -> KeyStore:
    """Return the OS keychain via the `keyring` library."""
    import keyring  # local import — only needed in production paths

    return keyring  # keyring module satisfies the Protocol shape


@dataclass
class Identity:
    """Holds an Ed25519 private/public key pair."""

    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls) -> Identity:
        return cls(private_key=Ed25519PrivateKey.generate())

    @classmethod
    def from_private_pem(cls, pem: str) -> Identity:
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("expected Ed25519 private key")
        return cls(private_key=key)

    def to_private_pem(self) -> str:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def fingerprint(self) -> str:
        """Hex-encoded sha256 of the raw public-key bytes. Stable per install."""
        return hashlib.sha256(self.public_key_bytes()).hexdigest()

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)

    def verify(self, signature: bytes, message: bytes) -> bool:
        try:
            self.public_key.verify(signature, message)
            return True
        except Exception:  # noqa: BLE001 — verify returns False on any failure
            return False


def load_or_create_identity(store: KeyStore | None = None) -> Identity:
    """Load the persisted identity or generate + persist a fresh one."""
    keystore = store if store is not None else _system_keystore()
    pem = keystore.get_password(KEYCHAIN_SERVICE, KEYCHAIN_KEY)
    if pem:
        return Identity.from_private_pem(pem)
    identity = Identity.generate()
    keystore.set_password(KEYCHAIN_SERVICE, KEYCHAIN_KEY, identity.to_private_pem())
    log.info("generated new identity, fingerprint=%s", identity.fingerprint)
    return identity


def forget_identity(store: KeyStore | None = None) -> None:
    """Delete the persisted identity. Caller is responsible for confirmation UX."""
    keystore = store if store is not None else _system_keystore()
    keystore.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_KEY)


# In-memory keystore for tests + ephemeral CI use. Not for production.


class InMemoryKeyStore:
    """Dict-backed `KeyStore`. Useful in tests and `assist` ephemeral sessions."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        self._store[(service, username)] = value

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)
