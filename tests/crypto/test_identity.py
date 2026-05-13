"""Tests for the Ed25519 identity helpers."""

from __future__ import annotations

import pytest

from meetmind.crypto.identity import (
    KEYCHAIN_KEY,
    KEYCHAIN_SERVICE,
    Identity,
    InMemoryKeyStore,
    forget_identity,
    load_or_create_identity,
)


def test_generate_round_trip_signs_and_verifies():
    identity = Identity.generate()
    msg = b"some payload to sign"
    sig = identity.sign(msg)
    assert identity.verify(sig, msg)
    assert not identity.verify(sig, msg + b"!")


def test_pem_round_trip_preserves_keypair():
    a = Identity.generate()
    pem = a.to_private_pem()
    b = Identity.from_private_pem(pem)
    assert a.fingerprint == b.fingerprint
    assert a.public_key_bytes() == b.public_key_bytes()


def test_fingerprint_is_stable_64_hex_chars():
    identity = Identity.generate()
    fp = identity.fingerprint
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_differs_per_identity():
    a = Identity.generate()
    b = Identity.generate()
    assert a.fingerprint != b.fingerprint


def test_load_or_create_persists_to_keystore():
    store = InMemoryKeyStore()
    a = load_or_create_identity(store)
    b = load_or_create_identity(store)
    # Same identity returned both times.
    assert a.fingerprint == b.fingerprint
    # Persisted under expected service/key.
    assert store.get_password(KEYCHAIN_SERVICE, KEYCHAIN_KEY) is not None


def test_forget_identity_clears_keystore():
    store = InMemoryKeyStore()
    a = load_or_create_identity(store)
    forget_identity(store)
    assert store.get_password(KEYCHAIN_SERVICE, KEYCHAIN_KEY) is None
    b = load_or_create_identity(store)
    assert a.fingerprint != b.fingerprint  # fresh identity


def test_from_private_pem_rejects_non_ed25519():
    with pytest.raises((ValueError, TypeError)):
        Identity.from_private_pem("not a pem")
