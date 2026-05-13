"""Crypto + key management.

SQLCipher passphrase wrapping, OS keychain integration
(macOS Keychain / Windows DPAPI-NG / libsecret), XChaCha20-Poly1305
chunked Opus envelopes, Ed25519 transcript-bundle signing,
crypto-shred for backup-safe deletion.

Module boundary: leaf — never imports project-internal modules.
"""
