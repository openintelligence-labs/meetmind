"""Persistent storage.

SQLCipher AES-256 + LanceDB embedded. Per-DB DEK wrapped by OS keychain.
Hybrid search: BM25 + dense → RRF(k=60) + optional ColBERT rerank.

Module boundary: only this package opens SQLCipher connections.
"""
