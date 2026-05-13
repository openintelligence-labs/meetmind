"""Post-meeting analysis pipeline.

Summarization (Qwen3-30B-A3B Chain-of-Density), action-item extraction with
substring-validated `evidence_quote`, decision + dissenter extraction,
embedding (nomic-embed-text v2), topic clustering (BERTopic + HDBSCAN
partial_fit), live coaching loop (Qwen3-4B rolling 60s).

Module boundary: consumes `Transcript` (text + speakers + timestamps).
Never touches audio.
"""
