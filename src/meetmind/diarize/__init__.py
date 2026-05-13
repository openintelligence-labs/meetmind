"""Speaker diarization + voiceprint identity (opt-in).

Live: NVIDIA Streaming Sortformer 4spk-v2 (CoreML on Mac).
Offline polish: pyannote community-1 fused with live output via DOVER-Lap.
Voiceprint: ReDimNet-B3 ONNX, cosine on EMA centroid.

Module boundary: emits only `(start_ms, end_ms, speaker_id, confidence)`
tuples upward; never reaches into `capture/` or `stt/`.
"""
