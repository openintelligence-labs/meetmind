"""Voiceprint matcher.

Cosine similarity on **L2-normalized EMA centroids**. This is equivalent
to a properly tuned PLDA when embeddings are large-margin AAM-softmax
(which ReDimNet-B3, ECAPA-TDNN, and
ResNet293-LM all are) — and cosine has no domain-mismatch retraining,
so it's the right default for a consumer product.

Contract:
  • ``match(embedding, speakers)`` returns (speaker_id, score) for the
    best match if posterior > threshold, else None.
  • ``update(speaker, embedding, ...)`` does an EMA update on the
    centroid and a ring-buffer push (last 32 embeddings retained for
    re-centroiding after deletes).

Thresholds default to FAR ≈ 0.1% on ReDimNet-B3 — i.e. accept if cos ≥
0.58, queue for active-learning if 0.45–0.58, reject below 0.45. These
are tunable per-embedder; ECAPA-TDNN typically sits at τ ≈ 0.55 for
the same FAR.

This module is pure NumPy. The embedder lives in ``voiceprint.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from meetmind.models import Speaker


@dataclass(frozen=True)
class MatchDecision:
    """Outcome of `match()` against an enrolled speaker bank."""

    speaker_id: str | None
    cosine: float
    posterior: float
    band: str  # "accept" | "uncertain" | "reject"


@dataclass
class MatcherConfig:
    accept_threshold: float = 0.58  # FAR ≈ 0.1% on ReDimNet-B3
    uncertain_threshold: float = 0.45  # below → reject; above → uncertain
    posterior_threshold: float = 0.7
    softmax_temperature: float = 0.10  # cosine → posterior tempering
    unknown_prior: float = 0.1
    ema_alpha: float = 0.05
    update_min_cosine: float = 0.65
    update_min_snr_db: float = 10.0
    update_min_seconds: float = 3.0
    ring_capacity: int = 32


def _to_unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize, with a 0-vector → 0-vector escape."""
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v.astype(np.float32, copy=False)
    return (v / norm).astype(np.float32, copy=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two raw vectors (in [-1, 1])."""
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.dot(_to_unit(a), _to_unit(b)))


def _decode_centroid(speaker: Speaker) -> np.ndarray | None:
    if speaker.voiceprint_centroid is None or len(speaker.voiceprint_centroid) == 0:
        return None
    arr = np.frombuffer(speaker.voiceprint_centroid, dtype=np.float32)
    return arr if arr.size > 0 else None


def _encode_centroid(vec: np.ndarray) -> bytes:
    return _to_unit(vec.astype(np.float32, copy=False)).tobytes()


def _encode_ring(samples: Iterable[np.ndarray]) -> list[bytes]:
    return [_to_unit(s.astype(np.float32, copy=False)).tobytes() for s in samples]


def _decode_ring(speaker: Speaker) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for blob in speaker.voiceprint_ring or []:
        if blob is None:
            continue
        if isinstance(blob, str):
            blob = bytes.fromhex(blob)
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size > 0:
            out.append(arr)
    return out


@dataclass
class Matcher:
    """Cosine matcher + EMA centroid updater.

    Stateless aside from the `MatcherConfig`; centroid + ring storage
    lives on `Speaker` rows in the SQLCipher store. All updates return
    a fresh `Speaker` value rather than mutating in-place so the caller
    decides when to upsert.
    """

    config: MatcherConfig = field(default_factory=MatcherConfig)

    # ---------------------------------------------------------------------
    # Matching
    # ---------------------------------------------------------------------

    def match(
        self,
        embedding: np.ndarray,
        speakers: Iterable[Speaker],
        *,
        priors: dict[str, float] | None = None,
    ) -> MatchDecision:
        """Find the best speaker for ``embedding`` from ``speakers``.

        ``priors`` is an optional ``{speaker_id: weight}`` map (e.g.
        from `calendar_prior.bayesian_priors`). Missing speakers
        receive a uniform prior over the unprovided set.
        """
        emb_unit = _to_unit(embedding)
        candidates: list[tuple[str, float, float]] = []  # (id, cos, prior)
        speakers_list = [s for s in speakers if _decode_centroid(s) is not None]
        if not speakers_list:
            return MatchDecision(speaker_id=None, cosine=0.0, posterior=0.0, band="reject")

        if priors is None:
            uniform = (1.0 - self.config.unknown_prior) / len(speakers_list)
            priors = {s.id: uniform for s in speakers_list}

        for sp in speakers_list:
            centroid = _decode_centroid(sp)
            assert centroid is not None  # narrow above
            cos = float(np.dot(emb_unit, _to_unit(centroid)))
            candidates.append((sp.id, cos, float(priors.get(sp.id, 0.0))))

        # Posterior via softmax on (cos / τ) weighted by priors. UNKNOWN
        # gets a synthetic candidate at cos=accept_threshold, prior=alpha.
        tau = max(self.config.softmax_temperature, 1e-6)
        unknown_logit = self.config.accept_threshold / tau + math.log(
            max(self.config.unknown_prior, 1e-12)
        )
        logits = []
        for _, cos, prior in candidates:
            logits.append(cos / tau + math.log(max(prior, 1e-12)))
        max_logit = max(logits + [unknown_logit])
        exps = [math.exp(x - max_logit) for x in logits]
        unknown_exp = math.exp(unknown_logit - max_logit)
        Z = sum(exps) + unknown_exp
        posteriors = [e / Z for e in exps]

        best_idx = max(range(len(candidates)), key=lambda i: posteriors[i])
        best_id, best_cos, _ = candidates[best_idx]
        best_post = posteriors[best_idx]

        if (
            best_cos >= self.config.accept_threshold
            and best_post >= self.config.posterior_threshold
        ):
            band = "accept"
        elif best_cos >= self.config.uncertain_threshold:
            band = "uncertain"
        else:
            band = "reject"

        return MatchDecision(
            speaker_id=best_id if band == "accept" else None,
            cosine=best_cos,
            posterior=best_post,
            band=band,
        )

    # ---------------------------------------------------------------------
    # EMA update
    # ---------------------------------------------------------------------

    def should_update(
        self,
        cos: float,
        *,
        snr_db: float | None,
        duration_seconds: float,
    ) -> bool:
        """Gate centroid updates: only on confident, clean, long-enough samples."""
        if cos < self.config.update_min_cosine:
            return False
        if duration_seconds < self.config.update_min_seconds:
            return False
        return not (snr_db is not None and snr_db < self.config.update_min_snr_db)

    def update_centroid(
        self,
        speaker: Speaker,
        embedding: np.ndarray,
    ) -> Speaker:
        """Return a new ``Speaker`` with EMA-updated centroid + ring."""
        emb_unit = _to_unit(embedding)
        existing = _decode_centroid(speaker)
        if existing is None:
            new_centroid = emb_unit
        else:
            alpha = self.config.ema_alpha
            mixed = (1.0 - alpha) * existing + alpha * emb_unit
            new_centroid = _to_unit(mixed)

        ring = _decode_ring(speaker)
        ring.append(emb_unit.copy())
        ring = ring[-self.config.ring_capacity :]

        # Speaker is a Pydantic v2 model — use model_copy.
        return speaker.model_copy(
            update={
                "voiceprint_centroid": _encode_centroid(new_centroid),
                "voiceprint_ring": _encode_ring(ring),
            }
        )

    # ---------------------------------------------------------------------
    # Re-centroid (after deletes / migrations)
    # ---------------------------------------------------------------------

    def recentroid(self, speaker: Speaker) -> Speaker:
        """Recompute centroid from the ring buffer.

        Useful after a privacy delete where the centroid may have
        contained leaked information from an erased speaker, or
        after migrating to a new embedder.
        """
        ring = _decode_ring(speaker)
        if not ring:
            return speaker
        stacked = np.stack([_to_unit(v) for v in ring], axis=0)
        new_centroid = _to_unit(stacked.mean(axis=0))
        return speaker.model_copy(update={"voiceprint_centroid": _encode_centroid(new_centroid)})
