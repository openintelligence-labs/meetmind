"""Channel-prior gate.

When mic + loopback are kept separate end-to-end, diarization gets a
free accuracy boost from the channel of origin. The
mic stream is, by construction, a single speaker (the user). The
loopback stream contains everyone else.

`ChannelPrior` does two things:

  1. **Relabel** — replace opaque cluster ids with `self` / `remote-<X>`
     based purely on the originating channel.
  2. **Override** — when both per-segment mic and loopback RMS are
     known, flip the label if one channel dominates the other by more
     than `override_margin_db`. This catches the specific failure mode
     where the capture sidecar mis-routed energy (e.g. system audio
     leaked into the mic and got labelled as `remote` by the diarizer).

The override is the documented "free 30%" wedge: it rescues the
self/remote confusions that are the worst error class in real laptop
meeting apps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

from meetmind.diarize.base import DiarSegment
from meetmind.ipc import StreamId


def _rms(samples: Iterable[float]) -> float:
    seq = list(samples)
    if not seq:
        return 0.0
    return math.sqrt(sum(x * x for x in seq) / len(seq))


def _db(num: float, denom: float) -> float:
    """20·log10(num/denom), with denom→0 protection that returns +inf."""
    if denom <= 1e-12:
        return float("inf") if num > 1e-12 else 0.0
    return 20.0 * math.log10(num / denom)


@dataclass
class ChannelPrior:
    """Apply the mic-vs-loopback channel prior to diarization output."""

    override_margin_db: float = 6.0
    relabel_unknown_channel: bool = False

    def relabel(self, seg: DiarSegment) -> DiarSegment:
        """Return a new segment with cluster_id replaced by self/remote-X."""
        if seg.channel is StreamId.MIC:
            return replace(seg, cluster_id="self")
        if seg.channel is StreamId.LOOPBACK:
            return replace(seg, cluster_id=f"remote-{seg.cluster_id}")
        if self.relabel_unknown_channel:
            return replace(seg, cluster_id="unknown")
        return seg

    def override_with_rms(
        self,
        seg: DiarSegment,
        mic_rms_samples: Iterable[float],
        loopback_rms_samples: Iterable[float],
    ) -> DiarSegment:
        """Channel prior with extra per-frame RMS evidence covering the segment.

        If one channel dominates by `override_margin_db`, the segment's
        label is forced to that channel's identity, regardless of which
        channel the diarizer originally attributed it to.
        """
        mic = _rms(mic_rms_samples)
        loop = _rms(loopback_rms_samples)
        margin = self.override_margin_db

        if mic > 0 or loop > 0:
            if _db(mic, loop) >= margin:
                return replace(seg, channel=StreamId.MIC, cluster_id="self")
            if _db(loop, mic) >= margin:
                return replace(
                    seg, channel=StreamId.LOOPBACK, cluster_id=f"remote-{seg.cluster_id}"
                )

        return self.relabel(seg)
