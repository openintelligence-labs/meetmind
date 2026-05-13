"""Deterministic canonicalization of transcript objects → bytes for signing.

We need bit-identical output across runs and across machines so that
verifying a signed bundle always reproduces the same hash. JSON's
default serialization is not sufficient (key ordering, whitespace,
non-ASCII handling all vary).

Approach:
  • RFC 8785 / I-JSON style: sort keys recursively, no extra whitespace,
    UTF-8 with `ensure_ascii=False` and explicit (',', ':') separators.
  • Floats: pin to fixed precision (we don't have any in the schema we
    sign, but if added, callers should round before passing).
  • Path / datetime / Enum: caller is responsible for serializing those
    to plain str/int before passing.

`canonicalize_meeting` is the bundle-specific helper that picks exactly
the fields a signature must cover.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Return a deterministic UTF-8 JSON encoding."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_default_serializer,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _default_serializer(value: Any) -> Any:
    # Last-resort fallback for things json can't natively encode.
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "as_posix"):
        return value.as_posix()
    return str(value)


def canonicalize_meeting_for_signing(meeting_dict: dict[str, Any]) -> dict[str, Any]:
    """Project a meeting dict to just the fields the signature covers.

    Excludes mutable / derived / signature fields (the signature can't
    cover itself). Locked schema for v0.13.
    """
    keep = {
        "id",
        "title",
        "started_at",
        "ended_at",
        "duration_seconds",
        "template",
        "calendar_event_id",
        "transcript",
        "decisions",
        "summary",
    }
    return {k: meeting_dict[k] for k in sorted(keep) if k in meeting_dict}
