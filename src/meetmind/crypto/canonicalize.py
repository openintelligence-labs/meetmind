"""Deterministic canonicalization of transcript objects into signable bytes.

RFC 8785 / I-JSON style: recursively sorted keys, no incidental whitespace,
UTF-8 with ``ensure_ascii=False``. Default JSON serialization is not enough,
since key ordering, whitespace and non-ASCII handling all vary between runs
and machines, and a signed bundle must re-hash identically anywhere.

Callers should round floats and pre-serialize datetimes and enums.
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
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "as_posix"):
        return value.as_posix()
    return str(value)


def canonicalize_meeting_for_signing(meeting_dict: dict[str, Any]) -> dict[str, Any]:
    """Project a meeting dict to just the fields the signature covers.

    Excludes mutable and derived fields, and the signature itself.
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
