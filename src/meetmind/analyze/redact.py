"""Regex redaction profiles for transcripts.

``raw`` passes text through, ``team_internal`` strips emails/phones/cards but
keeps names, and ``public_share`` also strips person names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["raw", "team_internal", "public_share"]


@dataclass
class RedactionResult:
    text: str
    redaction_count: int
    profile: ProfileName


_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# Conservative phone matcher — international E.164-ish and US 10-digit.
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
# 13–19 digit groups with optional dashes/spaces, anchored on word boundaries.
_CREDIT = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# Person-name heuristic: 2+ capitalized tokens. Misses lowercase names.
_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def redact(text: str, *, profile: ProfileName = "team_internal") -> RedactionResult:
    """Apply a redaction profile to ``text``. Pure function."""
    if profile == "raw":
        return RedactionResult(text=text, redaction_count=0, profile=profile)

    n = 0
    out = text

    out, c = _EMAIL.subn("[email]", out)
    n += c
    # Card must run before phone: both match long digit runs, and the more
    # specific card pattern (13-19 digits) has to claim its matches first.
    out, c = _CREDIT.subn("[card]", out)
    n += c
    out, c = _PHONE.subn("[phone]", out)
    n += c

    if profile == "public_share":
        out, c = _NAME.subn("[name]", out)
        n += c

    return RedactionResult(text=out, redaction_count=n, profile=profile)


def available_profiles() -> list[ProfileName]:
    return ["raw", "team_internal", "public_share"]
