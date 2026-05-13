"""Tests for voiceprint enrollment + consent flow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meetmind.crypto.identity import Identity
from meetmind.diarize.enroll import (
    CURRENT_DISCLOSURE_VERSION,
    DEFAULT_RETENTION_DAYS,
    consent_event_to_summary,
    enroll,
    export_event,
    forget,
    revoke,
    speaker_to_summary,
    verify_consent_event,
)
from meetmind.diarize.matcher import _decode_centroid
from meetmind.memory.store import Store


def _embed(seed: int, dim: int = 192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Store, Identity]:
    store = Store.open(tmp_path / "audit.db")
    identity = Identity.generate()
    return store, identity


def test_enroll_creates_speaker_with_centroid_and_audit_event(fixture):
    store, identity = fixture
    result = enroll(name="Sam Chen", embedding=_embed(1), store=store, identity=identity)
    assert result.speaker.display_name == "Sam Chen"
    assert result.speaker.consent_ts is not None
    assert result.speaker.consent_disclosure_version == CURRENT_DISCLOSURE_VERSION
    assert _decode_centroid(result.speaker) is not None
    assert result.speaker.retention_until is not None

    event = result.event
    assert event.action == "enroll"
    assert event.actor_speaker_id == result.speaker.id
    assert event.signature is not None
    assert verify_consent_event(event, identity) is True


def test_consent_event_signature_rejects_other_identities(fixture):
    store, identity = fixture
    result = enroll(name="Sam", embedding=_embed(1), store=store, identity=identity)
    other = Identity.generate()
    assert verify_consent_event(result.event, other) is False


def test_revoke_clears_voiceprint_keeps_speaker_row(fixture):
    store, identity = fixture
    result = enroll(name="Sam", embedding=_embed(1), store=store, identity=identity)
    revoke(speaker_id=result.speaker.id, store=store, identity=identity)
    sp = store.get_speaker(result.speaker.id)
    assert sp is not None
    assert sp.voiceprint_centroid is None
    assert sp.consent_ts is None
    events = store.list_consent_events(result.speaker.id)
    assert [e.action for e in events] == ["enroll", "revoke"]


def test_forget_deletes_speaker_but_retains_audit_log(fixture):
    store, identity = fixture
    result = enroll(name="Priya", embedding=_embed(2), store=store, identity=identity)
    forget(speaker_id=result.speaker.id, store=store, identity=identity)
    assert store.get_speaker(result.speaker.id) is None
    events = store.list_consent_events(result.speaker.id)
    assert [e.action for e in events] == ["enroll", "delete"]
    assert all(verify_consent_event(e, identity) for e in events)


def test_export_logs_a_signed_event(fixture):
    store, identity = fixture
    result = enroll(name="Sam", embedding=_embed(1), store=store, identity=identity)
    event = export_event(speaker_id=result.speaker.id, store=store, identity=identity)
    assert event.action == "export"
    assert verify_consent_event(event, identity) is True


def test_revoke_on_unknown_speaker_raises(fixture):
    store, identity = fixture
    with pytest.raises(ValueError):
        revoke(speaker_id="01NOPE", store=store, identity=identity)


def test_forget_on_unknown_speaker_raises(fixture):
    store, identity = fixture
    with pytest.raises(ValueError):
        forget(speaker_id="01NOPE", store=store, identity=identity)


def test_speaker_to_summary_excludes_centroid_bytes(fixture):
    store, identity = fixture
    result = enroll(name="Sam", embedding=_embed(1), store=store, identity=identity)
    summary = speaker_to_summary(result.speaker)
    assert summary["display_name"] == "Sam"
    assert summary["has_voiceprint"] is True
    assert summary["ring_size"] >= 1
    assert "voiceprint_centroid" not in summary


def test_consent_event_summary_round_trip(fixture):
    store, identity = fixture
    result = enroll(name="Sam", embedding=_embed(1), store=store, identity=identity)
    summary = consent_event_to_summary(result.event)
    assert summary["action"] == "enroll"
    assert summary["signed"] is True
    assert summary["actor_speaker_id"] == result.speaker.id


def test_default_retention_window_is_one_year(fixture):
    store, identity = fixture
    result = enroll(name="Sam", embedding=_embed(1), store=store, identity=identity)
    sp = result.speaker
    assert sp.retention_until is not None
    today = result.event.ts.date()
    delta_days = (sp.retention_until - today).days
    assert delta_days == DEFAULT_RETENTION_DAYS


def test_re_enrollment_requires_new_consent_event(fixture):
    store, identity = fixture
    a = enroll(name="Sam", embedding=_embed(1), store=store, identity=identity)
    revoke(speaker_id=a.speaker.id, store=store, identity=identity)
    b = enroll(name="Sam (new)", embedding=_embed(7), store=store, identity=identity)
    assert b.speaker.id != a.speaker.id
    assert b.event.action == "enroll"
