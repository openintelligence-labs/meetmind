"""Tests for the Obsidian integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from meetmind.integrations.obsidian import export_meeting
from meetmind.memory.store import Store
from meetmind.models import (
    ActionItem,
    ChannelKind,
    Decision,
    Meeting,
    TranscriptSegment,
)


@pytest.fixture
def populated_store(tmp_path: Path) -> Store:
    store = Store.open(tmp_path / "store.db")
    m = Meeting(id="01OBS", title="Snowflake migration kickoff")
    store.upsert_meeting(m)
    store.append_segment(
        m.id,
        TranscriptSegment(
            start_seconds=0.0,
            end_seconds=4.0,
            text="Sam: I'll send the deck Friday.",
            channel=ChannelKind.MIC,
            speaker_id="remote-A",
        ),
    )
    store.upsert_action_item(
        m.id,
        ActionItem(
            description="Send the migration deck",
            owner="remote-A",
            due="2026-05-08",
            evidence_quote="I'll send the deck Friday.",
            status="open",
        ),
    )
    store.upsert_decision(
        m.id, Decision(decision="Adopt LanceDB", rationale="Better at 1M+ vectors")
    )
    return store


def test_export_writes_markdown_with_frontmatter(populated_store, tmp_path):
    vault = tmp_path / "vault"
    result = export_meeting(populated_store, "01OBS", vault=vault)

    assert result.note_path.exists()
    text = result.note_path.read_text("utf-8")
    assert text.startswith("---\n")
    assert "meeting_id: 01OBS" in text
    assert "tags:" in text
    assert "  - meetmind" in text
    assert "## Decisions" in text
    assert "Adopt LanceDB" in text
    assert "## Action items" in text
    assert "Send the migration deck" in text
    assert "## Transcript" in text


def test_export_creates_subfolder(populated_store, tmp_path):
    vault = tmp_path / "vault"
    result = export_meeting(populated_store, "01OBS", vault=vault, folder="Notes/Meetings")
    assert result.note_path.parent == vault / "Notes" / "Meetings"


def test_export_refuses_overwrite_by_default(populated_store, tmp_path):
    vault = tmp_path / "vault"
    export_meeting(populated_store, "01OBS", vault=vault)
    with pytest.raises(FileExistsError):
        export_meeting(populated_store, "01OBS", vault=vault)


def test_export_overwrite_replaces(populated_store, tmp_path):
    vault = tmp_path / "vault"
    first = export_meeting(populated_store, "01OBS", vault=vault)
    first.note_path.write_text("clobbered")
    second = export_meeting(populated_store, "01OBS", vault=vault, overwrite=True)
    assert "clobbered" not in second.note_path.read_text("utf-8")


def test_unknown_meeting_raises(populated_store, tmp_path):
    with pytest.raises(LookupError):
        export_meeting(populated_store, "01NOPE", vault=tmp_path / "vault")
