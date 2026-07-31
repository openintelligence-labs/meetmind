"""Tests for compliance.dpia."""

from __future__ import annotations

from pathlib import Path

from meetmind.compliance.dpia import DpiaInputs, generate_dpia
from meetmind.memory.store import Store
from meetmind.models import Meeting


def test_generate_dpia_renders_required_sections(tmp_path: Path):
    db = tmp_path / "store.db"
    Store.open(db).close()
    md = generate_dpia(
        DpiaInputs(
            db_path=db,
            organization="Acme GmbH",
            controller_contact="dpo@acme.test",
        )
    )
    for header in (
        "# Data Protection Impact Assessment",
        "## 1. Purpose of processing",
        "## 2. Categories of data",
        "## 3. Lawful basis",
        "## 4. Recipients & third-party transfers",
        "## 5. Data subject rights",
        "## 6. Security measures",
        "## 7. Risks identified",
        "## 8. Configuration snapshot",
    ):
        assert header in md, f"missing section: {header}"
    assert "Acme GmbH" in md
    assert "dpo@acme.test" in md


def test_generate_dpia_reports_zero_meetings_for_empty_store(tmp_path: Path):
    db = tmp_path / "store.db"
    Store.open(db).close()
    md = generate_dpia(DpiaInputs(db_path=db))
    assert "Meetings: **0**" in md


def test_generate_dpia_counts_meetings_and_speakers(tmp_path: Path):
    db = tmp_path / "store.db"
    with Store.open(db) as store:
        store.upsert_meeting(Meeting(title="kickoff"))
        store.upsert_meeting(Meeting(title="weekly"))
    md = generate_dpia(DpiaInputs(db_path=db))
    assert "Meetings: **2**" in md


def test_generate_dpia_warns_when_provider_is_remote(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEETMIND_LLM_PROVIDER", "openai")
    db = tmp_path / "store.db"
    Store.open(db).close()
    md = generate_dpia(DpiaInputs(db_path=db))
    assert "Active hosted LLM" in md
    assert "openai" in md


def test_dpia_handles_missing_db(tmp_path: Path):
    md = generate_dpia(DpiaInputs(db_path=tmp_path / "nope.db"))
    assert "# Data Protection Impact Assessment" in md
