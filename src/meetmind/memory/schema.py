"""SQLCipher / SQLite schema (DDL only, no driver-specific code).

Schema version is pinned in `SCHEMA_VERSION`. Migrations are applied by
`apply_schema(conn)` which is idempotent — safe to call on every open.

All identifiers use ULIDs from the data model. Foreign keys cascade on
delete so `forget_speaker(speaker_id)` and `forget_meeting(meeting_id)`
are single-row operations that wipe everything downstream.
"""

from __future__ import annotations

from typing import Final

# Bumped on every additive change to SCHEMA_SQL. `MIGRATIONS` below carries
# the per-version upgrade DDL applied by `apply_schema()` on an existing DB.
SCHEMA_VERSION: Final[int] = 2


SCHEMA_SQL: Final[str] = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meetings (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    started_at          TEXT,
    ended_at            TEXT,
    duration_seconds    REAL,
    template            TEXT,
    calendar_event_id   TEXT,
    audio_path_mic      TEXT,
    audio_path_loopback TEXT,
    transcript_hash     BLOB,
    signature           BLOB,
    cost_usd            REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS speakers (
    id                          TEXT PRIMARY KEY,
    display_name                TEXT,
    consent_ts                  TEXT,
    consent_disclosure_version  TEXT,
    voiceprint_centroid         BLOB,
    voiceprint_ring             BLOB,  -- json-encoded list of bytes (base64) for now
    aliases                     BLOB,  -- json-encoded list of strings
    enrolled_at                 TEXT NOT NULL,
    confidence                  REAL DEFAULT 0,
    retention_until             TEXT
);

-- transcript_segments.speaker_id is a *cluster label* (e.g. "self",
-- "remote-A", or a real ULID once enrollment lands). It deliberately
-- has no FK; the resolver in voiceprint.py joins to speakers when an
-- identity exists. The diarizer emits only (start, end, speaker_id)
-- tuples and never holds onto the underlying voiceprint embedding.
CREATE TABLE IF NOT EXISTS transcript_segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id      TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    channel         TEXT,
    speaker_id      TEXT,
    text            TEXT NOT NULL,
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    confidence      REAL,
    language        TEXT DEFAULT 'en'
);

CREATE INDEX IF NOT EXISTS idx_segments_meeting   ON transcript_segments(meeting_id);
CREATE INDEX IF NOT EXISTS idx_segments_speaker   ON transcript_segments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_segments_time      ON transcript_segments(meeting_id, start_ms);

CREATE TABLE IF NOT EXISTS action_items (
    id                       TEXT PRIMARY KEY,
    meeting_id               TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    description              TEXT NOT NULL,
    owner_speaker_id         TEXT,  -- cluster label or speaker_id; no FK
    deadline                 TEXT,
    source_segment_id        INTEGER REFERENCES transcript_segments(id) ON DELETE SET NULL,
    evidence_quote           TEXT,
    status                   TEXT NOT NULL DEFAULT 'open',
    closed_in_meeting_id     TEXT REFERENCES meetings(id) ON DELETE SET NULL,
    closed_evidence_quote    TEXT
);

CREATE INDEX IF NOT EXISTS idx_action_items_meeting ON action_items(meeting_id);
CREATE INDEX IF NOT EXISTS idx_action_items_status  ON action_items(status);

CREATE TABLE IF NOT EXISTS decisions (
    id                  TEXT PRIMARY KEY,
    meeting_id          TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    decision            TEXT NOT NULL,
    rationale           TEXT,
    dissenters          BLOB,  -- json-encoded list of speaker_ids
    source_segment_ids  BLOB   -- json-encoded list of ints
);

CREATE INDEX IF NOT EXISTS idx_decisions_meeting ON decisions(meeting_id);

CREATE TABLE IF NOT EXISTS consent_events (
    id                   TEXT PRIMARY KEY,
    ts                   TEXT NOT NULL,
    actor_speaker_id     TEXT NOT NULL,
    action               TEXT NOT NULL,
    disclosure_version   TEXT NOT NULL,
    signature            BLOB
);

CREATE INDEX IF NOT EXISTS idx_consent_actor ON consent_events(actor_speaker_id);

CREATE TABLE IF NOT EXISTS summaries (
    meeting_id   TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    tl_dr        TEXT NOT NULL,
    topics_json  TEXT NOT NULL DEFAULT '[]',
    model        TEXT,
    created_at   TEXT NOT NULL
);

-- v2: speed the dashboard list query (ORDER BY started_at DESC).
CREATE INDEX IF NOT EXISTS idx_meetings_started ON meetings(started_at DESC);
"""


# Ordered migrations from (version - 1) → version. The runner picks up
# any version greater than the stored `schema_meta.schema_version` and
# applies them inside a single transaction. Migrations must be
# idempotent (use IF NOT EXISTS / IF EXISTS) so a partial-apply
# followed by a retry is safe.
MIGRATIONS: Final[dict[int, str]] = {
    # 1→2: add idx_meetings_started. Already created by SCHEMA_SQL on
    # fresh DBs but harmless to re-run on existing DBs via IF NOT EXISTS.
    2: "CREATE INDEX IF NOT EXISTS idx_meetings_started ON meetings(started_at DESC);",
}
