"""Encrypted persistent store (SQLCipher / sqlite-compatible).

`pysqlcipher3.dbapi2` is API-compatible with stdlib sqlite3, so the same DDL
and DAL run on either driver; without the `meetmind[encrypted]` extra the
store falls back to plain sqlite3. All persistence flows through `Store`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from meetmind.memory import schema as _schema
from meetmind.memory.schema import MIGRATIONS, SCHEMA_SQL, SCHEMA_VERSION  # noqa: F401
from meetmind.models import (
    ActionItem,
    ChannelKind,
    ConsentEvent,
    Decision,
    Meeting,
    MeetingTemplate,
    Speaker,
    TranscriptSegment,
)

log = logging.getLogger(__name__)


def open_connection(path: Path | str, *, passphrase: str | None = None) -> sqlite3.Connection:
    """Open a connection to the given DB path with the standard PRAGMAs.

    `passphrase` is forwarded to the SQLCipher driver when available; the
    stdlib driver ignores it and the database is left unencrypted.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(str(path), passphrase=passphrase)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL is incompatible with :memory: and some network filesystems; fall back
    # to truncate journaling if WAL can't be enabled rather than failing open.
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError as e:
        log.warning("WAL not available (%s); using truncate journal", e)
        conn.execute("PRAGMA journal_mode = TRUNCATE")
    conn.execute("PRAGMA synchronous = NORMAL")
    # Recorder, UI, and summarize can all hold the DB at once.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


_PYSQLCIPHER_WARNED = False  # One-shot guard so the hint logs once per process.


def _connect(path: str, *, passphrase: str | None) -> sqlite3.Connection:
    """Connect via SQLCipher when a passphrase is given and the driver is
    installed, else stdlib sqlite3. Either way the Connection is stdlib-shaped.
    """
    if passphrase:
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher  # noqa: PLC0415

            c = sqlcipher.connect(path, isolation_level=None)
            # PRAGMA key must be the first statement on the connection.
            quoted = passphrase.replace("'", "''")
            c.execute(f"PRAGMA key = '{quoted}'")
            # Reading sqlite_master raises here if the key is wrong.
            c.execute("SELECT count(*) FROM sqlite_master").fetchone()
            return c
        except ImportError:
            global _PYSQLCIPHER_WARNED  # noqa: PLW0603
            if not _PYSQLCIPHER_WARNED:
                _PYSQLCIPHER_WARNED = True
                log.warning(
                    "pysqlcipher3 not installed; opening DB unencrypted. "
                    "Install with: pip install 'meetmind[encrypted]'"
                )
        except Exception as e:
            log.error(
                "SQLCipher open failed (%s); refusing to fall back to unencrypted "
                "because a passphrase was supplied. Check the passphrase or remove it.",
                e,
            )
            raise
    return sqlite3.connect(path, isolation_level=None)


def is_encrypted(conn: sqlite3.Connection) -> bool:
    """Best-effort check: SQLCipher exposes `cipher_version`; stdlib doesn't."""
    try:
        row = conn.execute("PRAGMA cipher_version").fetchone()
    except sqlite3.DatabaseError:
        return False
    return bool(row and row[0])


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply DDL and run any pending forward migrations.

    Fresh DBs get `SCHEMA_SQL` and are stamped immediately; existing DBs run
    only the migrations they are missing.
    """
    conn.executescript(_schema.SCHEMA_SQL)
    current = _read_schema_version(conn)
    target = _schema.SCHEMA_VERSION
    if current >= target:
        # Either a brand-new DB or one written by a newer build. Never
        # silently downgrade: stamp at the target version and warn instead.
        if current > target:
            log.warning(
                "DB schema_version=%d is newer than this build (%d). "
                "Opening read-write anyway; new columns may be ignored.",
                current,
                target,
            )
        _stamp_version(conn, target)
        return
    pending = sorted(v for v in _schema.MIGRATIONS if current < v <= target)
    if not pending:
        _stamp_version(conn, target)
        return
    # The loop can't run inside one transaction: `executescript` commits
    # implicitly on the stdlib driver. Stamping per version instead leaves a
    # mid-sequence failure pointing at the last step that did apply.
    for version in pending:
        log.info("applying schema migration → v%d", version)
        conn.executescript(MIGRATIONS[version])
        _stamp_version(conn, version)


def _read_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.DatabaseError:
        return 0
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def _stamp_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(version)),
    )


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_iso(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _json_or_none(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _row_to_meeting(row: sqlite3.Row) -> Meeting:
    return Meeting(
        id=row["id"],
        title=row["title"],
        created_at=_parse_iso(row["created_at"]) or datetime.now(UTC),
        started_at=_parse_iso(row["started_at"]),
        ended_at=_parse_iso(row["ended_at"]),
        duration_seconds=row["duration_seconds"],
        template=MeetingTemplate(row["template"]) if row["template"] else None,
        calendar_event_id=row["calendar_event_id"],
        audio_path_mic=Path(row["audio_path_mic"]) if row["audio_path_mic"] else None,
        audio_path_loopback=(
            Path(row["audio_path_loopback"]) if row["audio_path_loopback"] else None
        ),
        transcript_hash=row["transcript_hash"],
        signature=row["signature"],
        cost_usd=row["cost_usd"] or 0.0,
    )


def _json_to_list(blob: bytes | str | None) -> list:
    if blob is None:
        return []
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    return json.loads(blob)


class Store:
    """Thin DAL over a sqlite3 connection.

    Use `Store.open(path)` as a context manager for a self-contained scope, or
    construct it with an already-open connection to share one.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @classmethod
    def open(
        cls,
        path: Path | str,
        *,
        passphrase: str | None = None,
        use_keychain: bool = True,
    ) -> Store:
        """Open the store, applying schema migrations.

        An explicit ``passphrase`` wins; otherwise the keychain DEK is used
        when ``use_keychain`` is set, and failing that the DB opens
        unencrypted. Tests pass ``use_keychain=False`` to stay hermetic.
        """
        if passphrase is None and use_keychain:
            from meetmind.memory.keyring import get_or_create_dek  # noqa: PLC0415

            passphrase = get_or_create_dek()
        conn = open_connection(path, passphrase=passphrase)
        apply_schema(conn)
        return cls(conn)

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self.conn.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.conn.execute("BEGIN")
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # -- Meetings -----------------------------------------------------------

    def upsert_meeting(self, m: Meeting) -> None:
        # Not INSERT OR REPLACE: REPLACE deletes first, cascading through every
        # FK referencing meetings(id) and silently dropping the child rows.
        self.conn.execute(
            """
            INSERT INTO meetings (
                id, title, created_at, started_at, ended_at, duration_seconds,
                template, calendar_event_id, audio_path_mic, audio_path_loopback,
                transcript_hash, signature, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title               = excluded.title,
                started_at          = excluded.started_at,
                ended_at            = excluded.ended_at,
                duration_seconds    = excluded.duration_seconds,
                template            = excluded.template,
                calendar_event_id   = excluded.calendar_event_id,
                audio_path_mic      = excluded.audio_path_mic,
                audio_path_loopback = excluded.audio_path_loopback,
                transcript_hash     = excluded.transcript_hash,
                signature           = excluded.signature,
                cost_usd            = excluded.cost_usd
            """,
            (
                m.id,
                m.title,
                _iso(m.created_at),
                _iso(m.started_at),
                _iso(m.ended_at),
                m.duration_seconds,
                m.template.value if m.template is not None else None,
                m.calendar_event_id,
                str(m.audio_path_mic) if m.audio_path_mic else None,
                str(m.audio_path_loopback) if m.audio_path_loopback else None,
                m.transcript_hash,
                m.signature,
                m.cost_usd,
            ),
        )

    def get_meeting(self, meeting_id: str) -> Meeting | None:
        row = self.conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if row is None:
            return None
        return _row_to_meeting(row)

    def list_meetings(self, *, limit: int | None = None) -> list[Meeting]:
        sql = "SELECT * FROM meetings ORDER BY COALESCE(started_at, created_at) DESC, id DESC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_meeting(r) for r in rows]

    def forget_meeting(self, meeting_id: str) -> None:
        """Cascading delete of the meeting and all its children."""
        with self.transaction():
            self.conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))

    # -- Segments -----------------------------------------------------------

    def append_segment(self, meeting_id: str, seg: TranscriptSegment) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO transcript_segments (
                meeting_id, channel, speaker_id, text, start_ms, end_ms,
                confidence, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meeting_id,
                seg.channel.value if seg.channel is not None else None,
                seg.speaker_id,
                seg.text,
                seg.start_ms,
                seg.end_ms,
                seg.confidence,
                seg.language,
            ),
        )
        return cur.lastrowid or 0

    def list_segments(self, meeting_id: str) -> list[TranscriptSegment]:
        rows = self.conn.execute(
            "SELECT * FROM transcript_segments WHERE meeting_id = ? ORDER BY start_ms",
            (meeting_id,),
        ).fetchall()
        return [
            TranscriptSegment(
                start_seconds=row["start_ms"] / 1000.0,
                end_seconds=row["end_ms"] / 1000.0,
                speaker=row["speaker_id"],
                text=row["text"],
                channel=ChannelKind(row["channel"]) if row["channel"] else None,
                speaker_id=row["speaker_id"],
                confidence=row["confidence"],
                language=row["language"] or "en",
            )
            for row in rows
        ]

    # -- Speakers + consent -------------------------------------------------

    def upsert_speaker(self, sp: Speaker) -> None:
        # ON CONFLICT rather than REPLACE — see `upsert_meeting`.
        self.conn.execute(
            """
            INSERT INTO speakers (
                id, display_name, consent_ts, consent_disclosure_version,
                voiceprint_centroid, voiceprint_ring, aliases, enrolled_at,
                confidence, retention_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name               = excluded.display_name,
                consent_ts                 = excluded.consent_ts,
                consent_disclosure_version = excluded.consent_disclosure_version,
                voiceprint_centroid        = excluded.voiceprint_centroid,
                voiceprint_ring            = excluded.voiceprint_ring,
                aliases                    = excluded.aliases,
                enrolled_at                = excluded.enrolled_at,
                confidence                 = excluded.confidence,
                retention_until            = excluded.retention_until
            """,
            (
                sp.id,
                sp.display_name,
                _iso(sp.consent_ts),
                sp.consent_disclosure_version,
                sp.voiceprint_centroid,
                _json_or_none(
                    [v.hex() if isinstance(v, bytes | bytearray) else v for v in sp.voiceprint_ring]
                ),
                _json_or_none(sp.aliases),
                _iso(sp.enrolled_at),
                sp.confidence,
                sp.retention_until.isoformat() if sp.retention_until is not None else None,
            ),
        )

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        row = self.conn.execute("SELECT * FROM speakers WHERE id = ?", (speaker_id,)).fetchone()
        if row is None:
            return None
        ring_raw = _json_to_list(row["voiceprint_ring"])
        return Speaker(
            id=row["id"],
            display_name=row["display_name"],
            consent_ts=_parse_iso(row["consent_ts"]),
            consent_disclosure_version=row["consent_disclosure_version"],
            voiceprint_centroid=row["voiceprint_centroid"],
            voiceprint_ring=[bytes.fromhex(x) if isinstance(x, str) else x for x in ring_raw],
            aliases=_json_to_list(row["aliases"]),
            enrolled_at=_parse_iso(row["enrolled_at"]) or datetime.now(UTC),
            confidence=row["confidence"] or 0.0,
            retention_until=date.fromisoformat(row["retention_until"])
            if row["retention_until"]
            else None,
        )

    def append_consent_event(self, event: ConsentEvent) -> None:
        self.conn.execute(
            """
            INSERT INTO consent_events (
                id, ts, actor_speaker_id, action, disclosure_version, signature
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                _iso(event.ts),
                event.actor_speaker_id,
                event.action,
                event.disclosure_version,
                event.signature,
            ),
        )

    def list_consent_events(self, actor_speaker_id: str) -> list[ConsentEvent]:
        rows = self.conn.execute(
            "SELECT * FROM consent_events WHERE actor_speaker_id = ? ORDER BY ts",
            (actor_speaker_id,),
        ).fetchall()
        return [
            ConsentEvent(
                id=r["id"],
                ts=_parse_iso(r["ts"]) or datetime.now(UTC),
                actor_speaker_id=r["actor_speaker_id"],
                action=r["action"],
                disclosure_version=r["disclosure_version"],
                signature=r["signature"],
            )
            for r in rows
        ]

    def forget_speaker(self, speaker_id: str) -> None:
        """Delete the speaker; ON DELETE SET NULL detaches segments + actions.

        Consent events are *retained* for audit accountability — proof
        of prior consent must outlive the speaker row (GDPR Art. 7(1)).
        """
        with self.transaction():
            self.conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))

    # -- Action items -------------------------------------------------------

    def upsert_action_item(self, meeting_id: str, item: ActionItem) -> None:
        self.conn.execute(
            """
            INSERT INTO action_items (
                id, meeting_id, description, owner_speaker_id, deadline,
                source_segment_id, evidence_quote, status, closed_in_meeting_id,
                closed_evidence_quote
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                meeting_id            = excluded.meeting_id,
                description           = excluded.description,
                owner_speaker_id      = excluded.owner_speaker_id,
                deadline              = excluded.deadline,
                source_segment_id     = excluded.source_segment_id,
                evidence_quote        = excluded.evidence_quote,
                status                = excluded.status,
                closed_in_meeting_id  = excluded.closed_in_meeting_id,
                closed_evidence_quote = excluded.closed_evidence_quote
            """,
            (
                item.id,
                meeting_id,
                item.description,
                item.owner,
                item.due,
                item.source_segment_id,
                item.evidence_quote,
                item.status,
                item.closed_in_meeting_id,
                item.closed_evidence_quote,
            ),
        )

    def list_action_items(
        self,
        *,
        status: str | None = None,
        meeting_id: str | None = None,
        owner: str | None = None,
    ) -> list[ActionItem]:
        sql = "SELECT * FROM action_items WHERE 1=1"
        args: list[Any] = []
        if status:
            sql += " AND status = ?"
            args.append(status)
        if meeting_id:
            sql += " AND meeting_id = ?"
            args.append(meeting_id)
        if owner:
            sql += " AND owner_speaker_id = ?"
            args.append(owner)
        sql += " ORDER BY id"
        rows = self.conn.execute(sql, args).fetchall()
        return [
            ActionItem(
                id=r["id"],
                description=r["description"],
                owner=r["owner_speaker_id"],
                due=r["deadline"],
                source_segment_id=r["source_segment_id"],
                evidence_quote=r["evidence_quote"],
                status=r["status"],
                closed_in_meeting_id=r["closed_in_meeting_id"],
                closed_evidence_quote=r["closed_evidence_quote"],
            )
            for r in rows
        ]

    # -- Decisions ----------------------------------------------------------

    def upsert_decision(self, meeting_id: str, dec: Decision) -> None:
        self.conn.execute(
            """
            INSERT INTO decisions (
                id, meeting_id, decision, rationale, dissenters, source_segment_ids
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                meeting_id         = excluded.meeting_id,
                decision           = excluded.decision,
                rationale          = excluded.rationale,
                dissenters         = excluded.dissenters,
                source_segment_ids = excluded.source_segment_ids
            """,
            (
                dec.id,
                meeting_id,
                dec.decision,
                dec.rationale,
                _json_or_none(dec.dissenters),
                _json_or_none(dec.source_segment_ids),
            ),
        )

    def list_decisions(self, meeting_id: str) -> list[Decision]:
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE meeting_id = ?", (meeting_id,)
        ).fetchall()
        return [
            Decision(
                id=r["id"],
                decision=r["decision"],
                rationale=r["rationale"] or "",
                dissenters=_json_to_list(r["dissenters"]),
                source_segment_ids=_json_to_list(r["source_segment_ids"]),
            )
            for r in rows
        ]

    def list_all_decisions(self, *, limit: int = 200) -> list[tuple[str, Decision]]:
        """Return the most recent decisions across all meetings.

        Yields ``(meeting_id, Decision)`` pairs so callers can group or link
        back without a second query.
        """
        rows = self.conn.execute(
            "SELECT * FROM decisions ORDER BY rowid DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [
            (
                r["meeting_id"],
                Decision(
                    id=r["id"],
                    decision=r["decision"],
                    rationale=r["rationale"] or "",
                    dissenters=_json_to_list(r["dissenters"]),
                    source_segment_ids=_json_to_list(r["source_segment_ids"]),
                ),
            )
            for r in rows
        ]

    # -- Summaries ----------------------------------------------------------

    def upsert_summary(
        self,
        meeting_id: str,
        *,
        tl_dr: str,
        topics: list[str] | None = None,
        model: str | None = None,
    ) -> None:
        """Persist the summary for a meeting, overwriting any prior one."""
        topics_json = json.dumps(topics or [])
        self.conn.execute(
            """
            INSERT INTO summaries (meeting_id, tl_dr, topics_json, model, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(meeting_id) DO UPDATE SET
                tl_dr       = excluded.tl_dr,
                topics_json = excluded.topics_json,
                model       = excluded.model,
                created_at  = excluded.created_at
            """,
            (
                meeting_id,
                tl_dr,
                topics_json,
                model,
                datetime.now(UTC).isoformat(),
            ),
        )

    def get_summary(self, meeting_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT tl_dr, topics_json, model, created_at FROM summaries WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "tl_dr": row["tl_dr"],
            "topics": _json_to_list(row["topics_json"]),
            "model": row["model"],
            "created_at": row["created_at"],
        }
