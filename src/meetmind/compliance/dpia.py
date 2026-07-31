"""DPIA (Data Protection Impact Assessment) generator.

Introspects an install and emits a Markdown DPIA covering what data is
stored, where, for how long, under what lawful basis, and with which
recipients. No LLM and no network: it reads the on-disk store, the
environment, and the configured providers.

The output captures the install's technical state as a starting template;
it is not a substitute for legal review of the policy fields.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import meetmind
from meetmind.analyze.llm import llm_config_summary


@dataclass
class DpiaInputs:
    """What the generator needs to know. Caller fills these in."""

    db_path: Path
    install_dir: Path | None = None
    organization: str = "[organization name]"
    controller_contact: str = "[data protection officer or controller email]"
    retention_days_meetings: int = 1095  # 3y default
    retention_days_voiceprints: int = 365  # 1y default
    purpose: str = "Internal meeting summarization, action-item extraction, and search."
    lawful_basis: str = (
        "Legitimate interest (Art. 6(1)(f) GDPR) + explicit consent for voiceprints (Art. 9(2)(a))."
    )
    third_country_transfers: str = "None when the local-Ollama default is used."


def _store_inventory(db_path: Path) -> dict[str, Any]:
    """Best-effort read of the local store for counts; tolerates missing DB."""
    if not db_path.exists():
        return {"db_present": False, "meetings": 0, "speakers": 0, "consent_events": 0}
    from meetmind.memory.store import Store  # noqa: PLC0415

    try:
        with Store.open(db_path) as store:
            meetings = store.list_meetings(limit=10000)
            speakers = store.conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
            consent = store.conn.execute("SELECT COUNT(*) FROM consent_events").fetchone()[0]
        return {
            "db_present": True,
            "meetings": len(meetings),
            "speakers": int(speakers),
            "consent_events": int(consent),
        }
    except Exception:  # noqa: BLE001
        return {"db_present": True, "error": "could not read store"}


def generate_dpia(inputs: DpiaInputs) -> str:
    """Render the Markdown DPIA. Pure: no I/O beyond what `inputs` references."""
    cfg = llm_config_summary()
    inv = _store_inventory(inputs.db_path)
    now = datetime.now(UTC).date().isoformat()
    install_dir = inputs.install_dir or inputs.db_path.parent

    transfers_section = inputs.third_country_transfers
    if cfg["provider"] != "ollama":
        transfers_section = (
            f"**Active hosted LLM**: provider=`{cfg['provider']}`. "
            "Prompted text (transcripts, action items) leaves the device when "
            "this provider is invoked. Audio + voiceprints + the SQLCipher "
            "archive remain local. Configure via `MEETMIND_LLM_PROVIDER`."
        )

    return f"""# Data Protection Impact Assessment — MeetMind

> Generated: **{now}**  ·  MeetMind version: **{meetmind.__version__}**
> Host: `{platform.platform()}` · Install dir: `{install_dir}`
> Organization: **{inputs.organization}**
> Controller contact: **{inputs.controller_contact}**

This document is a **technical snapshot** of a MeetMind install for use
in a GDPR Article 35 DPIA. It is not legal advice. Combine it with
your organization's policy review before signing.

## 1. Purpose of processing

{inputs.purpose}

## 2. Categories of data

| Category | Source | Storage | Retention |
|---|---|---|---|
| Audio recordings | System audio capture (loopback) + microphone | `{install_dir}/audio/` (when persistence enabled) | {inputs.retention_days_meetings} days, configurable |
| Transcripts | Local STT (Parakeet / Whisper.cpp) | SQLCipher DB at `{inputs.db_path}` | {inputs.retention_days_meetings} days |
| Action items + decisions | Local LLM extraction (default Ollama) | SQLCipher DB | {inputs.retention_days_meetings} days |
| Speaker voiceprints | Voiceprint embedder | SQLCipher DB (`speakers.voiceprint_centroid`) | {inputs.retention_days_voiceprints} days OR until consent withdrawn |
| Consent events | User opt-in / opt-out actions | Append-only audit log; survives speaker deletion (tombstoned) | Indefinite (Art. 7(1) GDPR proof-of-consent) |
| Live transcript stream | In-process pub/sub | Memory only (not persisted) | Until process exits |

Current store contents:
- Meetings: **{inv.get("meetings", "?")}**
- Speakers: **{inv.get("speakers", "?")}**
- Consent events: **{inv.get("consent_events", "?")}**

Voiceprints are special-category biometric data under **GDPR Art. 9** and
fall under **BIPA / CUBI** in the United States. MeetMind treats them as
opt-in only; consent is captured before any voiceprint is created and
logged immutably as a `ConsentEvent`.

## 3. Lawful basis (Art. 6 / Art. 9)

{inputs.lawful_basis}

## 4. Recipients & third-party transfers

{transfers_section}

When the LLM provider is `ollama` (the default), the address is
`{cfg["base_url"]}` — by default a loopback URL on the same host. No
data leaves the device under default operation.

The HTTP API server binds **127.0.0.1 only** and is enforced at the
`uvicorn` config level. It cannot be exposed on 0.0.0.0 without
modifying source.

## 5. Data subject rights (Arts. 15–22)

| Right | How MeetMind supports it |
|---|---|
| Access (Art. 15) | `meetmind meetings`, `meetmind get-meeting <id>` — full export as JSON / Markdown bundle. |
| Rectification (Art. 16) | Direct edit of SQLCipher DB or via re-summarization. |
| Erasure (Art. 17) | `meetmind forget <speaker-id>` cascades through every FK; `meetmind compliance retention-sweep` enforces TTLs. ConsentEvent is retained as a tombstone for proof. |
| Restriction (Art. 18) | Recording can be paused / disabled per meeting via `--no-store`. |
| Portability (Art. 20) | `meetmind export-bundle <id>` — Ed25519-signed tar.gz with canonical JSON manifest. |
| Object (Art. 21) | Stop opting in, run the erasure cascade. |
| Automated decision-making (Art. 22) | None — extraction is suggestion-only and surfaced for human review. |

## 6. Security measures (Art. 32)

- **Encryption at rest**: SQLCipher AES-256 (DEK wrapped by OS keychain).
- **Encryption in transit**: API binds loopback; bearer-token auth.
- **Access control**: per-launch ephemeral bearer token written
  `~/.meetmind/token` mode 0600.
- **Bundle integrity**: Ed25519 signatures over canonical JSON
  (RFC 8785 style) for any exported bundle.
- **Minimum-data**: only signed-off meetings + opted-in voiceprints
  retained.

## 7. Risks identified

| Risk | Mitigation |
|---|---|
| Local-host compromise → DB read | SQLCipher AES-256 + OS-keychain DEK wrap; cold-storage backups crypto-shred via DEK wipe. |
| Voiceprint leakage | Stored only with explicit consent; centroid is a 192-d L2-normalized embedding (not a recoverable representation of speech). |
| Hosted-LLM accidental enablement | Default is local Ollama; switching providers requires explicit env var; `meetmind status` prints the active provider. |
| Outbound exfiltration | CI test (`tests/security/test_no_outbound_calls.py`) verifies no non-loopback connect occurs under default config. |

## 8. Configuration snapshot

```
LLM provider     : {cfg["provider"]}
LLM model        : {cfg["model"] or "(actants default)"}
LLM base URL     : {cfg["base_url"]}
Has API key set  : {cfg["has_api_key"]}
Database path    : {inputs.db_path}
Token file       : {Path.home() / ".meetmind" / "token"} (mode 0600)
```

---

*Generated by `meetmind compliance dpia`. Combine with your DPO's
review and your organization's standard DPIA template.*
"""


def default_db_path() -> Path:
    """Same default the CLI uses: ``$MEETMIND_HOME/data/meetmind.db``,
    falling back to ``~/.meetmind/data/meetmind.db``."""
    base = Path(os.environ.get("MEETMIND_HOME", str(Path.home() / ".meetmind")))
    return base / "data" / "meetmind.db"
