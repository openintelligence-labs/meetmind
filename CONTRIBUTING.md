# Contributing to MeetMind

Thanks for considering a contribution. MeetMind is a small,
opinionated codebase — the README anti-features list is a contract
with users, and PRs that violate it will not be merged. Read both
before opening a large change.

## Quick start

```bash
git clone https://github.com/openintelligence-labs/meetmind
cd meetmind
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,api,audio,storage]'
pytest -q
ruff check src tests
lint-imports --config pyproject.toml
```

If those three commands are green you have a working dev environment.
`meetmind selftest` is a one-shot smoke test that exercises the
storage, LLM config, integrations, and end-to-end paths without
network — paste its `--json` output into any GitHub issue.

## What we accept

**Yes — please send these:**
- Bug fixes with a regression test (one test per fix).
- Sidecar binaries for Linux / Windows that match the IPC protocol.
- Real translations for `src/meetmind/i18n/_strings_de.py` and
  `_strings_fr.py` (currently English placeholders).
- Performance improvements with a `scripts/bench_*.py` number to back
  them up.
- New integrations as leaf modules in `src/meetmind/integrations/`.
  Each must satisfy the module-boundary linter (no imports from
  capture/stt/diarize/analyze/memory/api/assist — pass `_StoreLike` via
  structural typing, like `obsidian.py` and `slack.py` do).
- MCP tool additions in `src/meetmind/api/mcp_server.py` with tests.

**No — please don't send these:**
- Telemetry, error reporting, usage analytics. The codebase has no
  telemetry call to disable because it doesn't exist; keep it that way.
- Network calls in the default code path. New egress must be triggered
  by an explicit user choice and must not break the
  `tests/security/test_no_outbound_calls.py` guarantee.
- Code that captures, stores, or routes voiceprints without going
  through the existing `ConsentEvent` audit log.
- Backends that send audio or transcripts to a hosted service. The
  audio/transcripts/voiceprints/archive must stay on the device; only
  LLM-prompted text is allowed to egress, and only with opt-in.
- Large refactors without a corresponding issue + design discussion.
- Anything that breaks the import-linter contracts in `pyproject.toml`
  (capture/stt/diarize/analyze/integrations/crypto leaves).

## Development workflow

1. **Open an issue first** for non-trivial changes — saves you from
   building something we'd reject for architectural reasons.
2. **Branch from `main`**, name it `kind/short-description`
   (`fix/wal-corruption`, `feat/notion-export`).
3. **Write the test first** if it's a bug fix. We don't merge bug
   fixes without a failing-then-passing test.
4. **Run the gates locally** before pushing:
   ```bash
   pytest -q
   ruff check src tests
   ruff format --check src tests
   lint-imports --config pyproject.toml
   ```
5. **CHANGELOG.md** gets an entry under the `## [Unreleased]` section
   (until v1.0 ships), in present tense ("Add Slack export", not
   "Added Slack export"). After v1.0 ships, new entries go under a
   new `## [x.y.z]` heading dated for the release.
6. **Open the PR**, link the issue.

## What "Definition of Done" looks like

A PR is mergeable when:
- [ ] All existing tests pass.
- [ ] ≥ 1 happy-path test + ≥ 1 edge-case test for new behavior.
- [ ] `ruff check` and `ruff format --check` are clean.
- [ ] `lint-imports` reports all 6 contracts kept.
- [ ] No new outbound network call in the default code path
      (`test_no_outbound_calls.py` still green).
- [ ] Public API / CLI changes are reflected in `README.md`.
- [ ] CHANGELOG entry under the new version.

## Architecture

The README has a 1-page architecture diagram. For deeper context,
read these before non-trivial changes:

- **Module map**: `pyproject.toml` `[tool.importlinter]` contracts.
- **Storage**: `src/meetmind/memory/schema.py` is the single source of
  truth. Schema bumps need a migration in `MIGRATIONS` and a test in
  `tests/memory/test_migrations.py`.
- **IPC**: `src/meetmind/ipc/protocol.py` length-prefixed binary frames.
  Sidecars in `sidecars/macos/` consume the same wire format.
- **LLM transport**: `src/meetmind/analyze/llm.py` forwards
  `MEETMIND_LLM_*` env vars to `ACTANTS_*`. Don't add per-provider
  code paths here — that's `actants`' job.

## Code style

- Python 3.12+, type-annotated.
- 100-char lines (`ruff` enforced).
- No emojis in code unless explicitly asked.
- Comments explain **why** something is non-obvious, not **what** the
  code does. Don't leave PR-history references in comments
  ("added for issue #123") — those belong in the commit message.
- Avoid `# type: ignore` and `noqa` unless the diagnostic is a known
  false positive; if you use one, add a one-line comment saying why.

## Tests

- New code lives next to its mirror in `tests/`: `src/meetmind/foo/bar.py`
  → `tests/foo/test_bar.py`.
- Tests must be hermetic: no real keychain (`use_keychain=False` on
  `Store.open`, or rely on the `MEETMIND_DISABLE_ENCRYPTION=1` set in
  `tests/conftest.py`), no real network, no real OS audio devices.
- Async tests use the `asyncio_mode = "auto"` config in
  `pyproject.toml` — declare them with `async def` and they'll run.

## Releasing (maintainers only)

The first public release will be **v1.0.0**. Until then the version
stays at `0.1.0.dev0` and the CHANGELOG keeps everything under
`## [Unreleased]`.

When you do cut a release:

1. Bump `pyproject.toml` and `src/meetmind/__init__.py::__version__` together.
2. Rename `## [Unreleased]` to `## [x.y.z] — YYYY-MM-DD` in `CHANGELOG.md`,
   then add a fresh empty `## [Unreleased]` at the top.
3. Tag: `git tag -s vX.Y.Z -m "vX.Y.Z"`.
4. Push: `git push && git push --tags`.
5. CI builds + uploads to PyPI on tag push.

## Questions

If you're not sure whether something belongs in MeetMind, open a
discussion on GitHub first. The opinions-per-feature ratio is high
here — we'd rather say "no, but here's a different approach" up front
than reject a PR after you've built it.
