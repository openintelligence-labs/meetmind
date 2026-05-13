<!-- Read CONTRIBUTING.md before opening — the "What we accept / don't" list is enforced. -->

## What does this PR do?

A one-paragraph summary. Link the issue this addresses.

Fixes #

## How was this tested?

- [ ] `pytest -q` is green locally
- [ ] `ruff check src tests` is clean
- [ ] `ruff format --check src tests` is clean
- [ ] `lint-imports --config pyproject.toml` reports 6/6 contracts kept
- [ ] `meetmind selftest` runs to completion
- [ ] New behaviour has ≥ 1 happy-path test + ≥ 1 edge-case test

## Anti-features check

- [ ] No telemetry added.
- [ ] No new outbound network call in the default code path.
- [ ] Audio + transcripts + voiceprints + archive still stay on the device.
- [ ] `tests/security/test_no_outbound_calls.py` still passes.

## Reviewer notes

Anything reviewers should pay extra attention to (perf trade-offs,
unusual API shapes, breaking changes, migration concerns).
