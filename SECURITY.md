# Security Policy

## Reporting a vulnerability

If you find a security issue in MeetMind, **please do not open a public
GitHub issue**. Email **security@openintelligence-labs.dev** with:

- a description of the bug,
- the smallest reproducer you can produce (script, config, audio clip),
- the version of MeetMind (`meetmind --version`),
- your OS + Python version.

We aim to acknowledge within 3 business days and fix or document a
mitigation within 30 days for confirmed exploitable issues. Reporters
of confirmed vulnerabilities are credited in the release notes unless
they prefer to remain anonymous.

## Threat model

MeetMind is local-first; the audio, transcripts, voiceprints, and
archive never leave the device under default configuration. The
threats we explicitly defend against:

1. **Offline disk theft.** Mitigated by SQLCipher (`meetmind[encrypted]`)
   wrapping the data store, with the DEK held in the OS keychain. Without
   that extra installed the store is plain SQLite — `meetmind status`
   reports which mode you're in.
2. **Same-host malicious browser tab.** The local API binds 127.0.0.1
   only, requires a per-launch bearer token (file mode 0600), and the
   CORS allow-list is narrowed to the bound port. Cross-origin attempts
   are rejected at the middleware layer.
3. **Accidental egress.** CI test `tests/security/test_no_outbound_calls.py`
   asserts that under default config no non-loopback socket is opened.
   Three opt-in code paths *can* talk to the network — they are all
   triggered by an explicit user choice:
   - LLM provider (`MEETMIND_LLM_PROVIDER`/`MEETMIND_LLM_BASE_URL` non-loopback),
   - `meetmind export-github` (shells out to the `gh` CLI),
   - `meetmind export-slack` (POST to `SLACK_WEBHOOK_URL`).

## Out of scope

- **Same-host malicious local user with elevated privileges.** The
  token file is 0600 but a root-equivalent attacker can read anything.
- **Audio capture permission abuse.** If you grant a different
  application Core Audio Tap / ScreenCaptureKit access, MeetMind has no
  defense against it — that's the OS's job.
- **Voiceprint impersonation.** The voiceprint matcher is a
  similarity-based heuristic, not a biometric authenticator. Don't use
  it for access control.
- **LLM provider behavior.** If you point MeetMind at a hosted LLM, what
  that provider does with the prompted text is between you and them.

## Known dual-use code

- `src/meetmind/integrations/slack.py` — outbound POST when explicitly
  configured.
- `src/meetmind/integrations/github.py` — shells out to the user's `gh`
  CLI, which has its own authentication.
- `src/meetmind/analyze/llm.py` — outbound to the configured LLM provider.

## Disclosure policy

We follow coordinated disclosure: please give us 30 days to ship a fix
before publishing details. We don't pursue legal action against
researchers acting in good faith. If you've found something serious and
need to share preliminary details with a third party (e.g. a downstream
distro), email us first and we'll coordinate.
