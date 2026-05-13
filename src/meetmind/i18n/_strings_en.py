"""English strings (default + canonical source). Other locales fall back here."""

STRINGS = {
    "cli": {
        "recording_started": "recording started",
        "recording_stopped": "recording stopped",
        "no_meetings": "no meetings stored yet — run `meetmind record` first",
        "wrote_file": "wrote {path}",
    },
    "ui": {
        "waiting_for_transcripts": "Waiting for transcripts.",
        "connecting": "connecting…",
        "connected": "connected",
        "disconnected": "disconnected",
    },
    "consent": {
        "enroll_disclosure": (
            "Enrolling your voiceprint stores a 192-d biometric embedding "
            "on this device. It is special-category data under GDPR Art. 9. "
            "You can revoke at any time with `meetmind forget`."
        ),
    },
}
