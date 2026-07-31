"""German strings — placeholder scaffold.

TODO: needs review by a native speaker before release. Absent keys fall back
to English via the lookup chain in ``meetmind.i18n.t``.
"""

STRINGS = {
    "cli": {
        "recording_started": "Aufnahme gestartet",
        "recording_stopped": "Aufnahme beendet",
        "no_meetings": "Noch keine Meetings gespeichert — bitte zuerst `meetmind record` ausführen",
        "wrote_file": "{path} geschrieben",
    },
    "ui": {
        "waiting_for_transcripts": "Warte auf Transkripte.",
        "connecting": "verbinde…",
        "connected": "verbunden",
        "disconnected": "getrennt",
    },
    "consent": {
        "enroll_disclosure": (
            "Beim Anlegen Ihres Stimmprofils wird ein 192-dimensionaler "
            "biometrischer Vektor auf diesem Gerät gespeichert. Dies ist "
            "eine besondere Datenkategorie nach Art. 9 DSGVO. Sie können "
            "die Einwilligung jederzeit mit `meetmind forget` widerrufen."
        ),
    },
}
