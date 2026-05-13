"""French strings — placeholder. Translation by a native speaker is TODO.

See `_strings_de.py` for the same caveat.
"""

STRINGS = {
    "cli": {
        "recording_started": "enregistrement démarré",
        "recording_stopped": "enregistrement arrêté",
        "no_meetings": "aucune réunion enregistrée — exécutez d'abord `meetmind record`",
        "wrote_file": "{path} écrit",
    },
    "ui": {
        "waiting_for_transcripts": "En attente des transcriptions.",
        "connecting": "connexion en cours…",
        "connected": "connecté",
        "disconnected": "déconnecté",
    },
    "consent": {
        "enroll_disclosure": (
            "L'enregistrement de votre empreinte vocale stocke un vecteur "
            "biométrique de 192 dimensions sur cet appareil. Il s'agit "
            "d'une catégorie particulière de données au sens de l'art. 9 "
            "du RGPD. Vous pouvez révoquer à tout moment via `meetmind forget`."
        ),
    },
}
