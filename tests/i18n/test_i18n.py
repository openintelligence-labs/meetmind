"""Tests for the i18n scaffolding (S15.5)."""

from __future__ import annotations

from meetmind.i18n import active_locale, available_locales, t


def test_default_locale_is_english(monkeypatch):
    monkeypatch.delenv("MEETMIND_LOCALE", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    assert active_locale() == "en"
    assert t("cli.recording_started") == "recording started"


def test_explicit_locale_override():
    assert t("cli.recording_started", locale="de") == "Aufnahme gestartet"
    assert t("cli.recording_started", locale="fr") == "enregistrement démarré"


def test_unknown_key_falls_back_to_key_name():
    assert t("nope.does.not.exist") == "nope.does.not.exist"


def test_format_kwargs_substitute():
    assert t("cli.wrote_file", path="/tmp/x.md") == "wrote /tmp/x.md"


def test_missing_key_in_locale_falls_back_to_english(monkeypatch):
    # Simulate a locale that hasn't translated a particular key.
    from meetmind.i18n import _strings_de

    saved = _strings_de.STRINGS.pop("cli", None)
    try:
        assert t("cli.recording_started", locale="de") == "recording started"
    finally:
        if saved is not None:
            _strings_de.STRINGS["cli"] = saved


def test_locale_picked_from_env(monkeypatch):
    monkeypatch.setenv("MEETMIND_LOCALE", "fr_FR.UTF-8")
    assert active_locale() == "fr"


def test_available_locales():
    assert set(available_locales()) == {"de", "en", "fr"}
