"""Tests for analyze.redact (S11.9)."""

from __future__ import annotations

from meetmind.analyze.redact import available_profiles, redact


def test_raw_passes_text_through():
    text = "email me at sam@example.com or call +1 555 123 4567"
    out = redact(text, profile="raw")
    assert out.text == text
    assert out.redaction_count == 0


def test_team_internal_strips_email_phone_card():
    text = "Sam: email sam@example.com, call +1-555-123-4567, card 4111 1111 1111 1111"
    out = redact(text, profile="team_internal")
    assert "[email]" in out.text
    assert "[phone]" in out.text
    assert "[card]" in out.text
    assert "sam@example.com" not in out.text
    assert "Sam" in out.text  # name preserved at this tier
    assert out.redaction_count >= 3


def test_public_share_also_strips_names():
    text = "Sam Williams said the deck is ready."
    out = redact(text, profile="public_share")
    assert "[name]" in out.text
    assert "Sam Williams" not in out.text


def test_public_share_includes_team_internal_redactions():
    text = "Sam Williams emailed sam@example.com about the deck."
    out = redact(text, profile="public_share")
    assert "[email]" in out.text
    assert "[name]" in out.text


def test_available_profiles_lists_three():
    assert available_profiles() == ["raw", "team_internal", "public_share"]
