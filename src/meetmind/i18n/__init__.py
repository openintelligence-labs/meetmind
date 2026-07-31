"""Lightweight i18n scaffolding for ``en`` (default), ``de`` and ``fr``.

Strings live as nested dicts in per-locale modules; lookup falls back from
the requested locale to English to the key itself. Plain dicts rather than
gettext, since this is a handful of CLI strings and needs no .mo shipping.

The active locale comes from ``MEETMIND_LOCALE``, else ``LC_ALL``/``LANG``.
"""

from __future__ import annotations

import os
from typing import Any

from meetmind.i18n._strings_de import STRINGS as STRINGS_DE
from meetmind.i18n._strings_en import STRINGS as STRINGS_EN
from meetmind.i18n._strings_fr import STRINGS as STRINGS_FR

_LOCALES: dict[str, dict[str, Any]] = {
    "en": STRINGS_EN,
    "de": STRINGS_DE,
    "fr": STRINGS_FR,
}


def active_locale() -> str:
    """Pick the active locale from env. Falls back to ``en``."""
    env = os.environ.get("MEETMIND_LOCALE")
    if env:
        prefix = env.split("_", 1)[0].lower()
        if prefix in _LOCALES:
            return prefix
    for var in ("LC_ALL", "LANG"):
        v = os.environ.get(var, "")
        prefix = v.split("_", 1)[0].lower()
        if prefix in _LOCALES:
            return prefix
    return "en"


def t(key: str, *, locale: str | None = None, **kwargs: object) -> str:
    """Look up a translation by dotted key. Falls back to English then to the key."""
    loc = (locale or active_locale()).lower()
    bag = _LOCALES.get(loc, STRINGS_EN)
    parts = key.split(".")
    value: Any = bag
    for p in parts:
        if isinstance(value, dict) and p in value:
            value = value[p]
        else:
            value = None
            break
    if value is None and loc != "en":
        value = STRINGS_EN
        for p in parts:
            if isinstance(value, dict) and p in value:
                value = value[p]
            else:
                value = None
                break
    if not isinstance(value, str):
        return key
    if kwargs:
        try:
            return value.format(**kwargs)
        except KeyError:
            return value
    return value


def available_locales() -> list[str]:
    return sorted(_LOCALES.keys())
