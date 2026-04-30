"""FonadaLabs TTS model types — languages and voices available on the API.

.. note::
    **This file is auto-generated.**
    Re-run ``python scripts/generate_fonadalabs_models.py`` to refresh it
    whenever FonadaLabs adds new voices or languages.

    Generated: 2026-04-02 00:00 UTC
    Source:    https://api.fonada.ai/supported-voices

These ``Literal`` types give IDEs autocomplete and let mypy catch typos at
type-check time.  The :class:`~livekit.plugins.fonadalabs.TTS` class accepts
either a typed constant *or* a plain ``str``, so you can still pass a raw
string if the API adds new voices before this file is regenerated.

Example::

    from livekit.plugins import fonadalabs

    tts = fonadalabs.TTS(
        language="Hindi",
        voice="Vaanee",
    )
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

TTSLanguages = Literal["English", "Hindi", "Tamil", "Telugu"]
"""Supported language display names accepted by the FonadaLabs TTS API."""

# ---------------------------------------------------------------------------
# Per-language voice literals
# ---------------------------------------------------------------------------

TTSEnglishVoice = Literal[
    "Dhruv",
    "Vaanee",
    "Swastik",
    "Laksh",
    "Raag",
    "Sarvagya",
    "Komal",
    "Meghra",
    "Pancham",
    "Tara",
    "Sharad",
    "Kritika",
    "Mandra",
    "Karn",
    "Gauri",
    "Ruhi",
    "Roshini",
    "Parikshit",
]
"""18 English voices available on the FonadaLabs TTS API."""

TTSHindiVoice = Literal[
    "Dhruv",
    "Vaanee",
    "Swastik",
    "Laksh",
    "Raag",
    "Sarvagya",
    "Komal",
    "Meghra",
    "Pancham",
    "Tara",
    "Sharad",
    "Kritika",
    "Mandra",
    "Karn",
    "Gauri",
    "Ruhi",
    "Roshini",
    "Parikshit",
]
"""18 Hindi voices available on the FonadaLabs TTS API."""

TTSTamilVoice = Literal[
    "Vaani",
    "Isai",
    "Thalam",
    "Swaram",
    "Madhuri",
    "Naadham",
    "Rachna",
    "Pallavi",
    "Mrityunjay",
    "Malika",
    "Yamini",
    "Tilak",
    "Dhruv",
    "Sanket",
    "Rudraksh",
]
"""15 Tamil voices available on the FonadaLabs TTS API."""

TTSTeluguVoice = Literal[
    "Ansh",
    "Dhruv",
    "Aadhira",
    "Aahana",
    "Aakriti",
    "Ridhima",
    "Vaani",
    "Shaury",
    "Bhavyaa",
    "Tanuj",
    "Utkarsh",
    "Adyaa",
    "Raagini",
    "Kanika",
    "Madhav",
    "Malini",
    "Priya",
    "Mala",
    "Dhairy",
    "Shruti",
    "Tara",
    "Shubhra",
    "Mandara",
    "Tanya",
    "Sara",
    "Rudrika",
    "Ruhi",
    "Sameer",
    "Geetika",
    "Naitik",
    "Kartik",
    "Naira",
    "Meghvani",
    "Sneha",
    "Sonika",
    "Nikunj",
    "Sonia",
    "Mridul",
    "Shobha",
    "Gunika",
    "Yuvaan",
    "Abhishek",
    "Dhruvika",
    "Hardik",
    "Raj",
    "Raghav",
    "Sanatan",
    "Tanmay",
    "Sanjana",
    "Niharika",
    "Ruchika",
    "Sonakshi",
    "Aaliya",
    "Aanchal",
    "Vartika",
    "Rihaan",
    "Divya",
    "Taarini",
    "Divyansh",
    "Pavani",
]
"""60 Telugu voices available on the FonadaLabs TTS API."""

# ---------------------------------------------------------------------------
# Combined voice type (union of all per-language literals)
# ---------------------------------------------------------------------------

TTSVoice = TTSEnglishVoice | TTSHindiVoice | TTSTamilVoice | TTSTeluguVoice
"""Union of all voices across every supported language.

Pass this (or a plain ``str``) as the ``voice`` argument to :class:`TTS`.
"""
