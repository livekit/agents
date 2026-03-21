"""FonadaLabs TTS model types — languages and voices available on the API.

.. note::
    **This file is auto-generated.**
    Re-run ``python scripts/generate_fonadalabs_models.py`` to refresh it
    whenever FonadaLabs adds new voices or languages.

    Generated: 2026-03-21 04:53 UTC
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
    "Naad",
    "Dhwani",
    "Vaanee",
    "Swara",
    "Taal",
    "Laya",
    "Raaga",
    "Geetika",
    "Swarini",
    "Geet",
    "Sangeeta",
    "Raagini",
    "Madhura",
    "Komal",
    "Sangeet",
    "Meghra",
    "Gandhar",
    "Madhyam",
    "Shruti",
    "Pancham",
    "Dhaivat",
    "Nishad",
    "Tara",
    "Shadja",
    "Komalika",
    "Rishabh",
    "Mandra",
    "Tarana",
    "Swarika",
    "Komala",
    "Geetini",
    "Teevra",
    "Chaitra",
    "Madhur",
    "Raagika",
    "Swarita",
    "Vibhaag",
    "Gitanjali",
    "Aalap",
    "Sangeeti",
    "Taan",
    "Meend",
    "Raagita",
    "Gamak",
    "Murki",
    "Khatka",
    "Andolan",
    "Sparsh",
    "Kampan",
    "Shrutika",
    "Swaranjali",
    "Nada",
    "Lahar",
    "Tarang",
    "Dhwaniya",
    "Shrutini",
    "Swar",
    "Geetanjali",
    "Raaginika",
    "Sangeetika",
    "Ninada",
    "Swaroopa",
    "Geetimala",
    "Naadayana",
    "Swarayana",
    "Layakari",
    "Taalayana",
    "Raag",
    "Swaranjana",
    "Naadanika",
    "Dhwanika",
    "Swaraka",
    "Sangeetara",
    "Layabaddha",
]
"""74 English voices available on the FonadaLabs TTS API."""

TTSHindiVoice = Literal[
    "Naad",
    "Dhwani",
    "Vaanee",
    "Swara",
    "Taal",
    "Laya",
    "Raaga",
    "Geetika",
    "Swarini",
    "Geet",
    "Sangeeta",
    "Raagini",
    "Madhura",
    "Komal",
    "Sangeet",
    "Meghra",
    "Gandhar",
    "Madhyam",
    "Shruti",
    "Pancham",
    "Dhaivat",
    "Nishad",
    "Tara",
    "Shadja",
    "Komalika",
    "Rishabh",
    "Mandra",
    "Tarana",
    "Swarika",
    "Komala",
    "Geetini",
    "Teevra",
    "Chaitra",
    "Madhur",
    "Raagika",
    "Swarita",
    "Vibhaag",
    "Gitanjali",
    "Aalap",
    "Sangeeti",
    "Taan",
    "Meend",
    "Raagita",
    "Gamak",
    "Murki",
    "Khatka",
    "Andolan",
    "Sparsh",
    "Kampan",
    "Shrutika",
    "Swaranjali",
    "Nada",
    "Lahar",
    "Tarang",
    "Dhwaniya",
    "Shrutini",
    "Swar",
    "Geetanjali",
    "Raaginika",
    "Sangeetika",
    "Ninada",
    "Swaroopa",
    "Geetimala",
    "Naadayana",
    "Swarayana",
    "Layakari",
    "Taalayana",
    "Raag",
    "Swaranjana",
    "Naadanika",
    "Dhwanika",
    "Swaraka",
    "Sangeetara",
    "Layabaddha",
]
"""74 Hindi voices available on the FonadaLabs TTS API."""

TTSTamilVoice = Literal[
    "Vaani",
    "Isai",
    "Thalam",
    "Swaram",
    "Madhuram",
    "Naadham",
    "Ragam",
    "Pallavi",
    "Komalam",
    "Raagamalika",
    "Geetham",
    "Taalam",
    "Dhwani",
    "Sangeetham",
    "Raagaratna",
    "Shruti",
]
"""16 Tamil voices available on the FonadaLabs TTS API."""

TTSTeluguVoice = Literal[
    "Naadamu",
    "Dhwani",
    "Taalam",
    "Geetamu",
    "Raagamalika",
    "Sangeetamu",
    "Vaani",
    "Swaramu",
    "Layamu",
    "Taalabaddha",
    "Raagapriya",
    "Swarajathi",
    "Raagini",
    "Komala",
    "Naada",
    "Meghamalini",
    "Sangeetapriya",
    "Raagamala",
    "Dhwaniya",
    "Shruti",
    "Tara",
    "Komalavani",
    "Mandara",
    "Taana",
    "Swarajati",
    "Raagaanjali",
    "Raagika",
    "Swaranjali",
    "Geetika",
    "Swaramala",
    "Aalapana",
    "Raagaratnam",
    "Meghavani",
    "Swarita",
    "Geetavani",
    "Taala",
    "Layakari",
    "Murki",
    "Sangeetavani",
    "Geetamala",
    "Naadapriya",
    "Dhwanika",
    "Dhwanimala",
    "Sangeetanjali",
    "Gamaka",
    "Raagasudha",
    "Sangeetaratna",
    "Taalamani",
    "Sangeetasundari",
    "Naadayana",
    "Raagavalli",
    "Swarasudha",
    "Sangeetaswarna",
    "Raagamanjari",
    "Swaravara",
    "Naadeshwara",
    "Dhwanividya",
    "Taalapala",
    "Dhwanipala",
    "Swarapala",
]
"""60 Telugu voices available on the FonadaLabs TTS API."""

# ---------------------------------------------------------------------------
# Combined voice type (union of all per-language literals)
# ---------------------------------------------------------------------------

TTSVoice = TTSEnglishVoice | TTSHindiVoice | TTSTamilVoice | TTSTeluguVoice
"""Union of all voices across every supported language.

Pass this (or a plain ``str``) as the ``voice`` argument to :class:`TTS`.
"""
