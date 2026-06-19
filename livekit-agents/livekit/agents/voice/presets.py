"""Public expressive presets.

A preset is a *use-case* (customer service, casual) that is
provider-agnostic at the call site:

    from livekit.agents.voice import presets

    session = AgentSession(tts=inference.TTS("inworld/inworld-tts-2"), expressive=presets.CASUAL)

Each ``presets.*`` constant is just an :class:`~livekit.agents.voice.ExpressiveOptions`
carrying a ``preset``. At session start the framework resolves it against the active TTS
provider (via ``tts.markup._provider_key()``) and injects the variant tuned for that
provider's markup tags. A provider with no tuned preset falls back to the agnostic default,
which still injects that provider's tag reference through the
``{tts.markup.llm_instructions}`` placeholder — so a preset always does something sensible
and can never disagree with the markup pipeline (both read the same provider key).

Customize by spreading a constant into a new dict (don't mutate the constant in place):

    expressive={**presets.CUSTOMER_SERVICE, "tts_instructions_append": "Confirm the name."}
    expressive={**presets.CASUAL, "audio_recognition_instructions_template": "..."}
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from ..llm.chat_context import Instructions
from ..tts import _provider_format as _pf

if TYPE_CHECKING:
    from .agent_session import ExpressiveOptions


class Preset(enum.Enum):
    """The domain a preset is tuned for. Used to key the per-provider registry."""

    CUSTOMER_SERVICE = "customer_service"
    CASUAL = "casual"


# (provider key as returned by ``tts.markup._provider_key()``) -> preset -> body
_REGISTRY: dict[str, dict[Preset, ExpressiveOptions]] = {
    "inworld": {
        Preset.CUSTOMER_SERVICE: _pf._INWORLD_CUSTOMER_SERVICE,
        Preset.CASUAL: _pf._INWORLD_CASUAL,
    },
    "cartesia": {
        Preset.CUSTOMER_SERVICE: _pf._CARTESIA_CUSTOMER_SERVICE,
        Preset.CASUAL: _pf._CARTESIA_CASUAL,
    },
}


def _append(template: Instructions | str, extra: str) -> Instructions:
    # concatenate the *raw* template text so any {placeholders} survive until render()
    if isinstance(template, Instructions):
        return Instructions(
            template.common + "\n\n" + extra, audio=template.audio, text=template.text
        )
    return Instructions(template + "\n\n" + extra)


def resolve_options(
    expr: ExpressiveOptions, *, provider_key: str, default: ExpressiveOptions
) -> ExpressiveOptions:
    """Resolve a user ``ExpressiveOptions`` to a concrete options dict for a provider.

    If ``expr`` carries a ``preset``, start from that provider's tuned preset (or
    ``default`` when the provider has none); otherwise start from ``default``. Then apply
    any explicit template overrides and ``tts_instructions_append``. The returned dict
    always has both template keys and never the ``preset`` / ``tts_instructions_append``
    helper keys.
    """
    preset = expr.get("preset")
    if preset is not None:
        base = _REGISTRY.get(provider_key, {}).get(preset, default)
    else:
        base = default

    tts_tmpl = expr.get("tts_instructions_template", base["tts_instructions_template"])
    ar_tmpl = expr.get(
        "audio_recognition_instructions_template",
        base["audio_recognition_instructions_template"],
    )
    append = expr.get("tts_instructions_append")
    if append:
        tts_tmpl = _append(tts_tmpl, append)

    result: ExpressiveOptions = {
        "tts_instructions_template": tts_tmpl,
        "audio_recognition_instructions_template": ar_tmpl,
    }
    return result


CUSTOMER_SERVICE: ExpressiveOptions = {"preset": Preset.CUSTOMER_SERVICE}
CASUAL: ExpressiveOptions = {"preset": Preset.CASUAL}

__all__ = ["Preset", "CUSTOMER_SERVICE", "CASUAL"]
