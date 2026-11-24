"""Agent-under-test presets.

A preset fixes everything about the agent that a benchmark run compares: models, turn handling
and tool limits. Benchmarks supply only the instructions, the tools and the caller audio, so two
presets can be compared on equal terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    name: str
    stt: str = "deepgram/flux-general"
    stt_language: str = "en"
    llm: str = "google/gemma-4-31b-it"
    llm_reasoning: str | None = None
    tts: str = "cartesia/sonic-3"
    tts_voice: str = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
    tts_word_timestamps: bool = True
    """Ask the TTS for word timings, so the spoken transcript (and the chat history after an
    interruption) is cut at the word actually reached. Measurement fidelity, not behavior."""
    noise_cancellation: str | None = None
    """``"krisp_telephony"``: Krisp voice isolation for telephony audio on the caller's input
    (LiveKit Cloud auth), as a room's ``AudioInputOptions(noise_cancellation=...)`` would apply.
    BVC/BVCTelephony cannot be used here: they only run on a track in a connected room."""
    max_tool_steps: int = 10
    preemptive_generation: bool = True
    resume_false_interruption: bool = True
    false_interruption_timeout: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The repo's starter agent (examples/voice_agents/basic_agent.py) with the LiveKit gateway's
# gemma-4. Deviations from the starter, each for a benchmark-specific reason:
# - max_tool_steps 10 (default 3): tau2/EVA tasks chain lookups before the agent next speaks, and
#   the default would measure the cap rather than the agent.
# - no aec_warmup_duration: the caller's audio is a clean separate channel, there is no echo,
#   and the warmup would block every barge-in for the first 3 s of each agent turn.
# - no keyterm detection: the starter's keyterms are LiveKit-specific.
# - tts_word_timestamps on: see AgentConfig.tts_word_timestamps.
BASELINE_GEMMA4 = AgentConfig(name="baseline-gemma4")

PRESETS: dict[str, AgentConfig] = {
    BASELINE_GEMMA4.name: BASELINE_GEMMA4,
    "baseline-gemma4-krisp": replace(
        BASELINE_GEMMA4, name="baseline-gemma4-krisp", noise_cancellation="krisp_telephony"
    ),
}


def get_preset(name: str) -> AgentConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown agent preset {name!r}; known: {sorted(PRESETS)}") from None
