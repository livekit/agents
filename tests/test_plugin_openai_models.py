from typing import get_args

import pytest

from livekit.plugins.openai.llm import ServiceTier as ChatServiceTier
from livekit.plugins.openai.models import (
    ChatModels,
    RealtimeModels,
    ResponsesModels,
    STTModels,
    TTSModels,
    TTSVoices,
)
from livekit.plugins.openai.realtime.gpt_live_model import GPTLiveModels, GPTLiveVoices
from livekit.plugins.openai.realtime.gpt_live_types import ServiceTier as GPTLiveServiceTier
from livekit.plugins.openai.responses.llm import ServiceTier as ResponsesServiceTier

pytestmark = pytest.mark.plugin("openai")


def _literal_values(type_: object) -> set[str]:
    values: set[str] = set()
    for value in get_args(type_):
        if isinstance(value, str):
            values.add(value)
        else:
            values.update(_literal_values(value))
    return values


def test_model_enums_cover_current_openapi_spec() -> None:
    shared_models = set(
        """
        gpt-6-astra gpt-6-sol gpt-6-luna gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna
        gpt-5.5 gpt-5.5-2026-04-23 gpt-5.4 gpt-5.4-mini gpt-5.4-nano
        gpt-5.4-mini-2026-03-17 gpt-5.4-nano-2026-03-17 gpt-5.3-chat-latest gpt-5.2
        gpt-5.2-2025-12-11 gpt-5.2-chat-latest gpt-5.2-pro gpt-5.2-pro-2025-12-11
        gpt-5.1 gpt-5.1-2025-11-13 gpt-5.1-codex gpt-5.1-mini gpt-5.1-chat-latest
        gpt-5 gpt-5-mini gpt-5-nano gpt-5-2025-08-07 gpt-5-mini-2025-08-07
        gpt-5-nano-2025-08-07 gpt-5-chat-latest gpt-4.1 gpt-4.1-mini gpt-4.1-nano
        gpt-4.1-2025-04-14 gpt-4.1-mini-2025-04-14 gpt-4.1-nano-2025-04-14
        o4-mini o4-mini-2025-04-16 o3 o3-2025-04-16 o3-mini o3-mini-2025-01-31
        o1 o1-2024-12-17 o1-preview o1-preview-2024-09-12 o1-mini o1-mini-2024-09-12
        gpt-4o gpt-4o-2024-11-20 gpt-4o-2024-08-06 gpt-4o-2024-05-13
        gpt-audio-mini gpt-audio-mini-2025-12-15 gpt-4o-audio-preview
        gpt-4o-audio-preview-2024-10-01 gpt-4o-audio-preview-2024-12-17
        gpt-4o-audio-preview-2025-06-03 gpt-4o-mini-audio-preview
        gpt-4o-mini-audio-preview-2024-12-17 gpt-4o-search-preview
        gpt-4o-mini-search-preview gpt-4o-search-preview-2025-03-11
        gpt-4o-mini-search-preview-2025-03-11 chatgpt-4o-latest codex-mini-latest
        gpt-4o-mini gpt-4o-mini-2024-07-18 gpt-4-turbo gpt-4-turbo-2024-04-09
        gpt-4-0125-preview gpt-4-turbo-preview gpt-4-1106-preview gpt-4-vision-preview
        gpt-4 gpt-4-0314 gpt-4-0613 gpt-4-32k gpt-4-32k-0314 gpt-4-32k-0613
        gpt-3.5-turbo gpt-3.5-turbo-16k gpt-3.5-turbo-0301 gpt-3.5-turbo-0613
        gpt-3.5-turbo-1106 gpt-3.5-turbo-0125 gpt-3.5-turbo-16k-0613
        """.split()
    )
    responses_only = set(
        """
        o1-pro o1-pro-2025-03-19 o3-pro o3-pro-2025-06-10 o3-deep-research
        o3-deep-research-2025-06-26 o4-mini-deep-research o4-mini-deep-research-2025-06-26
        computer-use-preview computer-use-preview-2025-03-11 gpt-5.5-pro
        gpt-5.5-pro-2026-04-23 gpt-5-codex gpt-5-pro gpt-5-pro-2025-10-06
        gpt-5.1-codex-max gpt-daybreak-blue-latest gpt-daybreak-red-latest gpt-5.6-cyber
        gpt-rosalind-research
        """.split()
    )

    assert shared_models <= _literal_values(ChatModels)
    assert shared_models | responses_only <= _literal_values(ResponsesModels)
    assert {
        "whisper-1",
        "gpt-transcribe",
        "gpt-live-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-12-15",
        "gpt-4o-transcribe",
        "gpt-4o-transcribe-diarize",
        "gpt-realtime-whisper",
    } <= _literal_values(STTModels)
    assert {
        "tts-1",
        "tts-1-hd",
        "gpt-4o-mini-tts",
        "gpt-4o-mini-tts-2025-12-15",
    } <= _literal_values(TTSModels)


def test_realtime_model_enums_cover_current_openapi_spec() -> None:
    expected = set(
        """
        gpt-realtime gpt-realtime-1.5 gpt-realtime-2 gpt-realtime-2.1
        gpt-realtime-2.1-mini gpt-realtime-2025-08-28 gpt-4o-realtime-preview
        gpt-4o-realtime-preview-2024-10-01 gpt-4o-realtime-preview-2024-12-17
        gpt-4o-realtime-preview-2025-06-03 gpt-4o-mini-realtime-preview
        gpt-4o-mini-realtime-preview-2024-12-17 gpt-realtime-mini
        gpt-realtime-mini-2025-10-06 gpt-realtime-mini-2025-12-15 gpt-audio-1.5
        gpt-audio-mini gpt-audio-mini-2025-10-06 gpt-audio-mini-2025-12-15
        """.split()
    )
    assert expected <= _literal_values(RealtimeModels)
    assert _literal_values(GPTLiveModels) == {"gpt-live-1"}


def test_voice_and_live_service_tier_enums_cover_current_openapi_spec() -> None:
    assert {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    } <= _literal_values(TTSVoices)
    assert _literal_values(GPTLiveVoices) == {
        "aster",
        "beacon",
        "cinder",
        "marin",
        "stone",
        "vesper",
    }
    assert _literal_values(GPTLiveServiceTier) == {
        "auto",
        "default",
        "fast_tier_temp_pilot",
        "flex",
        "priority",
        "ultrafast",
    }
    assert _literal_values(ChatServiceTier) == {
        "auto",
        "default",
        "flex",
        "scale",
        "priority",
        "fast",
    }
    assert _literal_values(ResponsesServiceTier) == {
        "auto",
        "default",
        "flex",
        "scale",
        "priority",
        "fast",
        "ultrafast",
    }
