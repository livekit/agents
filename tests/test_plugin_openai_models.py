from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest

from livekit.agents.llm._realtime.openai_types import RealtimeModels
from livekit.plugins.openai.models import (
    ChatModels,
    DalleModels,
    EmbeddingModels,
    ResponsesModels,
    STTModels,
    TTSModels,
    TTSVoices,
)
from livekit.plugins.openai.realtime import GPTLiveModels, GPTLiveVoices

pytestmark = pytest.mark.unit


def _values(annotation: object) -> set[str]:
    if get_origin(annotation) in (Literal, Union, UnionType):
        values: set[str] = set()
        for arg in get_args(annotation):
            if isinstance(arg, str):
                values.add(arg)
            else:
                values.update(_values(arg))
        return values
    return set()


def test_models_cover_current_openai_spec() -> None:
    assert _values(STTModels) >= {
        "whisper-1",
        "gpt-transcribe",
        "gpt-live-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-12-15",
        "gpt-4o-transcribe",
        "gpt-4o-transcribe-diarize",
        "gpt-realtime-whisper",
    }
    assert _values(TTSModels) == {
        "tts-1",
        "tts-1-hd",
        "gpt-4o-mini-tts",
        "gpt-4o-mini-tts-2025-12-15",
    }
    assert _values(EmbeddingModels) == {
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    }
    assert _values(DalleModels) >= {
        "gpt-image-1.5",
        "gpt-image-2",
        "gpt-image-2-2026-04-21",
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-sunburst-2026-09-08",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-flare-2026-09-08",
        "dall-e-2",
        "dall-e-3",
        "gpt-image-1",
        "gpt-image-1-mini",
        "chatgpt-image-latest",
    }
    assert _values(GPTLiveModels) == {"gpt-live-1"}


def test_chat_and_responses_models_include_current_additions() -> None:
    shared = {
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.1-mini",
        "gpt-audio-mini-2025-12-15",
    }
    assert _values(ChatModels) >= shared
    assert _values(ResponsesModels) >= shared | {
        "gpt-5.5-pro",
        "gpt-5.5-pro-2026-04-23",
        "gpt-daybreak-blue-latest",
        "gpt-daybreak-red-latest",
        "gpt-5.6-cyber",
        "gpt-rosalind-research",
    }


def test_realtime_models_cover_current_openai_spec() -> None:
    assert _values(RealtimeModels) >= {
        "gpt-realtime",
        "gpt-realtime-1.5",
        "gpt-realtime-2",
        "gpt-realtime-2.1",
        "gpt-realtime-2.1-mini",
        "gpt-realtime-2025-08-28",
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-realtime-preview-2024-12-17",
        "gpt-4o-realtime-preview-2025-06-03",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-realtime-preview-2024-12-17",
        "gpt-realtime-mini",
        "gpt-realtime-mini-2025-10-06",
        "gpt-realtime-mini-2025-12-15",
        "gpt-audio-1.5",
        "gpt-audio-mini",
        "gpt-audio-mini-2025-10-06",
        "gpt-audio-mini-2025-12-15",
    }


def test_voice_enums_cover_current_openai_spec() -> None:
    assert _values(TTSVoices) >= {
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
    }
    assert _values(GPTLiveVoices) >= {
        "alloy",
        "ash",
        "ballad",
        "beacon",
        "bossa",
        "cedar",
        "cinder",
        "coral",
        "delta",
        "echo",
        "gleam",
        "marin",
        "meridian",
        "quartz",
        "ripple",
        "sage",
        "shimmer",
        "stone",
        "tempo",
        "verse",
        "vesper",
        "willow",
    }
