from __future__ import annotations

import asyncio

import pytest

from plugins.interrupt_handler import InterruptHandler, InterruptHandlerConfig


class _Stopper:
    def __init__(self) -> None:
        self.calls: int = 0

    def interrupt(self) -> asyncio.Future[None]:
        self.calls += 1
        fut: asyncio.Future[None] = asyncio.Future()
        fut.set_result(None)
        return fut


def _create_handler(**config_kwargs: object) -> InterruptHandler:
    cfg = InterruptHandlerConfig(**config_kwargs)
    return InterruptHandler(stop_callback=_Stopper().interrupt, config=cfg)


def test_classification_ignores_single_filler() -> None:
    handler = _create_handler()
    result = handler.classification_for_segment(["uh"], [0.95], tts_is_speaking=True)
    assert result == "FILLER"


def test_classification_accepts_command_phrase() -> None:
    handler = _create_handler()
    result = handler.classification_for_segment(
        ["umm", "okay", "stop"], [0.4, 0.7, 0.95], tts_is_speaking=True
    )
    assert result == "VALID_SPEECH"


def test_classification_accepts_when_agent_quiet() -> None:
    handler = _create_handler()
    result = handler.classification_for_segment(["umm"], [0.1], tts_is_speaking=False)
    assert result == "VALID_SPEECH"


def test_classification_uncertain_on_low_conf_non_filler() -> None:
    handler = _create_handler(confidence_threshold=0.6)
    result = handler.classification_for_segment(
        ["hmm", "yeah"], [0.2, 0.3], tts_is_speaking=True
    )
    assert result == "UNCERTAIN"


@pytest.mark.asyncio
async def test_integration_interrupts_on_valid_speech() -> None:
    stopper = _Stopper()
    handler = InterruptHandler(stop_callback=stopper.interrupt, config=InterruptHandlerConfig())

    handler.on_tts_state(True)
    ignored = await handler.on_transcription(
        "uh",
        words_meta=[{"text": "uh", "confidence": 0.99}],
        metadata={"is_final": False},
    )
    assert ignored is False
    assert stopper.calls == 0

    accepted = await handler.on_transcription(
        "please stop",
        words_meta=[{"text": "please", "confidence": 0.65}, {"text": "stop", "confidence": 0.93}],
        metadata={"is_final": True},
    )
    assert accepted is True
    assert stopper.calls == 1

    handler.on_tts_state(False)
    passthrough = await handler.on_transcription(
        "umm", words_meta=[{"text": "umm", "confidence": 0.2}], metadata={"is_final": True}
    )
    assert passthrough is False
    assert stopper.calls == 1

