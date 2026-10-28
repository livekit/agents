"""Test inference STT alignment claims and adaptive-interruption eligibility.

The gateway's Cartesia Ink-2 translator emits ``Words: []`` with ``Start`` = the
audio-time offset and ``Duration`` = billed audio bytes / bytes-per-second — a billing
figure, not the utterance span. Cartesia's Turns API sends no word timings to forward.

Adaptive interruption uses local STT arrival times and VAD state. The alignment claim
still describes provider output, but it does not control adaptive-interruption eligibility.
"""

from __future__ import annotations

import pytest

from livekit.agents import inference

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "fake")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "fakesecret")
    monkeypatch.setenv("CARTESIA_API_KEY", "fake")


def test_inference_and_plugin_agree_about_ink2(_fake_credentials: None) -> None:
    """The same Cartesia model must report the same alignment through either integration.

    Verified against the live Turns API: every event is {type, transcript, turn_id,
    request_id}, on both the version the plugin pins and the one the gateway pins, and
    neither `add_timestamps` nor `timestamp_granularities[]` changes that.
    """
    cartesia = pytest.importorskip("livekit.plugins.cartesia")

    gateway_stt = inference.STT(model="cartesia/ink-2")
    plugin_stt = cartesia.STT(model="ink-2")

    assert plugin_stt.capabilities.aligned_transcript is False
    assert gateway_stt.capabilities.aligned_transcript is False


def test_models_that_do_send_words_keep_their_claim(_fake_credentials: None) -> None:
    """The fix must not strip alignment from the providers that genuinely forward words."""
    assert inference.STT(model="cartesia/ink-whisper").capabilities.aligned_transcript == "word"
    assert inference.STT(model="deepgram/nova-3").capabilities.aligned_transcript == "word"
    assert inference.STT(
        model="assemblyai/universal-streaming"
    ).capabilities.aligned_transcript == ("word")
    # the provider is resolved server-side per language, so alignment can't be promised
    assert inference.STT(model="auto").capabilities.aligned_transcript is False
    assert inference.STT(model="inworld/inworld-stt-1").capabilities.aligned_transcript is False


def test_alignment_is_recomputed_when_the_model_changes(_fake_credentials: None) -> None:
    """`update_options` recomputes it, as it already does for keyterms and chat_context."""
    stt_impl = inference.STT(model="deepgram/nova-3")
    assert stt_impl.capabilities.aligned_transcript == "word"

    stt_impl.update_options(model="cartesia/ink-2")
    assert stt_impl.capabilities.aligned_transcript is False


def test_unknown_models_do_not_claim_alignment(_fake_credentials: None) -> None:
    stt_impl = inference.STT(model="new-provider/new-turn-model")
    assert stt_impl.capabilities.aligned_transcript is False


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        pytest.param("cartesia/ink-whisper", "word", id="aligned-fallback"),
        pytest.param("cartesia/ink-2", False, id="unaligned-fallback"),
        pytest.param("new-provider/new-turn-model", False, id="unknown-fallback"),
    ],
)
def test_fallback_models_constrain_alignment_claim(
    fallback: str, expected: object, _fake_credentials: None
) -> None:
    stt_impl = inference.STT(model="deepgram/nova-3", fallback=fallback)
    assert stt_impl.capabilities.aligned_transcript == expected


def test_alignment_update_still_accounts_for_fallback(_fake_credentials: None) -> None:
    stt_impl = inference.STT(model="cartesia/ink-2", fallback="new-provider/new-turn-model")

    stt_impl.update_options(model="deepgram/nova-3")

    assert stt_impl.capabilities.aligned_transcript is False
