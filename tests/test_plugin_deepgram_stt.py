from __future__ import annotations

from unittest import mock

import pytest

pytestmark = pytest.mark.plugin("deepgram")


def _capture_request_params(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the Deepgram URL builder so we can inspect the request params dict
    the plugin assembles, without performing any network request."""
    from livekit.plugins.deepgram import stt as dg_stt

    captured: dict = {}

    def fake_to_deepgram_url(opts: dict, base_url: str, *, websocket: bool) -> str:
        captured.clear()
        captured.update(opts)
        return f"{base_url}?captured"

    monkeypatch.setattr(dg_stt, "_to_deepgram_url", fake_to_deepgram_url)
    return captured


async def test_diarize_model_in_prerecorded_request_params(monkeypatch: pytest.MonkeyPatch):
    from livekit import rtc
    from livekit.agents import APIConnectionError
    from livekit.plugins.deepgram import STT

    captured = _capture_request_params(monkeypatch)
    stt = STT(api_key="test-key", diarize_model="v2")

    buffer = rtc.AudioFrame(
        data=b"\x00\x00" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
    )

    with mock.patch.object(stt, "_ensure_session") as ensure_session:
        ensure_session.return_value.post.side_effect = RuntimeError("stop-before-network")
        with pytest.raises(APIConnectionError):
            await stt._recognize_impl(buffer)

    assert captured.get("diarize_model") == "v2"


async def test_diarize_model_in_live_request_params(monkeypatch: pytest.MonkeyPatch):
    from livekit.agents import APIConnectionError
    from livekit.plugins.deepgram import STT

    captured = _capture_request_params(monkeypatch)

    session = mock.MagicMock()

    async def fake_ws_connect(url, **kwargs):
        raise APIConnectionError("stop-before-network")

    session.ws_connect = fake_ws_connect

    stt = STT(api_key="test-key", diarize_model="latest", http_session=session)
    stream = stt.stream(language="en-US")

    with pytest.raises(APIConnectionError):
        await stream._connect_ws()

    assert captured.get("diarize_model") == "latest"
    await stream.aclose()


async def test_update_options_diarize_model():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key")
    assert stt._opts.diarize_model is None or stt._opts.diarize_model == ""
    stt.update_options(diarize_model="v2")
    assert stt._opts.diarize_model == "v2"


async def test_diarize_model_reports_diarization_capability():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key", diarize_model="latest")
    assert stt.capabilities.diarization is True


async def test_enable_diarization_reports_diarization_capability():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key", enable_diarization=True)
    assert stt.capabilities.diarization is True


async def test_no_diarization_reports_no_capability():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key")
    assert stt.capabilities.diarization is False


async def test_update_options_uses_stored_language_for_model_validation():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key", language="fr")
    stt.update_options(model="nova-2-meeting")
    assert stt._opts.model == "nova-2-general"


async def test_update_options_explicit_language_overrides_stored():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key", language="fr")
    stt.update_options(model="nova-2-meeting", language="en-US")
    assert stt._opts.model == "nova-2-meeting"


async def test_update_options_no_language_set_keeps_en_only_model():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="test-key")
    stt.update_options(model="nova-2-meeting")
    assert stt._opts.model == "nova-2-meeting"
