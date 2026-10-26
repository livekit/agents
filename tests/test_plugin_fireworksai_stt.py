from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.plugin("fireworksai")


async def test_update_options_rejects_text_timeout_out_of_range(monkeypatch) -> None:
    from livekit.plugins.fireworksai import stt as fireworks_stt

    async def _idle_run(self: fireworks_stt.SpeechStream) -> None:
        await asyncio.Event().wait()

    # keep the stream off the network; only its options are under test
    monkeypatch.setattr(fireworks_stt.SpeechStream, "_run", _idle_run)

    stt = fireworks_stt.STT(api_key="test-key", http_session=object())  # type: ignore[arg-type]
    stream = stt.stream()
    try:
        stt_snapshot = dict(vars(stt._opts))
        stream_snapshot = dict(vars(stream._opts))

        with pytest.raises(ValueError, match="text_timeout_seconds must be between"):
            stt.update_options(model="whisper-v3-turbo", text_timeout_seconds=30.0)
        assert vars(stt._opts) == stt_snapshot
        assert vars(stream._opts) == stream_snapshot

        with pytest.raises(ValueError, match="text_timeout_seconds must be between"):
            stream.update_options(text_timeout_seconds=0.5)
        assert vars(stream._opts) == stream_snapshot
        assert not stream._reconnect_event.is_set()

        stt.update_options(model="whisper-v3-turbo", text_timeout_seconds=29.0)
        assert stt._opts.model == stream._opts.model == "whisper-v3-turbo"
        assert stt._opts.text_timeout_seconds == stream._opts.text_timeout_seconds == 29.0
    finally:
        await stream.aclose()
