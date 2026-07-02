from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.plugin("deepgram")


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_str(self, data: str) -> None:
        self.sent.append(data)


def _make_flux_stream(*, ws=None, **opts_kwargs):
    # exercise SpeechStreamv2's update logic without starting its connection task
    from livekit.agents.types import NOT_GIVEN
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2, STTOptions

    opts = STTOptions(
        model="flux-general-en",
        sample_rate=16000,
        keyterm=[],
        endpoint_url="wss://api.deepgram.com/v2/listen",
        eot_threshold=opts_kwargs.get("eot_threshold", NOT_GIVEN),
        eager_eot_threshold=opts_kwargs.get("eager_eot_threshold", NOT_GIVEN),
        eot_timeout_ms=opts_kwargs.get("eot_timeout_ms", NOT_GIVEN),
        language_hint=opts_kwargs.get("language_hint", []),
    )
    opts.keyterm = opts_kwargs.get("keyterm", [])
    stream = SimpleNamespace(
        _opts=opts,
        _reconnect_event=asyncio.Event(),
        _reconfigure_atask=None,
        _ws=ws,
    )
    stream._send_configure = SpeechStreamv2._send_configure.__get__(stream)
    return stream


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


async def test_flux_live_fields_reconfigure_without_reconnect():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, eot_threshold=0.7)
    SpeechStreamv2.update_options(
        stream, eot_threshold=0.85, keyterm=["LiveKit"], language_hint=["en"]
    )

    # a Configure send is scheduled in-band, no reconnect
    assert not stream._reconnect_event.is_set()
    assert stream._reconfigure_atask is not None
    await stream._reconfigure_atask

    assert stream._opts.eot_threshold == 0.85
    assert len(ws.sent) == 1
    assert json.loads(ws.sent[0]) == {
        "type": "Configure",
        "thresholds": {"eot_threshold": 0.85},
        "keyterms": ["LiveKit"],
        "language_hints": ["en"],
    }


async def test_flux_reconnect_fields_skip_inband_configure():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws)
    SpeechStreamv2.update_options(stream, model="flux-general-multi", eot_threshold=0.8)

    # model can't be tuned in-band; a reconnect carries every option instead
    assert stream._reconnect_event.is_set()
    assert stream._reconfigure_atask is None
    assert ws.sent == []


async def test_flux_configure_sends_only_changed_fields():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    # stream already has a threshold and keyterms configured
    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, eot_threshold=0.7, keyterm=["existing"])

    # only keyterms change: the Configure delta must omit the unchanged threshold
    SpeechStreamv2.update_options(stream, keyterm=["LiveKit", "Deepgram"])
    await stream._reconfigure_atask

    assert json.loads(ws.sent[0]) == {"type": "Configure", "keyterms": ["LiveKit", "Deepgram"]}


async def test_flux_configure_thresholds_only_delta():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, keyterm=["existing"])

    SpeechStreamv2.update_options(stream, eot_timeout_ms=5000)
    await stream._reconfigure_atask

    assert json.loads(ws.sent[0]) == {"type": "Configure", "thresholds": {"eot_timeout_ms": 5000}}


async def test_flux_configure_sends_are_ordered():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    ws = _FakeWS()
    stream = _make_flux_stream(ws=ws, eot_threshold=0.7)

    # rapid successive updates chain off each other and reach the server in order
    SpeechStreamv2.update_options(stream, eot_threshold=0.8)
    SpeechStreamv2.update_options(stream, eot_threshold=0.9)
    await stream._reconfigure_atask

    assert [json.loads(m)["thresholds"]["eot_threshold"] for m in ws.sent] == [0.8, 0.9]


async def test_flux_configure_noop_when_disconnected():
    from livekit.plugins.deepgram.stt_v2 import SpeechStreamv2

    stream = _make_flux_stream(ws=None)
    # no active connection: the next reconnect carries the latest options instead
    SpeechStreamv2.update_options(stream, keyterm=["LiveKit"])
    await stream._reconfigure_atask

    assert stream._opts.keyterm == ["LiveKit"]
