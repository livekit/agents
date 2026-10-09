from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.sarvam import stt as sarvam_stt, stt_streaming

pytestmark = pytest.mark.plugin("sarvam")


def _query_params(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def _make_realtime_stream() -> stt_streaming.RealtimeSpeechStream:
    stream = object.__new__(stt_streaming.RealtimeSpeechStream)
    stream._opts = stt_streaming.RealtimeSTTOptions(language="hi-IN", api_key="sk_test")
    stream._logger = stt_streaming.logger.getChild("RealtimeSpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._pending_config_update = None
    return stream


def test_keyterms_are_validated_and_copied() -> None:
    keyterms = ["Sarvam", "New Delhi"]
    opts = sarvam_stt.SarvamSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        model="saaras:v4",
        keyterms=keyterms,
    )
    keyterms.append("Vistaar")

    assert opts.keyterms == ["Sarvam", "New Delhi"]

    with pytest.raises(ValueError, match="at most 50 terms"):
        sarvam_stt.SarvamSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            model="saaras:v4",
            keyterms=[str(index) for index in range(51)],
        )

    with pytest.raises(ValueError, match="at most 64 characters"):
        sarvam_stt.SarvamSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            model="saaras:v4",
            keyterms=["x" * 65],
        )

    with pytest.raises(ValueError, match="must be distinct"):
        sarvam_stt.SarvamSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            model="saaras:v4",
            keyterms=["Sarvam", "Sarvam"],
        )

    with pytest.raises(ValueError, match="only supported for model saaras:v4"):
        sarvam_stt.SarvamSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            model="saaras:v3",
            keyterms=["Sarvam"],
        )


def test_legacy_websocket_url_encodes_keyterms_as_json_array() -> None:
    opts = sarvam_stt.SarvamSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        model="saaras:v4",
        keyterms=["Sarvam", "New Delhi"],
    )

    params = _query_params(
        sarvam_stt._build_websocket_url(sarvam_stt.SARVAM_STT_STREAMING_URL, opts)
    )

    assert params["keyterms"] == ['["Sarvam","New Delhi"]']


async def test_rest_request_sends_keyterms_as_one_json_form_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        status = 200

        async def __aenter__(self) -> _Response:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def json(self) -> dict[str, str]:
            return {"transcript": "namaste", "language_code": "hi-IN"}

        async def text(self) -> str:
            return ""

    class _Session:
        def post(self, **kwargs: object) -> _Response:
            captured.update(kwargs)
            return _Response()

    monkeypatch.setattr(
        sarvam_stt.rtc,
        "combine_audio_frames",
        lambda _: SimpleNamespace(to_wav_bytes=lambda: b"wav"),
    )
    stt = sarvam_stt.STT(
        api_key="sk_test",
        model="saaras:v4",
        keyterms=["Sarvam", "New Delhi"],
        http_session=_Session(),  # type: ignore[arg-type]
    )

    await stt._recognize_impl([], conn_options=DEFAULT_API_CONNECT_OPTIONS)

    form_data = captured["data"]
    fields = {field[0]["name"]: field[2] for field in form_data._fields}
    assert fields["keyterms"] == '["Sarvam","New Delhi"]'


def test_realtime_websocket_supports_v4_keyterms() -> None:
    opts = stt_streaming.RealtimeSTTOptions(
        language="hi-IN",
        api_key="sk_test",
        model="saaras:v4",
        keyterms=["Sarvam", "New Delhi"],
    )

    params = _query_params(
        stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)
    )

    assert params["model"] == ["saaras:v4"]
    assert params["keyterms"] == ['["Sarvam","New Delhi"]']


def test_legacy_instance_update_replaces_keyterms() -> None:
    stt = sarvam_stt.STT(
        api_key="sk_test",
        model="saaras:v4",
        keyterms=["Sarvam"],
    )
    updated = ["New Delhi"]

    stt.update_options(keyterms=updated)
    updated.append("Vistaar")

    assert stt._opts.keyterms == ["New Delhi"]


def test_realtime_v3_rejects_keyterms() -> None:
    with pytest.raises(ValueError, match="only supported for model saaras:v4"):
        stt_streaming.RealtimeSTTOptions(
            language="hi-IN",
            api_key="sk_test",
            keyterms=["Sarvam"],
        )


def test_realtime_model_and_keyterms_updates_are_connection_only() -> None:
    stream = _make_realtime_stream()

    stream.update_options(model="saaras:v4", keyterms=["Sarvam"])

    assert stream._opts.model == "saaras:v3-realtime"
    assert stream._opts.keyterms is None
    assert stream._pending_config_update is None


def test_instance_update_applies_v4_keyterms_to_new_realtime_streams() -> None:
    stt = stt_streaming.STTRealtime(
        api_key="sk_test",
        http_session=object(),  # type: ignore[arg-type]
    )

    stt.update_options(model="saaras:v4", keyterms=["Sarvam"])

    assert stt.model == "saaras:v4"
    assert stt._opts.keyterms == ["Sarvam"]


def test_instance_keyterm_update_does_not_reconfigure_an_existing_v3_stream() -> None:
    stt = stt_streaming.STTRealtime(
        api_key="sk_test",
        http_session=object(),  # type: ignore[arg-type]
    )
    stream = _make_realtime_stream()
    stt._streams.add(stream)

    stt.update_options(model="saaras:v4")
    stt.update_options(keyterms=["Sarvam"])

    assert stt._opts.model == "saaras:v4"
    assert stt._opts.keyterms == ["Sarvam"]
    assert stream._opts.model == "saaras:v3-realtime"
    assert stream._opts.keyterms is None
