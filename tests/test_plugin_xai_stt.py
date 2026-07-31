from __future__ import annotations

from typing import Any, cast

import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import is_given

pytestmark = pytest.mark.plugin("xai")


class _FakeStream:
    def __init__(self, *, speaking: bool = False) -> None:
        self._speaking = speaking
        self._pending_keyterm: list[str] | None = None
        self.updated_keyterms: list[list[str]] = []

    def update_options(self, *, keyterm: Any = NOT_GIVEN, **_: Any) -> None:
        if is_given(keyterm):
            self.updated_keyterms.append(list(keyterm))
        self._pending_keyterm = None


def _add_fake_stream(instance: Any, *, speaking: bool = False) -> _FakeStream:
    stream = _FakeStream(speaking=speaking)
    instance._streams.add(cast(Any, stream))
    return stream


def test_reports_keyterm_capability() -> None:
    from livekit.plugins.xai import STT

    instance = STT(api_key="test-key")
    assert instance.capabilities.keyterms is True


def test_merges_user_and_session_keyterms() -> None:
    from livekit.plugins.xai import STT

    instance = STT(api_key="test-key", keyterm=["LiveKit", "shared"])
    stream = _add_fake_stream(instance)

    instance._update_session_keyterms(["shared", "Krisp"])

    assert instance._opts.keyterm == ["LiveKit", "shared", "Krisp"]
    assert stream.updated_keyterms == [["LiveKit", "shared", "Krisp"]]

    instance.update_options(keyterm=["Agents"])
    assert instance._opts.keyterm == ["Agents", "shared", "Krisp"]
    assert stream.updated_keyterms[-1] == ["Agents", "shared", "Krisp"]

    instance._update_session_keyterms([])
    assert instance._opts.keyterm == ["Agents"]
    assert stream.updated_keyterms[-1] == ["Agents"]


def test_defers_session_keyterm_reconnect_while_speaking() -> None:
    from livekit.plugins.xai import STT
    from livekit.plugins.xai.stt import SpeechStream

    instance = STT(api_key="test-key", keyterm=["LiveKit"])
    stream = _add_fake_stream(instance, speaking=True)

    instance._update_session_keyterms(["Krisp"])

    assert stream.updated_keyterms == []
    assert stream._pending_keyterm == ["LiveKit", "Krisp"]

    SpeechStream._on_end_of_speech(cast(Any, stream))
    assert stream.updated_keyterms == [["LiveKit", "Krisp"]]
    assert stream._pending_keyterm is None


def test_validates_user_keyterm_limits() -> None:
    from livekit.plugins.xai import STT

    with pytest.raises(ValueError, match="at most 100 keyterms"):
        STT(api_key="test-key", keyterm=[f"term-{i}" for i in range(101)])

    with pytest.raises(ValueError, match="at most 50 characters"):
        STT(api_key="test-key", keyterm=["x" * 51])


def test_session_keyterms_respect_provider_limits(caplog: pytest.LogCaptureFixture) -> None:
    from livekit.plugins.xai import STT

    instance = STT(api_key="test-key", keyterm=["user"])
    session_keyterms = ["x" * 51, *[f"session-{i}" for i in range(101)]]

    instance._update_session_keyterms(session_keyterms)

    assert len(instance._opts.keyterm) == 100
    assert instance._opts.keyterm[0] == "user"
    assert instance._opts.keyterm[-1] == "session-98"
    assert "longer than 50 characters" in caplog.text
    assert "beyond the 100-term limit" in caplog.text
