# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AsyncAI TTS: settings that are baked into the websocket init payload.

``_connect_ws`` sends ``model_id``, ``voice`` and ``language`` once, right after the
handshake. The connection then lives in a ``utils.ConnectionPool`` created with
``mark_refreshed_on_get=True``, so its ``max_session_duration`` restarts on every
acquire and a reused connection can outlive a settings change indefinitely. Changing
any of those three therefore has to invalidate the pool, or synthesis silently
continues with the previous voice/model/language.
"""

from __future__ import annotations

import pytest

from livekit.plugins.asyncai import TTS

pytestmark = pytest.mark.plugin("asyncai")


def _tts_with_recording_pool() -> tuple[TTS, list[bool]]:
    tts = TTS(api_key="test-key")
    calls: list[bool] = []
    tts._pool.invalidate = lambda: calls.append(True)  # type: ignore[method-assign]
    return tts, calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voice", "some-other-voice-id"),
        ("model", "async_flash_v1.0"),
        ("language", "es"),
    ],
)
def test_update_options_invalidates_pool_for_connection_params(field: str, value: str) -> None:
    """Each of the three fields is sent in the init payload, so each must invalidate."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options(**{field: value})

    assert getattr(tts._opts, field) == value
    assert calls == [True], f"changing {field} must invalidate the pooled connection"


def test_update_options_noop_leaves_pool_alone() -> None:
    """A call that changes nothing must not tear down a healthy pooled connection."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options()

    assert calls == []


def test_update_options_invalidates_once_for_several_changes() -> None:
    """Changing all three together still only needs a single invalidation."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options(model="async_flash_v1.0", language="fr", voice="another-voice")

    assert calls == [True]
    assert tts._opts.model == "async_flash_v1.0"
    assert tts._opts.language == "fr"
    assert tts._opts.voice == "another-voice"
