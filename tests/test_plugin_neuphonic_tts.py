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

"""Neuphonic TTS: settings that are baked into the websocket URL.

``_connect_ws`` builds its url as
``/speak/{lang_code}?speed={speed}&lang_code={lang_code}&encoding=...&voice_id={voice_id}``,
so ``lang_code``, ``speed`` and ``voice_id`` are all fixed at connection time. The
socket then lives in a ``utils.ConnectionPool`` created with
``mark_refreshed_on_get=True``, which restarts ``max_session_duration`` on every
acquire, so a reused connection can outlive a settings change indefinitely. Changing
any of the three must therefore invalidate the pool.
"""

from __future__ import annotations

import pytest

from livekit.plugins.neuphonic import TTS

pytestmark = pytest.mark.plugin("neuphonic")


def _tts_with_recording_pool() -> tuple[TTS, list[bool]]:
    tts = TTS(api_key="test-key")
    calls: list[bool] = []
    tts._pool.invalidate = lambda: calls.append(True)  # type: ignore[method-assign]
    return tts, calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voice_id", "some-other-voice-id"),
        ("speed", 1.25),
        ("lang_code", "es"),
    ],
)
def test_update_options_invalidates_pool_for_connection_params(field: str, value: object) -> None:
    """Each of the three fields is a url parameter, so each must invalidate."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options(**{field: value})

    assert calls == [True], f"changing {field} must invalidate the pooled connection"


def test_update_options_noop_leaves_pool_alone() -> None:
    """A call that changes nothing must not tear down a healthy pooled connection."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options()

    assert calls == []


def test_update_options_invalidates_once_for_several_changes() -> None:
    """Changing all three together still only needs a single invalidation."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options(lang_code="fr", voice_id="another-voice", speed=0.9)

    assert calls == [True]
    assert tts._opts.voice_id == "another-voice"
    assert tts._opts.speed == 0.9
