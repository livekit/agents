from __future__ import annotations

import pytest

from livekit import rtc
from livekit.agents import APIStatusError

pytestmark = pytest.mark.plugin("deepgram")


class _Resp:
    status = 401
    headers = {"dg-request-id": "req-1"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self, **kwargs):
        return {"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials."}


class _Session:
    def post(self, **kwargs):
        return _Resp()


async def test_prerecorded_http_error_surfaces_as_status_error():
    from livekit.plugins.deepgram import STT

    stt = STT(api_key="bad", language="en-US", http_session=_Session())  # type: ignore[arg-type]
    frame = rtc.AudioFrame(b"\x00\x00" * 1600, 16000, 1, 1600)
    with pytest.raises(APIStatusError) as exc:
        await stt._recognize_impl([frame])
    assert exc.value.status_code == 401
    assert exc.value.request_id == "req-1"
