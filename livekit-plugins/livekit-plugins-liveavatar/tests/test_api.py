import pytest

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.plugins.liveavatar.api import LiveAvatarAPI

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, *, status: int, body: str) -> None:
        self.status = status
        self.ok = False
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self) -> str:
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def post(self, *, url: str, headers: dict, json: dict):
        return self.response


async def test_post_preserves_non_retryable_status_error():
    api = LiveAvatarAPI(
        api_key="test-key",
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        session=_FakeSession(_FakeResponse(status=401, body='{"error":"bad api key"}')),
    )

    with pytest.raises(APIStatusError) as exc:
        await api._post(endpoint="/token", payload={}, headers={})

    assert exc.value.status_code == 401
    assert exc.value.body == '{"error":"bad api key"}'
    assert exc.value.retryable is False
