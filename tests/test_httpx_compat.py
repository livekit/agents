import httpx
import httpx2
import pytest

from livekit.agents.utils import httpx_compat

pytestmark = pytest.mark.unit


def test_httpx2_timeout_passes_through() -> None:
    timeout = httpx2.Timeout(connect=1, read=2, write=3, pool=4)

    assert httpx_compat.to_httpx2_timeout(timeout) is timeout


def test_legacy_timeout_warning_and_conversion_are_separate() -> None:
    timeout = httpx.Timeout(connect=1, read=2, write=3, pool=4)

    with pytest.warns(DeprecationWarning, match="no longer be supported.*2.0"):
        httpx_compat.warn_on_legacy_timeout(timeout)

    converted = httpx_compat.to_httpx2_timeout(timeout)

    assert isinstance(converted, httpx2.Timeout)
    assert converted.connect == 1
    assert converted.read == 2
    assert converted.write == 3
    assert converted.pool == 4


def test_httpx2_timeout_converts_for_legacy_sdk_without_warning() -> None:
    timeout = httpx2.Timeout(connect=1, read=2, write=3, pool=4)

    converted = httpx_compat.to_legacy_timeout(timeout)

    assert isinstance(converted, httpx.Timeout)
    assert converted.connect == 1
    assert converted.read == 2
    assert converted.write == 3
    assert converted.pool == 4


@pytest.mark.asyncio
async def test_legacy_client_converts_httpx2_configuration() -> None:
    client = httpx_compat.legacy_async_client(
        timeout=httpx2.Timeout(connect=1, read=2, write=3, pool=4),
        limits=httpx2.Limits(
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=30,
        ),
    )

    try:
        assert isinstance(client, httpx.AsyncClient)
        assert client.timeout.connect == 1
        assert client.timeout.read == 2
        assert client.timeout.write == 3
        assert client.timeout.pool == 4
    finally:
        await client.aclose()
