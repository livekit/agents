from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import spatius as spatius_sdk

from livekit.plugins.spatius import warmup

pytestmark = pytest.mark.unit


def _sdk_prewarm_result() -> MagicMock:
    result = MagicMock()
    result.region = "us-east"
    result.tls_warmed = ["console.us-east.spatius.ai", "api.us-east.spatius.ai"]
    result.session_token_prefetched = True
    return result


def _patch_sdk_prewarm():
    return patch.object(spatius_sdk, "prewarm", new=AsyncMock(return_value=_sdk_prewarm_result()))


def test_prewarm_delegates_to_sdk_with_env_config() -> None:
    proc = MagicMock()
    with (
        patch.dict(
            os.environ,
            {"SPATIUS_APP_ID": "test-app", "SPATIUS_API_KEY": "test-key"},
            clear=True,
        ),
        _patch_sdk_prewarm() as sdk_prewarm,
    ):
        warmup.prewarm(proc)

    sdk_prewarm.assert_awaited_once()
    kwargs = sdk_prewarm.await_args.kwargs
    assert kwargs["app_id"] == "test-app"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["region"] == "auto"
    assert kwargs["console_endpoint_url"] == ""
    assert kwargs["ingress_endpoint_url"] == ""
    assert kwargs["prefetch_session_token"] is True


def test_prewarm_forwards_region_and_endpoints() -> None:
    proc = MagicMock()
    with (
        patch.dict(
            os.environ,
            {
                "SPATIUS_APP_ID": "test-app",
                "SPATIUS_API_KEY": "test-key",
                "SPATIUS_REGION": "us-west",
                "SPATIUS_CONSOLE_ENDPOINT": "https://console.example.com",
                "SPATIUS_INGRESS_ENDPOINT": "wss://api.example.com",
            },
            clear=True,
        ),
        _patch_sdk_prewarm() as sdk_prewarm,
    ):
        warmup.prewarm(proc, prefetch_session_token=False)

    kwargs = sdk_prewarm.await_args.kwargs
    assert kwargs["region"] == "us-west"
    assert kwargs["console_endpoint_url"] == "https://console.example.com"
    assert kwargs["ingress_endpoint_url"] == "wss://api.example.com"
    assert kwargs["prefetch_session_token"] is False


def test_prewarm_skips_without_app_id() -> None:
    proc = MagicMock()
    with patch.dict(os.environ, {}, clear=True), _patch_sdk_prewarm() as sdk_prewarm:
        warmup.prewarm(proc)

    sdk_prewarm.assert_not_awaited()


def test_prewarm_disables_token_prefetch_without_api_key() -> None:
    proc = MagicMock()
    with (
        patch.dict(os.environ, {"SPATIUS_APP_ID": "test-app"}, clear=True),
        _patch_sdk_prewarm() as sdk_prewarm,
    ):
        warmup.prewarm(proc)

    kwargs = sdk_prewarm.await_args.kwargs
    assert kwargs["api_key"] is None
    assert kwargs["prefetch_session_token"] is False


def test_prewarm_never_raises_on_sdk_failure() -> None:
    proc = MagicMock()
    with (
        patch.dict(
            os.environ,
            {"SPATIUS_APP_ID": "test-app", "SPATIUS_API_KEY": "test-key"},
            clear=True,
        ),
        patch.object(spatius_sdk, "prewarm", new=AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        warmup.prewarm(proc)  # logs and swallows
