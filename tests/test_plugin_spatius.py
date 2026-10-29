from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import spatius as spatius_sdk

from livekit.plugins.spatius import warmup

pytestmark = pytest.mark.unit


def test_prewarm_needs_an_app_id_and_prefetches_a_token_only_with_an_api_key() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(spatius_sdk, "prewarm", new=AsyncMock()) as sdk_prewarm,
    ):
        warmup.prewarm(MagicMock())
    sdk_prewarm.assert_not_awaited()

    with (
        patch.dict(os.environ, {"SPATIUS_APP_ID": "test-app"}, clear=True),
        patch.object(spatius_sdk, "prewarm", new=AsyncMock()) as sdk_prewarm,
    ):
        warmup.prewarm(MagicMock())
    kwargs = sdk_prewarm.await_args.kwargs
    assert kwargs["api_key"] is None and kwargs["prefetch_session_token"] is False


def test_prewarm_never_raises_on_sdk_failure() -> None:
    with (
        patch.dict(
            os.environ, {"SPATIUS_APP_ID": "test-app", "SPATIUS_API_KEY": "test-key"}, clear=True
        ),
        patch.object(spatius_sdk, "prewarm", new=AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        warmup.prewarm(MagicMock())
