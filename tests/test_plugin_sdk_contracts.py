from __future__ import annotations

from collections.abc import Callable

import httpx2 as httpx
import pytest

from livekit.plugins import anthropic, openai

pytestmark = pytest.mark.unit


class PublicSDKClient:
    base_url = httpx.URL("https://provider.example:8443/v1")


@pytest.mark.parametrize("factory", [openai.LLM, openai.STT, openai.TTS, anthropic.LLM])
def test_provider_identity_uses_the_sdk_public_base_url(factory: Callable) -> None:
    model = factory(api_key="test", client=PublicSDKClient())
    assert model.provider == "provider.example:8443"
