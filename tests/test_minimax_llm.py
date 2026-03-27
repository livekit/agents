"""Tests for the MiniMax LLM plugin."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestMiniMaxLLMInit:
    """Unit tests for MiniMax LLM initialization."""

    def test_init_with_api_key(self):
        """LLM should accept an explicit api_key."""
        from livekit.plugins.minimax import LLM

        llm = LLM(api_key="test-key")
        assert llm.model == "MiniMax-M2.7"

    def test_init_with_env_api_key(self):
        """LLM should read MINIMAX_API_KEY from environment."""
        from livekit.plugins.minimax import LLM

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-test-key"}):
            llm = LLM()
            assert llm.model == "MiniMax-M2.7"

    def test_init_missing_api_key_raises(self):
        """LLM should raise ValueError when no API key is provided."""
        from livekit.plugins.minimax import LLM

        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("MINIMAX_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                    LLM()

    def test_default_model(self):
        """Default model should be MiniMax-M2.7."""
        from livekit.plugins.minimax import LLM

        llm = LLM(api_key="test-key")
        assert llm.model == "MiniMax-M2.7"

    def test_custom_model(self):
        """LLM should accept custom model names."""
        from livekit.plugins.minimax import LLM

        llm = LLM(api_key="test-key", model="MiniMax-M2.5")
        assert llm.model == "MiniMax-M2.5"

    def test_highspeed_model(self):
        """LLM should support the highspeed model variant."""
        from livekit.plugins.minimax import LLM

        llm = LLM(api_key="test-key", model="MiniMax-M2.5-highspeed")
        assert llm.model == "MiniMax-M2.5-highspeed"

    def test_provider_name(self):
        """Provider should report as 'openai' (inherits from OpenAI LLM)."""
        from livekit.plugins.minimax import LLM

        llm = LLM(api_key="test-key")
        # Inherits from OpenAILLM, provider is reported by the base class
        assert hasattr(llm, "provider")

    def test_exports(self):
        """Package should export both TTS and LLM."""
        from livekit.plugins import minimax

        assert hasattr(minimax, "TTS")
        assert hasattr(minimax, "LLM")
        assert "TTS" in minimax.__all__
        assert "LLM" in minimax.__all__


class TestMiniMaxLLMModels:
    """Tests for model type definitions."""

    def test_llm_models_type(self):
        """LLMModels type should include all supported models."""
        from livekit.plugins.minimax.llm import LLMModels

        # Verify the Literal type exists
        assert LLMModels is not None


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxLLMIntegration:
    """Integration tests that require a real MiniMax API key."""

    async def test_simple_chat(self):
        """Test a simple chat completion."""
        from livekit.agents import llm, utils
        from livekit.plugins.minimax import LLM

        utils.http_context._new_session_ctx()
        try:
            minimax_llm = LLM(model="MiniMax-M2.5-highspeed")
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="user", content="Say hello in one word.")

            stream = minimax_llm.chat(chat_ctx=chat_ctx)
            text = ""
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content

            assert len(text) > 0
        finally:
            await utils.http_context._close_http_ctx()

    async def test_chat_with_temperature(self):
        """Test chat with custom temperature."""
        from livekit.agents import llm, utils
        from livekit.plugins.minimax import LLM

        utils.http_context._new_session_ctx()
        try:
            minimax_llm = LLM(model="MiniMax-M2.5-highspeed", temperature=0.5)
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="user", content="What is 2+2? Answer with just the number.")

            stream = minimax_llm.chat(chat_ctx=chat_ctx)
            text = ""
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content

            assert "4" in text
        finally:
            await utils.http_context._close_http_ctx()

    async def test_chat_m27_model(self):
        """Test chat with MiniMax-M2.7 model."""
        from livekit.agents import llm, utils
        from livekit.plugins.minimax import LLM

        utils.http_context._new_session_ctx()
        try:
            minimax_llm = LLM(model="MiniMax-M2.7")
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="user", content="Say 'hi' and nothing else.")

            stream = minimax_llm.chat(chat_ctx=chat_ctx)
            text = ""
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content

            assert len(text) > 0
        finally:
            await utils.http_context._close_http_ctx()
