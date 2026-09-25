# Copyright 2024-2026, Daily
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LiteLLM LLM plugin."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from livekit.plugins.litellm import LLM
from livekit.plugins.litellm.llm import _LiteLLMClientShim


def _make_response(content="Hello!", finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].delta = MagicMock()
    resp.choices[0].delta.content = content
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


class TestLiteLLMShim(unittest.TestCase):
    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_shim_dispatches_to_litellm(self, mock_acompletion):
        import asyncio

        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(model="anthropic/claude-sonnet-4-6", messages=[])
        )
        mock_acompletion.assert_called_once()
        kw = mock_acompletion.call_args.kwargs
        self.assertEqual(kw["model"], "anthropic/claude-sonnet-4-6")
        self.assertTrue(kw["drop_params"])

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_drop_params_default_true(self, mock_acompletion):
        import asyncio

        shim = _LiteLLMClientShim()

        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(model="openai/gpt-4o", messages=[])
        )
        self.assertTrue(mock_acompletion.call_args.kwargs["drop_params"])

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_drop_params_can_be_overridden(self, mock_acompletion):
        import asyncio

        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(model="openai/gpt-4o", messages=[], drop_params=False)
        )
        self.assertFalse(mock_acompletion.call_args.kwargs["drop_params"])

    def test_shim_has_base_url(self):
        shim = _LiteLLMClientShim()
        self.assertEqual(shim._base_url.netloc, b"litellm")

    def test_shim_close_is_noop(self):
        import asyncio

        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(shim.close())


class TestLiteLLMLLM(unittest.TestCase):
    def test_llm_instantiates(self):
        llm = LLM(model="anthropic/claude-sonnet-4-6")
        self.assertEqual(llm.model, "anthropic/claude-sonnet-4-6")
        self.assertIsInstance(llm._client, _LiteLLMClientShim)

    def test_default_model(self):
        llm = LLM()
        self.assertEqual(llm.model, "openai/gpt-4o")

    def test_provider_is_litellm(self):
        llm = LLM(model="anthropic/claude-sonnet-4-6")
        self.assertEqual(llm.provider, "litellm")


if __name__ == "__main__":
    unittest.main()
