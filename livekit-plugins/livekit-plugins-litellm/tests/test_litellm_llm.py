# Copyright 2024-2026, Daily
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LiteLLM LLM plugin."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai

from livekit.plugins.litellm import LLM
from livekit.plugins.litellm.llm import _is_openai_sentinel, _LiteLLMClientShim


def _make_response(content="Hello!"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


class TestLiteLLMShim(unittest.TestCase):
    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_shim_dispatches_to_litellm(self, mock_acompletion):
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
        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(model="openai/gpt-4o", messages=[])
        )
        self.assertTrue(mock_acompletion.call_args.kwargs["drop_params"])

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_drop_params_can_be_overridden(self, mock_acompletion):
        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(model="openai/gpt-4o", messages=[], drop_params=False)
        )
        self.assertFalse(mock_acompletion.call_args.kwargs["drop_params"])

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_openai_omit_sentinel_stripped(self, mock_acompletion):
        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(
                model="openai/gpt-4o",
                messages=[],
                tools=openai.NOT_GIVEN,
            )
        )
        self.assertNotIn("tools", mock_acompletion.call_args.kwargs)

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_httpx_timeout_converted_to_float(self, mock_acompletion):
        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(
                model="openai/gpt-4o",
                messages=[],
                timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            )
        )
        timeout_val = mock_acompletion.call_args.kwargs["timeout"]
        self.assertIsInstance(timeout_val, float)
        self.assertEqual(timeout_val, 30.0)

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_extra_headers_stripped(self, mock_acompletion):
        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(
                model="openai/gpt-4o",
                messages=[],
                extra_headers={"X-LiveKit-Something": "value"},
            )
        )
        self.assertNotIn("extra_headers", mock_acompletion.call_args.kwargs)

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_api_key_forwarded_to_litellm(self, mock_acompletion):
        shim = _LiteLLMClientShim(api_key="sk-test-123")
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(model="openai/gpt-4o", messages=[])
        )
        self.assertEqual(mock_acompletion.call_args.kwargs["api_key"], "sk-test-123")

    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_response())
    def test_no_api_key_when_not_provided(self, mock_acompletion):
        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(
            shim.chat.completions.create(model="openai/gpt-4o", messages=[])
        )
        self.assertNotIn("api_key", mock_acompletion.call_args.kwargs)

    def test_shim_has_base_url(self):
        shim = _LiteLLMClientShim()
        self.assertEqual(shim._base_url.netloc, b"litellm")

    def test_shim_close_is_noop(self):
        shim = _LiteLLMClientShim()
        asyncio.get_event_loop().run_until_complete(shim.close())

    def test_openai_sentinel_detection(self):
        self.assertTrue(_is_openai_sentinel(openai.NOT_GIVEN))
        self.assertFalse(_is_openai_sentinel("hello"))
        self.assertFalse(_is_openai_sentinel(42))
        self.assertFalse(_is_openai_sentinel(None))


class TestLiteLLMLLM(unittest.TestCase):
    def test_llm_instantiates(self):
        llm_inst = LLM(model="anthropic/claude-sonnet-4-6")
        self.assertEqual(llm_inst.model, "anthropic/claude-sonnet-4-6")
        self.assertIsInstance(llm_inst._client, _LiteLLMClientShim)

    def test_default_model(self):
        llm_inst = LLM()
        self.assertEqual(llm_inst.model, "openai/gpt-4o")

    def test_provider_is_litellm(self):
        llm_inst = LLM(model="anthropic/claude-sonnet-4-6")
        self.assertEqual(llm_inst.provider, "litellm")

    def test_api_key_reaches_shim(self):
        llm_inst = LLM(model="openai/gpt-4o", api_key="sk-test-456")
        self.assertEqual(llm_inst._client.chat.completions._api_key, "sk-test-456")


if __name__ == "__main__":
    unittest.main()
