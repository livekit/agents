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

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from livekit.agents import APIConnectionError, llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import Tool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .log import logger


@dataclass(frozen=True)
class _LLMOptions:
    model: str
    device: str
    torch_dtype: str
    temperature: float
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    trust_remote_code: bool
    turboquant_enabled: bool
    turboquant_bits: int


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "float16",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.0,
        trust_remote_code: bool = False,
        turboquant: bool = False,
        turboquant_bits: int = 3,
    ) -> None:
        super().__init__()

        self._opts = _LLMOptions(
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            trust_remote_code=trust_remote_code,
            turboquant_enabled=turboquant,
            turboquant_bits=turboquant_bits,
        )

        self._model: Any = None
        self._tokenizer: Any = None
        self._tq_engine: Any = None
        self._load_lock = threading.Lock()
        self._gen_semaphore: asyncio.Semaphore | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "huggingface"

    def prewarm(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading model %s on %s", self._opts.model, self._opts.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._opts.model,
            trust_remote_code=self._opts.trust_remote_code,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._opts.model,
            torch_dtype=getattr(torch, self._opts.torch_dtype),
            device_map=self._opts.device,
            trust_remote_code=self._opts.trust_remote_code,
        )
        self._model.eval()

        if self._opts.turboquant_enabled:
            try:
                from turboquant_gpu import TurboQuantEngine
            except ImportError:
                raise ImportError(
                    "turboquant-gpu is required when turboquant=True. "
                    "Install with: pip install livekit-plugins-hf[turboquant]"
                ) from None

            self._tq_engine = TurboQuantEngine(
                head_dim=self._model.config.head_dim,
                total_bits=self._opts.turboquant_bits,
                device=self._opts.device,
            )
            self._tq_engine.auto_tune(seq_len=2048)
            logger.info(
                "TurboQuant-GPU initialized (%d-bit, head_dim=%d)",
                self._opts.turboquant_bits,
                self._model.config.head_dim,
            )

        logger.info("model %s loaded successfully", self._opts.model)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self.prewarm()

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        self._ensure_model()

        # Convert ChatContext to OpenAI-style messages (what HF apply_chat_template expects)
        messages, _ = chat_ctx.to_provider_format(format="openai")

        # Convert tools to OpenAI function-calling schema
        tool_schemas: list[dict[str, Any]] = []
        if tools:
            tool_ctx = llm.ToolContext(tools)
            tool_schemas = tool_ctx.parse_function_tools("openai")

        if self._gen_semaphore is None:
            self._gen_semaphore = asyncio.Semaphore(1)

        return LLMStream(
            self,
            messages=messages,
            tool_schemas=tool_schemas,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        if self._model is not None:
            import torch

            del self._model
            self._model = None
            del self._tokenizer
            self._tokenizer = None
            self._tq_engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: LLM,
        *,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._hf_llm = llm_instance
        self._messages = messages
        self._tool_schemas = tool_schemas

    async def _run(self) -> None:
        try:
            assert self._hf_llm._gen_semaphore is not None
            async with self._hf_llm._gen_semaphore:
                if self._hf_llm._tq_engine is not None:
                    await self._run_turboquant()
                else:
                    await self._run_streamer()
        except Exception as e:
            raise APIConnectionError(retryable=False) from e

    async def _run_streamer(self) -> None:
        """Standard generation path using TextIteratorStreamer."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        tokenizer = self._hf_llm._tokenizer
        model = self._hf_llm._model
        opts = self._hf_llm._opts

        # Apply chat template
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._tool_schemas:
            try:
                template_kwargs["tools"] = self._tool_schemas
            except Exception:
                logger.warning("model tokenizer does not support tools in chat template")

        input_text = tokenizer.apply_chat_template(self._messages, **template_kwargs)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Streamer for token-by-token output
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Stop flag for cancellation
        stop_event = threading.Event()

        class _CancelCriteria(StoppingCriteria):
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
            ) -> bool:
                return stop_event.is_set()

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": opts.max_new_tokens,
            "do_sample": opts.temperature > 0,
            "repetition_penalty": opts.repetition_penalty,
            "stopping_criteria": StoppingCriteriaList([_CancelCriteria()]),
        }
        if opts.temperature > 0:
            gen_kwargs["temperature"] = opts.temperature
            gen_kwargs["top_p"] = opts.top_p

        request_id = str(uuid.uuid4())[:12]

        # Run blocking generate() in a thread
        loop = asyncio.get_running_loop()
        gen_future = loop.run_in_executor(None, lambda: model.generate(**gen_kwargs))

        # Consume streamer tokens asynchronously
        full_text = ""
        sentinel = object()
        iter_streamer = iter(streamer)

        def _next_token() -> Any:
            try:
                return next(iter_streamer)
            except StopIteration:
                return sentinel

        try:
            while True:
                token_text = await loop.run_in_executor(None, _next_token)
                if token_text is sentinel:
                    break
                if token_text:
                    full_text += token_text
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=request_id,
                            delta=llm.ChoiceDelta(content=token_text, role="assistant"),
                        )
                    )
        except asyncio.CancelledError:
            stop_event.set()
            raise
        finally:
            await gen_future

        # Parse tool calls from full output
        tool_calls = _parse_tool_calls(full_text)
        if tool_calls:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=tool_calls),
                )
            )

        # Accurate token count via re-tokenization
        completion_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=request_id,
                usage=llm.CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        )

    async def _run_turboquant(self) -> None:
        """Step-by-step generation with TurboQuant-GPU KV cache compression."""
        import torch

        tokenizer = self._hf_llm._tokenizer
        model = self._hf_llm._model
        tq_engine = self._hf_llm._tq_engine
        opts = self._hf_llm._opts

        # Apply chat template
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._tool_schemas:
            try:
                template_kwargs["tools"] = self._tool_schemas
            except Exception:
                logger.warning("model tokenizer does not support tools in chat template")

        input_text = tokenizer.apply_chat_template(self._messages, **template_kwargs)
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(model.device)
        prompt_tokens = input_ids.shape[1]

        request_id = str(uuid.uuid4())[:12]
        full_text = ""
        past_key_values: Any = None
        loop = asyncio.get_running_loop()

        def _forward_step(ids: torch.Tensor, past_kv: Any) -> tuple[torch.Tensor, str, Any, bool]:
            with torch.no_grad():
                output = model(ids, past_key_values=past_kv, use_cache=True)

            # Compress KV cache via TurboQuant
            compressed = tq_engine.compress_kv_cache(output.past_key_values)
            new_past_kv = tq_engine.build_cache(compressed)

            # Sample next token
            logits = output.logits[:, -1, :]
            if opts.temperature > 0:
                logits = logits / opts.temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

            token_text = tokenizer.decode(next_token, skip_special_tokens=True)
            is_eos = next_token.item() == tokenizer.eos_token_id
            return next_token.unsqueeze(0), token_text, new_past_kv, is_eos

        try:
            for _ in range(opts.max_new_tokens):
                next_token, token_text, past_key_values, is_eos = await loop.run_in_executor(
                    None, _forward_step, input_ids, past_key_values
                )

                if token_text:
                    full_text += token_text
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=request_id,
                            delta=llm.ChoiceDelta(content=token_text, role="assistant"),
                        )
                    )

                if is_eos:
                    break

                input_ids = next_token
        except asyncio.CancelledError:
            raise

        # Parse tool calls
        tool_calls = _parse_tool_calls(full_text)
        if tool_calls:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=tool_calls),
                )
            )

        # Usage
        completion_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=request_id,
                usage=llm.CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        )


# Tool call patterns emitted by various HuggingFace models
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>\s*(\{.*?\})(?:<\|eom_id\|>|$)", re.DOTALL)


def _parse_tool_calls(text: str) -> list[llm.FunctionToolCall]:
    """Extract tool calls from model output text."""
    tool_calls: list[llm.FunctionToolCall] = []

    # Try <tool_call>...</tool_call> blocks
    matches = _TOOL_CALL_BLOCK_RE.findall(text)
    if not matches:
        # Try <|python_tag|> format (Llama 3.1)
        matches = _PYTHON_TAG_RE.findall(text)

    for raw_json in matches:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.debug("failed to parse tool call JSON: %s", raw_json)
            continue

        name = data.get("name", "")
        arguments = data.get("arguments") or data.get("parameters") or {}
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)

        tool_calls.append(
            llm.FunctionToolCall(
                name=name,
                arguments=arguments,
                call_id=str(uuid.uuid4())[:12],
            )
        )

    return tool_calls
