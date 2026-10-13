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
import threading
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
from livekit.agents.utils import is_given, shortuuid

from .log import logger
from .tool_parser import TOOL_CALL_OPEN, ToolCallStreamParser

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class _ModelKey:
    model: str
    revision: str | None
    device: str
    dtype: str
    trust_remote_code: bool


@dataclass
class _LoadedModel:
    model: Any
    tokenizer: Any
    # generate() is not safe to run concurrently on one model, and jobs in a thread
    # executor share the model across event loops, so this is a thread lock
    generate_lock: threading.Lock


# weights are shared by every LLM instance in the process (one copy per job process,
# or one copy per worker when jobs run in a thread executor)
_models: dict[_ModelKey, _LoadedModel] = {}
_load_locks: dict[_ModelKey, threading.Lock] = {}
# guards the two dicts only; loads hold their model's own lock so they don't block others
_models_lock = threading.Lock()


@dataclass
class _LLMOptions:
    model: str
    revision: str | None
    device: str
    dtype: str
    temperature: float
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    trust_remote_code: bool
    local_files_only: bool

    @property
    def key(self) -> _ModelKey:
        return _ModelKey(
            model=self.model,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.0,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
    ) -> None:
        """Run a Hugging Face chat model locally with ``transformers``.

        Weights are loaded in a background thread when the ``AgentSession`` prewarms the
        LLM, or synchronously by calling :meth:`load` (e.g. from the server's
        ``setup_fnc``). They are cached per process and shared by every instance using
        the same model.

        Args:
            model: Hugging Face Hub repo id or path to a local model directory.
            revision: Hub revision (branch, tag or commit) to load.
            device: ``device_map`` passed to ``from_pretrained``, e.g. ``"auto"``,
                ``"cuda"``, ``"mps"`` or ``"cpu"``.
            dtype: Torch dtype name, e.g. ``"auto"``, ``"float16"`` or ``"bfloat16"``.
            temperature: Sampling temperature. ``0`` uses greedy decoding.
            top_p: Nucleus sampling probability mass.
            max_new_tokens: Maximum number of tokens generated per reply.
            repetition_penalty: Repetition penalty passed to ``generate``.
            trust_remote_code: Allow models that ship their own modeling code.
            local_files_only: Load only from the local Hugging Face cache and never
                download at runtime. Fetch weights ahead of time with the
                ``download-files`` CLI command or :func:`download_model`.
        """
        super().__init__()

        self._opts = _LLMOptions(
            model=model,
            revision=revision,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "huggingface"

    def load(self) -> None:
        """Load the model weights into this process, blocking until they are ready."""
        self._get_model()

    async def _prewarm_impl(self) -> None:
        await asyncio.to_thread(self._get_model)

    def _get_model(self) -> _LoadedModel:
        key = self._opts.key
        with _models_lock:
            if (loaded := _models.get(key)) is not None:
                return loaded
            load_lock = _load_locks.setdefault(key, threading.Lock())

        with load_lock:
            with _models_lock:
                if (loaded := _models.get(key)) is not None:
                    return loaded
            loaded = _load_model(key, local_files_only=self._opts.local_files_only)
            with _models_lock:
                _models[key] = loaded
            return loaded

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
        tool_schemas: list[dict[str, Any]] = []
        required_tool: str | bool = False
        if tools and not (is_given(tool_choice) and tool_choice == "none"):
            tool_schemas = llm.ToolContext(tools).parse_function_tools("openai", strict=False)
            if is_given(tool_choice) and isinstance(tool_choice, dict):
                name = tool_choice["function"]["name"]
                tool_schemas = [t for t in tool_schemas if t["function"]["name"] == name]
                required_tool = name
            elif is_given(tool_choice) and tool_choice == "required":
                required_tool = True

        return LLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            tool_schemas=tool_schemas,
            required_tool=required_tool,
            parallel_tool_calls=parallel_tool_calls is not False,
            conn_options=conn_options,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else {},
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        tool_schemas: list[dict[str, Any]],
        required_tool: str | bool,
        parallel_tool_calls: bool,
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._hf_llm = llm
        self._tool_schemas = tool_schemas
        # True when any tool call is required, or the name of the required tool
        self._required_tool = required_tool
        self._parallel_tool_calls = parallel_tool_calls
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        try:
            loaded = await asyncio.to_thread(self._hf_llm._get_model)
            await self._generate(loaded)
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError(
                f"local generation failed: {type(e).__name__}", retryable=False
            ) from e

    async def _generate(self, loaded: _LoadedModel) -> None:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        tokenizer = loaded.tokenizer
        model = loaded.model
        opts = self._hf_llm._opts

        messages = _to_chat_template_messages(self._chat_ctx)
        template_kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if self._tool_schemas:
            template_kwargs["tools"] = self._tool_schemas

        # there is no constrained decoding here, so a required tool call is enforced by
        # starting the reply with the call ourselves where the template has a call tag,
        # by instruction otherwise, and by validating the result either way
        prefill = ""
        if self._required_tool:
            if TOOL_CALL_OPEN in str(getattr(tokenizer, "chat_template", None) or ""):
                prefill = TOOL_CALL_OPEN + "\n"
                if isinstance(self._required_tool, str):
                    prefill += f'{{"name": {json.dumps(self._required_tool)}, "arguments": '
            else:
                messages = _with_tool_instruction(messages, self._required_tool)

        prompt = tokenizer.apply_chat_template(messages, **template_kwargs) + prefill
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        streamer: Any = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop_event = threading.Event()

        class _StopOnCancel(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return stop_event.is_set()

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": opts.max_new_tokens,
            "repetition_penalty": opts.repetition_penalty,
            "stopping_criteria": StoppingCriteriaList([_StopOnCancel()]),
            "do_sample": opts.temperature > 0,
        }
        if opts.temperature > 0:
            gen_kwargs["temperature"] = opts.temperature
            gen_kwargs["top_p"] = opts.top_p
        gen_kwargs.update(self._extra_kwargs)

        output: list[Any] = []

        def _generate_blocking() -> None:
            try:
                with loaded.generate_lock:
                    if stop_event.is_set():
                        return
                    output.append(model.generate(**gen_kwargs))
            finally:
                # unblocks the streamer consumer when generate() raises or never runs
                streamer.end()

        loop = asyncio.get_running_loop()
        text_ch: asyncio.Queue[str | None] = asyncio.Queue()

        def _consume_blocking() -> None:
            try:
                for text in streamer:
                    loop.call_soon_threadsafe(text_ch.put_nowait, text)
            finally:
                loop.call_soon_threadsafe(text_ch.put_nowait, None)

        request_id = shortuuid("hf_")
        parser = ToolCallStreamParser(
            tool_names=[t["function"]["name"] for t in self._tool_schemas],
            hold_content=bool(self._required_tool),
        )
        parser.feed(prefill)
        completion = ""

        gen_fut = loop.run_in_executor(None, _generate_blocking)
        consume_fut = loop.run_in_executor(None, _consume_blocking)
        try:
            while (text := await text_ch.get()) is not None:
                completion += text
                self._send_content(request_id, parser.feed(text))
        finally:
            # stops generate() early when the stream is cancelled mid-reply
            stop_event.set()
            await asyncio.gather(gen_fut, consume_fut, return_exceptions=True)

        # surface generate() errors that ended the stream early
        gen_fut.result()

        content, tool_calls = parser.flush()
        if not self._parallel_tool_calls:
            tool_calls = tool_calls[:1]
        if self._required_tool and not tool_calls:
            # nothing was emitted yet, so a retry starts clean
            raise APIConnectionError("model did not make the required tool call", retryable=True)

        self._send_content(request_id, content)
        if tool_calls:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            llm.FunctionToolCall(
                                name=call.name,
                                arguments=call.arguments,
                                call_id=shortuuid("call_"),
                            )
                            for call in tool_calls
                        ],
                    ),
                )
            )

        completion_tokens = _generated_token_count(output, prompt_tokens)
        if completion_tokens is None:
            completion_tokens = len(
                tokenizer.encode(prefill + completion, add_special_tokens=False)
            )
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

    def _send_content(self, request_id: str, content: str) -> None:
        if content:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(role="assistant", content=content),
                )
            )


def _to_chat_template_messages(chat_ctx: ChatContext) -> list[dict[str, Any]]:
    """Convert a ChatContext to the message format Hugging Face chat templates expect.

    The OpenAI format is close, but chat templates take tool call arguments as a dict
    and message content as plain text.
    """
    messages, _ = chat_ctx.to_provider_format(format="openai")
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = "\n".join(
                part["text"] for part in content if part.get("type") == "text"
            )

        for tool_call in msg.get("tool_calls") or []:
            function = tool_call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    pass
    return messages


def _with_tool_instruction(
    messages: list[dict[str, Any]], required_tool: str | bool
) -> list[dict[str, Any]]:
    instruction = (
        f"You must reply by calling the `{required_tool}` tool."
        if isinstance(required_tool, str)
        else "You must reply by calling one of the provided tools."
    )
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = f"{first.get('content') or ''}\n\n{instruction}".strip()
        return [first, *messages[1:]]
    return [{"role": "system", "content": instruction}, *messages]


def _generated_token_count(output: list[Any], prompt_tokens: int) -> int | None:
    """Count generated token ids, special tokens included; None if generate() didn't return."""
    if not output:
        return None
    sequences = getattr(output[0], "sequences", output[0])
    shape = getattr(sequences, "shape", None)
    if not shape:
        return None
    return max(int(shape[-1]) - prompt_tokens, 0)


def _load_model(key: _ModelKey, *, local_files_only: bool) -> _LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading model", extra={"model": key.model, "device": key.device})

    kwargs: dict[str, Any] = {
        "revision": key.revision,
        "trust_remote_code": key.trust_remote_code,
        "local_files_only": local_files_only,
    }
    try:
        tokenizer: Any = AutoTokenizer.from_pretrained(key.model, **kwargs)
        model: Any = AutoModelForCausalLM.from_pretrained(
            key.model, dtype=key.dtype, device_map=key.device, **kwargs
        )
    except OSError as e:
        if not local_files_only:
            raise
        raise RuntimeError(
            f"model {key.model!r} is not in the local Hugging Face cache. Download it "
            "ahead of time with `python <your_agent>.py download-files` (set "
            "LIVEKIT_HF_MODELS to the models you use) or "
            "`livekit.plugins.hf.download_model()`, or pass local_files_only=False"
        ) from e

    model.eval()
    _warm_up(model, tokenizer)
    logger.info("model loaded", extra={"model": key.model})
    return _LoadedModel(model=model, tokenizer=tokenizer, generate_lock=threading.Lock())


def _warm_up(model: Any, tokenizer: Any) -> None:
    """Generate one token so the first reply doesn't pay for lazy weight paging and
    kernel compilation (seconds on MPS and CUDA), which would otherwise land on the
    caller's first turn."""
    import torch

    with torch.inference_mode():
        inputs = tokenizer("hi", return_tensors="pt").to(model.device)
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
