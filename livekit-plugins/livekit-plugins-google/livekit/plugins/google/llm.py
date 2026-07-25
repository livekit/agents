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

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import google.auth.credentials
from google.auth._default_async import default_async
from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, llm, utils
from livekit.agents.llm import ToolChoice, utils as llm_utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import ChatModels
from .utils import create_tools_config, to_response_format
from .version import __version__


def _is_gemini_3_model(model: str) -> bool:
    """Check if model is Gemini 3 series"""
    return "gemini-3" in model.lower() or model.lower().startswith("gemini-3")


def _function_calling_config(
    tool_choice: NotGivenOr[ToolChoice], function_tools: Iterable[str]
) -> types.FunctionCallingConfig | None:
    """Map a LiveKit `tool_choice` to Gemini's FunctionCallingConfig (None if unset)."""
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=[tool_choice["function"]["name"]],
        )
    if tool_choice == "required":
        return types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=list(function_tools) or None,
        )
    if tool_choice == "auto":
        return types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
    if tool_choice == "none":
        return types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
    return None


def _is_gemini_3_flash_model(model: str) -> bool:
    """Check if model is Gemini 3 Flash"""
    m = model.lower()
    return m.startswith("gemini-3") and "flash" in m


def _requires_thought_signatures(model: str) -> bool:
    """Check if model requires thought_signature handling for multi-turn function calling.

    Gemini 2.5+ models require thought signatures to be stored from responses and
    passed back in subsequent requests for proper multi-turn function calling.
    """
    if _is_gemini_3_model(model):
        return True
    model_lower = model.lower()
    return "gemini-2.5" in model_lower or model_lower.startswith("gemini-2.5")


@dataclass
class _LLMOptions:
    model: ChatModels | str
    temperature: NotGivenOr[float]
    tool_choice: NotGivenOr[ToolChoice]
    vertexai: NotGivenOr[bool]
    project: NotGivenOr[str]
    location: NotGivenOr[str]
    max_output_tokens: NotGivenOr[int]
    top_p: NotGivenOr[float]
    top_k: NotGivenOr[float]
    presence_penalty: NotGivenOr[float]
    frequency_penalty: NotGivenOr[float]
    thinking_config: NotGivenOr[types.ThinkingConfigOrDict]
    retrieval_config: NotGivenOr[types.RetrievalConfigOrDict]
    automatic_function_calling_config: NotGivenOr[types.AutomaticFunctionCallingConfigOrDict]
    http_options: NotGivenOr[types.HttpOptions]
    seed: NotGivenOr[int]
    safety_settings: NotGivenOr[list[types.SafetySettingOrDict]]
    service_tier: NotGivenOr[types.ServiceTier]
    cached_content: NotGivenOr[str]
    media_resolution: NotGivenOr[types.MediaResolution]


BLOCKED_REASONS = [
    types.FinishReason.SAFETY,
    types.FinishReason.SPII,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.LANGUAGE,
    types.FinishReason.RECITATION,
]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: ChatModels | str = "gemini-2.5-flash",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        thinking_config: NotGivenOr[types.ThinkingConfigOrDict] = NOT_GIVEN,
        retrieval_config: NotGivenOr[types.RetrievalConfigOrDict] = NOT_GIVEN,
        automatic_function_calling_config: NotGivenOr[
            types.AutomaticFunctionCallingConfigOrDict
        ] = NOT_GIVEN,
        http_options: NotGivenOr[types.HttpOptions] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        safety_settings: NotGivenOr[list[types.SafetySettingOrDict]] = NOT_GIVEN,
        service_tier: NotGivenOr[types.ServiceTier] = NOT_GIVEN,
        cached_content: NotGivenOr[str] = NOT_GIVEN,
        media_resolution: NotGivenOr[types.MediaResolution] = NOT_GIVEN,
        credentials: google.auth.credentials.Credentials | None = None,
    ) -> None:
        """
        Create a new instance of Google GenAI LLM.

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file or use any of the other Google Cloud auth methods.
        The Google Cloud project and location can be set via `project` and `location` arguments or the environment variables
        `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. By default, the project is inferred from the service account key file,
        and the location defaults to "us-central1".
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        Args:
            model (ChatModels | str, optional): The model name to use. Defaults to "gemini-2.0-flash-001".
            api_key (str, optional): The API key for Google Gemini. If not provided, it attempts to read from the `GOOGLE_API_KEY` environment variable.
            vertexai (bool, optional): Whether to use VertexAI. If not provided, it attempts to read from the `GOOGLE_GENAI_USE_VERTEXAI` environment variable. Defaults to False.
                project (str, optional): The Google Cloud project to use (only for VertexAI). Defaults to None.
                location (str, optional): The location to use for VertexAI API requests. Defaults value is "us-central1".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_output_tokens (int, optional): Maximum number of tokens to generate in the output. Defaults to None.
            top_p (float, optional): The nucleus sampling probability for response generation. Defaults to None.
            top_k (int, optional): The top-k sampling value for response generation. Defaults to None.
            presence_penalty (float, optional): Penalizes the model for generating previously mentioned concepts. Defaults to None.
            frequency_penalty (float, optional): Penalizes the model for repeating words. Defaults to None.
            tool_choice (ToolChoice, optional): Specifies whether to use tools during response generation. Defaults to "auto".
            thinking_config (ThinkingConfigOrDict, optional): The thinking configuration for response generation. Defaults to None.
            retrieval_config (RetrievalConfigOrDict, optional): The retrieval configuration for response generation. Defaults to None.
            automatic_function_calling_config (AutomaticFunctionCallingConfigOrDict, optional): The automatic function calling configuration for response generation. Defaults to None.
            http_options (HttpOptions, optional): The HTTP options to use for the session.
            seed (int, optional): Random seed for reproducible generation. Defaults to None.
            safety_settings (list[SafetySettingOrDict], optional): Safety settings for content filtering. Defaults to None.
            service_tier (types.ServiceTier, optional): The service tier for the request (e.g. types.ServiceTier.PRIORITY). Defaults to None.
            cached_content (str, optional): Resource name of an explicit context cache to attach to every request from this LLM instance, e.g. ``"cachedContents/abc123"`` for the Gemini API or ``"projects/<project>/locations/<location>/cachedContents/abc123"`` for VertexAI. The cache must already exist — create it via ``client.caches.create(...)`` and pass the returned ``name``. Gemini rejects ``generateContent`` requests that combine ``cached_content`` with ``system_instruction``, ``tools``, or ``tool_config``, so when this option is set the plugin bakes those fields out of every outgoing request; the cache resource itself must contain whichever of them the model needs (typically the system prompt and the tool schemas). Useful for long-lived static prefixes where implicit caching is unreliable. See https://ai.google.dev/gemini-api/docs/caching for details and minimum prefix-token requirements. Defaults to None.
            media_resolution (types.MediaResolution, optional): The media resolution for the request. Defaults to None.
        """  # noqa: E501
        super().__init__()
        gcp_project = project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location: str | None = (
            location
            if is_given(location)
            else os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        )
        use_vertexai = (
            vertexai
            if is_given(vertexai)
            else os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in ["true", "1"]
        )
        gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")

        if use_vertexai:
            if not gcp_project:
                _, gcp_project = default_async(  # type: ignore
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if not gcp_project or not gcp_location:
                raise ValueError(
                    "Project is required for VertexAI via project kwarg or GOOGLE_CLOUD_PROJECT environment variable"  # noqa: E501
                )
            gemini_api_key = None  # VertexAI does not require an API key

        else:
            gcp_project = None
            gcp_location = None
            if credentials is not None:
                logger.warning(
                    "'credentials' is only applicable to VertexAI and will be ignored for the Gemini API"
                )
                credentials = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"  # noqa: E501
                )

        # Validate thinking_config
        if is_given(thinking_config):
            _thinking_budget = None
            _thinking_level = None
            if isinstance(thinking_config, dict):
                _thinking_budget = thinking_config.get("thinking_budget")
                _thinking_level = thinking_config.get("thinking_level")
            elif isinstance(thinking_config, types.ThinkingConfig):
                _thinking_budget = thinking_config.thinking_budget
                _thinking_level = getattr(thinking_config, "thinking_level", None)

            if _thinking_budget is not None:
                if not isinstance(_thinking_budget, int):
                    raise ValueError("thinking_budget inside thinking_config must be an integer")

        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            vertexai=use_vertexai,
            project=project,
            location=location,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            thinking_config=thinking_config,
            retrieval_config=retrieval_config,
            automatic_function_calling_config=automatic_function_calling_config,
            http_options=http_options,
            seed=seed,
            safety_settings=safety_settings,
            service_tier=service_tier,
            cached_content=cached_content,
            media_resolution=media_resolution,
        )
        self._client = Client(
            api_key=gemini_api_key,
            vertexai=use_vertexai,
            project=gcp_project,
            location=gcp_location,
            credentials=credentials,
        )
        # Store thought_signatures for Gemini 2.5+ multi-turn function calling
        self._thought_signatures: dict[str, bytes] = {}

    async def _prewarm_impl(self) -> None:
        # also fetches auth tokens ahead of time on vertexai
        await self._client.aio.models.list(config={"page_size": 1})

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        if self._client.vertexai:
            return "Vertex AI"
        else:
            return "Gemini"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            types.SchemaUnion | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}

        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        tool_ctx = llm.ToolContext(tools or [])
        # Mixing built-in (provider) tools with function tools is only supported on the
        # Gemini 3 Developer API (not Vertex).
        # https://ai.google.dev/gemini-api/docs/tool-combination
        is_gemini_3_api = _is_gemini_3_model(self._opts.model) and not self._client.vertexai

        # cached_content may arrive from the constructor or via extra_kwargs; `_run()` keys
        # off the merged request, so match it here.
        using_cache = is_given(self._opts.cached_content) or "cached_content" in extra
        if using_cache:
            # tools/tool_config/system_instruction must be baked into the CachedContent
            # resource, not sent on the request. Don't build them here, and drop any the
            # caller passed directly via extra_kwargs.
            dropped = [k for k in ("tools", "tool_config") if k in extra]
            for key in dropped:
                extra.pop(key, None)
            if tool_ctx.function_tools or tool_ctx.provider_tools or dropped:
                logger.warning(
                    "gemini llm: ignoring tools; bake them into the CachedContent resource",
                    extra={"model": self._opts.model},
                )
        else:
            retrieval_config = (
                self._opts.retrieval_config if is_given(self._opts.retrieval_config) else None
            )
            if isinstance(retrieval_config, dict):
                retrieval_config = types.RetrievalConfig.model_validate(retrieval_config)

            # create_tools_config is the single authority on whether built-in and function
            # tools are actually combined; `mixed` gates the server-side invocation flag.
            tools_config, mixed = create_tools_config(tool_ctx, allow_mixed_tools=is_gemini_3_api)
            fcc = _function_calling_config(tool_choice, tool_ctx.function_tools)

            if fcc is not None or retrieval_config or mixed:
                extra["tool_config"] = types.ToolConfig(
                    function_calling_config=fcc,
                    retrieval_config=retrieval_config,
                    include_server_side_tool_invocations=mixed or None,
                )

            if tools_config:
                extra["tools"] = tools_config

        if is_given(response_format):
            extra["response_schema"] = to_response_format(response_format)
            extra["response_mime_type"] = "application/json"

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature
        if is_given(self._opts.max_output_tokens):
            extra["max_output_tokens"] = self._opts.max_output_tokens
        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p
        if is_given(self._opts.top_k):
            extra["top_k"] = self._opts.top_k
        if is_given(self._opts.presence_penalty):
            extra["presence_penalty"] = self._opts.presence_penalty
        if is_given(self._opts.frequency_penalty):
            extra["frequency_penalty"] = self._opts.frequency_penalty
        if is_given(self._opts.seed):
            extra["seed"] = self._opts.seed

        # Handle thinking_config based on model version
        if is_given(self._opts.thinking_config):
            is_gemini_3 = _is_gemini_3_model(self._opts.model)
            is_gemini_3_flash = _is_gemini_3_flash_model(self._opts.model)
            thinking_cfg = self._opts.thinking_config

            # Extract both parameters
            _budget = None
            _level: str | types.ThinkingLevel | None = None
            if isinstance(thinking_cfg, dict):
                _budget = thinking_cfg.get("thinking_budget")
                _level = thinking_cfg.get("thinking_level")
            elif isinstance(thinking_cfg, types.ThinkingConfig):
                _budget = thinking_cfg.thinking_budget
                _level = getattr(thinking_cfg, "thinking_level", None)

            if is_gemini_3:
                # Gemini 3: only support thinking_level
                if _budget is not None and _level is None:
                    logger.warning(
                        f"Model {self._opts.model} is Gemini 3 which does not support thinking_budget. "
                        "Please use thinking_level ('low' or 'high') instead. Ignoring thinking_budget."
                    )
                if _level is None:
                    # If no thinking_level is provided, use the fastest thinking level
                    if is_gemini_3_flash:
                        _level = "minimal"
                    else:
                        _level = "low"
                # Use thinking_level only (pass as dict since SDK may not have this field yet)
                extra["thinking_config"] = {"thinking_level": _level}

            else:
                # Gemini 2.5 and earlier: only support thinking_budget
                if _level is not None and _budget is None:
                    raise ValueError(
                        f"Model {self._opts.model} does not support thinking_level. "
                        "Please use thinking_budget (int) instead for Gemini 2.5 and earlier models."
                    )
                if _budget is not None:
                    # Use thinking_budget only
                    extra["thinking_config"] = types.ThinkingConfig(thinking_budget=_budget)
                else:
                    # Pass through original config if no specific handling needed
                    extra["thinking_config"] = self._opts.thinking_config

        if is_given(self._opts.automatic_function_calling_config):
            extra["automatic_function_calling"] = self._opts.automatic_function_calling_config

        if is_given(self._opts.safety_settings):
            extra["safety_settings"] = self._opts.safety_settings

        if is_given(self._opts.service_tier):
            extra["service_tier"] = self._opts.service_tier

        if is_given(self._opts.cached_content):
            extra["cached_content"] = self._opts.cached_content

        if is_given(self._opts.media_resolution):
            extra["media_resolution"] = self._opts.media_resolution

        return LLMStream(
            self,
            client=self._client,
            model=self._opts.model,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_v: LLM,
        *,
        client: Client,
        model: str | ChatModels,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        tools: list[llm.Tool],
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm_v, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._client = client
        self._model = model
        self._llm: LLM = llm_v
        self._extra_kwargs = extra_kwargs
        self._tool_ctx = llm.ToolContext(tools)

    async def _run(self) -> None:
        retryable = True
        request_id = utils.shortuuid()

        try:
            # Pass thought_signatures for Gemini 2.5+ multi-turn function calling
            thought_sigs = (
                self._llm._thought_signatures if _requires_thought_signatures(self._model) else None
            )
            turns_dict, extra_data = self._chat_ctx.to_provider_format(
                format="google", thought_signatures=thought_sigs
            )

            turns = [types.Content.model_validate(turn) for turn in turns_dict]
            # Request shaping (tools/tool_config) is done in `chat()`. When a cache is
            # attached, `system_instruction` must also live inside the CachedContent
            # resource, so it's dropped from the outgoing request below.
            using_cache = "cached_content" in self._extra_kwargs
            if is_given(self._llm._opts.http_options):
                http_options = self._llm._opts.http_options.model_copy()
                if http_options.timeout is None:
                    http_options.timeout = int(self._conn_options.timeout * 1000)
            else:
                http_options = types.HttpOptions(timeout=int(self._conn_options.timeout * 1000))

            headers = dict(http_options.headers or {})
            headers["x-goog-api-client"] = f"livekit-agents/{__version__}"
            http_options.headers = headers
            config = types.GenerateContentConfig(
                system_instruction=(
                    None
                    if using_cache
                    else (
                        [types.Part(text=content) for content in extra_data.system_messages]
                        if extra_data.system_messages
                        else None
                    )
                ),
                http_options=http_options,
                **self._extra_kwargs,
            )

            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=cast(types.ContentListUnion, turns),
                config=config,
            )

            response_generated = False
            finish_reason: types.FinishReason | None = None
            async for response in stream:
                if response.prompt_feedback:
                    raise APIStatusError(
                        response.prompt_feedback.model_dump_json(),
                        retryable=False,
                        request_id=request_id,
                    )

                if response.usage_metadata is not None:
                    usage = response.usage_metadata
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=request_id,
                            usage=llm.CompletionUsage(
                                completion_tokens=usage.candidates_token_count or 0,
                                prompt_tokens=usage.prompt_token_count or 0,
                                prompt_cached_tokens=usage.cached_content_token_count or 0,
                                total_tokens=usage.total_token_count or 0,
                            ),
                        )
                    )

                if not response.candidates:
                    continue

                if len(response.candidates) > 1:
                    logger.warning(
                        "gemini llm: there are multiple candidates in the response, returning response from the first one."  # noqa: E501
                    )

                candidate = response.candidates[0]

                if candidate.finish_reason is not None:
                    finish_reason = candidate.finish_reason
                    if candidate.finish_reason in BLOCKED_REASONS:
                        raise APIStatusError(
                            f"generation blocked by gemini: {candidate.finish_reason}",
                            retryable=False,
                            request_id=request_id,
                        )

                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    chat_chunk = self._parse_part(request_id, part)
                    if chat_chunk is not None:
                        # Only count parts that actually produce output. Thought
                        # summaries (dropped in _parse_part) must not mark the
                        # response as generated, otherwise a thought-only turn would
                        # be treated as successful and skip the retry below.
                        response_generated = True
                        retryable = False
                        self._event_ch.send_nowait(chat_chunk)

            if not response_generated:
                raise APIStatusError(
                    "no response generated",
                    retryable=retryable,
                    request_id=request_id,
                    body=f"finish reason: {finish_reason}",
                )

        except ClientError as e:
            raise APIStatusError(
                "gemini llm: client error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                request_id=request_id,
                retryable=True if e.code in {429, 499} else False,
            ) from e
        except ServerError as e:
            raise APIStatusError(
                "gemini llm: server error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                request_id=request_id,
                retryable=retryable,
            ) from e
        except APIError as e:
            raise APIStatusError(
                "gemini llm: api error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                request_id=request_id,
                retryable=retryable,
            ) from e
        except (APIStatusError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError(
                f"gemini llm: error generating content {str(e)}",
                retryable=retryable,
            ) from e

    def _parse_part(self, id: str, part: types.Part) -> llm.ChatChunk | None:
        if part.function_call:
            tool_call = llm.FunctionToolCall(
                arguments=json.dumps(part.function_call.args),
                name=part.function_call.name,
                call_id=part.function_call.id or utils.shortuuid("function_call_"),
            )

            # Store thought_signature for Gemini 2.5+ multi-turn function calling
            if (
                _requires_thought_signatures(self._model)
                and hasattr(part, "thought_signature")
                and part.thought_signature
            ):
                self._llm._thought_signatures[tool_call.call_id] = part.thought_signature

            chat_chunk = llm.ChatChunk(
                id=id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    tool_calls=[tool_call],
                    content=None,
                ),
            )
            return chat_chunk

        if not part.text:
            return None

        # Drop thought summaries: when include_thoughts is enabled Gemini streams its
        # reasoning back as parts flagged part.thought=True. These must not reach
        # ChoiceDelta.content, otherwise TTS would speak the model's internal reasoning.
        if part.thought:
            return None

        return llm.ChatChunk(
            id=id,
            delta=llm.ChoiceDelta(content=part.text, role="assistant"),
        )
