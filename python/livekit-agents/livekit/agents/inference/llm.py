from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, get_args

from .. import _core, llm
from .._bindings import APIConnectionError, APIStatusError, APITimeoutError
from .._generated import facades_inference_llm as _facades, public_inference_llm as _public
from .._generated.public_inference_llm import *  # noqa: F403
from ..llm.tool_context import Tool
from ..types import APIConnectOptions

if TYPE_CHECKING:
    import openai


__all__ = [
    "LLM",
    "LLMStream",
]
__all__ += _public.__all__

_REALTIME_MODEL_IDS: frozenset[str] = frozenset(
    model for literal in get_args(RealtimeModels) for model in get_args(literal)
)


def is_realtime_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a realtime model in ``RealtimeModels``."""
    return model in _REALTIME_MODEL_IDS


def drop_unsupported_params(
    model: str, params: dict[str, Any], tools: list[Any] | None = None
) -> dict[str, Any]:
    """Remove parameters that are not supported by the given model.

    Strips any provider prefix (e.g. ``openai/o3-pro`` -> ``o3-pro``) before
    matching against known model prefixes.
    """
    effort = params.get("reasoning_effort")
    drop = _core.llm_unsupported_params(
        model,
        has_tools=bool(tools),
        reasoning_effort=effort if isinstance(effort, str) else None,
        reasoning=getattr(params.get("reasoning"), "effort", None),
    )
    return {k: v for k, v in params.items() if k not in drop}


def min_reasoning_effort(model: str) -> ReasoningEffort | None:
    """Lowest reasoning effort the model supports, or None if the model has no
    reasoning-effort control.

    Strips any provider prefix (e.g. ``openai/gpt-5`` -> ``gpt-5``) before matching.
    """
    return cast("ReasoningEffort | None", _core.min_reasoning_effort(model))


class LLM(_facades.LLM):
    """LiveKit Inference LLM."""

    @classmethod
    def from_model_string(cls, model: str) -> LLM:
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"


class LLMStream(llm.LLMStream):
    """A streamed OpenAI-compatible chat completion; Azure OpenAI and the Google plugin run theirs
    on it with their own client."""

    def __init__(
        self,
        llm_v: llm.LLM,
        *,
        model: LLMModels | str,
        strict_tool_schema: bool,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm_v, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._strict_tool_schema = strict_tool_schema
        self._client = client
        self._extra_kwargs = drop_unsupported_params(model, extra_kwargs, tools=tools)
        self._tool_ctx = llm.ToolContext(tools)

    async def _run(self) -> None:
        # lazy: only the provider plugins running this stream need the OpenAI SDK
        import httpx2 as httpx
        import openai
        from openai.types.chat import ChatCompletionToolParam

        options = self._conn_options
        timeout = httpx.Timeout(
            connect=options.connect_timeout,
            read=options.read_timeout,
            write=options.write_timeout,
            pool=options.pool_timeout,
        )
        client = self._client.with_options(max_retries=0)
        parser = _core.LlmChoiceParser()
        try:
            messages, _ = await self._chat_ctx.to_provider_format(format="openai")
            schemas = self._tool_ctx.parse_function_tools("openai", strict=self._strict_tool_schema)
            extra = dict(self._extra_kwargs)
            if not self._tools:
                extra.pop("tool_choice", None)  # a tool choice without tools is refused
            stream = await client.chat.completions.create(
                messages=messages,
                tools=cast(list[ChatCompletionToolParam], schemas) or openai.omit,
                model=self._model,
                stream_options={"include_usage": True},
                stream=True,
                timeout=timeout,
                **extra,
            )
            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        if (parsed := parser.parse(chunk.id, choice.model_dump())) is not None:
                            self.send(parsed)
                    if chunk.usage is not None and (
                        usage := _core.llm_usage_chunk(chunk.model_dump())
                    ):
                        self.send(usage)
                if (closed := parser.finish()) is not None:
                    self.send(closed)
        except (openai.APITimeoutError, httpx.TimeoutException) as e:
            raise APITimeoutError() from e
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message, status_code=e.status_code, request_id=e.request_id, body=e.body
            ) from e
        except (openai.APIConnectionError, httpx.TransportError) as e:
            raise APIConnectionError() from e


def llm_from_model_string(model: str) -> llm.LLM | llm.RealtimeModel:
    """The inference model a ``llm=`` string names: a realtime model or an LLM."""
    if is_realtime_model(model):
        from .realtime import RealtimeModel

        return RealtimeModel.from_model_string(model)
    return LLM.from_model_string(model)
