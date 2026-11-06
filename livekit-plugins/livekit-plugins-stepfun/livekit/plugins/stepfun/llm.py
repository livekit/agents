"""StepFun chat models, over the OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Any

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import _utils
from .log import logger
from .models import (
    DEFAULT_LLM_MODEL,
    REASONING_EFFORT_LLM_MODELS,
    Cluster,
    LLMModels,
    ReasoningEffort,
)

try:
    import livekit.plugins.openai as _openai
except ImportError as exc:
    raise ImportError(
        "stepfun.LLM requires the OpenAI plugin, which provides the "
        "OpenAI-compatible chat client it builds on.\n"
        "    pip install 'livekit-plugins-stepfun[llm]'\n"
        "or\n"
        "    pip install livekit-plugins-openai\n"
        "The STT and TTS components do not need it."
    ) from exc


class LLM(_openai.LLM):  # type: ignore[misc, unused-ignore]
    """StepFun chat completion models."""

    def __init__(
        self,
        *,
        model: LLMModels | str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        cluster: Cluster | None = None,
        base_url: str | None = None,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        """Create an OpenAI-compatible StepFun chat client.

        Args:
            model: Chat model ID, defaulting to step-3.7-flash.
            api_key: API key, falling back to STEPFUN_API_KEY.
            cluster: "global" or "china" (mainland China); defaults to
                STEPFUN_CLUSTER, then global. Credentials are cluster-specific.
            base_url: Override the HTTP base URL; /v1 is appended if absent.
            reasoning_effort: low, medium, or high. Undocumented models log a warning.
            kwargs: Additional options for the OpenAI plugin's LLM client.
        """
        resolved_cluster = _utils.resolve_cluster(cluster)
        resolved_key = _utils.resolve_api_key(api_key)
        resolved_url = _utils.resolve_llm_base_url(base_url, resolved_cluster)

        if is_given(reasoning_effort):
            if model not in REASONING_EFFORT_LLM_MODELS:
                logger.warning(
                    "reasoning_effort is only documented for %s; the server may "
                    "ignore it on this model",
                    ", ".join(sorted(REASONING_EFFORT_LLM_MODELS)),
                    extra={"model": model},
                )
            kwargs["reasoning_effort"] = reasoning_effort

        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=resolved_url,
            **kwargs,
        )

        self._stepfun_cluster = resolved_cluster

    @property
    def provider(self) -> str:
        # The parent reports the base URL's host, which is accurate but reads as
        # an endpoint rather than a vendor in metrics and traces.
        return "StepFun"

    @property
    def cluster(self) -> Cluster:
        return self._stepfun_cluster
