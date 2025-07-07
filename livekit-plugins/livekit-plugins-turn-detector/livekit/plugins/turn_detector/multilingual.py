from __future__ import annotations

import os
from time import perf_counter

import aiohttp

from livekit.agents import llm, utils
from livekit.agents.inference_runner import _InferenceRunner

from .base import EOUModelBase, _EUORunnerBase
from .log import logger

REMOTE_INFERENCE_TIMEOUT = 2


class _EUORunnerMultilingual(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_multilingual"

    def __init__(self) -> None:
        super().__init__("multilingual")


class MultilingualModel(EOUModelBase):
    def __init__(self, *, unlikely_threshold: float | None = None):
        super().__init__(
            model_type="multilingual",
            unlikely_threshold=unlikely_threshold,
            load_languages=_remote_inference_url() is None,
        )

    def _inference_method(self) -> str:
        return _EUORunnerMultilingual.INFERENCE_METHOD

    async def unlikely_threshold(self, language: str | None) -> float | None:
        if not language:
            return None

        threshold = await super().unlikely_threshold(language)
        if threshold is None:
            if url := _remote_inference_url():
                async with utils.http_context.http_session().post(
                    url=url,
                    json={
                        "language": language,
                    },
                    timeout=aiohttp.ClientTimeout(total=REMOTE_INFERENCE_TIMEOUT),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    threshold = data.get("threshold")
                    if threshold:
                        self._languages[language] = {"threshold": threshold}

        return threshold

    async def predict_end_of_turn(
        self, chat_ctx: llm.ChatContext, *, timeout: float | None = 3
    ) -> float:
        url = _remote_inference_url()
        if not url:
            return await super().predict_end_of_turn(chat_ctx, timeout=timeout)

        started_at = perf_counter()
        async with utils.http_context.http_session().post(
            url=url,
            json=chat_ctx.to_dict(exclude_function_call=True),
            timeout=aiohttp.ClientTimeout(total=REMOTE_INFERENCE_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            probability = data.get("probability")
            if isinstance(probability, float) and probability >= 0:
                logger.debug(
                    "eou prediction",
                    extra={
                        "eou_probability": probability,
                        "duration": perf_counter() - started_at,
                    },
                )
                return probability
            else:
                # default to indicate no prediction
                return 1


def _remote_inference_url() -> str | None:
    url_base = os.getenv("LIVEKIT_REMOTE_EOT_URL")
    if not url_base:
        return None
    return f"{url_base}/eot/multi"


if not _remote_inference_url():
    _InferenceRunner.register_runner(_EUORunnerMultilingual)
