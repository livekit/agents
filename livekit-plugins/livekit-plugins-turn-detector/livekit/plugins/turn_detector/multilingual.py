from __future__ import annotations

import os

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
            load_languages=remote_inference_url() is None,
        )

    def _inference_method(self) -> str:
        return _EUORunnerMultilingual.INFERENCE_METHOD

    async def unlikely_threshold(self, language: str | None) -> float | None:
        threshold = await super().unlikely_threshold(language)
        if threshold is None:
            if url := remote_inference_url():
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
                        self._languages[language] = {"threshold": data["threshold"]}

        return threshold

    async def predict_end_of_turn(
        self, chat_ctx: llm.ChatContext, *, timeout: float | None = 3
    ) -> float:
        url = remote_inference_url()
        if not url:
            return super().predict_end_of_turn(chat_ctx, timeout=timeout)

        async with utils.http_context.http_session().post(
            url=url,
            json=chat_ctx.to_dict(exclude_function_call=True),
            timeout=aiohttp.ClientTimeout(total=REMOTE_INFERENCE_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            probability = data.get("probability")
            if probability:
                logger.debug(
                    "eou prediction",
                    extra=data,
                )
                return probability
            else:
                # default to indicate no prediction
                return 1


def remote_inference_url() -> str | None:
    return os.getenv("LIVEKIT_EOT_MULTI_URL")


if not remote_inference_url():
    _InferenceRunner.register_runner(_EUORunnerMultilingual)
