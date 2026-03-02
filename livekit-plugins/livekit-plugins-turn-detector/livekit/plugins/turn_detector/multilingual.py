from __future__ import annotations

import os
from time import perf_counter

import aiohttp

from livekit.agents import Language, Plugin, get_job_context, llm, utils
from livekit.agents.inference_runner import _InferenceRunner

from .base import MAX_HISTORY_TURNS, EOUModelBase, EOUPlugin, _EUORunnerBase
from .log import logger
from .models import EOUModelType

REMOTE_INFERENCE_TIMEOUT = 2


class _EUORunnerMultilingual(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_multilingual"

    @classmethod
    def model_type(cls) -> EOUModelType:
        return "multilingual"


class MultilingualModel(EOUModelBase):
    def __init__(self, *, unlikely_threshold: float | None = None):
        super().__init__(
            model_type="multilingual",
            unlikely_threshold=unlikely_threshold,
            load_languages=_remote_inference_url() is None,
        )

    def _inference_method(self) -> str:
        return _EUORunnerMultilingual.INFERENCE_METHOD

    async def unlikely_threshold(self, language: Language | None) -> float | None:
        if not language:
            return None

        threshold = await super().unlikely_threshold(language)
        if threshold is None:
            try:
                if url := _remote_inference_url():
                    async with utils.http_context.http_session().post(
                        url=url,
                        json={
                            "language": language.iso,
                        },
                        timeout=aiohttp.ClientTimeout(total=REMOTE_INFERENCE_TIMEOUT),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        threshold = data.get("threshold")
                        if threshold:
                            # cache by base language so the base class lookup finds it
                            self._languages[language.language] = {"threshold": threshold}
            except Exception as e:
                logger.warning("Error fetching threshold for language %s", language, exc_info=e)

        return threshold

    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext,
        *,
        timeout: float | None = 3,
    ) -> float:
        url = _remote_inference_url()
        if not url:
            return await super().predict_end_of_turn(chat_ctx, timeout=timeout)

        messages = chat_ctx.copy(
            exclude_function_call=True, exclude_instructions=True, exclude_empty_message=True
        ).truncate(max_items=MAX_HISTORY_TURNS)

        ctx = get_job_context()
        request = messages.to_dict(exclude_image=True, exclude_audio=True, exclude_timestamp=True)
        request["jobId"] = ctx.job.id
        request["workerId"] = ctx.worker_id
        agent_id = os.getenv("LIVEKIT_AGENT_ID")
        if agent_id:
            request["agentId"] = agent_id

        started_at = perf_counter()
        async with utils.http_context.http_session().post(
            url=url,
            json=request,
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
Plugin.register_plugin(EOUPlugin(_EUORunnerMultilingual))
