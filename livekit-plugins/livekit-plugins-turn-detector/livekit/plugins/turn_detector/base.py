from __future__ import annotations

import asyncio
import json
import math
import time
from abc import ABC, abstractmethod
from typing import Any

from livekit.agents import llm
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.inference_executor import InferenceExecutor
from livekit.agents.job import get_job_context
from livekit.agents.utils import hw

from .log import logger
from .models import HG_MODEL, MODEL_REVISIONS, ONNX_FILENAME, EOUModelType

MAX_HISTORY_TOKENS = 128
MAX_HISTORY_TURNS = 6


def _download_from_hf_hub(repo_id: str, filename: str, **kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    return local_path


class _EUORunnerBase(_InferenceRunner):
    def __init__(self, model_type: EOUModelType):
        super().__init__()
        self._model_revision = MODEL_REVISIONS[model_type]

    def _format_chat_ctx(self, chat_ctx: list[dict[str, Any]]) -> str:
        new_chat_ctx = []
        last_msg: dict[str, Any] | None = None
        for msg in chat_ctx:
            content = msg["content"]
            if not content:
                continue

            # need to combine adjacent turns together to match training data
            if last_msg and last_msg["role"] == msg["role"]:
                last_msg["content"] += content
            else:
                msg["content"] = content
                new_chat_ctx.append(msg)
                last_msg = msg

        convo_text = self._tokenizer.apply_chat_template(
            new_chat_ctx,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )

        # remove the EOU token from current utterance
        ix = convo_text.rfind("<|im_end|>")
        text = convo_text[:ix]
        return text  # type: ignore

    def initialize(self) -> None:
        import onnxruntime as ort  # type: ignore
        from huggingface_hub import errors
        from transformers import AutoTokenizer  # type: ignore

        try:
            local_path_onnx = _download_from_hf_hub(
                HG_MODEL,
                ONNX_FILENAME,
                subfolder="onnx",
                revision=self._model_revision,
                local_files_only=True,
            )
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = max(
                1, math.ceil(hw.get_cpu_monitor().cpu_count()) // 2
            )
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry("session.dynamic_block_base", "4")
            self._session = ort.InferenceSession(
                local_path_onnx, providers=["CPUExecutionProvider"], sess_options=sess_options
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                HG_MODEL,
                revision=self._model_revision,
                local_files_only=True,
                truncation_side="left",
            )

        except (errors.LocalEntryNotFoundError, OSError):
            logger.error(
                f"Could not find model {HG_MODEL} with revision {self._model_revision}. "
                "Make sure you have downloaded the model before running the agent. "
                "Use `python3 your_agent.py download-files` to download the models."
            )
            raise RuntimeError(
                "livekit-plugins-turn-detector initialization failed. "
                f"Could not find model {HG_MODEL} with revision {self._model_revision}."
            ) from None

    def run(self, data: bytes) -> bytes | None:
        data_json = json.loads(data)
        chat_ctx = data_json.get("chat_ctx", None)

        if not chat_ctx:
            raise ValueError("chat_ctx is required on the inference input data")

        start_time = time.perf_counter()

        text = self._format_chat_ctx(chat_ctx)
        inputs = self._tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )
        # Run inference
        outputs = self._session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})
        eou_probability = outputs[0].flatten()[-1]
        end_time = time.perf_counter()

        result: dict[str, Any] = {
            "eou_probability": float(eou_probability),
            "input": text,
            "duration": round(end_time - start_time, 3),
        }
        return json.dumps(result).encode()


class EOUModelBase(ABC):
    def __init__(
        self,
        model_type: EOUModelType = "en",  # default to smaller, english-only model
        inference_executor: InferenceExecutor | None = None,
        # if set, overrides the per-language threshold tuned for accuracy.
        # not recommended unless you're confident in the impact.
        unlikely_threshold: float | None = None,
    ) -> None:
        self._model_type = model_type
        self._executor = inference_executor or get_job_context().inference_executor

        config_fname = _download_from_hf_hub(
            HG_MODEL,
            "languages.json",
            revision=MODEL_REVISIONS[self._model_type],
            local_files_only=True,
        )
        with open(config_fname) as f:
            self._languages = json.load(f)

        self._unlikely_threshold = unlikely_threshold

    @abstractmethod
    def _inference_method(self) -> str: ...

    def unlikely_threshold(self, language: str | None) -> float | None:
        if language is None:
            return None

        lang = language.lower()
        # try the full language code first
        lang_data = self._languages.get(lang)

        # try the base language if the full language code is not found
        if lang_data is None and "-" in lang:
            base_lang = lang.split("-")[0]
            lang_data = self._languages.get(base_lang)

        if not lang_data:
            logger.warning(f"Language {language} not supported by EOU model")
            return None
        # if a custom threshold is provided, use it
        if self._unlikely_threshold is not None:
            return self._unlikely_threshold
        else:
            return lang_data["threshold"]  # type: ignore

    def supports_language(self, language: str | None) -> bool:
        return self.unlikely_threshold(language) is not None

    async def predict_eou(self, chat_ctx: llm.ChatContext) -> float:
        return await self.predict_end_of_turn(chat_ctx)

    # our EOU model inference should be fast, 3 seconds is more than enough
    async def predict_end_of_turn(
        self, chat_ctx: llm.ChatContext, *, timeout: float | None = 3
    ) -> float:
        messages: list[dict[str, Any]] = []

        for item in chat_ctx.items:
            if item.type != "message":
                continue

            if item.role not in ("user", "assistant"):
                continue

            for cnt in item.content:
                if isinstance(cnt, str):
                    messages.append(
                        {
                            "role": item.role,
                            "content": cnt,
                        }
                    )
                    break

        messages = messages[-MAX_HISTORY_TURNS:]

        json_data = json.dumps({"chat_ctx": messages}).encode()

        result = await asyncio.wait_for(
            self._executor.do_inference(self._inference_method(), json_data),
            timeout=timeout,
        )

        assert result is not None, "end_of_utterance prediction should always returns a result"

        result_json: dict[str, Any] = json.loads(result.decode())
        logger.debug(
            "eou prediction",
            extra=result_json,
        )
        return result_json["eou_probability"]  # type: ignore
