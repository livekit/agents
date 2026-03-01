from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from huggingface_hub import errors

from livekit.agents import Language, Plugin, llm
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.inference_executor import InferenceExecutor
from livekit.agents.job import get_job_context
from livekit.agents.utils import hw

from .log import logger
from .models import HG_MODEL, MODEL_REVISIONS, ONNX_FILENAME, EOUModelType
from .version import __version__

MAX_HISTORY_TOKENS = 128
MAX_HISTORY_TURNS = 6


def _download_from_hf_hub(repo_id: str, filename: str, **kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    except (errors.LocalEntryNotFoundError, OSError):
        logger.error(
            f'Could not find file "{filename}". '
            "Make sure you have downloaded the model before running the agent. "
            "Use `python3 your_agent.py download-files` to download the model."
        )
        raise RuntimeError(
            "livekit-plugins-turn-detector initialization failed. "
            f'Could not find file "{filename}".'
        ) from None
    return local_path


class _EUORunnerBase(_InferenceRunner):
    @classmethod
    @abstractmethod
    def model_type(cls) -> EOUModelType: ...

    @classmethod
    def model_revision(cls) -> str:
        return MODEL_REVISIONS[cls.model_type()]

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text.lower())
        text = "".join(
            ch
            for ch in text
            if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"])
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _format_chat_ctx(self, chat_ctx: list[dict[str, Any]]) -> str:
        new_chat_ctx = []
        last_msg: dict[str, Any] | None = None
        for msg in chat_ctx:
            if not msg["content"]:
                continue

            content = self._normalize_text(msg["content"])

            # need to combine adjacent turns together to match training data
            if last_msg and last_msg["role"] == msg["role"]:
                last_msg["content"] += f" {content}"
            else:
                msg["content"] = content
                new_chat_ctx.append(msg)
                last_msg = msg

        convo_text = self._tokenizer.apply_chat_template(
            new_chat_ctx, add_generation_prompt=False, add_special_tokens=False, tokenize=False
        )

        # remove the EOU token from current utterance
        ix = convo_text.rfind("<|im_end|>")
        text = convo_text[:ix]
        return text  # type: ignore

    def initialize(self) -> None:
        logger = logging.getLogger("transformers")

        class _SuppressSpecific(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                return not msg.startswith(
                    "None of PyTorch, TensorFlow >= 2.0, or Flax have been found."
                )

        filt = _SuppressSpecific()
        # filter this log since it conflicts with the console CLI (since it directly prints to stdout)
        logger.addFilter(filt)
        try:
            import onnxruntime as ort  # type: ignore
            from huggingface_hub import errors
            from transformers import AutoTokenizer
        finally:
            logger.removeFilter(filt)

        revision = self.__class__.model_revision()
        try:
            local_path_onnx = _download_from_hf_hub(
                HG_MODEL,
                ONNX_FILENAME,
                subfolder="onnx",
                revision=revision,
                local_files_only=True,
            )
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = max(
                1, min(math.ceil(hw.get_cpu_monitor().cpu_count()) // 2, 4)
            )
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry("session.dynamic_block_base", "4")
            self._session = ort.InferenceSession(
                local_path_onnx, providers=["CPUExecutionProvider"], sess_options=sess_options
            )
            self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                HG_MODEL,
                revision=revision,
                local_files_only=True,
                truncation_side="left",
            )

        except (errors.LocalEntryNotFoundError, OSError):
            logger.error(
                f"Could not find model {HG_MODEL} with revision {revision}. "
                "Make sure you have downloaded the model before running the agent. "
                "Use `python3 your_agent.py download-files` to download the models."
            )
            raise RuntimeError(
                "livekit-plugins-turn-detector initialization failed. "
                f"Could not find model {HG_MODEL} with revision {revision}."
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
        # run inference
        outputs = self._session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})
        eou_probability = outputs[0].flatten()[-1]
        end_time = time.perf_counter()

        result: dict[str, Any] = {
            "eou_probability": float(eou_probability),
            "duration": round(end_time - start_time, 3),
            "input": text,
        }
        return json.dumps(result).encode()

    @classmethod
    def _download_files(cls) -> None:
        from transformers import AutoTokenizer

        # ensure the tokenizer is downloaded
        AutoTokenizer.from_pretrained(HG_MODEL, revision=cls.model_revision())  # type: ignore[no-untyped-call]
        _download_from_hf_hub(
            HG_MODEL, ONNX_FILENAME, subfolder="onnx", revision=cls.model_revision()
        )
        _download_from_hf_hub(HG_MODEL, "languages.json", revision=cls.model_revision())


class EOUPlugin(Plugin):
    def __init__(self, runner: type[_EUORunnerBase]) -> None:
        super().__init__(__name__, __version__, __package__, logger)
        self._runner_class = runner

    def download_files(self) -> None:
        self._runner_class._download_files()


class EOUModelBase(ABC):
    def __init__(
        self,
        model_type: EOUModelType = "en",  # default to smaller, english-only model
        inference_executor: InferenceExecutor | None = None,
        # if set, overrides the per-language threshold tuned for accuracy.
        # not recommended unless you're confident in the impact.
        unlikely_threshold: float | None = None,
        load_languages: bool = True,
    ) -> None:
        self._model_type = model_type
        self._executor = inference_executor or get_job_context().inference_executor
        self._unlikely_threshold = unlikely_threshold
        self._languages: dict[str, Any] = {}

        if load_languages:
            config_fname = _download_from_hf_hub(
                HG_MODEL,
                "languages.json",
                revision=MODEL_REVISIONS[self._model_type],
                local_files_only=True,
            )
            with open(config_fname) as f:
                self._languages = json.load(f)

    @property
    def model(self) -> str:
        return self._model_type

    @property
    def provider(self) -> str:
        return "livekit"

    @abstractmethod
    def _inference_method(self) -> str: ...

    async def unlikely_threshold(self, language: Language | None) -> float | None:
        if language is None:
            return None

        # try the full language code first
        lang_data = self._languages.get(language.iso)

        # try the base language if the full language code is not found
        if lang_data is None:
            lang_data = self._languages.get(language.language)

        if not lang_data:
            return None

        # if a custom threshold is provided, use it
        if self._unlikely_threshold is not None:
            return self._unlikely_threshold
        else:
            return lang_data["threshold"]  # type: ignore

    async def supports_language(self, language: Language | None) -> bool:
        return await self.unlikely_threshold(language) is not None

    # our EOU model inference should be fast, 3 seconds is more than enough
    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext,
        *,
        timeout: float | None = 3,
    ) -> float:
        messages: list[dict[str, Any]] = []
        for msg in chat_ctx.messages():
            if msg.role not in ("user", "assistant"):
                continue

            text_content = msg.text_content
            if text_content:
                messages.append(
                    {
                        "role": msg.role,
                        "content": text_content,
                    }
                )

        messages = messages[-MAX_HISTORY_TURNS:]
        json_data = json.dumps({"chat_ctx": messages}).encode()

        result = await asyncio.wait_for(
            self._executor.do_inference(self._inference_method(), json_data), timeout=timeout
        )

        if result is None:
            logger.warning("Inference executor returned None, using default probability")
            return 1.0  # Default to indicating no end-of-turn

        result_json: dict[str, Any] = json.loads(result.decode())
        logger.debug("eou prediction", extra=result_json)
        return result_json["eou_probability"]  # type: ignore
