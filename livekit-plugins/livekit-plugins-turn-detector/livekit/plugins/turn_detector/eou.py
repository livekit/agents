from __future__ import annotations

import asyncio
import json
import time

from livekit.agents import llm
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.inference_executor import InferenceExecutor
from livekit.agents.job import get_current_job_context

from .log import logger

HG_MODEL = "livekit/turn-detector"
ONNX_FILENAME = "model_q8.onnx"
MODEL_REVISIONS = {"en": "v1.2.1", "multilingual": "v0.1.0-intl"}
MAX_HISTORY_TOKENS = 512
MAX_HISTORY_TURNS = 6


def _download_from_hf_hub(repo_id, filename, **kwargs):
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    return local_path


class _EUORunnerBase(_InferenceRunner):
    def __init__(self, model_revision: str):
        super().__init__()
        self._model_revision = model_revision

    def _format_chat_ctx(self, chat_ctx: dict):
        new_chat_ctx = []
        for msg in chat_ctx:
            content = msg["content"]
            if not content:
                continue

            msg["content"] = content
            new_chat_ctx.append(msg)

        convo_text = self._tokenizer.apply_chat_template(
            new_chat_ctx,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )

        # remove the EOU token from current utterance
        ix = convo_text.rfind("<|im_end|>")
        text = convo_text[:ix]
        return text

    def initialize(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import errors
        from transformers import AutoTokenizer

        try:
            local_path_onnx = _download_from_hf_hub(
                HG_MODEL,
                ONNX_FILENAME,
                subfolder="onnx",
                revision=self._model_revision,
                local_files_only=True,
            )
            self._session = ort.InferenceSession(
                local_path_onnx, providers=["CPUExecutionProvider"]
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                HG_MODEL,
                revision=self._model_revision,
                local_files_only=True,
                truncation_side="left",
            )

        except (errors.LocalEntryNotFoundError, OSError):
            logger.error(
                (
                    f"Could not find model {HG_MODEL} with revision {self._model_revision}. Make sure you have downloaded the model before running the agent. "
                    "Use `python3 your_agent.py download-files` to download the models."
                )
            )
            raise RuntimeError(
                f"livekit-plugins-turn-detector initialization failed. Could not find model {HG_MODEL} with revision {self._model_revision}."
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
        outputs = self._session.run(
            None, {"input_ids": inputs["input_ids"].astype("int64")}
        )
        eou_probability = outputs[0][0]
        end_time = time.perf_counter()

        data = {
            "eou_probability": float(eou_probability),
            "input": text,
            "duration": round(end_time - start_time, 3),
        }
        return json.dumps(data).encode()


class _EUORunnerEn(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_en"

    def __init__(self):
        super().__init__(MODEL_REVISIONS["en"])


class _EUORunnerMultilingual(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_multilingual"

    def __init__(self):
        super().__init__(MODEL_REVISIONS["multilingual"])


class EOUModel:
    def __init__(
        self,
        english_only: bool = False,  # "en" or "multilingual"
        inference_executor: InferenceExecutor | None = None,
        unlikely_threshold: float = 0.0289,
    ) -> None:
        self._english_only = english_only
        self._executor = (
            inference_executor or get_current_job_context().inference_executor
        )

        if not self._english_only:
            self._inference_method = _EUORunnerMultilingual.INFERENCE_METHOD
            self._languages = _load_languages(self._model_revision)
        else:
            self._inference_method = _EUORunnerEn.INFERENCE_METHOD
            self._languages = {"en": {"threshold": unlikely_threshold}}

    def unlikely_threshold(self, language: str | None) -> float | None:
        if language is None:
            return None
        lang = language.lower()
        if lang in self._languages:
            return self._languages[lang]["threshold"]
        if "-" in lang:
            parts = lang.split("-")
            if parts[0] in self._languages:
                return self._languages[parts[0]]["threshold"]
        return None

    def supports_language(self, language: str | None) -> bool:
        if language is None:
            return False
        lang = language.lower()
        if lang in self._languages:
            return True
        if "-" in lang:
            parts = lang.split("-")
            if parts[0] in self._languages:
                return True
        return False

    async def predict_eou(self, chat_ctx: llm.ChatContext) -> float:
        return await self.predict_end_of_turn(chat_ctx)

    # our EOU model inference should be fast, 3 seconds is more than enough
    async def predict_end_of_turn(
        self, chat_ctx: llm.ChatContext, *, timeout: float | None = 3
    ) -> float:
        messages = []

        for msg in chat_ctx.messages:
            if msg.role not in ("user", "assistant"):
                continue

            if isinstance(msg.content, str):
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )
            elif isinstance(msg.content, list):
                for cnt in msg.content:
                    if isinstance(cnt, str):
                        messages.append(
                            {
                                "role": msg.role,
                                "content": cnt,
                            }
                        )
                        break

        messages = messages[-MAX_HISTORY_TURNS:]

        json_data = json.dumps({"chat_ctx": messages}).encode()

        result = await asyncio.wait_for(
            self._executor.do_inference(self._inference_method, json_data),
            timeout=timeout,
        )

        assert result is not None, (
            "end_of_utterance prediction should always returns a result"
        )

        result_json = json.loads(result.decode())
        logger.debug(
            "eou prediction",
            extra=result_json,
        )
        return result_json["eou_probability"]


def _load_languages(model_revision: str) -> dict:
    lang_names = {
        "fr": "French",
        "id": "Indonesian",
        "ru": "Russian",
        "tr": "Turkish",
        "nl": "Dutch",
        "pt-br": "Portuguese (Brazil)",
        "pt-pt": "Portuguese (Portugal)",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "ko": "Korean",
        "en-us": "English (United States)",
        "en": "English",
        "ja": "Japanese",
        "zh-hant": "Chinese (Traditional)",
        "zh-hans": "Chinese (Simplified)",
    }

    fname = _download_from_hf_hub(
        HG_MODEL,
        "languages.json",
        revision=model_revision,
        local_files_only=True,
    )
    with open(fname, "r") as f:
        languages = {k.lower(): v for k, v in json.load(f).items()}

    # we add language names to the languages dict bc openai STT returns language names
    codes = languages.keys()
    for code in codes:
        conf = languages[code]
        if code in lang_names:
            lang_name = lang_names[code]
            # Add title case, lower case
            languages[lang_name] = conf
            languages[lang_name.lower()] = conf
            # Add first word if multiple words
            languages[lang_name.split()[0]] = conf
            languages[lang_name.split()[0].lower()] = conf
    return languages
