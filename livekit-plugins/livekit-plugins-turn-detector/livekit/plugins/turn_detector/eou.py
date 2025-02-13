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
MODEL_REVISION = "v1.2.0"
MAX_HISTORY = 4
MAX_HISTORY_TOKENS = 512


def _download_from_hf_hub(repo_id, filename, **kwargs):
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    return local_path


class _EUORunner(_InferenceRunner):
    INFERENCE_METHOD = "lk_end_of_utterance"

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
                revision=MODEL_REVISION,
                local_files_only=True,
            )
            self._session = ort.InferenceSession(
                local_path_onnx, providers=["CPUExecutionProvider"]
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                HG_MODEL,
                revision=MODEL_REVISION,
                local_files_only=True,
                truncation_side="left",
            )
        except (errors.LocalEntryNotFoundError, OSError):
            logger.error(
                (
                    f"Could not find model {HG_MODEL}. Make sure you have downloaded the model before running the agent. "
                    "Use `python3 your_agent.py download-files` to download the models."
                )
            )
            raise RuntimeError(
                f"livekit-plugins-turn-detector initialization failed. Could not find model {HG_MODEL}."
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

        logger.debug(
            "eou prediction",
            extra={
                "eou_probability": eou_probability,
                "input": text,
                "duration": round(end_time - start_time, 3),
            },
        )
        return json.dumps({"eou_probability": float(eou_probability)}).encode()


class EOUModel:
    def __init__(
        self,
        inference_executor: InferenceExecutor | None = None,
        unlikely_threshold: float = 0.008,
    ) -> None:
        self._executor = (
            inference_executor or get_current_job_context().inference_executor
        )
        self._unlikely_threshold = unlikely_threshold

    def unlikely_threshold(self) -> float:
        return self._unlikely_threshold

    def supports_language(self, language: str | None) -> bool:
        if language is None:
            return False
        parts = language.lower().split("-")
        # certain models use language codes (DG, AssemblyAI), others use full names (like OAI)
        return parts[0] == "en" or parts[0] == "english"

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

        messages = messages[-MAX_HISTORY:]

        json_data = json.dumps({"chat_ctx": messages}).encode()

        result = await asyncio.wait_for(
            self._executor.do_inference(_EUORunner.INFERENCE_METHOD, json_data),
            timeout=timeout,
        )

        assert result is not None, (
            "end_of_utterance prediction should always returns a result"
        )

        result_json = json.loads(result.decode())
        return result_json["eou_probability"]
