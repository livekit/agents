from __future__ import annotations

import json
import string
import time

import numpy as np
from livekit.agents import llm
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.inference_executor import InferenceExecutor
from livekit.agents.job import get_current_job_context

from .log import logger

HG_MODEL = "livekit/opt-125m-endpoint-detector-2"
PUNCS = string.punctuation.replace("'", "")
MAX_HISTORY = 4


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


class _EUORunner(_InferenceRunner):
    INFERENCE_METHOD = "lk_end_of_utterance"

    def _normalize(self, text):
        def strip_puncs(text):
            return text.translate(str.maketrans("", "", PUNCS))

        return " ".join(strip_puncs(text).lower().split())

    def _format_chat_ctx(self, chat_ctx: dict):
        new_chat_ctx = []
        for msg in chat_ctx:
            content = self._normalize(msg["content"])

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
        from huggingface_hub import errors
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                HG_MODEL, local_files_only=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                HG_MODEL, local_files_only=True
            )
            self._eou_index = self._tokenizer.encode("<|im_end|>")[-1]
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
            return_tensors="pt",
        )

        outputs = self._model(**inputs)
        logits = outputs.logits[0, -1, :].detach().numpy()
        output_probs = _softmax(logits)
        eou_probability = output_probs[self._eou_index]

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
    def __init__(self, inference_executor: InferenceExecutor | None = None) -> None:
        self._executor = (
            inference_executor or get_current_job_context().inference_executor
        )

    async def predict_eou(self, chat_ctx: llm.ChatContext) -> float:
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
        result = await self._executor.do_inference(
            _EUORunner.INFERENCE_METHOD, json_data
        )

        assert (
            result is not None
        ), "end_of_utterance prediction should always returns a result"

        result_json = json.loads(result.decode())
        return result_json["eou_probability"]
