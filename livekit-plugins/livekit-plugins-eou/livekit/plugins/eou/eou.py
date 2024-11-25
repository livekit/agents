from __future__ import annotations

import json
import string

import numpy as np
from livekit.agents import llm
from livekit.agents.inference_runner import _InferenceRunner

HG_MODEL = "livekit/opt-125m-endpoint-detector"
PUNCS = string.punctuation.replace("'", "")
MAX_HISTORY = 4


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


class _EUORunner(_InferenceRunner):
    METHOD = "lk_end_of_utterance"

    def __init__(self) -> None:
        pass

    def _normalize(self, text):
        def strip_puncs(text):
            return text.translate(str.maketrans("", "", PUNCS))

        return " ".join(strip_puncs(text).lower().split())

    def _format_chat_ctx(self, chat_ctx: dict):
        new_chat_ctx = []
        for msg in chat_ctx:
            if msg["role"] not in ["user", "assistant"]:
                continue

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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(HG_MODEL)
        self._tokenizer = AutoTokenizer.from_pretrained(HG_MODEL)
        self._eou_index = self._tokenizer.encode("<|im_end|>")[-1]

    def run(self, data: bytes) -> bytes | None:
        data_json = json.loads(data)
        chat_ctx = data_json.get("chat_ctx", None)

        if not chat_ctx:
            raise ValueError("chat_ctx is required on the inference input data")

        chat_ctx = chat_ctx[-MAX_HISTORY:]

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

        return json.dumps({"eou_probability": eou_probability}).encode()


class EOU:
    def __init__(self):
        pass

    def predict_eou(self, chat_ctx: llm.ChatContext) -> float:
        pass
