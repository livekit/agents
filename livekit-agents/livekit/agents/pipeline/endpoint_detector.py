import copy
import string
import time

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

from .log import logger

PUNCS = string.punctuation.replace("'","")


class EndpointDetector:
    def __init__(self, model_path='jeradf/opt-125m-eou'):
        self.model = ORTModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._eou_index = self.tokenizer.encode("<|im_end|>")[-1]

    def normalize(self, text):
        def strip_puncs(text):
            return text.translate(str.maketrans('', '', PUNCS))
        return " ".join(strip_puncs(text).lower().split())

    def apply_chat_template(self, convo):
        for msg in convo:
            msg['content'] = self.normalize(msg['content'])

        convo_text = self.tokenizer.apply_chat_template(
            convo,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )

        # remove the EOU token from current utterance
        ix = convo_text.rfind('<|im_end|>')
        text = convo_text[:ix]
        return text

    def tokenize(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors='pt',
        )

    def predict(self, utterance, convo=[]):
        start_time = time.time()

        convo_copy = copy.deepcopy(convo)
        convo_copy.append(dict(role='user', content=utterance))

        text = self.apply_chat_template(convo_copy)
        inputs = self.tokenize(text)

        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        result = probs[self._eou_index].item()

        end_time = time.time()
        latency = end_time - start_time
        
        logger.debug(
            "EndpointDetector prediction", 
            extra={
                "probability": round(result, 2), 
                "utterance": utterance,
                "latency": round(latency, 2),
            }
        )
        return result

