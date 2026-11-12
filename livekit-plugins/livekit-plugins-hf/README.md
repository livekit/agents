# Hugging Face plugin for LiveKit Agents

Run open models from the [Hugging Face Hub](https://huggingface.co/models) locally with
`transformers`, with no API key or hosted inference service:

- `hf.LLM`: chat models (Qwen, Llama, Mistral, SmolLM and others), with function tools.
- `hf.STT`: speech recognition models (Whisper and its fine-tunes, Moonshine, wav2vec2).

## Installation

```bash
pip install livekit-plugins-hf
```

Install the `torch` build that matches your hardware (CUDA, ROCm, Apple Silicon or CPU) by
following the [PyTorch instructions](https://pytorch.org/get-started/locally/).

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import hf

session = AgentSession(
    llm=hf.LLM(model="Qwen/Qwen2.5-7B-Instruct"),
    # stt=..., tts=..., vad=...
)
```

`model` accepts a Hub repo id or a path to a local model directory. Function tools work
with models whose chat template supports tools (Qwen 2.5, Llama 3.1+, Hermes, Mistral and
others).

For reasoning models such as Qwen3, pass `chat_template_kwargs={"enable_thinking": False}`
so the agent answers without a long silent think first. Text inside `<think>` tags is never spoken.

## Speech to text

```python
from livekit.plugins import hf, silero

session = AgentSession(
    stt=hf.STT(model="openai/whisper-large-v3-turbo", language="en"),
    vad=silero.VAD.load(),
    llm=hf.LLM(),
    # tts=...
)
```

`hf.STT` loads any model the `automatic-speech-recognition` pipeline supports. It is a batch
STT: the session's VAD cuts each utterance and the model transcribes it in one pass, so give
the session a `vad`. `language` is passed to Whisper models; leave it unset to let Whisper
detect it. `generate_kwargs` goes straight to `generate()`, for example `{"num_beams": 1}`.

Whisper fine-tunes that add languages on their own token ids take them in
`language_token_ids`. For example, [SunflowerASR](https://huggingface.co/Sunbird/SunflowerASR-51-african-languages)
(51 African languages, gated: accept its terms on the Hub and set `HF_TOKEN` to download):

```python
stt = hf.STT(
    model="Sunbird/SunflowerASR-51-african-languages",
    language="lug",
    language_token_ids={"lug": 50332, "swa": 50318, "kin": 50340, "eng": 50259},
    generate_kwargs={"num_beams": 1, "do_sample": False},
)
```

The token ids for every language are in the model card. A GPU is recommended for the
large Whisper models; on CPU, pick a small model such as `openai/whisper-base` or
`UsefulSensors/moonshine-base`.

## Downloading models ahead of time

By default the plugin never downloads at runtime (`local_files_only=True`). Fetch weights
before starting the agent, for example while building its Docker image:

```bash
# downloads Qwen/Qwen2.5-1.5B-Instruct, the default model
python my_agent.py download-files

# or choose the models to fetch
LIVEKIT_HF_MODELS="Qwen/Qwen2.5-7B-Instruct" python my_agent.py download-files
```

Add your STT model to `LIVEKIT_HF_MODELS` too, e.g.
`LIVEKIT_HF_MODELS="Qwen/Qwen2.5-7B-Instruct,openai/whisper-large-v3-turbo"`.

Pin a revision with `LIVEKIT_HF_MODELS="Qwen/Qwen2.5-7B-Instruct@<commit>"`, matching `LLM(revision=...)`.

You can also call `hf.download_model("Qwen/Qwen2.5-7B-Instruct")` from a build script.
Gated models (such as Llama) need `HF_TOKEN` set while downloading. Pass
`local_files_only=False` to let the model download on first use instead.

## Loading weights

Weights load in a background thread when the `AgentSession` starts, so the event loop stays
responsive, followed by a one-token warm-up so the first reply is fast. To have them ready
before a job is assigned, load them when the job process starts:

```python
from livekit.agents import AgentServer, JobProcess


def setup(proc: JobProcess) -> None:
    hf.LLM(model="Qwen/Qwen2.5-7B-Instruct").load()


server = AgentServer(setup_fnc=setup)
```

`hf.STT` has the same `load()` method. Loaded weights are cached per process and shared by
every instance using the same model.
Each job process holds its own copy, so on a single GPU run jobs in threads
(`job_executor_type=JobExecutorType.THREAD`) to keep one copy per worker. Generation on a
model runs one reply at a time.
