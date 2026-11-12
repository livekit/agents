# Hugging Face plugin for LiveKit Agents

Run open models from the [Hugging Face Hub](https://huggingface.co/models) locally with
`transformers`, with no API key or hosted inference service:

- `hf.LLM`: chat models (Qwen, Llama, Mistral, SmolLM and others), with function tools.
- `hf.STT`: speech recognition models (Whisper and its fine-tunes, Moonshine, wav2vec2).
- `hf.TTS`: speech synthesis with Orpheus models (streamed as they generate) and VITS/MMS voices.

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

## Text to speech

```bash
pip install "livekit-plugins-hf[orpheus]"
```

```python
session = AgentSession(
    tts=hf.TTS(model="canopylabs/orpheus-3b-0.1-ft", voice="tara"),
    # stt=..., llm=..., vad=...
)
```

`hf.TTS` supports two model families and picks one from the model's config:

- **Orpheus**: Llama models that speak in [SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz)
  audio codes, such as `canopylabs/orpheus-3b-0.1-ft` (the default; gated, set `HF_TOKEN`)
  and its fine-tunes. Audio is decoded and sent while the model is still generating, a few
  85 ms frames at a time (`chunk_frames`), so speech starts before the sentence is done.
  `voice` is prepended to the text as `"{voice}: {text}"`. Needs the `orpheus` extra and a GPU.
- **Text to waveform**: models `AutoModelForTextToWaveform` loads, such as the VITS-based
  [MMS voices](https://huggingface.co/models?search=facebook/mms-tts) for about 1,100
  languages (`facebook/mms-tts-eng`, `facebook/mms-tts-swh`, ...). Small enough for CPU.
  Check each model's license before using it commercially.

The session splits each reply into sentences and synthesizes one at a time. Audio is
resampled to `sample_rate` (24 kHz by default).

Multilingual Orpheus fine-tunes pick the language through the speaker. For example,
[Sunbird's model](https://huggingface.co/Sunbird/orpheus-3b-tts-multilingual) covers 20
African languages and English:

```python
tts = hf.TTS(model="Sunbird/orpheus-3b-tts-multilingual", voice="salt_lug_0001")  # Luganda
```

Speaker ids for each language are in the model card. `tts.update_options(voice=...)`
switches the speaker mid-session.

## Hosted inference

Running models in the agent process is simple, but in production it ties GPU memory to
every worker process and scales the models with your calls. To run them elsewhere, give
`hf.STT` or `hf.TTS` a `provider` or a `base_url`. Nothing loads locally then, and
`torch` is not used.

```python
# serverless, through Hugging Face Inference Providers (uses HF_TOKEN)
stt = hf.STT(model="openai/whisper-large-v3-turbo", provider="hf-inference")
tts = hf.TTS(model="hexgrad/Kokoro-82M", provider="fal-ai", voice="af_heart")

# a dedicated endpoint, such as a Hugging Face Inference Endpoint
stt = hf.STT(base_url="https://<your-endpoint>.endpoints.huggingface.cloud")
```

`provider` takes any [Inference Provider](https://huggingface.co/docs/inference-providers)
that serves the task (`"auto"`, `"hf-inference"`, `"fal-ai"`, `"deepinfra"`, `"replicate"`, ...).
Check the model page for which providers serve it; fine-tunes such as SunflowerASR and
Sunbird's Orpheus usually need a dedicated endpoint. `api_key` defaults to `HF_TOKEN`, and
`extra_body` passes provider-specific parameters. `language` (STT) and `voice` (TTS) are
sent as request parameters, except `language` on `hf-inference`.

Hosted calls are request and response: STT is batch behind the session's VAD as in local
mode, and TTS audio starts once the provider has synthesized the whole sentence, so local
Orpheus streaming starts sooner.

## Downloading models ahead of time

By default the plugin never downloads at runtime (`local_files_only=True`). Fetch weights
before starting the agent, for example while building its Docker image:

```bash
# downloads Qwen/Qwen2.5-1.5B-Instruct, the default model
python my_agent.py download-files

# or choose the models to fetch
LIVEKIT_HF_MODELS="Qwen/Qwen2.5-7B-Instruct" python my_agent.py download-files
```

Add your STT and TTS models to `LIVEKIT_HF_MODELS` too, e.g.
`LIVEKIT_HF_MODELS="Qwen/Qwen2.5-7B-Instruct,openai/whisper-large-v3-turbo,canopylabs/orpheus-3b-0.1-ft"`.
Orpheus models fetch the SNAC codec with them.

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

`hf.STT` and `hf.TTS` have the same `load()` method. Loaded weights are cached per process and shared by
every instance using the same model.
Each job process holds its own copy, so on a single GPU run jobs in threads
(`job_executor_type=JobExecutorType.THREAD`) to keep one copy per worker. Generation on a
model runs one reply at a time.
