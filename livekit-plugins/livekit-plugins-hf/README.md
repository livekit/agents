# Hugging Face plugin for LiveKit Agents

Run open chat models from the [Hugging Face Hub](https://huggingface.co/models) locally with
`transformers`, with no API key or hosted inference service.

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

## Downloading models ahead of time

By default the plugin never downloads at runtime (`local_files_only=True`). Fetch weights
before starting the agent, for example while building its Docker image:

```bash
# downloads Qwen/Qwen2.5-1.5B-Instruct, the default model
python my_agent.py download-files

# or choose the models to fetch
LIVEKIT_HF_MODELS="Qwen/Qwen2.5-7B-Instruct" python my_agent.py download-files
```

You can also call `hf.download_model("Qwen/Qwen2.5-7B-Instruct")` from a build script.
Gated models (such as Llama) need `HF_TOKEN` set while downloading. Pass
`local_files_only=False` to let the model download on first use instead.

## Loading weights

Weights load in a background thread when the `AgentSession` starts, so the event loop
stays responsive. To have them ready before a job is assigned, load them when the job
process starts:

```python
from livekit.agents import AgentServer, JobProcess

def setup(proc: JobProcess) -> None:
    hf.LLM(model="Qwen/Qwen2.5-7B-Instruct").load()

server = AgentServer(setup_fnc=setup)
```

Loaded weights are cached per process and shared by every `hf.LLM` using the same model.
Each job process holds its own copy, so on a single GPU run jobs in threads
(`job_executor_type=JobExecutorType.THREAD`) to keep one copy per worker. Generation on a
model runs one reply at a time.
