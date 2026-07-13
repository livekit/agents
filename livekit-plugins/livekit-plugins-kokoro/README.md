# Kokoro TTS plugin for LiveKit Agents

Support for [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), an open-weight TTS model, served by [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI). Audio is requested as a raw PCM stream and forwarded to the pipeline natively, without going through the OpenAI compatibility layer.

See [https://docs.livekit.io/agents/integrations/tts/](https://docs.livekit.io/agents/integrations/tts/) for more information.

## Installation

```bash
pip install livekit-plugins-kokoro
```

## Running a Kokoro-FastAPI server

```bash
# CPU (works on Apple Silicon too)
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest

# NVIDIA GPU
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

## Usage

```python
from livekit.plugins import kokoro

tts = kokoro.TTS(voice="af_heart")  # defaults to http://localhost:8880/v1
# or point at a remote deployment:
tts = kokoro.TTS(voice="af_bella(2)+af_sky(1)", base_url="http://my-kokoro-host:8880/v1")
```

The server address can also be set with the `KOKORO_BASE_URL` environment variable. No API key is required.
