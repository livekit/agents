# livekit-plugins-voxcpm

LiveKit Agents plugin for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) served through [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

## Requirements

- A running vLLM-Omni server with VoxCPM2 loaded, exposing the OpenAI-compatible Speech API.
- Python >= 3.10

Start a server:

```bash
vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8800
```

## Install

```bash
pip install livekit-plugins-voxcpm
```

Or from the monorepo workspace:

```bash
uv sync --all-extras --dev
```

## Usage

```python
from livekit.plugins import voxcpm

tts = voxcpm.TTS(
    base_url="http://127.0.0.1:8800/v1",
    model="openbmb/VoxCPM2",
    voice="default",
)
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_OMNI_URL` | `http://127.0.0.1:8800/v1` | vLLM-Omni OpenAI base URL |
| `VLLM_OMNI_MODEL` | `openbmb/VoxCPM2` | Model id |
| `VOXCPM_VOICE` | `default` | Preset or uploaded voice name |
| `VLLM_API_KEY` | unset | Optional bearer token |

### Voice cloning

Pass a reference clip at construction time or use a pre-uploaded voice via `POST /v1/audio/voices` on the server:

```python
tts = voxcpm.TTS(
    voice="my_speaker",
    ref_audio="/path/to/reference.wav",
    ref_text="Transcript of the reference clip.",
)
```

Voice design prefixes such as `(A warm female voice)Hello!` are passed through as plain text to the backend.

## API surface

- `synthesize(text)` uses HTTP streaming PCM (`POST /v1/audio/speech`).
- `stream()` uses the WebSocket endpoint (`/v1/audio/speech/stream`) for low-latency agent pipelines.

Output is mono 16-bit PCM at **48 kHz**.

## Links

- [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
- [VoxCPM vLLM-Omni deployment guide](https://voxcpm.readthedocs.io/en/latest/deployment/vllm_omni.html)
