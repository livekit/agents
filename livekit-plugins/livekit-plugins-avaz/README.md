# LiveKit Avaz TTS Plugin

LiveKit Agents plugin for [Avaz](https://github.com/Mank-Technology) text-to-speech over the dashboard WebSocket `stream-input` protocol.

## Installation

```bash
pip install "livekit-agents[avaz]~=1.6"
```

## Usage

```python
import os
from livekit.plugins import avaz

tts = avaz.TTS(
    api_key=os.environ["AVAZ_API_KEY"],
    base_url=os.environ["AVAZ_BASE_URL"],
    model_id=os.environ["AVAZ_AGENT_MODEL_ID"],
)
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `AVAZ_API_KEY` | yes (dashboard) | Dashboard API token (`X-API-Key` / Bearer) |
| `AVAZ_BASE_URL` | yes (dashboard) | Dashboard API base URL (e.g. `https://your-dashboard.example.com/api`) |
| `AVAZ_AGENT_MODEL_ID` | yes (dashboard) | Agent model UUID from your dashboard TTS catalog |
| `AVAZ_STREAM_MODEL` | no | Upstream WebSocket model string (`avaz1`, `avaz2`, `avaz3`); default `avaz3` |
| `TTS_WS_URI` | no | Direct WebSocket override for local TTS-Service (no auth) |

## Protocol

1. Connect to `{base_url}` → `wss://.../api/tts/stream-input`
2. Send `model_settings` + `voice_settings` (WebSocket `model_id` is the upstream string)
3. Stream `{"text": "..."}` chunks; receive base64 WAV in `{"audio": ...}`
4. Send `{"flush": true}` to finish the turn

HTTP synthesize (`POST /tts/synthesize`) uses the UUID `model_id` from the constructor; WebSocket init uses the upstream string (`stream_model`).
