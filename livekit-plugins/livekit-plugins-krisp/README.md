# Krisp VIVA Plugin for LiveKit Agents

Real-time noise reduction for LiveKit voice agents using [Krisp's VIVA SDK](https://krisp.ai).

## Features

- **`KrispVivaFilterFrameProcessor`**: Real-time noise reduction FrameProcessor for audio processing

## Installation

```bash
pip install livekit-plugins-krisp
```

That's it. The default backend is bundled with the plugin and authenticates through
LiveKit Cloud using the room's credentials — no separate SDK download, license key,
or model file is required.

> Using your own [Krisp license](#alternative-krisp-license-auth) instead?
> That path has additional prerequisites — see below.

## Quick Start

By default, `KrispVivaFilterFrameProcessor` uses **LiveKit Cloud** authentication: the
bundled backend ships the noise-reduction model and authenticates against LiveKit Cloud
using the room JWT the agent framework hands to the FrameProcessor automatically.

```python
from livekit.agents import AgentSession, Agent, JobContext, room_io
from livekit.plugins import krisp, silero, openai

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Default: LiveKit Cloud auth + bundled model. No keys or model files.
    processor = krisp.KrispVivaFilterFrameProcessor(
        noise_suppression_level=100,  # 0-100
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
    )

    # Start session with RoomIO and pass the FrameProcessor directly
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=processor,  # Pass FrameProcessor directly
            ),
        ),
    )
```

**Audio Pipeline:** `Room → RoomIO (with KrispVivaFilterFrameProcessor) → VAD → STT → LLM`

## Configuration

### KrispVivaFilterFrameProcessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auth_provider` | `LiveKitCloudAuthProvider \| KrispLicenseAuthProvider` | `LiveKitCloudAuthProvider` | Authentication backend. Defaults to LiveKit Cloud. See [the alternative](#alternative-krisp-license-auth). |
| `noise_suppression_level` | int | 100 | Noise reduction intensity (0-100) |
| `frame_duration_ms` | int | None | **Deprecated.** Input frames of any size are now buffered automatically. |
| `sample_rate` | int | None | **Deprecated.** The processor now adapts to the input sample rate automatically. |
| `model_path` | str | None | **Deprecated.** Use `auth_provider=krisp.auth.krisp_license(model_path=...)`. License-mode only. |


## Alternative: Krisp License Auth

> **Most users should use the default LiveKit Cloud path above.** This alternative is for
> running the public Krisp SDK directly with your own Krisp license — for example, when
> using Livekit OSS server.

This path uses the public `krisp_audio` wheel together with a Krisp license key and a
`.kef` model file that you obtain from Krisp.

### Prerequisites

1. **Krisp Audio SDK** — proprietary, not on public PyPI. Obtain and install it
   separately from [Krisp](https://krisp.ai/developers/):
   ```bash
   pip install krisp-audio
   ```
2. **License key**:
   ```bash
   export KRISP_VIVA_SDK_LICENSE_KEY=your-license-key-here
   ```
3. **Noise-reduction model** — a `.kef` model file from Krisp:
   ```bash
   export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/noise_model.kef
   ```

### Usage

Select the license backend by passing `auth_provider`:

```python
from livekit.plugins import krisp

processor = krisp.KrispVivaFilterFrameProcessor(
    auth_provider=krisp.auth.krisp_license(
        license_key="...",                    # or KRISP_VIVA_SDK_LICENSE_KEY
        model_path="/path/to/noise_model.kef",  # or KRISP_VIVA_FILTER_MODEL_PATH
    ),
    noise_suppression_level=100,
)
```

`license_key` and `model_path` fall back to the environment variables above when omitted.

## Troubleshooting

### `RuntimeError`: bundled backend missing
If the default (LiveKit Cloud) backend reports a missing wheel, the install is likely
broken. Reinstall the plugin:
```bash
pip install --force-reinstall livekit-plugins-krisp
```
Alternatively, fall back to the [Krisp license auth](#alternative-krisp-license-auth) path.

### "Krisp SDK initialization failed" or licensing errors *(license auth only)*
Make sure the license key is set:
```bash
export KRISP_VIVA_SDK_LICENSE_KEY=your-license-key-here
```

### "Model path must be provided" *(license auth only)*
```bash
export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/model.kef
```

### "Unsupported sample rate"
Supported: 8000, 16000, 24000, 32000, 44100, 48000 Hz

### Silent output
- Verify the model file is valid *(license auth only)*
- Test with known noisy audio

## License

The source code in this package (`livekit-plugins-krisp`) is licensed under the
**Apache-2.0** license.

The **default backend** is a separate, closed-source wheel (`livekit-plugins-krisp-internal`)
that is installed automatically as a dependency. It is **proprietary** and distributed under
the [LiveKit Terms of Service](https://livekit.io/legal/terms-of-service). That wheel bundles
the Krisp VIVA SDK along with its third-party open-source components, whose attribution
notices are shipped inside the wheel.

The **Krisp license alternative** (`KrispLicenseAuthProvider`) instead needs a manual install of the proprietary Krisp
Audio SDK together with your own Krisp license key and model file, governed by your agreement
with [Krisp](https://krisp.ai).
