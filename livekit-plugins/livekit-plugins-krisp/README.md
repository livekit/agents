# Krisp VIVA Plugin for LiveKit Agents

Real-time noise reduction for LiveKit voice agents using [Krisp's VIVA SDK](https://krisp.ai).

## Features

- **`KrispVivaFilterFrameProcessor`**: Real-time noise reduction FrameProcessor for audio processing

## Installation

```bash
# Install the plugin
pip install livekit-plugins-krisp

# Install krisp-audio SDK separately (required for actual usage)
```

**Note:** The `krisp-audio` package is a proprietary SDK not available on public PyPI. 
It must be obtained and installed separately from Krisp (https://krisp.ai/developers/).

## Prerequisites

### Required for All Features

1. **Krisp Audio SDK**: `pip install krisp-audio`
2. **License Key**: Obtain a license key from Krisp and set it as an environment variable:
   ```bash
   export KRISP_VIVA_SDK_LICENSE_KEY=your-license-key-here
   ```

### For Noise Reduction

1. **Noise Reduction Model**: Obtain a noise reduction or voice isolation `.kef` model file from Krisp
2. **Set environment variable**:
   ```bash
   export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/noise_model.kef
   ```

## Quick Start

### Human-to-Bot Noise Cancellation / Voice Isolation (Recommended)

For cleaning up user audio before STT/VAD processing using the FrameProcessor approach:

```python
from livekit.agents import AgentSession, Agent, JobContext, room_io
from livekit.plugins import krisp, silero, openai

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Create Krisp FrameProcessor
    processor = krisp.KrispVivaFilterFrameProcessor(
        noise_suppression_level=100,  # 0-100
        frame_duration_ms=10,
        sample_rate=16000,
    )
    
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
    )
    
    # Start session with RoomIO and pass FrameProcessor directly
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=16000,
                frame_size_ms=10,  # Must match Krisp frame_duration_ms
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
| `model_path` | str | env var | Path to noise reduction `.kef` model |
| `noise_suppression_level` | int | 100 | Noise reduction intensity (0-100) |
| `frame_duration_ms` | int | 10 | Frame size: 10, 15, 20, 30, or 32ms |
| `sample_rate` | int | None | Optional: pre-initialize with sample rate |

### Supported Sample Rates

8000, 16000, 24000, 32000, 44100, 48000 Hz


## Important Notes

### Frame Size Requirements

⚠️ **Frames must match the configured duration exactly**

- 10ms @ 16kHz = 160 samples
- 20ms @ 16kHz = 320 samples
- 20ms @ 32kHz = 640 samples

The filter validates frame sizes and raises `ValueError` if incorrect.

### Resource Management

- Session created once (on first use or if `sample_rate` provided)
- Call `close()` when done to free resources

### Shared SDK Management

The plugin uses `KrispSDKManager` to manage the Krisp SDK instance:

- **Singleton Pattern**: SDK initialized only once, shared across all components and sessions
- **Reference Counting**: Tracks active users (filters)
- **Automatic Cleanup**: SDK destroyed when last component releases its reference

## Troubleshooting

### "Krisp SDK initialization failed" or Licensing Errors
Make sure the license key is set:
```bash
export KRISP_VIVA_SDK_LICENSE_KEY=your-license-key-here
```

### "Model path must be provided"
```bash
export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/model.kef
```

### "Unsupported sample rate"
Supported: 8000, 16000, 24000, 32000, 44100, 48000 Hz


### "Frame size mismatch"
Ensure your audio frames match the configured `frame_duration_ms`.

For 20ms @ 16kHz, each frame must have exactly 320 samples.

### Silent output
- Verify model file is valid
- Test with known noisy audio
