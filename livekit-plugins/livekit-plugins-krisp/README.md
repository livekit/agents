# Krisp VIVA Plugin for LiveKit Agents

Real-time noise reduction and audio-based turn detection for LiveKit voice agents using [Krisp's VIVA SDK](https://krisp.ai/).

## Features

- **`KrispVivaFilter`**: Real-time noise reduction for cleaner audio output
- **`KrispVivaTurn`**: Audio-based turn detection for accurate end-of-turn detection

## Installation

```bash
pip install livekit-plugins-krisp
```

## Prerequisites

### For Noise Reduction

1. **Krisp Audio SDK**: `pip install krisp-audio`
2. **Noise Reduction Model**: Obtain a noise reduction `.kef` model file from Krisp
3. **Set environment variable**:
   ```bash
   export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/noise_model.kef
   ```

### For Turn Detection

1. **Krisp Audio SDK**: `pip install krisp-audio`
2. **Turn Detection Model**: Obtain a turn detection `.kef` model file from Krisp
3. **Set environment variable**:
   ```bash
   export KRISP_VIVA_TURN_MODEL_PATH=/path/to/turn_model.kef
   ```

Note: Noise reduction and turn detection use different model files.

## Quick Start

### Noise Reduction

```python
from livekit.agents import Agent
from livekit.plugins import krisp
from collections.abc import AsyncIterable
from livekit import rtc

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant.",
            llm=openai.realtime.RealtimeModel(),
        )
        # Create filter once, reuse for all audio
        self.krisp_filter = krisp.KrispVivaFilter(
            noise_suppression_level=100,  # 0-100
            frame_duration_ms=20,  # 10, 15, 20, 30, or 32
            sample_rate=16000,  # Optional: pre-load model
        )

    async def realtime_audio_output_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings
    ) -> AsyncIterable[rtc.AudioFrame]:
        # Filter output audio through Krisp
        async for frame in self.krisp_filter.process_stream(audio):
            yield frame
    
    async def aclose(self):
        self.krisp_filter.close()
        await super().aclose()
```

### Turn Detection

```python
from livekit.agents import AgentSession, Agent
from livekit.plugins import krisp, silero, deepgram, openai

async def entrypoint(ctx):
    # Create audio-based turn detector
    turn_detector = krisp.KrispVivaTurn(
        threshold=0.6,  # Higher = more conservative
        frame_duration_ms=20,
        sample_rate=16000,
    )
    
    session = AgentSession(
        allow_interruptions=True,
        turn_detection=turn_detector,  # Use Krisp for turn detection
        vad=silero.VAD.load(),  # VAD still needed for speech detection
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4"),
        tts=openai.TTS(),
    )
    
    agent = Agent(instructions="You are a helpful assistant.")
    await session.start(agent=agent, room=ctx.room)
```

## Configuration

### KrispVivaFilter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | env var | Path to noise reduction `.kef` model |
| `noise_suppression_level` | int | 100 | Noise reduction intensity (0-100) |
| `frame_duration_ms` | int | 10 | Frame size: 10, 15, 20, 30, or 32ms |
| `sample_rate` | int | None | Optional: pre-initialize with sample rate |

### KrispVivaTurn Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | env var | Path to turn detection `.kef` model |
| `threshold` | float | 0.5 | Turn probability threshold (0.0-1.0) |
| `frame_duration_ms` | int | 20 | Frame size: 10, 15, 20, 30, or 32ms |
| `sample_rate` | int | None | Optional: pre-initialize with sample rate |

### Supported Sample Rates

8000, 16000, 24000, 32000, 44100, 48000 Hz

## Testing

### Test with Audio Files

```bash
# Basic test
python test_audio_filtering.py input.wav output.wav

# With visualization (spectrograms)
python test_audio_filtering.py input.wav output.wav --visualize

# Custom parameters
python test_audio_filtering.py input.wav output.wav \
  --level 80 \
  --frame-duration 20 \
  --visualize
```

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
- SDK cleanup handled automatically by OS on process exit

### Shared SDK Management

The plugin uses `KrispSDKManager` to manage the Krisp SDK instance:

- **Singleton Pattern**: SDK initialized only once, shared across all components
- **Reference Counting**: Tracks active users (filters, turn detectors)
- **Automatic Cleanup**: SDK destroyed when last component releases its reference

**Using Multiple Components Together:**

```python
from livekit.plugins import krisp

# Both share the same SDK instance
noise_filter = krisp.KrispVivaFilter(noise_suppression_level=90)
turn_detector = krisp.KrispVivaTurn(threshold=0.5)

# Process audio through filter and turn detector
filtered_frame = noise_filter.filter(audio_frame)
turn_probability = turn_detector.process_audio(filtered_frame, is_speech=True)

# Cleanup - SDK destroyed only when both are closed
noise_filter.close()  # SDK still active (turn_detector holds reference)
turn_detector.close()  # SDK now destroyed (last reference released)
```

**Advanced SDK Management:**

```python
from livekit.plugins.krisp import KrispSDKManager

# Check SDK state
is_active = KrispSDKManager.is_initialized()
ref_count = KrispSDKManager.get_reference_count()

# Manual control (rarely needed - components manage this automatically)
KrispSDKManager.acquire()  # Increment reference
KrispSDKManager.release()  # Decrement reference
```

## Troubleshooting

### "Model path must be provided"
```bash
export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/model.kef
```

### "Unsupported sample rate"
Supported: 8000, 16000, 24000, 32000, 44100, 48000 Hz

Resample if needed:
```bash
ffmpeg -i input.wav -ar 16000 output.wav
```

### "Frame size mismatch"
Ensure your audio frames match the configured `frame_duration_ms`.

For 20ms @ 16kHz, each frame must have exactly 320 samples.

### Silent output
- Verify model file is valid
- Test with known noisy audio

## API Reference

### `KrispVivaFilter`

**Methods:**
- `async filter(frame: AudioFrame) -> AudioFrame` - Filter a single frame
- `async process_stream(audio_stream) -> AudioFrame` - Filter a stream
- `enable()` / `disable()` - Toggle filtering
- `close()` - Clean up resources

**Properties:**
- `is_enabled: bool` - Check if filtering is active

**Context Manager:**
```python
with KrispVivaFilter() as filter:
    # Automatic cleanup
    pass
```

### `KrispVivaTurn`

**Methods:**
- `process_audio(frame, *, is_speech) -> float` - Process audio frame, return turn probability
- `clear()` - Reset turn detection state
- `close()` - Clean up resources
- `async predict_end_of_turn(chat_ctx, timeout) -> float` - Protocol compatibility method
- `async supports_language(language) -> bool` - Always returns True (audio-based)
- `async unlikely_threshold(language) -> float | None` - Returns threshold

**Properties:**
- `model: str` - Model identifier ("krisp-viva-turn")
- `provider: str` - Provider name ("krisp")
- `threshold: float` - Turn probability threshold (get/set)
- `last_probability: float | None` - Last computed turn probability
- `frame_probabilities: list[float]` - All frame probabilities from last processing
- `speech_triggered: bool` - Whether speech has been detected

**Context Manager:**
```python
with KrispVivaTurn() as detector:
    # Automatic cleanup
    pass
```

## Turn Detection Notes

**Audio-based vs Text-based:**
- Krisp turn detection (`KrispVivaTurn`) works on audio frames
- LiveKit's built-in turn detector works on chat context (text)
- Both implement the same protocol and can be used interchangeably
- Audio-based detection has lower latency (no STT required for detection)
