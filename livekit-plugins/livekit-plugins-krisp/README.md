# Krisp VIVA Plugin for LiveKit Agents

Real-time noise reduction and audio-based turn detection for LiveKit voice agents using [Krisp's VIVA SDK](https://krisp.ai/).

## Features

- **`KrispVivaFilterFrameProcessor`**: Real-time noise reduction FrameProcessor for audio processing
- **`KrispVivaTurn`**: Audio-based turn detection for accurate end-of-turn detection

## Installation

```bash
# Install the plugin
pip install livekit-plugins-krisp

# Install krisp-audio SDK separately (required for actual usage)
pip install krisp-audio
```

**Note:** The `krisp-audio` package is a proprietary SDK not available on public PyPI. 
It must be obtained and installed separately from Krisp.

## Prerequisites

### Required for All Features

1. **Krisp Audio SDK**: `pip install krisp-audio`
2. **License Key**: Obtain a license key from Krisp and set it as an environment variable:
   ```bash
   export KRISP_VIVA_SDK_LICENSE_KEY=your-license-key-here
   ```

### For Noise Reduction

1. **Noise Reduction Model**: Obtain a noise reduction `.kef` model file from Krisp
2. **Set environment variable**:
   ```bash
   export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/noise_model.kef
   ```

### For Turn Detection

1. **Turn Detection Model**: Obtain a turn detection `.kef` model file from Krisp
2. **Set environment variable**:
   ```bash
   export KRISP_VIVA_TURN_MODEL_PATH=/path/to/turn_model.kef
   ```

Note: Noise reduction and turn detection use different model files.

## Quick Start

### Human-to-Bot Noise Cancellation (Recommended)

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

### KrispVivaFilterFrameProcessor Parameters

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
noise_processor = krisp.KrispVivaFilterFrameProcessor(noise_suppression_level=90)
turn_detector = krisp.KrispVivaTurn(threshold=0.5)

# Process audio through processor and turn detector
filtered_frame = noise_processor.process(audio_frame)  # FrameProcessor.process() is synchronous
turn_probability = turn_detector.process_audio(filtered_frame, is_speech=True)

# Cleanup - SDK destroyed only when both are closed
noise_processor.close()  # SDK still active (turn_detector holds reference)
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

### `KrispVivaFilterFrameProcessor`

**Purpose:** FrameProcessor implementation for Krisp noise reduction. Can be used directly with the `noise_cancellation` parameter in `AudioInputOptions` or `RoomInputOptions`.

**Constructor:**
```python
KrispVivaFilterFrameProcessor(
    model_path: str | None = None,
    noise_suppression_level: int = 100,
    frame_duration_ms: int = 10,
    sample_rate: int | None = None,
)
```

**Methods:**
- `process(frame: AudioFrame) -> AudioFrame` - Process a single frame (synchronous, required by FrameProcessor interface)
- `enable()` / `disable()` - Toggle filtering
- `close()` - Clean up resources

**Properties:**
- `is_enabled: bool` - Check if filtering is active

**Usage:**
```python
# Create processor
processor = krisp.KrispVivaFilterFrameProcessor(
    noise_suppression_level=100,
    frame_duration_ms=10,
)

# Use in AudioInputOptions
await session.start(
    agent=MyAgent(),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            sample_rate=16000,
            frame_size_ms=10,
            noise_cancellation=processor,  # Pass FrameProcessor directly
        ),
    ),
)
```

**Context Manager:**
```python
with KrispVivaFilterFrameProcessor() as processor:
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
