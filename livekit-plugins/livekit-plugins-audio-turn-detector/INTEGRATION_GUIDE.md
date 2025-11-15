# Audio Turn Detector Integration Guide

This guide explains how to integrate the audio-based turn detector into your LiveKit Agents project.

## Architecture Overview

### Flow Diagram

```
User Speech (Microphone)
         ↓
   rtc.AudioFrame (raw audio)
         ↓
   ┌─────────────────────────────────────┐
   │   AudioRecognition.push_audio()     │
   └─────────────────────────────────────┘
         ↓
   ┌────┬────────────┬────────────────┐
   │    │            │                │
   VAD  STT   (Audio frames buffered) │
   │    │                             │
   │    │                             │
   ▼    ▼                             │
   │    │                             │
   │    Transcript                    │
   │                                  │
   END_OF_SPEECH event                │
   (includes accumulated frames)──────┘
         ↓
   ┌─────────────────────────────────────┐
   │  _run_audio_turn_detection()        │
   │  - Combine frames                   │
   │  - Extract features                 │
   │  - ONNX inference                   │
   └─────────────────────────────────────┘
         ↓
   Audio Turn Probability (0.0 - 1.0)
         ↓
   ┌─────────────────────────────────────┐
   │  _run_eou_detection()               │
   │  - Compare with threshold           │
   │  - Adjust endpointing delay         │
   └─────────────────────────────────────┘
         ↓
   Decide End of Turn
```

### Key Components

1. **AudioTurnDetector** (`audio_turn_detector.py`)
   - Loads ONNX model
   - Extracts audio features (Whisper or raw)
   - Runs inference in thread pool (non-blocking)
   - Returns probability of end-of-turn

2. **AudioRecognition** (`audio_recognition.py`)
   - Modified to accept `audio_turn_detector` parameter
   - Calls `_run_audio_turn_detection()` on VAD END_OF_SPEECH
   - Passes probability to `_run_eou_detection()`

3. **AgentSession** (`agent_session.py`)
   - New parameter: `audio_turn_detection`
   - Stores and passes to AgentActivity

4. **AgentActivity** (`agent_activity.py`)
   - Passes `audio_turn_detection` to AudioRecognition

## Installation

### 1. Install the Plugin

```bash
# Basic installation
cd libs/agents/livekit-plugins/livekit-plugins-audio-turn-detector
pip install -e .

# Or with Whisper feature support
pip install -e ".[whisper]"
```

### 2. Verify Installation

```python
from livekit.plugins.audio_turn_detector import AudioTurnDetector
print(AudioTurnDetector.__doc__)
```

## Model Requirements

Your ONNX model should:

### Input Requirements

**Option 1: Whisper Features (Recommended)**
- Input shape: `(batch=1, mel_bins=80, time_steps=3000)`
- Feature extraction handled automatically by `WhisperFeatureExtractor`
- Example: smart-turn-v3 model

**Option 2: Raw Waveform**
- Input shape: `(batch=1, samples)`
- Samples = `sample_rate * max_audio_seconds`
- Example: `(1, 128000)` for 8 seconds @ 16kHz

### Output Requirements

- Output shape: `(batch=1, 1)` or `(1,)`
- Single probability value between 0.0 and 1.0
- Higher value = more likely to be end-of-turn

### Model Optimization Tips

```python
# When converting your model to ONNX, use:
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize to reduce size and improve speed
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

## Usage Examples

### Basic Usage

```python
from livekit.agents import AgentSession
from livekit.plugins.audio_turn_detector import AudioTurnDetector
from livekit.plugins import silero, deepgram, openai

# Load audio turn detector
audio_turn_detector = AudioTurnDetector(
    model_path="path/to/your/model.onnx",
    feature_type="whisper",
    max_audio_seconds=8,
    activation_threshold=0.5,
)

# Create session
session = AgentSession(
    vad=silero.VAD.load(),
    stt=deepgram.STT(),
    audio_turn_detection=audio_turn_detector,  # NEW!
    llm=openai.LLM(),
    tts=openai.TTS(),
)
```

### Hybrid Mode (Audio + Text)

```python
from livekit.plugins.turn_detector import MultilingualModel

# Both detectors
audio_detector = AudioTurnDetector(...)
text_detector = MultilingualModel()

session = AgentSession(
    audio_turn_detection=audio_detector,  # Primary
    turn_detection=text_detector,         # Secondary
    # Audio takes priority for endpointing delay adjustment
)
```

### Advanced Configuration

```python
audio_turn_detector = AudioTurnDetector(
    model_path="smart_turn_v3.onnx",
    feature_type="whisper",
    max_audio_seconds=8,
    sample_rate=16000,
    activation_threshold=0.5,      # Binary prediction threshold
    unlikely_threshold=0.3,        # If prob < 0.3, use max_endpointing_delay
    cpu_count=2,                   # Number of CPU threads for inference
)
```

## Integration with Pathors Agent

### Step 1: Modify Your Agent Initialization

```python
# In your inbound_agent.py or outbound_agent.py

from livekit.plugins.audio_turn_detector import AudioTurnDetector

def prewarm(proc: JobProcess):
    # Load VAD
    proc.userdata["vad"] = silero.VAD.load()

    # Load Audio Turn Detector
    audio_turn_detector = AudioTurnDetector(
        model_path="/path/to/smart_turn_model.onnx",
        feature_type="whisper",
        max_audio_seconds=8,
        unlikely_threshold=0.3,
        cpu_count=2,
    )
    proc.userdata["audio_turn_detector"] = audio_turn_detector
```

### Step 2: Pass to Assistant

```python
# In your PathorsAssistant or BaseEntrypointHandler

assistant = PathorsAssistant(
    config=config,
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    audio_turn_detection=audio_turn_detector,  # NEW!
    min_endpointing_delay=0.4,  # Shorter for confident detection
    max_endpointing_delay=2.0,  # Longer for uncertain cases
)
```

### Step 3: Update AgentSession

If you're using `AgentSession` directly:

```python
session = AgentSession(
    vad=vad,
    stt=stt,
    audio_turn_detection=audio_turn_detector,  # Add this
    llm=llm,
    tts=tts,
    # ... other params
)
```

## Configuration Parameters

### AudioTurnDetector

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to ONNX model file |
| `feature_type` | str | "whisper" | "whisper" or "raw" |
| `max_audio_seconds` | int | 8 | Max audio duration to analyze |
| `sample_rate` | int | 16000 | Audio sample rate |
| `activation_threshold` | float | 0.5 | Binary prediction threshold |
| `unlikely_threshold` | float | None | Low confidence threshold |
| `cpu_count` | int | 1 | CPU threads for inference |

### Endpointing Delays

**How it works:**

```python
if audio_turn_probability >= unlikely_threshold:
    # High confidence = end of turn likely
    endpointing_delay = min_endpointing_delay  # e.g., 0.4s
else:
    # Low confidence = might continue speaking
    endpointing_delay = max_endpointing_delay  # e.g., 2.0s
```

**Recommended values:**

- **Min delay**: 0.3 - 0.5s (for confident end-of-turn)
- **Max delay**: 1.5 - 2.5s (for uncertain cases)
- **Unlikely threshold**: 0.25 - 0.35 (depends on your model)

## Performance Considerations

### Inference Frequency

✅ **Low frequency** - Only runs on VAD END_OF_SPEECH events
- Typically 1-3 times per user turn
- Much lower than VAD (every 32ms)

### Latency

Audio turn detection adds:
- Feature extraction: ~10-50ms (Whisper features)
- ONNX inference: ~20-100ms (depends on model size)
- Total overhead: ~30-150ms per turn

**Tip**: Use quantized ONNX models (QUInt8) to reduce latency.

### Thread Safety

✅ Inference runs in `ThreadPoolExecutor` to avoid blocking event loop

```python
# Automatically handled
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(executor, self._predict_sync, ...)
```

## Debugging

### Enable Debug Logging

```python
import logging

logging.getLogger("livekit.plugins.audio_turn_detector").setLevel(logging.DEBUG)
logging.getLogger("livekit.agents.voice.audio_recognition").setLevel(logging.DEBUG)
```

### Check Audio Turn Results

```python
# In your logs, you'll see:
# "audio turn detection completed" - probability, num_frames, total_samples
# "audio turn detection result" - audio_turn_probability
```

### Verify Model Loading

```python
audio_turn_detector = AudioTurnDetector(...)
print(f"Model loaded: {audio_turn_detector.model}")
print(f"Provider: {audio_turn_detector.provider}")
```

## Troubleshooting

### Common Issues

**1. Model not found**
```
FileNotFoundError: [Errno 2] No such file or directory: 'model.onnx'
```
→ Use absolute path: `/full/path/to/model.onnx`

**2. ONNX Runtime error**
```
RuntimeError: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT
```
→ Check input shape matches model expectations

**3. High latency**
```
inference_duration: 0.5s
```
→ Use quantized model or reduce `max_audio_seconds`

**4. No audio frames**
```
WARNING: No audio frames provided for turn detection
```
→ Check VAD is enabled and producing END_OF_SPEECH events

### Performance Tuning

```python
# For faster inference
audio_turn_detector = AudioTurnDetector(
    model_path="model_quantized.onnx",  # Use quantized model
    max_audio_seconds=5,                # Reduce from 8
    cpu_count=4,                        # Use more CPU threads
)

# For better accuracy
audio_turn_detector = AudioTurnDetector(
    max_audio_seconds=10,               # Analyze more context
    unlikely_threshold=0.25,            # More conservative
)
```

## Comparison: Audio vs Text Turn Detection

| Aspect | Audio-based | Text-based |
|--------|-------------|------------|
| **Latency** | Lower (no STT wait) | Higher (requires STT) |
| **Accuracy** | Depends on model | High with good LLM context |
| **Language Support** | Universal (audio features) | Language-specific |
| **Context Awareness** | Limited to prosody | Full conversation context |
| **Computational Cost** | Medium (ONNX inference) | Medium (LLM inference) |
| **Best For** | Quick responses | Context-aware decisions |

## Best Practices

1. **Use both audio and text detection** for best results
2. **Tune thresholds** based on your model and use case
3. **Monitor latency** in production
4. **Log probabilities** for debugging
5. **Test with real conversations** to find optimal settings

## Next Steps

- [ ] Train or obtain an audio turn detection model
- [ ] Convert model to ONNX format
- [ ] Test with your agent
- [ ] Tune thresholds and delays
- [ ] Monitor performance in production

## Support

For issues or questions:
- Check the [examples](./examples/) directory
- Review the [README](./README.md)
- Open an issue on GitHub
