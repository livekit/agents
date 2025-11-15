# LiveKit Audio Turn Detector Plugin

Audio-based turn detection for LiveKit Agents using ONNX models.

## Overview

This plugin provides end-of-turn detection by analyzing raw audio frames, complementing or replacing text-based turn detection methods. It uses ONNX models for efficient on-device inference.

## Features

- **Audio-based detection**: Analyzes raw audio without requiring text transcription
- **VAD-triggered**: Works seamlessly with LiveKit's VAD for efficient processing
- **Flexible features**: Supports Whisper features or raw waveform input
- **Language-agnostic**: Audio features work across multiple languages
- **Async-ready**: Non-blocking inference using thread pools

## Installation

```bash
# Basic installation
pip install livekit-plugins-audio-turn-detector

# With Whisper feature extractor support
pip install livekit-plugins-audio-turn-detector[whisper]
```

## Usage

### Basic Example

```python
from livekit.agents import AgentSession, Agent
from livekit.plugins.audio_turn_detector import AudioTurnDetector
from livekit.plugins import silero, deepgram

# Initialize audio turn detector
audio_turn_detector = AudioTurnDetector(
    model_path="path/to/your/model.onnx",
    feature_type="whisper",  # or "raw"
    max_audio_seconds=8,
    activation_threshold=0.7,
)

# Use in agent session
session = AgentSession(
    vad=silero.VAD.load(),
    stt=deepgram.STT(),
    turn_detection=audio_turn_detector,  # Use audio-based turn detection
    # ... other configuration
)

await session.start(agent=Agent(...), room=ctx.room)
```

### Hybrid Mode (Audio + Text)

You can combine audio-based and text-based turn detection:

```python
from livekit.plugins.turn_detector import MultilingualModel
from livekit.plugins.audio_turn_detector import AudioTurnDetector

# Text-based turn detector
text_detector = MultilingualModel()

# Audio-based turn detector
audio_detector = AudioTurnDetector(
    model_path="path/to/model.onnx",
    unlikely_threshold=0.3,  # Low probability = use longer delay
)

# Use both (audio takes priority)
session = AgentSession(
    vad=silero.VAD.load(),
    stt=deepgram.STT(),
    turn_detection=audio_detector,  # Primary
    # You can also manually combine probabilities in custom logic
)
```

### Custom Model Integration

```python
# If you have your own ONNX model
detector = AudioTurnDetector(
    model_path="custom_model.onnx",
    feature_type="raw",  # Use raw waveform input
    max_audio_seconds=5,  # Adjust to your model's requirements
    sample_rate=16000,
    activation_threshold=0.5,
    cpu_count=2,  # Use 2 CPU threads for inference
)
```

## How It Works

1. **VAD Detection**: Silero VAD accumulates audio frames during speech
2. **END_OF_SPEECH Event**: When VAD detects silence, it triggers turn detection
3. **Audio Analysis**: Audio turn detector analyzes the accumulated frames
4. **Probability Output**: Returns probability of end-of-turn (0.0 to 1.0)
5. **Delay Adjustment**: High probability = short delay, low probability = longer delay

## Model Requirements

Your ONNX model should:
- Accept audio features as input (Whisper mel-spectrogram or raw waveform)
- Output a single probability value (0.0 to 1.0)
- Be optimized for CPU inference

### Example Model Input/Output

**Whisper Features:**
- Input: `(1, 80, 3000)` - Batch, Mel bins, Time steps
- Output: `(1, 1)` - Batch, Probability

**Raw Waveform:**
- Input: `(1, 128000)` - Batch, Samples (8 seconds @ 16kHz)
- Output: `(1, 1)` - Batch, Probability

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to ONNX model file |
| `feature_type` | str | "whisper" | "whisper" or "raw" |
| `max_audio_seconds` | int | 8 | Maximum audio duration to analyze |
| `sample_rate` | int | 16000 | Expected audio sample rate |
| `activation_threshold` | float | 0.5 | Threshold for binary prediction |
| `cpu_count` | int | 1 | Number of CPU threads |
| `unlikely_threshold` | float | None | Custom threshold for low-confidence detection |

## Performance Considerations

- **Inference Frequency**: Only runs on VAD END_OF_SPEECH events (low frequency)
- **Thread Pool**: Uses async executor to avoid blocking event loop
- **Audio Truncation**: Analyzes only last N seconds to control memory usage

## Comparison: Audio vs Text Turn Detection

| Feature | Audio-based | Text-based |
|---------|-------------|------------|
| **Latency** | Lower (no STT wait) | Higher (requires STT) |
| **Language Support** | Universal | Language-specific |
| **Context Awareness** | Limited to audio | Full conversation context |
| **Accuracy** | Depends on model | High with good LLM |

## License

Apache 2.0

## Contributing

Contributions welcome! Please ensure:
- Models are ONNX format
- Thread-safe inference
- Proper error handling
- Documentation updates
