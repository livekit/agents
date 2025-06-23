# Google Vertex AI MARS7 Plugin for LiveKit Agents

This plugin integrates Google Vertex AI's [MARS7](https://docs.camb.ai/) text-to-speech model with LiveKit Agents. MARS7 is CAMB.AI's latest generation speech synthesis model that creates hyper-realistic, prosodic, multilingual text-to-speech outputs with optional voice cloning capabilities.

## Installation

```bash
pip install livekit-plugins-google-vertex-camb
```

## Setup

### Prerequisites

1. **Google Cloud Project**: Access to a Google Cloud project with Vertex AI enabled
2. **Service Account**: A service account with appropriate Vertex AI permissions
3. **MARS7 Endpoint**: A deployed MARS7 model endpoint on Vertex AI (from `cambai` publisher)

### Authentication

Set up authentication using one of these methods:

#### Option 1: Environment Variable
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Option 2: Pass credentials path directly
```python
from livekit.plugins import google_vertex_camb

tts_engine = google_vertex_camb.TTS(
    endpoint_id="your_endpoint_id",
    credentials_path="/path/to/service-account-key.json"
)
```

## Usage

### Basic Text-to-Speech

```python
from livekit.plugins import google_vertex_camb

# Initialize the TTS engine
tts_engine = google_vertex_camb.TTS(
    endpoint_id="your_mars7_endpoint_id",
    project_id="cambai-public",  # Default project for MARS7
    location="us-central1",
    language=google_vertex_camb.Language.EN_US
)

# Synthesize speech
audio_stream = tts_engine.synthesize("Hello, this is MARS7 text-to-speech synthesis!")

# Process the audio stream
async for event in audio_stream:
    # Process audio frames (FLAC format, streamed in 8KB chunks)
    pass
```

### Voice Cloning with Reference Audio

```python
from livekit.plugins import google_vertex_camb

# Initialize with reference audio for voice cloning
tts_engine = google_vertex_camb.TTS(
    endpoint_id="your_mars7_endpoint_id",
    language=google_vertex_camb.Language.EN_US,
    audio_ref_path="/path/to/reference_voice.wav",
    ref_text="This is the transcription of the reference audio"  # Optional but recommended
)

# Synthesize with the cloned voice
audio_stream = tts_engine.synthesize("Hello, this will sound like the reference voice!")

async for event in audio_stream:
    # Process audio frames with cloned voice
    pass
```

## Configuration

### Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON file
- `PROJECT_ID`: Google Cloud project ID (defaults to "cambai-public")
- `LOCATION`: Google Cloud location (defaults to "us-central1")
- `ENDPOINT_ID`: Your deployed MARS7 endpoint ID

### Constructor Parameters

```python
TTS(
    endpoint_id: str,                     # Required: Vertex AI endpoint ID for MARS7
    project_id: str = "cambai-public",    # Optional: GCP project ID
    location: str = "us-central1",        # Optional: GCP location
    credentials_path: str = None,         # Optional: Path to credentials JSON
    language: Language = Language.EN_US,  # Optional: Target language
    audio_ref_path: str = None,           # Optional: Reference audio for voice cloning
    ref_text: str = None,                 # Optional: Reference audio transcription
)
```

## Supported Languages

- `Language.EN_US`: English (US)
- `Language.ES_ES`: Spanish (Spain)

## Technical Details

### Audio Format
- **Output Format**: FLAC (24kHz, mono)
- **Streaming**: Audio is streamed in 8KB chunks for optimal performance
- **Voice Cloning**: Supports reference audio in WAV format (2-10 seconds recommended)

### API Integration
- Uses Google Vertex AI's `raw_predict` method
- Handles base64 encoding/decoding of audio data
- Supports both basic synthesis and voice cloning workflows

## Features

- **High-quality synthesis**: MARS7 model provides natural, prosodic speech
- **Voice cloning**: Use reference audio to clone specific voices with optional transcription
- **Multilingual support**: Support for multiple languages
- **Streaming output**: Efficient chunked streaming of synthesized audio
- **Google Cloud integration**: Native integration with Vertex AI
- **Production ready**: Handles large audio files with optimized chunking

## Error Handling

The plugin handles various error conditions:

- **Authentication errors**: Invalid or missing Google Cloud credentials
- **Endpoint errors**: Endpoint not found, not accessible, or not deployed
- **API errors**: Vertex AI API-related issues
- **Audio processing errors**: Issues with reference audio or FLAC processing

## Requirements

- Python >= 3.9
- Google Cloud SDK with Vertex AI permissions
- Valid Google Cloud service account
- Deployed MARS7 model endpoint from `cambai` publisher
- Dependencies: `google-cloud-aiplatform>=1.98.0`, `soundfile>=0.13.1`

## License

Apache 2.0