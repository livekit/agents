# LiveKit NVIDIA STT Plugin

This plugin provides Speech-to-Text (STT) capabilities using NVIDIA's Riva ASR models through the NVIDIA Cloud Functions API.

## Features

- **Streaming Recognition**: Real-time speech-to-text conversion
- **Interim Results**: Get partial transcripts as users speak
- **Multiple Models**: Support for NVIDIA's Parakeet models
- **Automatic Punctuation**: Built-in punctuation support
- **Language Support**: Configurable language codes

## Installation

1. Install the nvidia-riva-client dependency:

```bash
pip install nvidia-riva-client
```

2. Set up your NVIDIA API key:

```bash
export NVIDIA_API_KEY="your_api_key_here"
```

## Usage

### Basic Usage

```python
import os
from livekit.plugins.nvidia import STT

# Initialize the STT
stt = STT(
    model="parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
    function_id="1598d209-5e27-4d3c-8079-4751568b1081",
    api_key=os.getenv("NVIDIA_API_KEY")  # or pass directly
)

# Create a recognition stream
stream = stt.stream(language="en-US")
```

### Configuration Options

- `model`: NVIDIA ASR model to use (default: parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer)
- `function_id`: NVIDIA function ID for the API (default: 1598d209-5e27-4d3c-8079-4751568b1081)
- `punctuate`: Enable automatic punctuation (default: True)
- `language_code`: Language code for recognition (default: en-US)
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `server`: NVIDIA server endpoint (default: grpc.nvcf.nvidia.com:443)
- `api_key`: NVIDIA API key (can also use NVIDIA_API_KEY env var)

### Available Models

Based on testing, the following models are confirmed to work:

**Streaming Models:**

- `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer` (Function ID: 1598d209-5e27-4d3c-8079-4751568b1081)

**Offline Models:**

- `parakeet-1.1b-en-US-asr-offline-silero-vad-sortformer` (Function ID: 1598d209-5e27-4d3c-8079-4751568b1081)

### Example Agent

```python
import asyncio
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.plugins.nvidia import STT

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Create STT instance
    stt = STT()

    # Process audio from participants
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        print(f"Participant connected: {participant.identity}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## Testing

Run the test script to verify your setup:

```bash
export NVIDIA_API_KEY="your_api_key_here"
python test_nvidia_stt.py
```

## Requirements

- Python 3.8+
- nvidia-riva-client
- livekit-agents
- Valid NVIDIA API key

## Troubleshooting

### Common Issues

1. **Import Error for riva.client**: Make sure nvidia-riva-client is installed
2. **API Key Error**: Ensure NVIDIA_API_KEY is set in your environment
3. **Model Not Available**: Check that you're using a supported model name and function ID
4. **Connection Issues**: Verify your internet connection and NVIDIA API access

### Debugging

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This plugin is licensed under the Apache License 2.0.
