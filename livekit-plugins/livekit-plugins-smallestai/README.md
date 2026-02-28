# LiveKit Plugins Smallest AI

LiveKit Agent Framework plugin for speech synthesis with the [Smallest AI](https://smallest.ai/) API ([documentation](https://waves-docs.smallest.ai/)).

## Installation

```bash
pip install livekit-plugins-smallestai
```

## Pre-requisites

You'll need an API key from Smallest AI. It can be set as an environment variable: `SMALLEST_API_KEY`

## Features

### Text-to-Speech (TTS)
Uses the Smallest AI Waves TTS API with support for multiple models (`lightning`, `lightning-large`, `lightning-v2`).

### Speech-to-Text (STT)
Uses the Smallest AI Pulse STT API with support for:
- **Pre-recorded transcription**: Batch audio file transcription via HTTP POST
- **Real-time streaming**: Low-latency WebSocket-based transcription (~64ms TTFT)
- **Word timestamps**: Precise timing information for each word
- **Speaker diarization**: Identify and label different speakers
- **32+ languages**: Including automatic language detection (`language="multi"`)