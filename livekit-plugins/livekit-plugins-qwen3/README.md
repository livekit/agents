# livekit-plugins-qwen3

[![PyPI version](https://badge.fury.io/py/livekit-plugins-qwen3.svg)](https://pypi.org/project/livekit-plugins-qwen3/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A [LiveKit Agents](https://docs.livekit.io/agents/) plugin for Qwen3 TTS and STT (Alibaba Cloud DashScope).

## Features

- **Text-to-Speech (TTS)**: Real-time streaming speech synthesis with multiple voices and languages
- **Speech-to-Text (STT)**: Real-time streaming speech recognition with interim results

## Installation

```bash
pip install livekit-plugins-qwen3
```

## Configuration

Set your DashScope API key as an environment variable:

```bash
export DASHSCOPE_API_KEY=your_api_key
```

Or copy `.env.example` to `.env` and fill in your credentials.

Get your API key at: https://dashscope.console.aliyun.com/

### International Users

By default, the plugin uses the China mainland endpoint. For international access, set:

```bash
export DASHSCOPE_BASE_URL=wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime
```

## Usage

### Text-to-Speech (TTS)

```python
from livekit.plugins import qwen3

tts = qwen3.TTS(
    voice="Kiki",      # or "Rocky", "Cherry", etc.
    language="auto",   # or "chinese", "english", "cantonese", etc.
)

# Streaming TTS
stream = tts.stream()
stream.push_text("Hello, world!")
stream.flush()

async for event in stream:
    # Handle audio events
    pass
```

### Speech-to-Text (STT)

```python
from livekit.plugins import qwen3

stt = qwen3.STT(
    language="zh",  # or "en", "yue" (Cantonese), etc.
)

# Streaming STT
stream = stt.stream()

# Push audio frames
stream.push_frame(audio_frame)

# Get transcription events
async for event in stream:
    if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
        print(f"Interim: {event.alternatives[0].text}")
    elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
        print(f"Final: {event.alternatives[0].text}")
```

### With LiveKit Agent

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import qwen3

agent = Agent(
    tts=qwen3.TTS(voice="Kiki"),
    stt=qwen3.STT(language="zh"),
)
```

## Available Options

### TTS Voices

| Category | Voices |
|----------|--------|
| Female | Kiki, Cherry, Jennifer |
| Male | Rocky, Ethan, Ryan |
| Regional | Sichuan-Sunny, Shanghai-Jada, Beijing-Yunxi |
| Cantonese | Cantonese_ProfessionalHost |

### TTS Languages

`auto`, `chinese`, `english`, `cantonese`, `german`, `italian`, `portuguese`, `spanish`, `japanese`, `korean`, `french`, `russian`

### STT Languages

`zh` (Chinese), `en` (English), `yue` (Cantonese), and more.

### Models

| Feature | Model |
|---------|-------|
| TTS | `qwen3-tts-flash-realtime` |
| STT | `qwen3-asr-flash-realtime` |

## API Reference

### TTS

```python
qwen3.TTS(
    model: str = "qwen3-tts-flash-realtime",
    voice: str = "Kiki",
    language: str = "auto",
    mode: str = "server_commit",
    sample_rate: int = 24000,
    api_key: str | None = None,
    base_url: str | None = None,
)
```

### STT

```python
qwen3.STT(
    model: str = "qwen3-asr-flash-realtime",
    language: str = "zh",
    sample_rate: int = 16000,
    api_key: str | None = None,
    base_url: str | None = None,
)
```

## Links

- [Qwen3 TTS API Reference](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-api)
- [Qwen3 TTS Guide](https://www.alibabacloud.com/help/en/model-studio/qwen-tts)
- [Qwen3 ASR Guide](https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run linting: `ruff check --fix && ruff format`
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
