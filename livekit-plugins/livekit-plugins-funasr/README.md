# LiveKit Plugins FunASR

Agent Framework plugin for local speech-to-text with [FunASR](https://github.com/modelscope/FunASR) models such as [SenseVoice](https://github.com/FunAudioLLM/SenseVoice).

SenseVoice is an open-source, fully-local, non-autoregressive multilingual ASR model (Chinese, Cantonese, English, Japanese, Korean and more) with leading Chinese accuracy and fast inference. The model runs locally, so no API key is required.

## Installation

```bash
pip install livekit-plugins-funasr
```

## Usage

```python
from livekit.plugins import funasr

stt = funasr.STT(model="iic/SenseVoiceSmall", device="cuda")
```

The first run downloads the model from ModelScope/Hugging Face. Use `language=None` (default) for automatic language detection, or set e.g. `language="zh"`.
