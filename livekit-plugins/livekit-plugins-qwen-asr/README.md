# Qwen3-ASR plugin for LiveKit Agents

Speech-to-text for a self-hosted [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) model served by [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html).

This is not the Alibaba Model Studio realtime API. Point `base_url` at your own vLLM server.

## Installation

```bash
pip install livekit-agents[qwen-asr]
```

## Run the model

Batch transcription works with a normal serve:

```bash
vllm serve Qwen/Qwen3-ASR-1.7B
```

Realtime also needs the realtime architecture. That process still serves batch transcription, including context prompts:

```bash
vllm serve Qwen/Qwen3-ASR-1.7B \
  --hf-overrides '{"architectures": ["Qwen3ASRRealtimeGeneration"]}'
```

## Batch

`use_realtime=False` is the default. LiveKit's VAD cuts each turn and `recognize()` posts it to `/v1/audio/transcriptions`. Leave `prompt` unset for an empty context. Set it to bias names and domain words; vLLM puts that text in Qwen's system turn. LiveKit keyterms are appended to the same text, because the model has no separate keywords field.

```python
from livekit.plugins import qwen_asr

stt = qwen_asr.STT(
    base_url="http://127.0.0.1:8000/v1",
    model="Qwen/Qwen3-ASR-1.7B",
    language="tr",
    prompt="Türkçe telefon görüşmesi. Fibabanka tüketici kredisi.",
)
```

Omit `language` to let the model detect it. Omit `prompt` to send no context.

## Realtime

`use_realtime=True` opens vLLM's `/v1/realtime` socket. Audio is resampled to 16 kHz PCM. The server does not detect end of speech, so the plugin uses the bundled Silero VAD and closes the turn itself. Pass `vad=None` and call `flush()` to close turns yourself.

```python
stt = qwen_asr.STT(
    base_url="http://127.0.0.1:8000/v1",
    model="Qwen/Qwen3-ASR-1.7B",
    language="tr",
    use_realtime=True,
)
```

`language` and `prompt` are sent on `session.update`. vLLM 0.30 reads the model name from that event and does not apply the prompt inside its realtime template, so context biasing currently applies to batch recognition. The plugin also drops the `language …<asr_text>` preamble that this vLLM version streams in front of the words.
