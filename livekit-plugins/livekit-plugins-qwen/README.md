# Qwen plugin for LiveKit Agents

Support for Qwen speech and language models on [Alibaba Cloud Model Studio](https://www.alibabacloud.com/en/product/modelstudio) (DashScope) in LiveKit Agents: realtime speech-to-text (`qwen3-asr-flash-realtime`), realtime text-to-speech (`qwen3-tts-flash-realtime`) and chat completions on the OpenAI-compatible endpoint.

See the [STT](https://docs.livekit.io/agents/models/stt/) and [TTS](https://docs.livekit.io/agents/models/tts/) model documentation for more information.

## Installation

```bash
pip install livekit-plugins-qwen
```

## Pre-requisites

You need a Model Studio API key. Create one in the Model Studio console and set it in your `.env`:

```
DASHSCOPE_API_KEY=<your_model_studio_api_key>
```

API keys are bound to a region. The plugin defaults to `region="intl"` (Singapore, keys created on alibabacloud.com). Pass `region="cn"` for Beijing keys created on aliyun.com. Model Studio also issues workspace-dedicated domains; pass one as `base_url` (it overrides `region`):

```python
qwen.STT(base_url="wss://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime")
```

For `qwen3-asr-flash-realtime`, enable the model for your workspace in the Model Studio console before the first call.

## Usage

### Speech-to-Text (STT)

```python
from livekit.agents import AgentSession
from livekit.plugins import qwen

session = AgentSession(
    stt=qwen.STT(),
    # ... llm, tts, etc.
)
```

Leave `language` unset to let the model detect the language, which is also Model Studio's recommended setting for speech that mixes Mandarin and English. Set `language="zh"` (or another ISO code) to pin it. `vad_silence_duration_ms` and `vad_threshold` tune Model Studio's server-side turn detection.

### Text-to-Speech (TTS)

```python
from livekit.plugins import qwen

session = AgentSession(
    tts=qwen.TTS(
        voice="Cherry",          # Cherry, Serena, Ethan and Chelsie speak both Mandarin and English
        language_type="Auto",    # or "Chinese", "English", ... for single-language text
    ),
    # ... stt, llm, etc.
)
```

Text is streamed to the model as it arrives from the LLM; the model closes sentences itself (`server_commit` mode), so playback starts before the turn's text is complete.

### LLM

```python
from livekit.plugins import qwen

session = AgentSession(
    llm=qwen.LLM(model="qwen-plus"),
    # ... stt, tts, etc.
)
```

`qwen.LLM` is `openai.LLM` pointed at Model Studio's OpenAI-compatible endpoint for the chosen region. Qwen's deep-thinking mode is off by default (`enable_thinking=False`): LiveKit's LLM stream does not read `reasoning_content`, so a thinking turn would play as silence. Pass `enable_thinking=True` (and optionally `thinking_budget`) to turn it on.

## More information and reference

- [Qwen-ASR realtime API: client events](https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-client-events) and [server events](https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-server-events)
- [Qwen-TTS realtime API: client events](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-client-events) and [server events](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-server-events)
- [Qwen-TTS voice list](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-list)
- [OpenAI compatibility](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope) and [deep thinking](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)
- [Regions and access domains](https://www.alibabacloud.com/help/en/model-studio/regions/)
