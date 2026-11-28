# Kitsch Labs plugin for LiveKit Agents

[Kitsch Labs](https://kitschlabs.com/en) provides Korean and Japanese speech services
for anime characters and other subculture content.

Use the plugin for HTTP or streaming text-to-speech, file transcription, and
realtime speech-to-text with interim and final transcripts.

## Installation

```bash
pip install livekit-plugins-kitsch
```

Set KITSCH_API_KEY to your workspace API key, or pass api_key to TTS and STT.
Choose a voice accessible to your workspace.

## Documentation

- Text-to-speech: https://docs.livekit.io/agents/models/tts/kitsch/
- Speech-to-text: https://docs.livekit.io/agents/models/stt/kitsch/

These documentation pages are pending the plugin release.

## Usage notes

Each TTS stream represents one utterance. Text beyond 5,000 characters is truncated
with a warning. Complete an utterance with flush() or end_input() and create a
new stream for the next utterance.

STT keeps accepting audio while a final transcript is pending. end_input() drains
outstanding finals, while aclose() cancels immediately. Interim transcripts replace
the previous hypothesis. Use your existing VAD for interruption handling.

STT respects APIConnectOptions, including retries. After a connection failure,
recovery opens a fresh session for subsequent audio. Use max_retry=0 to disable
recovery. File requests may be repeated when retrying, so account for
provider usage when choosing retry settings.

The plugin uses the LiveKit job HTTP session. Outside a worker, pass an
aiohttp.ClientSession using http_session and manage that session lifetime.

API keys, customer text, audio, and provider error bodies are excluded from plugin
logs. Recording and telemetry settings are controlled by the application.
