# BreezeBlue TTS plugin for LiveKit Agents

Use the [Breeze Developer API](https://docs.breezeblue.ai/) for text-to-speech in a LiveKit Agent.

## Quickstart

Install the `breezeblue` extra from a LiveKit Agents release that includes this plugin.

```sh
pip install 'livekit-agents[breezeblue]'
```

To try the plugin from source before a release, run this from the root of a checkout containing it.

```sh
python -m pip install ./livekit-plugins/livekit-plugins-breezeblue
```

Set `BREEZE_API_KEY` in the environment and pass a saved Breeze voice ID to your agent's TTS.

```python
from livekit.agents import AgentSession
from livekit.plugins import breezeblue

session = AgentSession(tts=breezeblue.TTS(voice="your-breeze-voice-id"))
```

## Streaming options

The default HTTP transport buffers one complete Agent reply and streams the synthesized PCM audio. Use `input_mode="sentence"` to begin synthesis at sentence boundaries. Text exceeding the provider's request limit is split as needed.

For streamed Agent replies, pass `input_chunk_length_schedule` to synthesize as soon as each chunk reaches its configured character count. It takes precedence over `input_mode` and works with HTTP and Realtime WebSocket.

```python
tts = breezeblue.TTS(voice="your-breeze-voice-id", input_chunk_length_schedule=[120, 160, 250])
```

This creates chunks of 120, 160, and then 250 characters each. Counts are per chunk and may cut across words or sentences. Pass a non-empty sequence of positive integers, no greater than `max_characters`. A LiveKit flush or end of input submits the remaining text. Each new stream starts at the first count. Leaving the schedule unset keeps the default whole-reply behavior.

Set `transport="websocket"` for Breeze Realtime sessions with server-side turn cancellation and connection reuse between turns. Keep a separate TTS instance per AgentSession when using this transport.

Set `word_timestamps=True` with the HTTP transport to receive aligned transcript metadata. The `instructions` and `voice_settings` options pass through the provider's synthesis controls. `update_options()` can change the voice or instructions for subsequent synthesis.
