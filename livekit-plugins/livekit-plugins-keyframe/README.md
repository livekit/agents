# livekit-plugins-keyframe

Agent Framework plugin for [Keyframe Labs](https://keyframelabs.com) avatars.

## Installation

```bash
pip install livekit-plugins-keyframe
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import keyframe

session = AgentSession(stt=..., llm=..., tts=...)

avatar = keyframe.AvatarSession(
    persona_slug="public:cosmo-2.0-live-preview",  # or persona_id="..."
    api_key="keyframe_sk_live_...",  # or set KEYFRAME_API_KEY env var
)

await avatar.start(session, room=ctx.room)
await session.start(room=ctx.room, agent=my_agent)
```

## Authentication

Set the following environment variables:

- `KEYFRAME_API_KEY` - Your Keyframe API key
- `LIVEKIT_URL` - LiveKit server URL
- `LIVEKIT_API_KEY` - LiveKit API key
- `LIVEKIT_API_SECRET` - LiveKit API secret

## Emotions

`persona-2.0-live` avatars support eleven expressions (`keyframe.EMOTIONS`):
`neutral`, `happy`, `sad`, `angry`, `surprised`, `concerned`, `confused`, `disgusted`, `excited`, `scared`, `skeptical`.
An expression holds until the next change.

### Sync expressions to speech

Override `tts_node` with `keyframe.emotion_tts_node` on every agent in the session, and add `keyframe.emotion_instructions()` to the prompt.
The LLM writes emotion markers into its reply, and the avatar changes expression on the marked word.

```python
class MyAgent(Agent):
    def tts_node(self, text, model_settings):
        return keyframe.emotion_tts_node(self, text, model_settings)
```

Use the configuration that matches your TTS:

- LiveKit Inference ([expressive mode](https://docs.livekit.io/agents/models/tts/expressive/)), where one `<expr>` marker steers both the voice and the face:

  ```python
  session = AgentSession(
      tts=inference.TTS("cartesia/sonic-3.6", extra_kwargs={"add_timestamps": True}),
      expressive={"tts_instructions_append": keyframe.emotion_instructions()},
      ...
  )
  ```

- ElevenLabs v3/v4 plugin, where one `[happy]` audio tag steers both the voice and the face. Pass `style="brackets"` to both functions, and set `use_tts_aligned_transcript=True` to keep tags out of transcripts:

  ```python
  class MyAgent(Agent):
      def tts_node(self, text, model_settings):
          return keyframe.emotion_tts_node(self, text, model_settings, style="brackets")

  agent = MyAgent(
      instructions=f"{MY_INSTRUCTIONS}\n\n{keyframe.emotion_instructions(style='brackets')}",
  )
  session = AgentSession(
      tts=elevenlabs.TTS(model="eleven_v3"),
      use_tts_aligned_transcript=True,
      ...
  )
  ```

- Any other TTS, where `<expr>` markers steer the face and are stripped before synthesis:

  ```python
  agent = MyAgent(instructions=f"{MY_INSTRUCTIONS}\n\n{keyframe.emotion_instructions()}")
  ```

### Set an expression directly

Use `set_emotion` for tool calls, vision events, or `persona-1.5-live` avatars (which support `neutral`, `happy`, `sad`, and `angry`):

```python
await avatar.set_emotion("happy")
```

The avatar switches on the next video frame that it renders.
During a reply, the avatar renders ahead of playback.
When a change must land on a specific word, use markers instead.

<details>
<summary>TTS support and timing details</summary>

Timing accuracy depends on what the TTS reports:

| TTS | When the expression changes |
|-----|-----------------------------|
| Streaming TTS with word timings: Cartesia (`add_timestamps=True`, plugin default), Inworld (`timestamp_type="WORD"`), ElevenLabs v3/v4 (plugin), Rime (WebSocket), Speechify, Smallest.ai, Baseten (`word_timestamps=True`) | On the marked word |
| Non-streaming TTS (Gemini, OpenAI, Hume) | At the start of the sentence |
| Streaming TTS without word timings (Deepgram, xAI, or ElevenLabs, Fish Audio, and Gemini through Inference) | Before the first word only (later markers are ignored) |

Provider notes:

- Inworld expressive mode accepts free-form labels: `"warm, concerned"` steers both the voice and the face, while `"sympathetic"` steers the voice only.
- Cartesia expressive mode covers nine of the eleven expressions (`concerned` and `skeptical` change the face only).
- SpaceXAI expressive mode has no expression labels, so markers change the face and are stripped before synthesis.
- In `style="brackets"`, only exact expression names (`[happy]`) change the face. Other audio tags (`[whispers]`, `[warmly]`) steer the voice only. In the default `"expr"` style, square brackets are plain text.

</details>

<details>
<summary>Custom emotion sets and custom TTS nodes</summary>

Pass `emotions=` to `emotion_tts_node` and `emotion_instructions` to limit which expressions a marker can select.

A custom `tts_node` can attach `keyframe.EmotionCue` items directly to audio frames.
`offset` is the time in seconds from the first sample of the frame and can be negative.

```python
frame.userdata[keyframe.USERDATA_EMOTION_CUES] = [
    keyframe.EmotionCue("happy", offset=0.25),
]
```

</details>
