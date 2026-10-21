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

`persona-2.0-live` avatars hold one of eleven expressions (`keyframe.EMOTIONS`): neutral,
happy, sad, angry, surprised, concerned, confused, disgusted, excited, scared, skeptical.
An expression holds until the next change. Three ways to change it, in order of
preference:

### 1. Expressive mode: one marker steers the voice and the face

If your TTS supports LiveKit's [expressive mode](https://docs.livekit.io/agents/models/tts/expressive/)
(a LiveKit Inference TTS from a provider on that page), the LLM already writes
`<expr type="expression" label="happy"/>` into its replies. `emotion_tts_node` reads the
same markers and changes the face on the word they precede:

```python
class MyAgent(Agent):
    def tts_node(self, text, model_settings):
        return keyframe.emotion_tts_node(self, text, model_settings)

session = AgentSession(
    tts=inference.TTS("cartesia/sonic-3.6", extra_kwargs={"add_timestamps": True}),
    expressive={"tts_instructions_append": keyframe.emotion_instructions()},
    ...
)
```

`emotion_instructions()` tells the LLM to use the avatar's eleven expression names as
labels. Whether the voice follows the same marker depends on the TTS's own vocabulary:

- **Inworld** takes free-form labels, so voice and face agree:
  `<expr type="expression" label="concerned"/>` sounds concerned and looks concerned. A
  label sets the face when it contains an expression name as a whole word, so "warm,
  concerned" works too; "sympathetic" steers the voice and leaves the face alone.
- **Cartesia** has a fixed vocabulary covering nine of the eleven names. `concerned` and
  `skeptical` change the face but not the voice.
- **SpaceXAI** has no expression labels at all. The marker is removed before the TTS, so
  the face changes and the voice does not.

Through Inference only Cartesia and Inworld report word timings. With Fish Audio, Gemini
or SpaceXAI the voice is expressive, but the face follows only a marker before the
reply's first word (see the table below).

Use `emotion_tts_node` as the `tts_node` of every agent the session runs, handoff targets
included. It is a drop-in for `Agent.default.tts_node`, so a `tts_node` that processes
the text before synthesis or the audio after calls it in the same place.

### 2. Any other TTS: markers steer the face only

Same `tts_node`; put the instructions in the agent's prompt instead. The markers are
stripped before synthesis:

```python
agent = MyAgent(instructions=f"{MY_INSTRUCTIONS}\n\n{keyframe.emotion_instructions()}")
```

How precisely a change lands depends on what the TTS reports about its audio:

| TTS | A change lands |
|-----|----------------|
| Streaming TTS with word timings on: Cartesia (`add_timestamps` through Inference, the plugin's default), Inworld (`timestamp_type="WORD"`), Rime (websocket), Speechify, Smallest.ai and Baseten (`word_timestamps=True`) | on the word |
| Non-streaming TTS (Gemini, OpenAI, Hume, ...), which the framework synthesizes sentence by sentence | at the start of the marker's sentence |
| Streaming TTS without word timings (Deepgram, xAI, ...) | only for a marker before the reply's first word; later markers are ignored |

### 3. `set_emotion`: a change that is not tied to the words

For a tool call, a vision event, anything outside the reply text:

```python
await avatar.set_emotion("happy")
```

The avatar switches from the next video frame it renders. While it is idle, or when the
call comes before a reply's audio (as a tool call the LLM makes before answering does),
that is as soon as the change can show. In the middle of a reply the avatar has usually
rendered ahead of playback, so the change shows when playback reaches that point, which
with a fast TTS and renderer can be the end of the reply. For a change on a particular
word, use a marker.

### Notes

- `persona-1.5-live` avatars take `set_emotion` (neutral, happy, sad, angry) but not
  markers.
- The `emotions=` argument of `emotion_tts_node` and `emotion_instructions` narrows the
  set a marker may pick from.
- A fully custom `tts_node` can time cues itself: attach a list of `keyframe.EmotionCue`
  under `keyframe.USERDATA_EMOTION_CUES` to the frames it yields. `offset` is seconds from
  the frame's first sample, and may be negative or run past the frame's end.

  ```python
  frame.userdata[keyframe.USERDATA_EMOTION_CUES] = [keyframe.EmotionCue("happy", offset=0.25)]
  ```
