# livekit-plugins-yoob

[Yoob](https://yoob.com) speech, transcription and avatars for [LiveKit Agents](https://docs.livekit.io/agents/).

LiveKit docs: [speech-to-text](https://docs.livekit.io/agents/models/stt/yoob/) and [text-to-speech](https://docs.livekit.io/agents/models/tts/yoob/) (live with the next release).

- `yoob.TTS`: Overtone expressive speech. Understands inline tags such as `[laughing]`, `[sigh]` and `[whisper] ... [/whisper]`, plus free-text `instructions`. Supports `synthesize()` (one request, audio streamed back) and `stream()` (text pushed as the LLM writes it, over one socket). 48 kHz mono.
- `yoob.AvatarSession`: A Yoob character as the agent's face, rendered on the user's own device (web, iOS or Android). No avatar server and no video track: the user's app plays the agent's voice through the character.
- `yoob.STT`: Transcription with Turbo or Pro. Streaming with interim results and start/end-of-speech events, plus batch `recognize()` of up to 60 s.

## Install

```bash
pip install livekit-plugins-yoob
```

Set `YOOB_API_KEY` (get one at [app.yoob.com](https://app.yoob.com)).

## Use

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import yoob

session = AgentSession(
    stt=yoob.STT(),
    llm=...,  # any LiveKit LLM
    tts=yoob.TTS(voice="w1243"),
    turn_handling=TurnHandlingOptions(turn_detection="stt"),
)
```

### `yoob.TTS(...)`

| Argument | Default | |
|---|---|---|
| `voice` | `w1243` | A built-in voice id or one of your cloned voices (`GET /v1/voices`). |
| `quality` | `best` | `fast` gives a quicker first sound. |
| `instructions` | none | Delivery, e.g. `"warm voice, softly spoken"`. |
| `speed` | none | 0.5 to 2.0. |
| `seed` | none | Makes renders repeatable. |
| `api_key`, `base_url`, `http_session` | env | `YOOB_API_KEY`, `YOOB_BASE_URL`. |

`update_options(voice=..., quality=..., instructions=..., speed=...)` changes them mid-session.

### `yoob.STT(...)`

| Argument | Default | |
|---|---|---|
| `model` | `yoob-asr-turbo-v1` | `yoob-asr-turbo-v1` (fastest, 25 languages) or `yoob-asr-pro-v1` (most accurate, 47 languages), for streaming and `recognize()`. |
| `language` | none | A code (`de`), regional tag (`de-DE`) or English name. Streams default to English; `recognize()` without it detects the language. `stream(language=...)` overrides it. |
| `interim_results` | `True` | Emit partial transcripts while the user speaks. |
| `api_key`, `base_url`, `http_session` | env | |

### `yoob.AvatarSession(...)`

The character renders in the user's app with a Yoob SDK (web, iOS or Android), from the agent's normal audio track, so the room
carries only audio and nothing renders on a server. The plugin is the agent's half: it announces the character in the
agent's attributes and opens the Yoob session the user's device needs, over an RPC, so the Yoob key stays with the
agent.

```python
from livekit.agents import AgentSession, JobContext
from livekit.plugins import yoob

async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    session = AgentSession(stt=yoob.STT(), llm=..., tts=yoob.TTS(voice="w0744"))
    avatar = yoob.AvatarSession(character="astrid")
    await avatar.start(session, room=ctx.room)
    await session.start(agent=..., room=ctx.room)
```

In a web app (`npm install @yoob/avatar livekit-client`):

```ts
import { YoobAvatar } from "@yoob/avatar";
import { YoobLiveKitSession, agentAvatar } from "@yoob/avatar/livekit";

await room.connect(url, token);
const { character, version, getCredentials } = await agentAvatar(room); // from the agent's attributes and RPC
const avatar = new YoobAvatar({ container, character, version, getCredentials });
const session = new YoobLiveKitSession(avatar, { room });
button.onclick = () => session.start(); // a click unlocks sound
```

iOS (`YoobLiveKit`): `let agent = try await YoobAgentAvatar.wait(in: room)`, then `YoobAvatar(agent.source)` and
`YoobLiveKitSession(avatar:room:)`. Android (`yoob-avatar-livekit`): `val agent = agentAvatar(room)`, then
`YoobAvatarOptions(agent.character, agent.getCredentials, agent.version)` and `YoobLiveKitSession(avatar, room)`.

| Argument | Default | |
|---|---|---|
| `character` | `astrid` | A Yoob character id, for example `astrid` or `luna-realistic`. |
| `version` | newest | Pin a character version. |
| `max_sessions_per_minute` | `6` | Sessions one participant may open per minute. |
| `api_key`, `api_base` | env | `YOOB_API_KEY`, `YOOB_API_BASE` (default `https://app.yoob.com`). |

Only the people in the call can open sessions (not agents). Avatar minutes are billed to the workspace of the Yoob API
key, from the device's heartbeats; the device ends its session when it leaves the call.

## Tests

Yoob runs in this repository's shared suites (`tests/test_tts.py`, `tests/test_stt.py`) against the live API:

```bash
YOOB_API_KEY=... uv run pytest tests/test_tts.py tests/test_stt.py -k yoob
```
