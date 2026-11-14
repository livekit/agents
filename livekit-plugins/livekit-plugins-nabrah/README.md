# Nabrah STT for LiveKit Agents

Use `nabrah.STT(recognition_model=...)` for all three backend models. It selects
one of two implementations internally:

```python
nabrah.STT(recognition_model="nabrah_stt_v1", api_key="YOUR_KEY")
nabrah.STT(recognition_model="eot_nabrah", api_key="YOUR_KEY")
nabrah.STT(recognition_model="nabrah_stt_conv", api_key="YOUR_KEY")
```

The original default remains `eot_nabrah`. Each model takes
its implementation's own options. `STTConv` fixes transcript metadata to `ar-SA`;
the original `STT` retains its production language handling, defaulting to
`ar-SA`. Neither implementation sends a language option to the backend.

## Conversational (`nabrah_stt_conv`)

The server decides when a turn ends and when to backchannel. Use
`turn_detection="stt"` and no VAD. The wire/metrics model name is `nabrah_stt_conv_v1`.

```python
import logging
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, TurnHandlingOptions
from livekit.plugins import cartesia, google, nabrah

load_dotenv()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    recognizer = nabrah.STT(
        recognition_model="nabrah_stt_conv",
        api_key=os.getenv("NABRAH_API_KEY"),
    )
    await recognizer.check_ready()

    session = AgentSession(
        stt=recognizer,
        vad=None,  # the server owns speech and end of turn
        llm=google.LLM(model="YOUR_LLM_MODEL"),
        tts=cartesia.TTS(model="YOUR_TTS_MODEL", voice="YOUR_VOICE_ID", language="ar"),
        turn_handling=TurnHandlingOptions(turn_detection="stt"),
    )
    backchannels = nabrah.BackchannelController(recognizer, session, text="ممم")

    async def cleanup() -> None:
        backchannels.close()

    ctx.add_shutdown_callback(cleanup)

    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant."),
        room=ctx.room,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
```

Options of `STTConv`. `None` means the server default.

| Option | Default | Meaning |
| --- | --- | --- |
| `api_key` | `NABRAH_API_KEY` | API key (`NABRAH_STT_V2_API_KEY` also works). |
| `eot_threshold` | `None` (0.915) | End-of-turn probability. |
| `eot_debounce` | `None` (2) | Consecutive 80 ms frames above the threshold. |
| `min_speech_ms` | `None` (350) | Speech needed before a turn end or backchannel. |
| `bc_threshold` | `None` (0.4) | Backchannel probability. |
| `bc_debounce` | `None` (2) | Consecutive frames above the backchannel threshold. |
| `bc_refractory_ms` | `None` (1200) | Minimum time between backchannels. |
| `no_text_reset_s` | `None` | Seconds without new words before the turn state resets (0 disables). |
| `preflight_threshold` | `0.5` | Probability that sends an early transcript for preemptive generation; `None` disables. |

`STTConv` sends 16 kHz mono PCM16 in 20 ms packets. The model processes 80 ms
frames. Transcript timing uses `audio_processed`; preflight uses the latest
`turn.frames[].p_end`. LiveKit must enable preemptive generation to use
preflight for early replies. Finals close the turn; custom `end_of_turn` and
`backchannel` events are also emitted.

## Legacy (`eot_nabrah`, `nabrah_stt_v1`)

The server returns utterances; the plugin joins them into a turn, strips `<eot>`,
and ends the turn on turn punctuation, after a confirm delay, or after an idle
timeout. The plugin owns the end of turn, so use `turn_detection="stt"` and no VAD.

```python
import logging
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, TurnHandlingOptions
from livekit.plugins import cartesia, google, nabrah

load_dotenv()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    recognizer = nabrah.STT(
        recognition_model="eot_nabrah",
        api_key=os.getenv("NABRAH_API_KEY"),
        priority_words=["نبرة"],
        priority_words_strength=0.5,
        end_of_turn_confirm_delay_seconds=0.4,
        end_of_turn_idle_timeout_seconds=1.5,
    )

    session = AgentSession(
        stt=recognizer,
        vad=None,  # the plugin ends the turn
        llm=google.LLM(model="YOUR_LLM_MODEL"),
        tts=cartesia.TTS(model="YOUR_TTS_MODEL", voice="YOUR_VOICE_ID", language="ar"),
        turn_handling=TurnHandlingOptions(turn_detection="stt"),
    )

    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant."),
        room=ctx.room,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
```

Options of the legacy `STT`:

| Option | Default | Meaning |
| --- | --- | --- |
| `recognition_model` | `"eot_nabrah"` | `"eot_nabrah"` or `"nabrah_stt_v1"`. |
| `api_key` | `NABRAH_API_KEY` | API key. |
| `language` | `"ar-SA"` | Language tag on returned transcripts; not sent to the backend. |
| `priority_words` | `None` (none) | Words the recognizer should favour. |
| `priority_words_strength` | `0.5` | How strongly to favour `priority_words`. |
| `end_of_turn_confirm_delay_seconds` | `0.4` | Wait this long after turn punctuation before ending the turn; `None` ends it at once. |
| `end_of_turn_idle_timeout_seconds` | `3.0` | End the turn when text exists but no new text arrived for this many seconds; `None` disables. |

Both classes read `NABRAH_API_KEY` when `api_key` is omitted.


