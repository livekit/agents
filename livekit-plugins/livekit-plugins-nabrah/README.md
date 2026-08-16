# Nabrah plugin for LiveKit Agents

Support for [Nabrah](https://nabrah.ai/) Speech-to-Text in LiveKit Agents, with
client-side end-of-turn detection tuned for Arabic conversation.

## Installation

```bash
pip install livekit-plugins-nabrah
```

## Pre-requisites

You'll need an API key from Nabrah. It can be set as an environment variable:
`NABRAH_API_KEY`. 

## Usage

Use Nabrah STT in an `AgentSession`:

```python
from livekit.agents import AgentSession
from livekit.plugins import nabrah

session = AgentSession(
    stt=nabrah.STT(
        recognition_model="eot_nabrah",
        language="ar-SA",
        end_of_turn_confirm_delay_seconds=0.4,
    ),
)
```

### Turn detection

`recognition_model="eot_nabrah"` emits the end-of-turn signal the plugin uses to
close a turn. `end_of_turn_confirm_delay_seconds` is how long it waits after that
signal before committing. Speaking again inside the window keeps the turn open.
`max_silence_before_finalize_seconds` (default `1.5`) is the fallback when no
signal arrives.

The default model (`recognition_model=""`) is more accurate but emits no
end-of-turn signal, leaving silence as the only turn detector.

### Word boosting

Bias recognition toward terms the model gets wrong:

```python
stt = nabrah.STT(
    priority_words=["مستشفى الملك فيصل التخصصي", "رقم الهوية الوطنية"],
    priority_words_strength=0.5,
)
```

`0.5` is the recommended strength. Higher values make boosted terms appear in
places they were not said, so keep the list to terms your callers actually say
and add one only after hearing it come out wrong. Multi-word phrases are boosted
as a unit.

#### Loading terms from a file

For anything beyond a handful of terms, keep them in a JSON file so they can be
edited without touching code. Create `boosting.json` next to your agent:

```json
{
  "boost_threshold": 0.5,
  "words": [
    ...
  ]
}
```

Load it at startup and pass it to the plugin:

```python
import json
import pathlib

from livekit.agents import AgentSession
from livekit.plugins import nabrah

boosting = json.loads(
    pathlib.Path(__file__).with_name("boosting.json").read_text(encoding="utf-8")
)

session = AgentSession(
    stt=nabrah.STT(
        recognition_model="eot_nabrah",
        language="ar-SA",
        end_of_turn_confirm_delay_seconds=0.4,
        priority_words=boosting["words"],
        priority_words_strength=boosting["boost_threshold"],
    ),
)
```

Read the file with `encoding="utf-8"`. Without it, Arabic terms fail to load on
platforms that default to a different encoding.

## Full example

A complete agent using Nabrah STT with word boosting loaded from a file:

```python
import json
import pathlib

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext
from livekit.plugins import nabrah

load_dotenv()

boosting = json.loads(
    pathlib.Path(__file__).with_name("boosting.json").read_text(encoding="utf-8")
)

server = AgentServer()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice assistant.")


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=nabrah.STT(
            recognition_model="eot_nabrah",
            language="ar-SA",
            end_of_turn_confirm_delay_seconds=0.4,
            max_silence_before_finalize_seconds=1.5,
            priority_words=boosting["words"],
            priority_words_strength=boosting["boost_threshold"],
        ),
        
        # llm
        # tts

        turn_handling=TurnHandlingOptions(
            turn_detection="stt",  
            endpointing={
                "min_delay": 0
            },
        )
    )

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    agents.cli.run_app(server)
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `recognition_model` | `"eot_nabrah"` | `"eot_nabrah"` emits the end-of-turn signal used for turn detection. `""` selects the default model, which is more accurate but emits no signal. |
| `end_of_turn_confirm_delay_seconds` | `0.4` | Hold after an end-of-turn signal before committing the turn. `None` commits immediately. |
| `max_silence_before_finalize_seconds` | `1.5` | Fallback when no end-of-turn signal arrives. `None` disables it. |
| `priority_words` | `[]` | Terms to bias recognition toward. |
| `priority_words_strength` | `0.5` | How strongly to bias. |
| `api_key` | `NABRAH_API_KEY` | API key, if not set in the environment. |
