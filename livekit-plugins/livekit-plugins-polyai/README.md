# PolyAI plugin for LiveKit Agents

Use [PolyAI](https://poly.ai)'s Dialog-RSN-1 as the realtime model of a LiveKit Agents session.
Dialog-RSN-1 is a speech-to-text model for voice agents: it hears the caller, confirms when their
turn is over, calls your tools and writes the reply. PolyAI's realtime API speaks the reply, or
streams it as text for a TTS plugin.

See the [Dialog-RSN-1 docs](https://docs.poly.ai/dialog-rsn-1/overview) for the model and the
[reference](https://docs.poly.ai/dialog-rsn-1/reference) for the realtime API.

## Installation

```bash
pip install livekit-plugins-polyai
```

## Pre-requisites

A Dialog-RSN-1 API key from [studio.poly.ai](https://studio.poly.ai) (API Keys tab), set as
`POLY_API_KEY`. A key from a regional Studio only works on that region's host: set
`DIALOG_REALTIME_URL`, for example `wss://api.us.poly.ai/v1/realtime` for `studio.us.poly.ai`.

## Usage

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import polyai

session = AgentSession(
    llm=polyai.RealtimeModel(
        knowledge=[polyai.KnowledgeTopic(name="Opening hours", content="Saturday 10am to 4pm.")],
        first_message="Thanks for calling Larkspur Bikes, this is Ivy. How can I help?",
    ),
)
await session.start(agent=Agent(instructions="You are Ivy, at Larkspur Bikes."), room=ctx.room)
```

The API speaks the reply, so the session needs no TTS plugin. Talking over the agent interrupts
it. `first_message` is said word for word as soon as the session is configured, so don't also
call `generate_reply()` to greet.

To speak the reply with your own TTS plugin, pass `modalities=["text"]` and a `tts=` to the
`AgentSession`.

### PolyAI options

| Argument | What it does |
| --- | --- |
| `voice` | A Cartesia voice id for replies the API speaks. Unset, the deployment's default voice speaks. |
| `knowledge` | `KnowledgeTopic`s the model answers from, sent as `poly_knowledge`. |
| `response_language` | The language to answer in, as an ISO 639 tag (`"es"`, `"pt-BR"`). |
| `first_message` | What the assistant says first, word for word. |
| `spoken_arguments` | Per tool name, one string argument the model says while the tool runs, such as a hold line. |
| `tool_speech` | Per tool name, `"none"` to keep the tool silent or `"required"` for a line on every call. |
| `on_response_done` | Called with each `response.done`, which can carry `poly_cited_topics` (the knowledge topics the reply used) and `poly_out_of_domain`. |

### Limits

Dialog-RSN-1 takes tools and `tool_choice` on the session only: per-reply `tools` and
`tool_choice` in `generate_reply()` are dropped with a warning, and `generate_reply(instructions=...)`
works. `tool_choice` takes `"auto"` only: `"none"`, which LiveKit asks for at the tool-call
limit, hides the session's tools until the choice is back to `"auto"`, and other values are
ignored. A tool takes `tool_speech` or a spoken argument, not both. Turn timing isn't tunable: only
`create_response` in `turn_detection` is read, and `interrupt_response=False` is rejected when the
model is created.
