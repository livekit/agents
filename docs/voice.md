# Voice

A call runs on a `LiveSession`: the runtime a room attaches to, with the
models, turn detection and audio. It runs an `Agent`, and it runs it in the
same thread a text endpoint uses, so a call and a text in one thread
share one history and one state.

## The voice door

```python
@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
    session = LiveSession(
        llm=openai.realtime.RealtimeModel(), state=Patient(), deps=deps, thread=ctx.thread
    )
    await session.start(agent=scheduler, room=ctx.room)
    await session.generate_reply(instructions="Greet the caller and ask about their appointment.")
```

The door is `{agent_name}/{endpoint}`: the agent's name from
`AgentServer(agent_name=)`, and `rtc_session(endpoint=)`, the agent's name
again when left empty.

| Parameter | What |
|---|---|
| `llm`, `stt`, `tts`, `vad` | the call's models; a realtime model replaces STT, LLM and TTS |
| `state` | the state instance for this thread, loaded from the thread's lane at `start()` |
| `deps` | per-run services for the call; never saved |
| `delegate` | a slow brain the model hands requests to; see below |
| `thread` | the call's durable thread, `ctx.thread`; omitted, the history is kept in memory and not saved |

`session.start(agent, chat_ctx=)` and `session.update_agent(agent, chat_ctx=)`
are the two places to hand in a different history; a handoff continues the
thread's history without them.

In a tool on a call, `ctx.live_session` is the call (`say()`,
`generate_reply()`), and `ctx.thread` is the durable thread, the same
object a text handler gets. Outside a call `ctx.live_session` is `None`.

## The call's thread

On a server with storage, the door resolves the call's thread before the
entrypoint runs: the `thread_id` key of the dispatch's JSON metadata, as
the web console sends it, else a new thread. It is `ctx.thread` on the
`JobContext`, `None` on a server without storage, and
`LiveSession(thread=ctx.thread)` is how the call gets it. A `LiveSession`
without `thread=` keeps its history in memory, not saved.
A call is not found by the caller's number; phone continuity is planned,
see [planned-channels.md](planned-channels.md).

At `start()` the call loads the agent's lane: the stored history when the
call was handed none, and the state into the object it was given. Each
voice turn then takes the lane from the end of the person's turn to the
commit of the reply, and releases it without draining, so a text that
arrived meanwhile is answered by the worker waiting on it. When a text
turn moved the lane during the call, the history and the state are
reloaded before the model reads them, so the call knows about the text. A
turn that produced no reply still releases the lane within 30 seconds.
Storage failures are logged and never end the call.

## The thin voice door

The realtime model is good at the floor: small talk, interruptions, pacing.
It is not where the tools and the state should live. An `A2ADelegate` in
the voice agent's `tools` gives it one tool, the brain behind the text
endpoint, named and described from that endpoint's card, and speaks the
brain's progress while it waits.

```python
back_office = A2ADelegate("http://127.0.0.1:7880/agents/dentist/default/backend")
voice = Agent(
    instructions="You are the practice's voice on the phone. Hand every request to the back office.",
    tools=[back_office],            # one tool, ask
)

@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = LiveSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)
```

The tool hands the request on with the call's conversation (messages only,
no tool calls) and the call's thread. Its first update releases the
turn with a short acknowledgment the model speaks, so the caller is not
left in silence; each progress message the expert sends after is spoken
while it works. The answer comes back as the tool's result and the model
says it in its own words. A question the expert asks (`input-required`) is
an answer like any other: the model relays it, and the caller's reply goes
to the expert in the next delegation. A failed delegation is a tool error.
The tool records the expert's task on the call
(`record_delegation`, [multi-agent.md](multi-agent.md)).

One brain, two doors: voice and text share one set of tools, one state type
and one history. The front desk answers a call today and a text tomorrow from
the same memory.

## Structured results in voice

A voice turn's reply is speech, so a typed result comes from an awaited
`AgentTask[T]`: the task takes the conversation, collects what it needs, and
returns the value to the tool that awaited it. See
[agent-tasks.md](agent-tasks.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py`,
`03_insurance_eligibility.py` and `07_hospital_main_line.py`.
