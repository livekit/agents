# Voice

A call runs on a `LiveSession`: the runtime a room attaches to, with the
models, turn detection and audio. It runs an `Agent`, and it runs it in the
same thread a text endpoint uses, so a call and a text in one thread
share one history and one state.

## The voice door

```python
@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    app = SchedulingContext(clinic=clinic_service)
    session = LiveSession(
        llm=openai.realtime.RealtimeModel(), context=app, thread=ctx.thread
    )
    await session.start(agent=scheduler, room=ctx.room)
    await session.store.load_context(app)
    await session.generate_reply(instructions="Greet the caller and ask about their appointment.")
```

The door is `{agent_name}/{endpoint}`: the agent's name from
`AgentServer(agent_name=)`, and `rtc_session(endpoint=)`, the agent's name
again when left empty.

| Parameter | What |
|---|---|
| `llm`, `stt`, `tts`, `vad` | the call's models; a realtime model replaces STT, LLM and TTS |
| `context` | the application object; marked fields require explicit loading and saving |
| `thread` | the call's durable thread, `ctx.thread`; omitted, the history is kept in memory and not saved |

`session.start(agent, chat_ctx=)` and `session.update_agent(agent, chat_ctx=)`
are the two places to supply a different history. An explicit agent switch
retains the current history when no replacement is supplied.

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
For phone-number routing and conversation continuity, see
[channel routing](planned-channels.md).

At `start()` the call loads the agent's lane: the stored history when the
call was handed none. Application context fields are explicitly loaded with
`session.store.load_context(app)` before generating a reply. Each
voice turn then takes the lane from the end of the person's turn to the
commit of the reply, and releases it without draining, so a text that
arrived meanwhile is answered by the worker waiting on it. When a text
turn moved the lane during the call, transport history is reloaded before the model reads it. Application code
reloads context fields at its turn boundary before using changes from other
activations. A
turn that produced no reply still releases the lane within 30 seconds.
Storage failures are logged and never end the call.

## The thin voice door

The realtime model is good at the floor: small talk, interruptions, pacing.
It is not where the tools and the state should live. An `A2AAgent(...).as_tool()` in
the voice agent's `tools` gives it one tool, the brain behind the text
endpoint, named and described from that endpoint's card, and speaks the
brain's progress while it waits.

```python
back_office = A2AAgent("http://127.0.0.1:7880/agents/dentist/default/backend")
voice = Agent(
    instructions="You are the practice's voice on the phone. Hand every request to the back office.",
    tools=[back_office.as_tool()],            # one tool, ask
)

@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = LiveSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)
```

The remote service owns its conversation and approval behavior. The adapter
forwards progress while it works, delivers questions to the person, and
correlates their replies with the remote task. Completion supplies the
parent's tool result; a failed remote task becomes a tool error.
Connection lifecycle and delegation metadata are managed internally. See
[remote agents as tools](agents.md#remote-agents-as-tools) for routing and
persistence ownership.

One brain, two doors: voice and text share one set of tools, explicitly persisted application data
and one history. The front desk answers a call today and a text tomorrow from
the same memory.

## Structured results in voice

A voice turn delivers conversational speech. An [AgentRouter](agent-router.md)
selects the active frame for multi-agent conversations. A child's accepted output
fills the parent tool call and starts a fresh parent run, whose response is
delivered as speech. Voice and
text use the same [task completion contract](agent-router.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py`,
`03_insurance_eligibility.py` and `07_hospital_main_line.py`.
