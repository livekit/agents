# TODO

Work the design pages describe that the code does not do yet.

## Expert events on a call

[voice-session.md](voice-session.md#expert-events-and-frontend-presentation) describes the
end state; these parts are not built:

- **Settle or cancel from the application.** When no agent exposing the expert's remote tool
  returns, the application needs an API to settle or cancel the pending interaction
  ([agent replacement and persistence](voice-session.md#agent-replacement-and-persistence)).
- **Queue persistence.** The frontend event queue is in-process only; optionally persisting it,
  separately from backend request batching, is not built.
- **Reply classification.** A reply to a delivered expert question is forwarded as is;
  recognizing a cancellation, a new request or an ambiguous answer instead of forwarding it
  ([routing the person's reply](voice-session.md#routing-the-persons-reply)) is not built.

## A call's thread

[threads.md](threads.md) describes the end state; these parts wait on other repositories:

- **Web console.** Its `@livekit/protocol` lacks `RoomAgentDispatch.threadId`, which is dropped
  as an unknown field, so web-console calls run in memory until the package is bumped.
- **Terminal console.** `lk agent console` sets no `thread_id` on its simulated job; the Go CLI
  needs to create the thread and pass it.
- **Protocol pin.** rust-sdks carries the `thread_id` fields as a local regeneration; re-pin to
  the livekit/protocol#1958 merge, and use `agent.Job(thread_id=)` in tests once
  livekit/python-sdks#929 lands.

## Public API to revisit

- Whether `Working` ([tools](tools.md#what-a-tool-returns)) and `ctx.execution_id`
  ([agents](agents.md#runcontext)) stay public.
- `RunResult.verbatim` ([agents](agents.md#running-a-turn)) and the `lk.approval_identity`
  history key ([approval](tools.md#approval)).
