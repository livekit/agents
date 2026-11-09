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

## Public API to revisit

- Whether `Working` ([tools](tools.md#what-a-tool-returns)) and `ctx.execution_id`
  ([agents](agents.md#runcontext)) stay public.
- `RunResult.verbatim` ([agents](agents.md#running-a-turn)) and the `lk.approval_identity`
  history key ([approval](tools.md#approval)).
