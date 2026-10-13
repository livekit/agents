# State and deps

Two objects travel with a run. One is saved with the thread, one is not.

| | `state` | `deps` |
|---|---|---|
| what | the thread's typed memory: what the agent has learned | services for one run: HTTP clients, connections, budgets |
| declared | `state_type=` on the agent, `state=` where the thread is opened | `deps_type=` on the agent |
| built | by the framework, loaded from the lane before every run | by whoever starts the run |
| saved | after every run that changed it, on the lane | never |
| read as | `ctx.state` | `ctx.deps` |
| shape | JSON: dataclasses, pydantic models, scalars, lists, dicts; at most 64 KiB | anything |

## State

```python
@dataclass
class Patient:
    appointment: Appointment | None = None

scheduler = Agent(..., state_type=Patient)

@server.a2a_session(endpoint="scheduling", description="Book or move an appointment.", state=Patient)
async def text(ctx: A2ASessionContext[Patient]) -> str:          # the type, loaded per run
    result = await scheduler.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
    return result.output

session = LiveSession(..., state=Patient())                      # an instance, for the call
```

State lives on the thread's lane (see [lanes.md](lanes.md)), so every
agent on that lane shares it. A fresh lane starts from `Patient()`: the
state type must construct with no arguments. Tools write it in place,
`ctx.state.appointment = ...`, and the next run on any worker sees it.
Render it into the prompt with `@agent.instructions`.

The state is dumped through pydantic. Saving fails with
`NotSerializable(field)` naming the offending field, almost always something
that belongs in deps, and with `TooLarge` over 64 KiB of JSON.

## Deps

```python
@dataclass
class Deps:
    clinic: httpx.AsyncClient

deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
await scheduler.run(message, ..., deps=deps)              # a turn
LiveSession(..., deps=deps)                               # a call
```

Whoever starts the run builds them and passes them in. Tests pass a `Deps` with fakes.

Both are typed: `deps_type=` and `state_type=` make the agent's tools take
`RunContext[Deps, State]`, so `ctx.deps.clinic` and `ctx.state.appointment`
are checked by the type checker, and the state is validated against its
type when it is loaded. `RunContext[None, Patient]` says an agent has no
deps.

See `examples/use-cases/01_appointments_by_phone_and_text.py` for both.
