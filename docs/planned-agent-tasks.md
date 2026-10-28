# Planned: agent tasks in the background

Status: designed, not built. Extends [agent-tasks.md](agent-tasks.md) and
[human-in-the-loop.md](human-in-the-loop.md).

A task that outlives the turn needs no primitive of its own. It is an A2A
task, which is what every `AgentTask` already is: it waits in `working` or
`input-required`, it is resumed by `message/send` with its task id, and a
caller learns it moved through `tasks/resubscribe` or a push notification.
What is not built is who sends those messages when no person is typing:
the caller that does not wait, the clock, and the system that posts back.

## Writing one

```python
from livekit.agents import AgentTask, Completed, InputRequired, RunContext, Working, function_tool

@dataclass
class Refill(AgentTask[str]):
    medication: str                                   # arguments: JSON-shaped, the task's only inputs

    def __post_init__(self) -> None:                  # a task builds its deps like a door does
        super().__init__(
            instructions="Get the patient's refill done.",
            llm="google/gemma-4-31b-it",
            deps=Deps(ehr=httpx.AsyncClient(base_url=EHR)),
        )

    async def on_enter(self, ctx: RunContext[Deps, Patient]) -> Working:
        await ctx.deps.ehr.post("/refill-requests", json={..., "callback_task": ctx.task_id})
        return Working(tool=self.handle_decision, timeout=timedelta(days=3),
                       message="Sent to the prescriber for approval.")

    async def on_timeout(self, ctx: RunContext[Deps, Patient]) -> Completed[str]:
        return Completed("no answer from the prescriber in three days")

    @function_tool
    async def handle_decision(self, ctx: RunContext[Deps, Patient], approved: bool) -> Completed[str] | InputRequired: ...
```

| Hook | When | Returns |
|---|---|---|
| `on_enter(ctx)` | once, when the task starts (built) | a transition, or `None` for a turn of the task's model |
| `on_timeout(ctx)` | at a deadline from `timeout=` | the next transition; `None` fails the task |
| `on_cancel(ctx)` | before the task is canceled | |
| `on_exit(ctx)` | once, after a terminal state | |

The rules of [agent-tasks.md](agent-tasks.md) hold: arguments are the only
inputs, the task builds its deps, a pure code task has no model, outside
calls are keyed on `ctx.task_id` since a hook may run again after a crash,
and code between two waits can be a [graph](graphs.md).

## Waiting on people and systems

- `InputRequired(question)` is built: the answer is a turn of the task's
  model. Planned around it: a tool that takes the answer, a timeout, and how
  the question reaches the person outside a live call
  ([planned-channels.md](planned-channels.md)).
- `Working(tool=, timeout=, message=)` waits on a system and delivers
  nothing; `message=` is what a client sees as the status.
- The named tool's parameter schema rides on the status as a data part, so
  a form knows what to render and a system knows what to post.
- `timeout=` schedules `on_timeout(ctx)`, which returns the next transition.

| The answer is | What happens |
|---|---|
| a `DataPart` matching the named tool's parameters, from a form, a client or a system posting `message/send` with the task id | the tool is called directly; no model |
| free text, from a chat box or a call | the tool's first string parameter, no model (built); a tool with none gets a turn of the task's own model, which reads it and calls the tool |
| a message in the main conversation that belongs to the waiting task | the main agent hands the conversation to the task, which holds it until it returns `Completed` |

## Starting one without waiting

```python
@clinic.tool
async def request_refill(ctx: RunContext[Deps, Patient], medication: str) -> str:
    started = await refills.send(f"refill {medication}")           # the delegate, not awaited
    return f"refill started, task {started.task_id}"
```

A background task is a delegation whose reply is not awaited:
`A2ADelegate.send()` returns once the expert's task is `submitted`, with its
id, and the expert may be an endpoint of the same server. The task runs on
its own lane of the thread, survives the turn, the worker and deploys, and
its result arrives later as a message on the caller's lane, which the
caller's next turn reads like any other. A chain is a task that sends to
its successor; a fan-out sends several.

## The clock

`delay=` on `send()`, for the check-in in a day, and `timeout=` on a
transition, for the refill nobody answered, are one feature: a
`message/send` to a task at a time, owned by agent-service, not by a
worker. It carries `livekit/timeout` so the task runs `on_timeout` rather
than a turn.

## Started from outside

A backend starts a task with no conversation and no model by `message/send`
carrying `livekit/task` with the class name and its arguments, the same
message a delegate sends. Every `AgentTask` subclass the server imports is
registered; `public=True` on the class lets callers of a public endpoint
start it. The backend API is in [planned-backend-api.md](planned-backend-api.md).

See `examples/use-cases/planned-05_back_in_stock.py`,
`planned-08_summarizing_the_simulations.py`,
`planned-11_post_procedure_check_in.py` and
`planned-13_prescription_refill.py`, written before `ctx.submit` was dropped
for `send()`.
