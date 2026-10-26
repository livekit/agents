# Agent tasks

An `AgentTask[Output]` is an agent with a job that ends with a value of
type `Output`: `await` on it returns an `Output`, and its tools return
`Completed[Output]`. A tool or a hook awaits it: the task takes the
conversation, on a call or in text, and hands it back when it completes.

Submitting a task to run in the background, and starting one from outside,
are designed in [planned-agent-tasks.md](planned-agent-tasks.md).

## Writing one

```python
from livekit.agents import AgentTask, Completed, Failed, RunContext, function_tool

class CaptureDateOfBirth(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(
            instructions="Ask for the caller's date of birth, then record it.",
            llm="google/gemma-4-31b-it",
        )

    async def on_enter(self, ctx: RunContext) -> None:
        self.session.generate_reply(instructions="Ask for the date of birth.")

    @function_tool
    async def record(self, ctx: RunContext, date_of_birth: str) -> Completed[str]:
        """Record the date of birth.

        Args:
            date_of_birth: YYYY-MM-DD
        """
        return Completed(date_of_birth)
```

The constructor takes what an `Agent` takes for a call (`instructions`,
`tools`, `llm`, `stt`, `tts`, `vad`, `turn_handling`) and
`preserve_function_call_history=`. A task is a new instance per use: its
constructor arguments are its input, and it is awaited once. An `Agent` is
the shared, stateless one. The hooks take the context explicitly,
`RunContext[Deps, State]`, for an awaited task and a submitted one alike.

A task that keeps state names its type on the class:

```python
@dataclass
class QState:
    answers: dict[str, str] = field(default_factory=dict)

class Questionnaire(AgentTask[Answers, QState]):
    ...
```

`self.state` is a `QState`, built empty when the task starts, saved by the
framework while the task waits and restored on the next turn.
`AgentTask[Output]` alone has no state. `ctx.state` inside the task's tools
is still the thread's state, for what must outlive the task.

| Hook | When | Returns |
|---|---|---|
| `on_enter(self, ctx)` | when the task takes the conversation | `None`, or a transition |
| `on_exit(self)` | when it leaves the conversation, on a call | |

`on_enter` takes the `RunContext` when it declares a parameter, on a call
and in text alike; an `on_enter(self)` with none still works on a call.

The task's tools and `on_enter` end it with a transition (see
[human-in-the-loop.md](human-in-the-loop.md)):

- `Completed(value)` ends the task; `value` is what `await` returns.
- `Failed(error)` or `Rejected(reason)` ends it; `await` raises `ToolError`
  with the text.
- `InputRequired(question)` speaks the question and waits for the person's
  next turn, which is the call's output; the task's model continues.

A task with a model completes only through a tool that returns
`Completed`; text from its model goes to the person. `on_enter` returning
`Failed("...")` ends the task before its model runs.

## Awaited: a guided sub-flow

```python
@front_desk.tool
async def verify_caller(ctx: RunContext[Deps, Patient]) -> str:
    """Verify the caller before touching their appointments."""
    ctx.state.date_of_birth = await CaptureDateOfBirth()
    return "verified"
```

A task is awaited inside a tool or inside an agent's `on_enter` or
`on_exit` on a call. While it runs, the agent that awaited it is paused and
the task's activity has the conversation. When it ends, what the task added
to the history stays in the thread's history, without its function calls
unless the task was built with `preserve_function_call_history=True`, and
the paused agent resumes. Wrap the await in `async with ctx.foreground():`
so it does not race with speech already queued; see [voice.md](voice.md).

`task.done()` says whether it has ended. `task.cancel()` interrupts it and
ends it; the awaiting code receives a `ToolError`.

## Awaited in text

```python
@dataclass
class BookRoom(AgentTask[RoomBooking, RoomDraft]):
    guests: int                                        # the arguments: JSON-shaped, the task's input

    def __post_init__(self) -> None:
        super().__init__(instructions="Book the room.", llm="google/gemma-4-31b-it")

    @function_tool
    async def confirm(self, ctx: RunContext[Hotel, Caller]) -> Completed[RoomBooking]: ...


@front_desk.tool
async def start_booking(ctx: RunContext[Hotel, Caller], guests: int) -> str:
    booking = await BookRoom(guests=guests)            # turns pass here
    return f"booked {booking.code}"
```

Inside `agent.run()` the await is a handoff plus a deferred call. The tool's
call stays pending in the history, the task is pushed on the lane's agent
stack with its arguments and its state, and the task's `on_enter` runs; what
the task's model says ends the turn `input-required`. Each later message runs
the task, on any worker: the stack entry rebuilds it from its class name and
arguments, as a dataclass, and restores `self.state`. `Completed(value)`
pops the task and writes the value as the pending call's output, and the
agent below continues its turn from there, phrasing the result; `Failed`
writes a `ToolError`. `InputRequired` from a task's tool waits exactly as it
does from a conversation tool: the person's answer is a turn of the task's
model ([tools.md](tools.md)). See
[lanes.md](lanes.md) for the stack.

A task class must be importable by the server that gets the next message:
the registry finds `AgentTask` subclasses by class name. See `examples/hotel-receptionist/`:
a room booking and a verification, awaited from the back office's tools.

## Structured results on a call

A voice turn's reply is speech, so a typed result on a call is an
`AgentTask[Output]`: the value its tool returns in `Completed(value)`.
