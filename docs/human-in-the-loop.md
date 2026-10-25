# Human in the loop

The return value of a tool or a hook is the transition it makes. One object
per state the conversation or a task can be in, and nothing else says
"wait" or "done".

```python
from livekit.agents import Completed, Failed, Rejected, InputRequired, Working

Completed(value)                 # done: the result, and the last message when text
Failed(error)                    # failed: the error text is the last message
Rejected(reason)                 # will not run this input
InputRequired(question)          # waiting on the person, with no timeout; asked in the facing agent's words
Working(message=None)            # waiting on a system; nothing delivered
```

On the wire these are the A2A task states `completed`, `failed`, `rejected`,
`input-required` and `working`.

## From a conversation tool

```python
@fnol.tool
async def request_photo(ctx: RunContext[None, Loss]) -> InputRequired:
    """Ask the policyholder for a photo of the damage."""
    return InputRequired("Could you send a photo of the damage? A wide shot and a close-up.")
```

The turn waits. What is stored: the function call, its output standing in
for the wait, `waiting for the person's answer to: <question>`, and the
question as an assistant message. The question is the turn's reply. Nothing
stays in memory.

## Who words it

The agent facing the person words what reaches them.

| | The agent facing the person | A run another agent asked |
|---|---|---|
| `InputRequired(question)` | asks in its own words: one model request with no tools in text, a `generate_reply` on a call | passes the question up as written; the agent facing the person words it |
| `InputRequired(question, verbatim=True)` | asks it exactly as written: the turn's reply in text, `say()` on a call | passes it up flagged `livekit/verbatim`; asked as written at the end |
| `ToolResult(text, verbatim=True)`, `Completed(value, verbatim=True)` | delivers the text as written; the model reads `delivered to the person as written: ...` and continues, unless `reply_required=False` ends the turn there | the same, and the exact text leads the answer, flagged `livekit/verbatim` |

A run is another agent's when its turn's messages carry
`livekit/kind: delegation`. An agent with no model, a graph or a code task,
asks as written. `result.verbatim` is the part of the answer or the question
that was exact; the door puts it on the task as `livekit/verbatim`, and the
delegate delivers that part as written and hands the rest to its model.
A read-back from `requires_approval` is verbatim.

The answer continues the waiting task ([delegation.md](delegation.md)),
whenever it comes and on whatever worker: the call keeps the question as
its output, the answer is the person's next message, and the task's model
runs on the history: it sees the question, the call and the answer, and
continues, calling the next tool or asking again. The tool that asked never
runs again by itself, so the person can refuse or change the subject. A
delegate's question is the one exception: its tool sends the answer as said
to the expert's task, whose own model reads it, in a call that names the
one that asked (`FunctionCall.extra["lk.answers"]`). An input that fixed
logic must collect is an `AgentTask`.

On a call the question is spoken and the next user turn is the answer. In
text, the turn's task ends `input-required` with the question:

| The next message | What happens |
|---|---|
| carries the waiting task's id | that task continues under its own id: the answer, then the reply or the next question; no row of its own, nothing merged |
| comes from the person and carries no task id | it goes to the question in front of the person, and that task continues |
| is a delegation and carries no task id | it is a new task; the question stays open |

A task pauses once: the questions of one step are one question, in order and
each once, verbatim only when every part is, and a question that work in the
background asks while its task waits joins that task's question, which is
asked again. A lane has one question in front of the person. A question of
another task asked while one is in front waits unasked (`livekit/queued`); a
follow-up of the answered task goes first; when the task in front ends or
its work goes on in the background, the reply asks the next one. On a call
the same rule holds for the questions of tools that run at once. A question
asked by text can be answered on a call, and the reverse.

What `run()` returns then: `result.input_required` is the pending question
and `result.output` is `None`. The framework ends the task `input-required`
with that question whatever the handler returns, so a handler returns
`result.output` as usual.

A conversation tool that returns `Failed(error)` or `Rejected(reason)`
records the text as an error output and the model replies to it.
`Completed(value)` from a conversation tool is that call's output; a
conversation turn completes when its handler returns, not by a transition.

## Approvals

A tool declared with `requires_approval=` asks before it runs: the call
returns its read-back as a verbatim `InputRequired`, and the person's next
message answers it. The body runs only when the model repeats the same call
after that answer; the schema has no approval flag the model could set. See
[tools.md](tools.md).

## From a task

An `AgentTask` on a call makes the same transitions from `on_enter` or one
of its tools:

| Transition | In a task |
|---|---|
| `Completed(value)` | ends the task; `value` is what the awaiting code receives |
| `Failed(error)`, `Rejected(reason)` | ends the task; the awaiting code receives an exception carrying the text |
| `InputRequired(question)` | speaks the question; the person's next turn is the call's output and the task's model continues |
| `Working(message=)` | records the status; the task stays where it is |

A task with a model completes only through a tool that returns `Completed`;
text from its model goes to the person. See [agent-tasks.md](agent-tasks.md).

Waiting on a system and deadlines are designed in
[planned-agent-tasks.md](planned-agent-tasks.md).

See `examples/use-cases/04_a_claim_with_a_photo.py`.
