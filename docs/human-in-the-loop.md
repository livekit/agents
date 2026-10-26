# Human in the loop

The return value of a tool or a hook is the transition it makes. One object
per state the conversation or a task can be in, and nothing else says
"wait" or "done".

```python
from livekit.agents import Completed, Failed, Rejected, InputRequired, Working

Completed(value)                 # done: the result, and the last message when text
Failed(error)                    # failed: the error text is the last message
Rejected(reason)                 # will not run this input
InputRequired(question)          # waiting on the person; asked in the facing agent's words
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

Use it only for an input the tool needs to go on: an email, a yes or no on
a read-back, a card number. A conversational question, such as "would you
like one of these slots?", is a plain reply: the task completes, and the
next message is a new task. The questions of one step's calls are one
question, each once and in order, exact only when every part is.

The turn waits. What is stored: the function call, its output,
`waiting for the person's answer to: <question>`, and the question as an
assistant message. The question is the turn's reply. Nothing stays in
memory.

## Who words it

The agent facing the person words what reaches them.

| | The agent facing the person | A run another agent asked |
|---|---|---|
| `InputRequired(question)` | asks in its own words: one model request with no tools in text, a `generate_reply` on a call | passes the question up as written; the agent facing the person words it |
| `InputRequired(question, verbatim=True)` | asks it exactly as written: the turn's reply in text, `say()` on a call | passes it up flagged `livekit/verbatim`; asked as written at the end |
| `ToolResult(text, verbatim=True)`, `Completed(value, verbatim=True)` | delivers the text as written; the model reads `delivered to the person as written: ...` and continues, unless `reply_required=False` ends the turn there | the same, sent as exact text: see [a2a-live-ext.md](a2a-live-ext.md#exact-text) |

A run is another agent's when its turn's messages carry
`livekit/kind: delegation`. An agent with no model, a graph or a code task,
asks as written. `result.verbatim` is the answer or the question when it is
exact, which the door flags and the delegate gives as written.
A read-back from `requires_approval` is verbatim.

The person's next message, whenever it comes and on whatever worker, is the
answer: one user message, and the call keeps its question as its output.
The model runs on the history: it sees the call, the question and the
answer, and continues, calling the next tool, or explaining and asking
again when the message did not answer. The tool that asked never runs
again. It has no continuation, which is why a conversation tool can ask and
cannot act on the answer; the model acts on it.

On a call the question is spoken and the next user turn is the answer. In
text, the turn's task ends `input-required` with the question:

| The next message | What happens |
|---|---|
| carries the waiting task's id | that task continues: the same task, no row of its own, the message added to its history; a resend of it runs nothing |
| carries no task id, from the person | the door sends it to the question in front of them, the oldest open one of the lane, which it continues |
| carries no task id, tagged `livekit/kind: delegation` | a new task: another agent names the task it continues |
| carries the id of a running task | a new task |

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

Waiting on a system, naming the tool that takes the answer, deadlines and
answers that call a tool with no model are designed in
[planned-agent-tasks.md](planned-agent-tasks.md).

See `examples/use-cases/04_a_claim_with_a_photo.py`.
