# Human in the loop

An agent invocation returns `RunResult`: either a completed application output
or a request for human input. Tools can request input with `InputRequired`;
the agent's `output_type` defines its accepted final application value.

Tool and handler transitions have distinct meanings. An ordinary tool's
`Completed(value)` supplies that tool's result to the model. A handler can
use transport transitions to answer its A2A task.

```python
from livekit.agents import Completed, Failed, Rejected, InputRequired, Working

Completed(value)                 # tool result, or a completed transport task
Failed(error)                    # failed: the error text is the last message
Rejected(reason)                 # will not run this input
InputRequired(question)          # waiting on the person; asked in the facing agent's words
Working(message=None)            # transport working status; no Python checkpoint
```

When returned as handler transitions, these map to A2A task states `completed`, `failed`, `rejected`,
`input-required` and `working`.

## From a conversation tool

```python
@fnol.tool
async def request_photo(ctx: RunContext[LossContext]) -> InputRequired:
    """Ask the policyholder for a photo of the damage."""
    return InputRequired("Could you send a photo of the damage? A wide shot and a close-up.")
```

The tool finishes with its question recorded as its outcome. The invocation
returns a result with `status="input_required"` and `question`. Save the
updated history and context, then deliver `result.question`.

## Who words it

The agent facing the person words what reaches them.

| | The agent facing the person | A run another agent asked |
|---|---|---|
| `InputRequired(question)` | asks in its own words: one model request with no tools in text, a `generate_reply` on a call | passes the question up as written; the agent facing the person words it |
| `InputRequired(question, verbatim=True)` | asks it exactly as written: the turn's reply in text, `say()` on a call | passes it up flagged `livekit/verbatim`; asked as written at the end |
| `ToolResult(text, verbatim=True)`, `Completed(value, verbatim=True)` | delivers the text as written; the model reads `delivered to the person as written: ...` and continues, unless `reply_required=False` ends the turn there | the same, and the exact text leads the answer, flagged `livekit/verbatim` |

A run is another agent's when its turn's messages carry
`livekit/kind: delegation`. A handler without a model asks as written. `result.verbatim` is the part of the answer or the question
that was exact; the door puts it on the task as `livekit/verbatim`, and the
delegate delivers that part as written and hands the rest to its model.
A read-back from `needs_approval` is verbatim.

The person's next message is appended to the saved history. A new invocation
reads that history and context. The model interprets the answer and chooses
its next tool or question. The tool that requested input has already finished.

On a call the question is spoken and the next user turn is the answer. In
text, the turn's task ends `input-required` with the question (`livekit/mergedInto` is in
[a2a-live-ext.md](a2a-live-ext.md)):

| The next message | What happens |
|---|---|
| carries the waiting task's id | that task continues: it gets the answer and the reply, and the row the message arrived under completes, with `livekit/mergedInto` pointing at that task |
| carries no task id | its own task carries the reply, and the waiting task completes, with `livekit/mergedInto` pointing at the new one |

Check `result.status == "input_required"` and read `result.question`.
The transport delivers that question and represents the turn as
`input-required`. A completed result exposes the declared application value
in `result.output`; the framework outcome stays outside `output_type`.

A conversation tool that returns `Failed(error)` or `Rejected(reason)`
records the text as an error output and the model replies to it.
`Completed(value)` from a conversation tool is that call's output; an
agent completes when its final output is accepted. The handler delivers the
returned framework outcome to the transport.

## Approvals

A tool declared with `needs_approval=` asks before it runs: the call
returns its read-back as a verbatim `InputRequired`, and the person's next
message answers it. The body runs only when the model repeats the same call
after that answer; the schema has no approval flag the model could set. See
[tools.md](tools.md).

An agent exposed with `as_tool(needs_approval=True)` asks permission before
starting the specialist. The router retains the pending invocation and its
arguments in routing state. The application explicitly saves that state; on
the next request it loads it and passes the person's reply to the router.
Approval to start the specialist does not authorize actions inside it: a
reservation or payment tool can require its own approval. See
[agent-tool approvals](agent-router.md#approval-before-starting-a-child).

## Routed conversations

[AgentRouter](agent-router.md) specifies task completion, human input, frames,
agent-tool calls, and child completion. It retains the active frame while waiting for a person.
The next message starts a new invocation of that frame's agent. Accepted child output fills the parent tool call and starts a fresh parent
invocation.
