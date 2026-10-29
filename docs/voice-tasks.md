# VoiceTask

`VoiceTask` is a temporary interaction run by the fast brain: the frontend
agent attached to an active voice session. It can collect an email address,
confirm a spoken name, gather DTMF input, or guide another bounded exchange
before returning a typed result to the frontend's calling tool.
`VoiceTask` runs an agent whose `output_type` is the typed result; helpers such
as `GetEmailTask` return their domain result through `.run()`.

Calling `.run()` resolves the active `VoiceSession` from the current frontend
invocation. No session argument or application-side session check is required.
The binding is invocation-local, so simultaneous sessions cannot share it.
Calling a live task without an active live-session interaction raises a clear
error before the task starts. Text-mode backend runs do not provide this binding.

The task temporarily takes the conversational floor. The parent frontend
waits in memory while the task receives live user turns and invokes its model
and tools. On accepted final output, the task returns its typed result and
restores the parent, whose model can continue from the tool result. This is
suspension and restoration of the same frontend activation: the parent's
`on_exit` and `on_enter` hooks are not called for the task transition.

The lifetime is the active `VoiceSession`. Completion, failure, cancellation,
and session closure end the task. Cleanup releases the floor and any
[override layers](voice-session.md#temporary-overrides) the task opened,
including on failure. Audio configuration remains owned by `VoiceSession`;
base changes the application makes during the interaction persist.

This is an in-memory frontend interaction, not a stateless workload or backend
execution primitive. Its parent await remains on the running Python stack.
Saving history or application fields does not restore that await after a
worker restart, deployment, or session loss. It does not provide background
execution, scheduling, or cross-request task recovery.

Use `Agent.run()` for stateless backend invocations and `TextSession` for
backend agent-tool conversations that continue across requests using saved
routing state. A frontend task and a backend delegation have separate owners:
temporarily collecting live input does not silently cancel, transfer, or
complete an outstanding remote task.

## Authoring a custom VoiceTask

Like `TextSession`, `VoiceTask` is not an `Agent`: it contains one.
`VoiceTask(agent=...)` runs that agent for one bounded live interaction. The
agent carries the instructions, tools and `output_type`; the task adds `.run()`
and the floor handling around it. A `VoiceTask` cannot be passed to
`update_agent()` or served on its own, and it exists only for the duration of
one `.run()`. The agent it contains is ordinary shared configuration, built once
at import like any other agent.

Declare the agent's `output_type`; returning `Completed(value)` from one of its
tools completes the live interaction. Ordinary tool values and conversational
speech keep it active. The framework validates the completed value before
returning it to the waiting caller and restoring that caller's live role.

A task can also fail: returning `Failed(reason)` from one of its tools ends
the interaction and restores the caller, whose `run()` raises
`VoiceTaskFailed` with that reason as its message. `VoiceTaskFailed` is a
`ToolError`, so a caller tool that does not catch it hands the reason to its
own model as the tool's error, and that model words the next step. A caller
that wants to recover catches it:

```python
@front_desk.tool
async def change_receipt_email(ctx: RunContext[AirlineContext]) -> str:
    try:
        result = await VoiceTask(agent=email_collector).run()
    except VoiceTaskFailed as failure:
        return f"The receipt email was not changed: {failure.message}"
    ...


@email_collector.tool
async def decline(ctx: RunContext[object], reason: str) -> Failed:
    """The passenger does not want to give an email address."""
    return Failed(reason)
```

Only the tools of the task's own agent end it this way; a `Failed` from the
caller's tools has its ordinary meaning. Cancellation and session closure
still cancel the caller's await (`asyncio.CancelledError`).

```python
from pydantic import BaseModel, EmailStr

from livekit.agents import Agent, Completed, Failed, RunContext, VoiceTask, VoiceTaskFailed


class EmailResult(BaseModel):
    email_address: EmailStr


email_collector = Agent(
    context_type=object,
    instructions=(
        "Collect the passenger's receipt email. Clarify spelling when needed. "
        "Call confirm_email with the address. Its approval read-back asks the "
        "passenger to confirm; finish only after they approve that exact address."
    ),
    output_type=EmailResult,
)


async def read_back(ctx: RunContext[object], email_address: EmailStr) -> str:
    return f"Should I use {email_address} for your refund receipt?"


@email_collector.tool(needs_approval=read_back)
async def confirm_email(
    ctx: RunContext[object], email_address: EmailStr
) -> Completed[EmailResult]:
    # EmailStr validates the address; the shared approval gate runs before
    # this body. Changing the address requires confirmation again.
    return Completed(EmailResult(email_address=email_address))


@front_desk.tool
async def change_receipt_email(ctx: RunContext[AirlineContext]) -> str:
    # EmailResult; the current session is implicit.
    result = await VoiceTask(agent=email_collector).run()
    # Completed restored front_desk. Its queued expert events become eligible
    # once this tool finishes, provided it still exposes the remote refund tool.
    await ctx.context.airline.set_refund_receipt_email(
        booking_reference=ctx.context.booking_reference,
        email=result.email_address,
    )
    return f"Refund receipt email updated to {result.email_address}."
```

Each call constructs a fresh `VoiceTask` around the shared agent. Its agent
changes only instructions, tools and the LLM; audio configuration belongs to the
session. A task that needs different audio settings opens
`async with session.override_options(...)` inside its code, released when the
task ends. Its tools can use the runtime wrapper without depending on the
parent's application data; this example types the unused application context
as `object`.

`Completed[EmailResult]` ends this live task, not the remote refund task or the
whole call. The email's confirmation authorizes using that address; it does
not authorize a refund. This task replaces the prebuilt `GetEmailTask` in
the airline refund example below and requires no expert-triggered collection behavior.

## Example: airline refund and receipt email

A passenger asks for a refund. The frontend calls the remote airline agent,
which checks the ticket and handles refund eligibility, approval, and execution.
While it works, the passenger asks to use a different email for the receipt.
The frontend collects that address with a local live task.

```python
from dataclasses import dataclass

from livekit.agents import Agent, RunContext
from livekit.agents.a2a import RemoteAgent
from livekit.agents.beta.workflows import GetEmailTask


@dataclass
class AirlineContext:
    booking_reference: str
    airline: AirlineServices


refunds = RemoteAgent(REFUND_AGENT_URL)
front_desk = Agent(
    context_type=AirlineContext,
    instructions=(
        "Help passengers with refunds through the airline refund expert. "
        "If the passenger asks to change their receipt email, collect and save it. "
        "Never promise a refund until the expert confirms it."
    ),
    tools=[refunds.as_tool()],
)


@front_desk.tool
async def change_receipt_email(ctx: RunContext[AirlineContext]) -> str:
    """Collect and confirm the email address for this booking's refund receipt."""
    # Passenger: "Please send the refund receipt to a different email."
    # The refund expert may still be checking the ticket.
    # INLINE TASK: temporarily owns the floor to collect and confirm the address.
    # It has no remote tool and cannot interact with the refund expert.
    # INBOX: expert progress/questions/results arrive in the frontend queue
    # and wake the session, but wait for email capture to finish.
    # The passenger's spelling and confirmation answers go only to this task.
    result = await GetEmailTask().run()

    # The task restores front_desk; no explicit update_agent() is needed.
    await ctx.context.airline.set_refund_receipt_email(
        booking_reference=ctx.context.booking_reference,
        email=result.email_address,
    )
    # INBOX: after this foreground tool finishes and the floor is available,
    # front_desk can present queued events because it exposes refunds.as_tool().
    # Another active agent with that same remote tool would also be eligible.
    # No polling or manual drain.
    # Progress can be combined. An expert question is phrased by front_desk;
    # the live session sends the next relevant answer to the recorded refund task.
    # A refund completion resolves the original remote call before the next
    # ordinary frontend model invocation announces the confirmed outcome.
    return f"The refund receipt email is now {result.email_address}."
```

For example, the refund expert may ask whether the passenger accepts a partial
refund while the email task is confirming a spelling. That question stays queued
until the task returns and the frontend tool finishes. The email confirmation
is not treated as acceptance of the refund. The frontend then asks the refund
question, and the live session sends its answer to the expert.

The frontend selects `change_receipt_email` for the passenger's explicit new
request, using a model-valid foreground view while the remote call remains
pending. Expert-event presentation alone does not invoke this tool. This
example handles the passenger's separate request to change their email; it
does not turn an expert's email question into a local collection task.
`AirlineServices` is application-provided; it saves the receipt destination in
the authoritative booking/refund record used by the expert. The service enforces
whether the receipt address can still change. Saving the address neither approves
nor executes a refund. The host supplies the authenticated booking context to
the live session. The frontend event queue is in-process and requires no DB.
