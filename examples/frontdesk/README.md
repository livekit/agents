# Front Desk Example

A front desk agent demonstrating customer service with calendar integration and appointment management.

For setup instructions and more details, see the [main examples README](../README.md).

## Overview

In this example, you will be able to schedule appointments (optionally with cal.com's API if `CAL_API_KEY` is set) and evaluate the agent's performance using `JudgeGroup`. The session will always begin with the agent saying "Hello, I can help you schedule an appointment!"

### Scheduling appointments 

The LLM will call list_available_slots before `schedule_appointment`, since `slot_id` is a required argument. 

`list_available_slots` will return slots like:

```bash
ST_abc123 - Saturday, January 1, 2000 at 14:00 PDT (in 5 days)
```

The slots are also cached as a lookup table for `schedule_appointment`. 

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/frontdesk/frontdesk_agent.py#L184


If the slot is invalid, we raise a `ToolError` to allow the LLM to self correct, which prevents the LLM from passing a hallucinated answer.

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/frontdesk/frontdesk_agent.py#L94-L95


The user's email is then collected via `GetEmailTask()`. If the agent is interrupted after the task completes, `schedule_appointment` is aborted before an API call is made to book the slot. After the task, the function is uninterruptible. 

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/frontdesk/frontdesk_agent.py#L97-L119


### Evaluations

After the session ends, we use a `JudgeGroup` with pre-built judges to score the conversation. 

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/frontdesk/frontdesk_agent.py#L200-L214

When the success criteria for an agent is clear, using judges can complete the evaluation by measuring the performance quality. 

### Simulations

`scenarios.yaml` contains 10 scenarios (happy paths and adversarial callers) that run the agent against a simulated user. All simulation glue lives in `simulation.py`; the agent code itself stays production-shaped.

Each scenario's `userdata` drives the whole run:

- `available_slots`: ISO datetimes seeding a deterministic `FakeCalendar` for that scenario. The entrypoint detects a simulated run via `ctx.simulation_context()` and swaps the data source.
- `expected_booking`: grades the run on final calendar state in `on_simulation_end`: the single slot the agent must have booked, `null` when the agent must not book anything, or omitted to grade on the conversation alone. This check can only veto a run the simulator passed (the effective result is the AND of both verdicts).
- `now`: an optional ISO datetime overriding the scenario clock (defaults to `simulation.SIMULATION_NOW`, `2026-06-12`).

The scenarios reference absolute dates, so under simulation the `FakeCalendar` runs on that fixed clock (`simulation.SIMULATION_NOW`, or a per-scenario `now`), keeping availability and expected bookings deterministic without any environment setup.

#### Tool mocking

Under simulation the agent's tools always run mocked, using the same `mock_tools` helper the tests use, but as a plain call targeting the live session instead of a context manager:

```python
mock_tools(FrontDeskAgent, simulation.tool_mocks(cal, tz), session=session)
```

The LLM keeps seeing the real tool schemas; only execution is intercepted, and a mock may declare any subset of the real tool's parameters. The mocks are dynamic: both close over the same `FakeCalendar`, so booking through the mocked `schedule_appointment` changes what the mocked `list_available_slots` returns on the next call (the "Booked slot disappears from later listings" scenario asserts exactly that). Passing a new dict replaces a session's mocks at any time; `{}` removes them.
