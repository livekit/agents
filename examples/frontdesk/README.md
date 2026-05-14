# Front Desk Example

A front desk agent demonstrating customer service with calendar integration and appointment management.

For setup instructions and more details, see the [main examples README](../README.md).

## Overview

In this example, you will be able to schedule appointments (optionally with cal.com's API if `CAL_API_KEY` is set) and evaluate the agent's performance using `JudgeGroup`. The session will always begin with the agent saying "Hello, I can help you schedule an appointment."

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
