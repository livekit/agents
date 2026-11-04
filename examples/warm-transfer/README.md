# Warm transfer example (supervisor escalation)

This example shows a warm transfer workflow for call centers when a customer requests escalation.

**Flow**:

1. Customer requests escalation
2. Agent places the customer on hold
3. Agent contacts the next escalation point (supervisor)
4. Agent briefs the supervisor with a summary
5. Agent connects the supervisor to the customer

**How it works with LiveKit**

- The agent creates a new Room to reach the supervisor and places a SIP call with `CreateSIPParticipant`.
- A separate `AgentSession` is used to share context with the supervisor.
- When the supervisor agrees, `MoveParticipant` API moves the supervisor into the customer's Room.

**Using the WarmTransferTask**

The `WarmTransferTask` from `livekit.agents.beta.workflows` simplifies the warm transfer flow. You don't need to implement the transfer logic yourself - just call the task with the target phone number and SIP trunk ID:

```python
result = await WarmTransferTask(
    target_phone_number=SUPERVISOR_PHONE_NUMBER,
    sip_trunk_id=SIP_TRUNK_ID,
    chat_ctx=self.chat_ctx,  # Provides conversation history to the supervisor
)
```


# Usage

**Prerequisites**

- A [LiveKit Cloud](https://livekit.io) account
- SIP trunks configured (inbound & outbound) [guide](https://docs.livekit.io/sip/quickstarts/configuring-sip-trunk/)
- Two phone numbers, one to call the agent, the other for escalation
- A SIP dispatch rule to trigger `sip-inbound` agent when dialed

**Environment variables**
- LIVEKIT_SIP_OUTBOUND_TRUNK: the outbound SIP trunk ID
- LIVEKIT_SUPERVISOR_PHONE_NUMBER: the phone number of the supervisor (including + and country-code)

**Run the agent**

```python
python warm_transfer.py dev
```

## Twilio connector transfers with the original caller ID

[`twilio_connector_warm_transfer.py`](twilio_connector_warm_transfer.py) dials the
supervisor through the [Twilio Connector](https://docs.livekit.io/telephony/connectors/twilio/)
and Twilio's Calls API. This requires LiveKit Cloud and Twilio credentials.

As shipped, this example uses the business number. It does not receive or store
inbound Twilio webhooks. To demonstrate caller-ID forwarding, implement
`inbound_caller_id()` using your server-side store for the current inbound `CallSid`;
the placeholder currently returns an empty number/token pair.

To show the inbound customer's number to the supervisor instead of your Twilio number,
pass `original_caller_number` (the inbound call's `From`) together with
`twilio_call_token` (its `CallToken`), both captured from the validated voice webhook:

```python
result = await TwilioConnectorWarmTransferTask(
    SUPERVISOR_PHONE_NUMBER,
    twilio_from_number=TWILIO_FROM_NUMBER,
    original_caller_number=inbound_from,
    twilio_call_token=inbound_call_token,
    chat_ctx=self.chat_ctx,
)
```

If Twilio rejects the preserved caller ID with HTTP 400 and
error [21210](https://www.twilio.com/docs/api/errors/21210) (From not verified) or
[21212](https://www.twilio.com/docs/api/errors/21212) (invalid From), the task retries
once from `twilio_from_number` without the token; other errors are not retried. Keep the
token in server-side state, out of prompts, chat history, logs, and participant attributes.

Keep the number/token pair together, keyed by the inbound `CallSid`;
the token does not authorize an arbitrary caller ID. Without a token (including an
empty token), the task uses the business number even if `original_caller_number`
is supplied. Transport timeouts and unrelated errors never trigger a second dial.

Cancellation and ringing timeouts return promptly while cleanup continues in the
background. If in-flight creation returns a call SID, the task attempts to cancel
that call while the worker is alive. Provider failure or worker shutdown can still
prevent cleanup.
