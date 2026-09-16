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

[`twilio_connector_warm_transfer.py`](twilio_connector_warm_transfer.py) uses the
[Twilio Connector](https://docs.livekit.io/telephony/connectors/twilio/) to dial the
supervisor through Twilio's Calls API and connect their audio to the consultation
room. This path requires LiveKit Cloud and the optional `twilio` Python package.

Passing `twilio_call_token` requires `twilio>=6.55.0`, which
[added CallToken to the Calls API client](https://github.com/twilio/twilio-python/releases/tag/6.55.0).
Install or upgrade with `pip install 'twilio>=6.55.0'`. If an older SDK lacks
this parameter, the task raises an upgrade error before contacting the connector
or dialing. Transfers without a token retain the existing behavior.

To show an inbound customer's phone number to the supervisor, capture `From` and
`CallToken` from that call's validated Twilio voice webhook. Keep them together in
your server-side state, keyed by the inbound `CallSid`, and supply them to the task
when that call requests a transfer:

```python
result = await TwilioConnectorWarmTransferTask(
    SUPERVISOR_PHONE_NUMBER,
    twilio_from_number=AGENT_PHONE_NUMBER,
    original_caller_number=inbound_from,
    twilio_call_token=inbound_call_token,
    chat_ctx=self.chat_ctx,
)
```

Twilio credentials are read from `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`, or
can be supplied through the task's corresponding constructor arguments. The task
passes the token unchanged to Twilio as `call_token`; it does not put it into the
connector request or TwiML. Retrieve it through your application's server-side
call context, and keep it out of prompts, chat history, logs, and participant
attributes.

`twilio_from_number` is the agent/business Twilio number or verified caller ID.
`original_caller_number` must match the original incoming call's `From`; the token
does not authorize an arbitrary caller ID. Supplying a nonempty `twilio_call_token`
requires `original_caller_number`. Without a token (or with an empty token), the task
uses the business number, even when the original caller number is available.

If Twilio explicitly rejects the preservation attempt with HTTP 400 and
[error 21210 (unverified From)](https://www.twilio.com/docs/api/errors/21210),
the task retries once from the business number without the token. Other errors,
including network timeouts and failures after call creation, are not retried to
avoid duplicate calls. A failed fallback is propagated to the transfer workflow.

The [Twilio Calls API](https://www.twilio.com/docs/voice/api/call-resource#create-a-call)
supports CallToken directly, so this workflow does not require a Twilio conference.
