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
    sip_call_to=SUPERVISOR_PHONE_NUMBER,
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

## Telnyx SIP trunk transfers with the original caller ID

[`telnyx_sip_trunk_warm_transfer.py`](telnyx_sip_trunk_warm_transfer.py) is a
Telnyx-only variant of `warm_transfer.py`. Use it only when your outbound trunk
terminates on Telnyx; other SIP providers have their own caller-ID rules, so keep
`warm_transfer.py` for them.

For an inbound call from customer A to your Telnyx number B, followed by a transfer
to supervisor C, [Telnyx requires](https://support.telnyx.com/en/articles/13117410-how-external-call-transfers-work):

- The original A-to-B call must still be active when dialing C.
- The outbound call must present A as the caller and include a SIP `Diversion`
  header identifying B.

`telnyx_caller_id()` does this with the existing `sip_number` and `sip_headers`
options of `WarmTransferTask`. It reads A from the caller's `sip.phoneNumber`
attribute and B from `sip.trunkPhoneNumber`, so both numbers come from the same
inbound call. No CallToken or Telnyx-specific SDK flag is needed.

1. Set `LIVEKIT_SIP_OUTBOUND_TRUNK` to an outbound trunk that terminates on Telnyx.
2. Run `python telnyx_sip_trunk_warm_transfer.py dev`.

Notes:

- The example uses the caller linked to the agent session. If that caller is not a
  SIP participant, or either number is missing or not in E.164 form, it keeps
  `LIVEKIT_SIP_NUMBER`.
- The customer's call stays connected while the supervisor is briefed, which
  satisfies the active-call requirement.
- There is no automatic fallback: if Telnyx rejects the call, the task fails and the
  caller conversation resumes.
- For `403 Unverified origination number D51`, check that the A-to-B call is still
  active and that `Diversion` names the Telnyx number B that received the call.
  Inspect the outbound SIP INVITE to confirm the header reaches Telnyx.
