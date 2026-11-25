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

## Twilio Elastic SIP Trunking transfers with the original caller ID

[`twilio_sip_trunk_warm_transfer.py`](twilio_sip_trunk_warm_transfer.py) is a
Twilio-only variant of `warm_transfer.py`. Use it only when your outbound trunk
terminates on Twilio Elastic SIP Trunking; other SIP providers do not use
this header, so keep `warm_transfer.py` for them.

With Twilio's [immutable call forwarding](https://www.twilio.com/en-us/changelog/elastic-sip-trunking---immutable-call-forwarding-with-calltoken-)
(public beta), Elastic SIP Trunking adds an `X-Twilio-CallToken` header to each
inbound trunk call. Sending that header back unchanged on the outbound call, with the
original caller's number as `From`, lets Twilio present the customer's caller ID to
the supervisor. `twilio_caller_id()` does this with the existing `sip_number` and
`sip_headers` options of `WarmTransferTask`.

1. Enroll your Twilio account in the beta. Without it, Twilio sends no token and
   rejects forwarded calls that present an unverified caller ID.
2. Set `LIVEKIT_SIP_OUTBOUND_TRUNK` to an outbound trunk that terminates on Twilio
   Elastic SIP Trunking.
3. Run `python twilio_sip_trunk_warm_transfer.py dev`.

No inbound trunk changes are needed: the example reads the header from the caller's
SIP leg with the [`lk.sip.GetRemoteHeaders`](https://docs.livekit.io/telephony/accepting-calls/workflow-setup/)
RPC, so the token is never stored in participant attributes.

Notes:

- The example uses the caller linked to the agent session. If that caller is not a
  SIP participant, the header is missing, or `sip.phoneNumber` is hidden by the
  inbound trunk, it keeps `LIVEKIT_SIP_NUMBER`.
- The token is only valid with the caller's number from the same inbound call. It
  does not authorize an arbitrary caller ID.
- There is no automatic fallback: if Twilio rejects the call, the task fails and the
  caller conversation resumes.
- Keep the token out of prompts, chat history, and logs. RPC responses are recorded in
  traces as PII (`lk.pii.rpc.response`), so keep PII redaction on if you export traces.
