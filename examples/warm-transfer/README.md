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

## Preserve the inbound caller's number with Telnyx

A warm transfer creates a new outbound call to the supervisor. To let the
supervisor identify the customer, an application can preserve the customer's
caller ID instead of presenting its business number. The existing `sip_number`
and `sip_headers` options support this with a Telnyx SIP trunk.

For an inbound call from customer A to your Telnyx number B, followed by a
transfer to supervisor C, [Telnyx requires](https://support.telnyx.com/en/articles/13117410-how-external-call-transfers-work):

- The original A-to-B call must still be active when dialing C.
- The outbound call must present A as the caller and include a SIP `Diversion`
  header identifying B.

In `SIPSupportAgent._start_transfer`, use the following options when creating
the task. `inbound_customer_number` and `inbound_telnyx_number` are application
variables: retrieve the original caller and called number from trusted,
server-side context for this specific inbound call. Normalize both to E.164
before constructing the header. `SIP_TRUNK_ID` must select your configured
Telnyx outbound trunk.

```python
return await WarmTransferTask(
    sip_call_to=SUPERVISOR_PHONE_NUMBER,
    sip_trunk_id=SIP_TRUNK_ID,
    sip_number=inbound_customer_number,
    sip_headers={
        "Diversion": f"<sip:{inbound_telnyx_number}@sip.telnyx.com>;reason=unconditional",
    },
    chat_ctx=self.chat_ctx,
    extra_instructions=SUMMARY_INSTRUCTIONS,
)
```

This uses the existing SIP warm-transfer flow; it does not require a CallToken
or a Telnyx-specific SDK flag. Keep the customer's original call connected
during consultation. Applications that want their business caller ID can keep
the existing example configuration.

### Verify and troubleshoot

Call B from phone A, request a transfer to phone C, then check the number shown
on C and complete the consultation and transfer. Caller-ID presentation also
depends on the receiving network. This snippet illustrates configuration; it
does not by itself verify carrier acceptance.

For `403 Unverified origination number D51`, check that the A-to-B call is still
active, the outbound caller number matches A, and `Diversion` identifies the
actual Telnyx number B that received the call. Inspect the outbound SIP INVITE
to confirm that the header reaches Telnyx. An arbitrary customer number or a
header copied from a different call does not establish a valid transfer.
