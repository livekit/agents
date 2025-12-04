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
