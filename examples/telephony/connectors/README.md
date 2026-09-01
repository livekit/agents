# Connector Examples

These examples connect phone and WhatsApp calls to a LiveKit agent with [LiveKit Connectors](https://docs.livekit.io/telephony/connectors/), with no SIP trunk to provision. Each call joins a LiveKit room as a regular participant, so the agent code is unchanged.

Both examples reuse the DTMF agent from [`../basic_dtmf_agent.py`](../basic_dtmf_agent.py). No new agent code is needed. The webhook servers dispatch that agent by its name (`my-telephony-agent` by default, or set `DTMF_AGENT_DISPATCH_NAME`).

## Overview

1. **Twilio (`twilio_connector.py`)**: A webhook server for [Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams). When a call comes in, it calls `ConnectTwilioCall`, dispatches the agent, and returns TwiML that bridges the call into the room. A `dial` command places outbound calls through the Twilio REST API.
2. **WhatsApp (`whatsapp_connector.py`)**: A webhook server for the [WhatsApp Business Calling API](https://developers.facebook.com/documentation/business-messaging/whatsapp/calling). It accepts inbound calls with `AcceptWhatsAppCall`, completes outbound calls with `ConnectWhatsAppCall`, and cleans up on the terminate event. A `dial` command starts outbound calls.

## Prerequisites

- A running agent. In one terminal: `python ../basic_dtmf_agent.py dev`
- A LiveKit Cloud project with connectors enabled, and `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` in your environment.
- A publicly reachable URL for the webhook server. For local development, a tunnel like ngrok works: `ngrok http 8080`.

Both servers verify webhook signatures by default and refuse to start without the secrets that verification needs: `TWILIO_AUTH_TOKEN` plus `TWILIO_WEBHOOK_URL` for Twilio, and `WHATSAPP_APP_SECRET` for WhatsApp. An unauthenticated webhook lets anyone trigger connector calls, so pass `serve --allow-unverified` only for local testing. Meta separately requires a verify token to register the webhook at all. The server always enforces that handshake.

## Twilio

| Variable | Purpose |
|---|---|
| `TWILIO_ACCOUNT_SID` | Account SID, for the `dial` command only |
| `TWILIO_AUTH_TOKEN` | Auth token. Signs inbound webhooks and authenticates the `dial` command |
| `TWILIO_FROM_NUMBER` | Your Twilio number, for the `dial` command only |
| `TWILIO_WEBHOOK_URL` | The exact webhook URL configured in the Twilio console. Twilio signs it, so verification uses it verbatim |

1. Run the server: `python twilio_connector.py serve` (add `--allow-unverified` to skip signature verification while testing)
2. In the [Twilio Console](https://console.twilio.com), set your phone number's "A call comes in" webhook to `https://<your-tunnel>/twilio/voice` with HTTP POST.
3. Call your Twilio number. The agent answers.

For an outbound call: `python twilio_connector.py dial --to +15551234567`

## WhatsApp

| Variable | Purpose |
|---|---|
| `WHATSAPP_PHONE_NUMBER_ID` | The business phone number ID from your Meta app |
| `WHATSAPP_API_KEY` | A Meta access token |
| `WHATSAPP_CLOUD_API_VERSION` | Cloud API version, default `26.0`. Use a version [the connector supports](https://docs.livekit.io/telephony/connectors/whatsapp/). |
| `WHATSAPP_VERIFY_TOKEN` | A string you choose. Meta requires it when registering the webhook and echoes it in the verification handshake |
| `WHATSAPP_APP_SECRET` | Your Meta app's App Secret (App Dashboard, Settings, Basic), which Meta uses to sign webhook payloads. Not the verify token |

1. Enable calling on your WhatsApp Business number and subscribe your app to the `calls` webhook field. See the [Meta setup guide](https://developers.facebook.com/docs/whatsapp/cloud-api/guides/set-up-webhooks).
2. Run the server: `python whatsapp_connector.py serve` (add `--allow-unverified` to skip signature verification while testing)
3. Register `https://<your-tunnel>/whatsapp/webhook` as the webhook URL with your verify token.
4. Call your business number from WhatsApp. The agent answers.

For an outbound call, keep the server running and run: `python whatsapp_connector.py dial --to 15551234567`

Outbound WhatsApp calling requires [user permission](https://developers.facebook.com/documentation/business-messaging/whatsapp/calling/user-call-permissions) and is not available in every country.

## Warm transfer

Twilio connector calls support the warm transfer workflow in [`../warm-transfer/`](../warm-transfer/). The customer leg needs no changes. There are two ways to reach the supervisor:

- `WarmTransferTask` dials the supervisor over a SIP trunk that LiveKit manages.
- `TwilioConnectorWarmTransferTask` dials the supervisor through the Twilio connector with your own Twilio credentials, so no trunk is needed. See [`../warm-transfer/twilio_connector_warm_transfer.py`](../warm-transfer/twilio_connector_warm_transfer.py).

To try it with these examples, set the Twilio variables above plus `LIVEKIT_SUPERVISOR_PHONE_NUMBER`, run the warm transfer agent instead of the DTMF agent, and point the webhook server at its dispatch name:

```bash
python ../warm-transfer/twilio_connector_warm_transfer.py dev   # registers as "telephony-support-agent"
DTMF_AGENT_DISPATCH_NAME=telephony-support-agent python twilio_connector.py serve
```

Calls then reach a support agent that escalates to a supervisor when asked, with the supervisor dialed through the connector too. Warm transfer on WhatsApp connector calls is not supported yet.

## Connectors and SIP

Connectors are one of two telephony paths. SIP trunking (see [`../amd.py`](../amd.py) and [`../bank-ivr/`](../bank-ivr/)) stays the recommended path when you are starting fresh: it works with any provider and LiveKit manages routing with dispatch rules. Use the connector when your call logic already lives in Twilio, or for WhatsApp, which has no phone number to trunk.

For setup instructions and more details, see the [main examples README](../../README.md).
