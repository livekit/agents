# India SIP trunk with region pinning

A voice agent that answers and places calls on Indian phone numbers, with SIP pinned to LiveKit's India region. [Vobiz](https://vobiz.ai) is the SIP carrier here, and the LiveKit configuration carries over to other Indian carriers.

The example shows the two settings that an India deployment needs and the default SIP setup leaves out:

- **Inbound:** the carrier sends calls to `<subdomain>.india.sip.livekit.cloud` instead of the global SIP endpoint.
- **Outbound:** the trunk sets `destination_country="in"`, so calls originate from LiveKit's India region.

See [Region pinning for telephony](https://docs.livekit.io/telephony/features/region-pinning/) for how these work.

## Prerequisites

- A LiveKit Cloud project and the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/)
- A Vobiz account with a phone number and an outbound SIP trunk with credentials

## Setup

Install the dependencies:

```bash
uv pip install "livekit-agents>=1.6" python-dotenv
```

Create a `.env` file in this directory:

```bash
LIVEKIT_URL="<your livekit ws url>"
LIVEKIT_API_KEY="<your livekit api key>"
LIVEKIT_API_SECRET="<your livekit api secret>"

# Project ID without the p_ prefix. This can differ from the subdomain in LIVEKIT_URL,
# so copy it from the SIP URI on your LiveKit project settings page.
LIVEKIT_SIP_SUBDOMAIN="<your sip subdomain>"

VOBIZ_PHONE_NUMBER="+91XXXXXXXXXX"
VOBIZ_SIP_DOMAIN="<sip domain of your vobiz trunk>"
VOBIZ_SIP_USERNAME="<sip username of your vobiz trunk>"
VOBIZ_SIP_PASSWORD="<sip password of your vobiz trunk>"
```

Create the trunks and the dispatch rule:

```bash
python setup_trunks.py
```

The script prints two values:

1. Add `SIP_OUTBOUND_TRUNK_ID` to your `.env` file.
2. In the Vobiz Console, set the **Primary URI** of your inbound trunk to the printed hostname. Enter the hostname only, without a `sip:` prefix.

Start the agent:

```bash
python agent.py dev
```

## Receive a call

Call your Vobiz number from any phone. The dispatch rule creates a room and the agent joins it.

## Place a call

Dispatch the agent with the number to call in its metadata:

```bash
lk dispatch create \
  --new-room \
  --agent-name vobiz-agent \
  --metadata '{"phone_number": "+91XXXXXXXXXX"}'
```

## Production notes

- The inbound trunk accepts any source address that presents your number. Before going live, set `allowed_addresses` to the [Vobiz SIP signaling IPs](https://www.vobiz.ai/docs/concepts/ip-whitelisting).
- Running `setup_trunks.py` a second time fails, because an inbound trunk already claims the number. Delete the old trunk with `lk sip inbound delete`, or reuse it.
- Outbound calling in India has its own rules on calling hours, DND and caller ID. Check these for your use case.
