# Telnyx SIP Trunk Provider Example

This example demonstrates how to configure LiveKit Agents to work with Telnyx as a SIP trunk provider for telephony integrations.

## Prerequisites

1. **LiveKit Cloud Account** - Sign up at [livekit.io](https://livekit.io)
2. **Telnyx Account** - Sign up at [telnyx.com](https://telnyx.com)
3. **Phone Number** - A Telnyx phone number for receiving/making calls

## Telnyx Configuration

### 1. Set Up Telnyx

1. Log in to the [Telnyx Mission Control Portal](https://portal.telnyx.com)
2. Create a new Connection (SIP Trunk)
3. Configure your inbound/outbound routes
4. Purchase or assign a phone number to your connection

### 2. Configure LiveKit SIP

In the LiveKit Cloud dashboard:

1. Navigate to **SIP** > **Trunks**
2. Create a new SIP trunk with Telnyx credentials:
   - **Trunk Type**: Outbound (for making calls)
   - **SIP Domain**: Your Telnyx SIP domain (e.g., `sip.telnyx.com`)
   - **Authentication**: Use Telnyx API credentials or IP-based auth
3. Note your **Trunk ID** (starts with `ST_`)

## Environment Variables

Create a `.env` file in the `telnyx` directory:

```bash
# LiveKit Configuration
LIVEKIT_URL="wss://your-project.livekit.cloud"
LIVEKIT_API_KEY="your_api_key"
LIVEKIT_API_SECRET="your_api_secret"

# Telnyx SIP Trunk Configuration
# The SIP trunk ID from LiveKit Cloud (configured with Telnyx as the provider)
LIVEKIT_SIP_OUTBOUND_TRUNK="ST_xxxxxxxxxxxxx"

# Phone number for caller ID (your Telnyx phone number)
LIVEKIT_SIP_NUMBER="+1xxxxxxxxxx"

# Optional: Supervisory phone number for transfers
LIVEKIT_SUPERVISOR_PHONE_NUMBER="+1xxxxxxxxxx"

# Model Configuration (using LiveKit Inference)
OPENAI_API_KEY="sk-xxxxxxx"
DEEPGRAM_API_KEY="xxxxxxxxx"
CARTESIA_API_KEY="xxxxxxxxx"
```

## Running the Example

```bash
# From the repository root
uv run examples/telnyx/basic_telnyx_agent.py console
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Caller    │────▶│   LiveKit   │────▶│   LiveKit   │────▶│  Telnyx     │
│   Phone     │     │   Cloud     │     │   Agents    │     │  SIP Trunk  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                            │                                      │
                            │                                      │
                            └─────────────▶ PSTN ◀──────────────────┘
```

## Key Configuration Points

### Telnyx Connection Settings

In Telnyx Mission Control Portal, ensure:

- **SIP Connection**: Enabled with appropriate codecs (G711, opus)
- **IP Authentication**: Add your LiveKit server IPs if using IP-based auth
- **Called Number Routes**: Configure patterns to route calls to your agents
- **Calling Number**: Set your Telnyx phone number as the caller ID

### LiveKit SIP Trunk Settings

In LiveKit Cloud:

- **SIP Domain**: `sip.telnyx.com` (or your dedicated Telnyx SIP domain)
- **Port**: 5060 (UDP/TCP) or 5061 (TLS)
- **Codecs**: G711, opus
- **Authentication**: Match your Telnyx configuration

## Example Features

This example includes:

- Basic telephony agent with voice input/output
- DTMF input support for IVR interactions
- Transfer capability using Telnyx trunk

## Troubleshooting

### Calls Not Connecting

1. Verify Telnyx connection status in Mission Control Portal
2. Check LiveKit SIP trunk configuration
3. Ensure firewall allows SIP traffic (5060/5061)

### One-Way Audio

1. Check NAT settings in Telnyx and LiveKit
2. Verify RTP port ranges are open
3. Ensure STUN/TURN servers are configured

### Authentication Failures

1. Verify SIP credentials match between Telnyx and LiveKit
2. Check IP allowlist if using IP-based auth
3. Review TLS certificates if using secure SIP