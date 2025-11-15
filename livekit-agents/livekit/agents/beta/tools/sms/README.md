# SMS Tools

Send SMS messages from voice agents with auto-detected provider support.
Supported: Twilio, Vonage, SignalWire

## Usage

```python
from livekit.agents import Agent
from livekit.agents.beta.tools.sms import create_sms_tool, SMSToolConfig

# Default: auto-detects recipient from SIP participant
# LLM only needs to provide the message text
sms_tool = create_sms_tool()

# Custom: change tool name, description, recipient or sender
sms_tool = create_sms_tool(SMSToolConfig(
    name="notify_customer",           # Custom function name
    description="Send confirmation",  # Custom function description
    to_number="+1234567890",          # Send to specific number
    from_number="+0987654321",        # Send from specific number
))

agent = Agent(
    instructions="You can send SMS messages to callers",
    tools=[sms_tool]
)
```

## Environment Variables

**Twilio:**

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
```

**SignalWire:**

```bash
SIGNALWIRE_PROJECT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
SIGNALWIRE_TOKEN=PTxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SIGNALWIRE_SPACE_URL=example.signalwire.com
SIGNALWIRE_PHONE_NUMBER=+1234567890
```

**Vonage:**

```bash
VONAGE_API_KEY=your_api_key
VONAGE_API_SECRET=your_api_secret
VONAGE_PHONE_NUMBER=1234567890
```
