# SMS Tools

Send SMS messages from voice agents with auto-detected provider support.
Supported: Twilio, Vonage, SignalWire

## Usage

```python
from livekit.agents import Agent
from livekit.agents.beta.tools.sms import create_sms_tool, SMSToolConfig

# Default: auto-detects the caller's number from the SIP participant
# LLM only needs to provide the message text
sms_tool = create_sms_tool()

# Custom: change tool name, description, recipient, or sender
sms_tool = create_sms_tool(SMSToolConfig(
    name="notify_customer",           # Custom function name
    description="Send confirmation",  # Custom function description
    to_number="+1234567890",          # Send to specific number
    from_number="+0987654321",        # Send from specific number
    auto_detect_caller=False,         # Require explicit "to" argument from the LLM
    execution_message="Sending a confirmation text right now.",
))

agent = Agent(
    instructions="You can send SMS messages to callers",
    tools=[sms_tool]
)
```

### Arguments surfaced to the LLM

`create_sms_tool` keeps the schema intentionally tiny:

- `message` — always required; this is the SMS body the LLM must supply.
- `to` — required **only** when you disable caller auto-detection _and_ don’t hardcode `to_number`. In every other configuration the tool determines the recipient on its own, so the LLM never sees or fills this field.

With `auto_detect_caller=True` (default) the tool looks up the SIP participant identity and omits the `to` argument entirely.

### Announcing executions

Set `SMSToolConfig.execution_message` when you want the agent to speak a short disclaimer before dispatching an SMS. The message is sent through `context.session.generate_reply()` with interruptions disabled so the user hears it exactly once.

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

If you supply `from_number` directly in `SMSToolConfig` it overrides the values above.
