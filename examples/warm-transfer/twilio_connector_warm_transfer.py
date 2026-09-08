import os

from dotenv import load_dotenv
from support_agent import SUMMARY_INSTRUCTIONS, SupportAgent, run

from livekit.agents.beta.workflows import TwilioConnectorWarmTransferTask, WarmTransferResult

load_dotenv()

# LiveKit credentials (read from env by the SDK): LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
# Twilio REST credentials + caller ID:
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")  # "ACxxxx..."
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")  # "xxxx..."
TWILIO_FROM_NUMBER = os.getenv(
    "TWILIO_FROM_NUMBER"
)  # "+15005006000" - your Twilio number, shown to supervisor
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
# This example places the outbound call with the Twilio REST SDK (pip install twilio):
# connect_twilio_call returns a connect_url that we hand to Twilio in <Stream url=...>
# TwiML; Twilio then streams the supervisor's call audio back to the connector.


class TwilioSupportAgent(SupportAgent):
    async def _start_transfer(self) -> WarmTransferResult:
        assert SUPERVISOR_PHONE_NUMBER is not None
        assert TWILIO_FROM_NUMBER is not None
        return await TwilioConnectorWarmTransferTask(
            SUPERVISOR_PHONE_NUMBER,
            twilio_from_number=TWILIO_FROM_NUMBER,
            twilio_account_sid=TWILIO_ACCOUNT_SID,
            twilio_auth_token=TWILIO_AUTH_TOKEN,
            chat_ctx=self.chat_ctx,
            # give up if the supervisor doesn't pick up within 25s:
            # ringing_timeout=25,
            # add extra instructions for summarization
            extra_instructions=SUMMARY_INSTRUCTIONS,
        )


if __name__ == "__main__":
    run(TwilioSupportAgent)
