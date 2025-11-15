import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli

logger = logging.getLogger("e2ee-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    e2ee_config = rtc.E2EEOptions(
        key_provider_options=rtc.KeyProviderOptions(
            shared_key=b"my_shared_key",
            # ratchet_salt=b"my_salt",
        ),
        encryption_type=rtc.EncryptionType.GCM,
    )

    # Connect to the room with end-to-end encryption (E2EE)
    # Only clients possessing the same shared key will be able to decode the published tracks
    await ctx.connect(e2ee=e2ee_config)


if __name__ == "__main__":
    cli.run_app(server)
