# gen_tokens.py  (run:  uv run python gen_tokens.py)
import os
from datetime import timedelta
from dotenv import load_dotenv
from livekit import api

load_dotenv()

ROOM = "translate-test"
TTL = timedelta(minutes=20)

def make(identity: str, language: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_attributes({"language": language})
        .with_grants(api.VideoGrants(
            room_join=True, room=ROOM, can_publish=True, can_subscribe=True,
        ))
        .with_ttl(TTL)
        .to_jwt()
    )

print("A (en):\n" + make("A", "en") + "\n")
print("B (de):\n" + make("B", "de") + "\n")