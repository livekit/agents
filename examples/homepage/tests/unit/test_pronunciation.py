import pytest
from filters.pronunciation import LIVEKIT_PHONEMES, LIVEKITS_PHONEMES, pronounce_livekit

pytestmark = pytest.mark.unit


async def _speak(*chunks: str) -> str:
    async def gen():
        for chunk in chunks:
            yield chunk

    return "".join([piece async for piece in pronounce_livekit(gen())])


@pytest.mark.asyncio
async def test_pronounce_livekit() -> None:
    assert await _speak("I love LiveKit.") == f"I love {LIVEKIT_PHONEMES}."
    assert await _speak("livekit is great") == f"{LIVEKIT_PHONEMES} is great"
    assert await _speak("Live", "Kit rocks") == f"{LIVEKIT_PHONEMES} rocks"
    assert await _speak("LiveKit's SDK") == f"{LIVEKITS_PHONEMES} SDK"
    assert await _speak("LiveKit’s SDK") == f"{LIVEKITS_PHONEMES} SDK"
    assert await _speak("LiveKitten is not a product") == "LiveKitten is not a product"
