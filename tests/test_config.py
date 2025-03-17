from livekit.plugins.openai.realtime.realtime_model import process_base_url


def test_process_base_url():
    assert (
        process_base_url("https://api.openai.com/v1", "gpt-4")
        == "wss://api.openai.com/v1/realtime?model=gpt-4"
    )
    assert (
        process_base_url("http://example.com", "gpt-4") == "ws://example.com/realtime?model=gpt-4"
    )
    assert (
        process_base_url(
            "wss://livekit.ai/voice/v1/chat/voice?client=oai&enable_noise_suppression=true",
            "gpt-4",
        )
        == "wss://livekit.ai/voice/v1/chat/voice?client=oai&enable_noise_suppression=true&model=gpt-4"
    )
