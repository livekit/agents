from livekit.plugins.openai.realtime.realtime_model import process_base_url


def test_process_base_url():
    assert (
        process_base_url("https://api.openai.com/v1", "gpt-4")
        == "wss://api.openai.com/v1/realtime?model=gpt-4"
    )
    assert (
        process_base_url("http://example.com", "gpt-4") == "ws://example.com/realtime?model=gpt-4"
    )
    assert (  # noqa: F631
        process_base_url(
            "wss://livekit.ai/voice/v1/chat/voice?client=oai&enable_noise_suppression=true",
            "gpt-4",
        )
        == "wss://livekit.ai/voice/v1/chat/voice?client=oai&enable_noise_suppression=true",
    )
    assert (
        process_base_url(
            "https://test.azure.com/openai",
            "gpt-4",
        )
        == "wss://test.azure.com/openai/realtime?model=gpt-4"
    )

    assert (
        process_base_url(
            "https://test.azure.com/openai",
            "gpt-4",
            is_azure=True,
            azure_deployment="my-deployment",
            api_version="2025-04-12",
        )
        == "wss://test.azure.com/openai/realtime?api-version=2025-04-12&deployment=my-deployment"
    )
    assert (
        process_base_url(
            "https://test.azure.com/custom/path",
            "gpt-4",
            api_version="2025-04-12",
        )
        == "wss://test.azure.com/custom/path?model=gpt-4"
    )
