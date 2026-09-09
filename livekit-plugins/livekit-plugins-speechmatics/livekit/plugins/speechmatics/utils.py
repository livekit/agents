from urllib.parse import urlencode

from speechmatics.agent_stt import __version__ as sdk_version

from .version import __version__ as lk_version


def get_tts_url(base_url: str, voice: str, sample_rate: int) -> str:
    """Format the TTS endpoint URL with voice, output format, and version params.

    Args:
        base_url: The base URL for the TTS endpoint.
        voice: The voice model to use.
        sample_rate: The audio sample rate.

    Returns:
        str: The formatted TTS endpoint URL.
    """
    query_params = {}
    query_params["output_format"] = f"pcm_{sample_rate}"
    # `sm-sdk` names the SDK, `sm-app` the application built on it. On the STT path the
    # SDK reports itself; this endpoint is a direct WebSocket, so it is set by hand.
    query_params["sm-sdk"] = f"python-agent-stt-sdk-v{sdk_version}"
    query_params["sm-app"] = f"livekit-plugins-{lk_version}"
    query = urlencode(query_params)

    return f"{base_url}/generate/{voice}?{query}"
