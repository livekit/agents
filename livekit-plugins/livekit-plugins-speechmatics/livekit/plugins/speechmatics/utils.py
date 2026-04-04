from urllib.parse import urlencode, urlparse

from speechmatics.voice import __version__ as sdk_version

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
    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError(f"Invalid base_url: {base_url}")

    if "/" in voice or "\\" in voice:
        raise ValueError(f"Invalid voice: {voice}")

    query_params = {}
    query_params["output_format"] = f"pcm_{sample_rate}"
    query_params["sm-sdk"] = f"livekit-plugins-{lk_version}"
    query_params["sm-app"] = f"livekit/{sdk_version}"
    query = urlencode(query_params)

    return f"{base_url}/generate/{voice}?{query}"
