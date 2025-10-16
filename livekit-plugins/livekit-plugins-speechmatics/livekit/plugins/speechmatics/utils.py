from urllib.parse import urlencode

from speechmatics.rt import __version__ as sdk_version

from .version import __version__ as lk_version


def get_endpoint_url(url: str) -> str:
    """Format the endpoint URL with the SDK and app versions.

    Args:
        url: The base URL for the endpoint.

    Returns:
        str: The formatted endpoint URL.
    """
    query_params = {}
    query_params["sm-sdk"] = f"livekit-plugins-{lk_version}"
    query_params["sm-app"] = f"livekit/{sdk_version}"
    query = urlencode(query_params)

    return f"{url}?{query}"
