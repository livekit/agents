import importlib.metadata
import os

import aiohttp


async def get_access_token(api_key: str) -> str:
    mp_api_url = os.getenv("SPEECHMATICS_MANAGEMENT_PLATFORM_URL", "https://mp.speechmatics.com")
    endpoint = f"{mp_api_url}/v1/api_keys"
    params = {"type": "rt", "sm-sdk": get_sdk_version()}
    json_body = {"ttl": 60}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, params=params, json=json_body, headers=headers) as resp:
            if resp.status == 201:
                try:
                    data = await resp.json()
                    return data["key_value"]
                except (ValueError, KeyError) as e:
                    raise Exception(f"Failed to parse Speechmatics access token response: {e}")  # noqa: B904
            else:
                error_message = await resp.text()
                raise Exception(
                    f"Failed to get Speechmatics access token. "
                    f"Status: {resp.status}, Error: {error_message}"
                )


def get_sdk_version():
    version = importlib.metadata.version("livekit-plugins-speechmatics")
    return f"livekit-plugins-{version}"


def sanitize_url(url, language):
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    parsed_url = urlparse(url)

    query_params = dict(parse_qsl(parsed_url.query))
    query_params["sm-sdk"] = get_sdk_version()
    updated_query = urlencode(query_params)

    url_path = parsed_url.path
    if not url_path.endswith(language):
        if url_path.endswith("/"):
            url_path += language
        else:
            url_path += f"/{language}"

    return urlunparse(parsed_url._replace(path=url_path, query=updated_query))
