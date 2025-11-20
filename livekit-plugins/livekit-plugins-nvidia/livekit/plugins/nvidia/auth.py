from typing import Optional

import riva.client

from livekit.agents.utils import is_given


def create_riva_auth(
    *,
    api_key: Optional[str],
    function_id: str,
    server: str,
    use_ssl: bool = True,
) -> riva.client.Auth:
    metadata_args = []

    if is_given(api_key) and api_key:
        metadata_args.append(["authorization", f"Bearer {api_key}"])

    metadata_args.append(["function-id", function_id])

    return riva.client.Auth(
        uri=server,
        use_ssl=use_ssl,
        metadata_args=metadata_args,
    )
