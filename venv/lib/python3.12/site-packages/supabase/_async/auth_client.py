from typing import Dict, Optional

from gotrue import (
    AsyncGoTrueClient,
    AsyncMemoryStorage,
    AsyncSupportedStorage,
    AuthFlowType,
)
from gotrue.http_clients import AsyncClient


class AsyncSupabaseAuthClient(AsyncGoTrueClient):
    """SupabaseAuthClient"""

    def __init__(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        storage_key: Optional[str] = None,
        auto_refresh_token: bool = True,
        persist_session: bool = True,
        storage: AsyncSupportedStorage = AsyncMemoryStorage(),
        http_client: Optional[AsyncClient] = None,
        flow_type: AuthFlowType = "implicit"
    ):
        """Instantiate SupabaseAuthClient instance."""
        if headers is None:
            headers = {}

        AsyncGoTrueClient.__init__(
            self,
            url=url,
            headers=headers,
            storage_key=storage_key,
            auto_refresh_token=auto_refresh_token,
            persist_session=persist_session,
            storage=storage,
            http_client=http_client,
            flow_type=flow_type,
        )
