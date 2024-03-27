from typing import Dict, Optional

from gotrue import (
    AuthFlowType,
    SyncGoTrueClient,
    SyncMemoryStorage,
    SyncSupportedStorage,
)
from gotrue.http_clients import SyncClient


class SyncSupabaseAuthClient(SyncGoTrueClient):
    """SupabaseAuthClient"""

    def __init__(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        storage_key: Optional[str] = None,
        auto_refresh_token: bool = True,
        persist_session: bool = True,
        storage: SyncSupportedStorage = SyncMemoryStorage(),
        http_client: Optional[SyncClient] = None,
        flow_type: AuthFlowType = "implicit"
    ):
        """Instantiate SupabaseAuthClient instance."""
        if headers is None:
            headers = {}

        SyncGoTrueClient.__init__(
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
