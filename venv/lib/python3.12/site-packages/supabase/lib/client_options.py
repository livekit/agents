from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from gotrue import AuthFlowType, SyncMemoryStorage, SyncSupportedStorage
from httpx import Timeout
from postgrest.constants import DEFAULT_POSTGREST_CLIENT_TIMEOUT
from storage3.constants import DEFAULT_TIMEOUT as DEFAULT_STORAGE_CLIENT_TIMEOUT

from supabase import __version__

DEFAULT_HEADERS = {"X-Client-Info": f"supabase-py/{__version__}"}


@dataclass
class ClientOptions:
    schema: str = "public"
    """
    The Postgres schema which your tables belong to.
    Must be on the list of exposed schemas in Supabase. Defaults to 'public'.
    """

    headers: Dict[str, str] = field(default_factory=DEFAULT_HEADERS.copy)
    """Optional headers for initializing the client."""

    auto_refresh_token: bool = True
    """Automatically refreshes the token for logged in users."""

    persist_session: bool = True
    """Whether to persist a logged in session to storage."""

    storage: SyncSupportedStorage = field(default_factory=SyncMemoryStorage)
    """A storage provider. Used to store the logged in session."""

    realtime: Optional[Dict[str, Any]] = None
    """Options passed to the realtime-py instance"""

    postgrest_client_timeout: Union[
        int, float, Timeout
    ] = DEFAULT_POSTGREST_CLIENT_TIMEOUT
    """Timeout passed to the SyncPostgrestClient instance."""

    storage_client_timeout: Union[int, float, Timeout] = DEFAULT_STORAGE_CLIENT_TIMEOUT
    """Timeout passed to the SyncStorageClient instance"""

    flow_type: AuthFlowType = "implicit"
    """flow type to use for authentication"""

    def replace(
        self,
        schema: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        auto_refresh_token: Optional[bool] = None,
        persist_session: Optional[bool] = None,
        storage: Optional[SyncSupportedStorage] = None,
        realtime: Optional[Dict[str, Any]] = None,
        postgrest_client_timeout: Union[
            int, float, Timeout
        ] = DEFAULT_POSTGREST_CLIENT_TIMEOUT,
        storage_client_timeout: Union[
            int, float, Timeout
        ] = DEFAULT_STORAGE_CLIENT_TIMEOUT,
        flow_type: Optional[AuthFlowType] = None,
    ) -> "ClientOptions":
        """Create a new SupabaseClientOptions with changes"""
        client_options = ClientOptions()
        client_options.schema = schema or self.schema
        client_options.headers = headers or self.headers
        client_options.auto_refresh_token = (
            auto_refresh_token or self.auto_refresh_token
        )
        client_options.persist_session = persist_session or self.persist_session
        client_options.storage = storage or self.storage
        client_options.realtime = realtime or self.realtime
        client_options.postgrest_client_timeout = (
            postgrest_client_timeout or self.postgrest_client_timeout
        )
        client_options.storage_client_timeout = (
            storage_client_timeout or self.storage_client_timeout
        )
        client_options.flow_type = flow_type or self.flow_type
        return client_options
