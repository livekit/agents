from postgrest import APIError as PostgrestAPIError
from postgrest import APIResponse as PostgrestAPIResponse
from storage3.utils import StorageException

from .__version__ import __version__
from ._sync.auth_client import SyncSupabaseAuthClient as SupabaseAuthClient
from ._sync.client import ClientOptions
from ._sync.client import SyncClient as Client
from ._sync.client import SyncStorageClient as SupabaseStorageClient
from ._sync.client import create_client
from .lib.realtime_client import SupabaseRealtimeClient

__all__ = [
    "PostgrestAPIError",
    "PostgrestAPIResponse",
    "StorageException",
    "SupabaseAuthClient",
    "__version__",
    "create_client",
    "Client",
    "ClientOptions",
    "SupabaseStorageClient",
    "SupabaseRealtimeClient",
]
