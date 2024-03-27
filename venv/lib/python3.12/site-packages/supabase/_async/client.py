import re
from typing import Any, Dict, Optional, Union

from gotrue import AsyncMemoryStorage
from gotrue.types import AuthChangeEvent, Session
from httpx import Timeout
from postgrest import (
    AsyncPostgrestClient,
    AsyncRequestBuilder,
    AsyncRPCFilterRequestBuilder,
)
from postgrest.constants import DEFAULT_POSTGREST_CLIENT_TIMEOUT
from storage3 import AsyncStorageClient
from storage3.constants import DEFAULT_TIMEOUT as DEFAULT_STORAGE_CLIENT_TIMEOUT
from supafunc import AsyncFunctionsClient

from ..lib.client_options import ClientOptions
from .auth_client import AsyncSupabaseAuthClient


# Create an exception class when user does not provide a valid url or key.
class SupabaseException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class AsyncClient:
    """Supabase client class."""

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        options: ClientOptions = ClientOptions(storage=AsyncMemoryStorage()),
    ):
        """Instantiate the client.

        Parameters
        ----------
        supabase_url: str
            The URL to the Supabase instance that should be connected to.
        supabase_key: str
            The API key to the Supabase instance that should be connected to.
        **options
            Any extra settings to be optionally specified - also see the
            `DEFAULT_OPTIONS` dict.
        """

        if not supabase_url:
            raise SupabaseException("supabase_url is required")
        if not supabase_key:
            raise SupabaseException("supabase_key is required")

        # Check if the url and key are valid
        if not re.match(r"^(https?)://.+", supabase_url):
            raise SupabaseException("Invalid URL")

        # Check if the key is a valid JWT
        if not re.match(
            r"^[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*$", supabase_key
        ):
            raise SupabaseException("Invalid API key")

        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self._auth_token = {
            "Authorization": f"Bearer {supabase_key}",
        }
        options.headers.update(self._get_auth_headers())
        self.options = options
        self.rest_url = f"{supabase_url}/rest/v1"
        self.realtime_url = f"{supabase_url}/realtime/v1".replace("http", "ws")
        self.auth_url = f"{supabase_url}/auth/v1"
        self.storage_url = f"{supabase_url}/storage/v1"
        self.functions_url = f"{supabase_url}/functions/v1"
        self.schema = options.schema

        # Instantiate clients.
        self.auth = self._init_supabase_auth_client(
            auth_url=self.auth_url,
            client_options=options,
        )
        # TODO: Bring up to parity with JS client.
        # self.realtime: SupabaseRealtimeClient = self._init_realtime_client(
        #     realtime_url=self.realtime_url,
        #     supabase_key=self.supabase_key,
        # )
        self.realtime = None
        self._postgrest = None
        self._storage = None
        self._functions = None
        self.auth.on_auth_state_change(self._listen_to_auth_events)

    @classmethod
    async def create(
        cls,
        supabase_url: str,
        supabase_key: str,
        options: ClientOptions = ClientOptions(),
    ):
        client = cls(supabase_url, supabase_key, options)
        client._auth_token = await client._get_token_header()
        return client

    def table(self, table_name: str) -> AsyncRequestBuilder:
        """Perform a table operation.

        Note that the supabase client uses the `from` method, but in Python,
        this is a reserved keyword, so we have elected to use the name `table`.
        Alternatively you can use the `.from_()` method.
        """
        return self.from_(table_name)

    def from_(self, table_name: str) -> AsyncRequestBuilder:
        """Perform a table operation.

        See the `table` method.
        """
        return self.postgrest.from_(table_name)

    def rpc(
        self, fn: str, params: Optional[Dict[Any, Any]] = None
    ) -> AsyncRPCFilterRequestBuilder:
        """Performs a stored procedure call.

        Parameters
        ----------
        fn : callable
            The stored procedure call to be executed.
        params : dict of any
            Parameters passed into the stored procedure call.

        Returns
        -------
        SyncFilterRequestBuilder
            Returns a filter builder. This lets you apply filters on the response
            of an RPC.
        """
        if params is None:
            params = {}
        return self.postgrest.rpc(fn, params)

    @property
    def postgrest(self):
        if self._postgrest is None:
            self.options.headers.update(self._auth_token)
            self._postgrest = self._init_postgrest_client(
                rest_url=self.rest_url,
                headers=self.options.headers,
                schema=self.options.schema,
                timeout=self.options.postgrest_client_timeout,
            )

        return self._postgrest

    @property
    def storage(self):
        if self._storage is None:
            headers = self._get_auth_headers()
            headers.update(self._auth_token)
            self._storage = self._init_storage_client(
                storage_url=self.storage_url,
                headers=headers,
                storage_client_timeout=self.options.storage_client_timeout,
            )
        return self._storage

    @property
    def functions(self):
        if self._functions is None:
            headers = self._get_auth_headers()
            headers.update(self._auth_token)
            self._functions = AsyncFunctionsClient(self.functions_url, headers)
        return self._functions

    #     async def remove_subscription_helper(resolve):
    #         try:
    #             await self._close_subscription(subscription)
    #             open_subscriptions = len(self.get_subscriptions())
    #             if not open_subscriptions:
    #                 error = await self.realtime.disconnect()
    #                 if error:
    #                     return {"error": None, "data": { open_subscriptions}}
    #         except Exception as e:
    #             raise e
    #     return remove_subscription_helper(subscription)

    # async def _close_subscription(self, subscription):
    #    """Close a given subscription

    #    Parameters
    #    ----------
    #    subscription
    #        The name of the channel
    #    """
    #    if not subscription.closed:
    #        await self._closeChannel(subscription)

    # def get_subscriptions(self):
    #     """Return all channels the client is subscribed to."""
    #     return self.realtime.channels

    # @staticmethod
    # def _init_realtime_client(
    #     realtime_url: str, supabase_key: str
    # ) -> SupabaseRealtimeClient:
    #     """Private method for creating an instance of the realtime-py client."""
    #     return SupabaseRealtimeClient(
    #         realtime_url, {"params": {"apikey": supabase_key}}
    #     )
    @staticmethod
    def _init_storage_client(
        storage_url: str,
        headers: Dict[str, str],
        storage_client_timeout: int = DEFAULT_STORAGE_CLIENT_TIMEOUT,
    ) -> AsyncStorageClient:
        return AsyncStorageClient(storage_url, headers, storage_client_timeout)

    @staticmethod
    def _init_supabase_auth_client(
        auth_url: str,
        client_options: ClientOptions,
    ) -> AsyncSupabaseAuthClient:
        """Creates a wrapped instance of the GoTrue Client."""
        return AsyncSupabaseAuthClient(
            url=auth_url,
            auto_refresh_token=client_options.auto_refresh_token,
            persist_session=client_options.persist_session,
            storage=client_options.storage,
            headers=client_options.headers,
            flow_type=client_options.flow_type,
        )

    @staticmethod
    def _init_postgrest_client(
        rest_url: str,
        headers: Dict[str, str],
        schema: str,
        timeout: Union[int, float, Timeout] = DEFAULT_POSTGREST_CLIENT_TIMEOUT,
    ) -> AsyncPostgrestClient:
        """Private helper for creating an instance of the Postgrest client."""
        return AsyncPostgrestClient(
            rest_url, headers=headers, schema=schema, timeout=timeout
        )

    def _create_auth_header(self, token: str):
        return {
            "Authorization": f"Bearer {token}",
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        """Helper method to get auth headers."""
        return {
            "apiKey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
        }

    async def _get_token_header(self):
        try:
            session = await self.auth.get_session()
            access_token = session.access_token
        except Exception as err:
            access_token = self.supabase_key

        return self._create_auth_header(access_token)

    def _listen_to_auth_events(
        self, event: AuthChangeEvent, session: Union[Session, None]
    ):
        access_token = self.supabase_key
        if event in ["SIGNED_IN", "TOKEN_REFRESHED", "SIGNED_OUT"]:
            # reset postgrest and storage instance on event change
            self._postgrest = None
            self._storage = None
            self._functions = None
            access_token = session.access_token if session else self.supabase_key

        self._auth_token = self._create_auth_header(access_token)


async def create_client(
    supabase_url: str,
    supabase_key: str,
    options: ClientOptions = ClientOptions(storage=AsyncMemoryStorage()),
) -> AsyncClient:
    """Create client function to instantiate supabase client like JS runtime.

    Parameters
    ----------
    supabase_url: str
        The URL to the Supabase instance that should be connected to.
    supabase_key: str
        The API key to the Supabase instance that should be connected to.
    **options
        Any extra settings to be optionally specified - also see the
        `DEFAULT_OPTIONS` dict.

    Examples
    --------
    Instantiating the client.
    >>> import os
    >>> from supabase import create_client, Client
    >>>
    >>> url: str = os.environ.get("SUPABASE_TEST_URL")
    >>> key: str = os.environ.get("SUPABASE_TEST_KEY")
    >>> supabase: Client = create_client(url, key)

    Returns
    -------
    Client
    """
    return await AsyncClient.create(
        supabase_url=supabase_url, supabase_key=supabase_key, options=options
    )
