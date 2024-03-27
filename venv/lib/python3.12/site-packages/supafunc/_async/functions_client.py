from typing import Any, Dict, Literal, Optional, Union

from httpx import HTTPError, Response

from ..errors import FunctionsHttpError, FunctionsRelayError
from ..utils import AsyncClient, __version__


class AsyncFunctionsClient:
    def __init__(self, url: str, headers: Dict):
        self.url = url
        self.headers = {
            "User-Agent": f"supabase-py/functions-py v{__version__}",
            **headers,
        }
        self._client = AsyncClient(base_url=self.url, headers=self.headers)

    async def _request(
        self,
        method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        url: str,
        headers: Union[Dict[str, str], None] = None,
        json: Optional[Dict[Any, Any]] = None,
    ) -> Response:
        response = await self._client.request(method, url, json=json, headers=headers)
        try:
            response.raise_for_status()
        except HTTPError as exc:
            raise FunctionsHttpError(
                response.json().get("error")
                or f"An error occurred while requesting your edge function at {exc.request.url!r}."
            ) from exc

        return response

    def set_auth(self, token: str) -> None:
        """Updates the authorization header

        Parameters
        ----------
        token : str
            the new jwt token sent in the authorization header
        """

        self.headers["Authorization"] = f"Bearer {token}"

    async def invoke(
        self, function_name: str, invoke_options: Optional[Dict] = None
    ) -> Union[Dict, bytes]:
        """Invokes a function

        Parameters
        ----------
        function_name : the name of the function to invoke
        invoke_options : object with the following properties
            `headers`: object representing the headers to send with the request
            `body`: the body of the request
            `responseType`: how the response should be parsed. The default is `json`
        """
        headers = self.headers
        if invoke_options is not None:
            headers.update(invoke_options.get("headers", {}))

        body = invoke_options.get("body") if invoke_options else None
        response_type = (
            invoke_options.get("responseType") if invoke_options else "text/plain"
        )

        if type(body) == str:
            headers["Content-Type"] = "text/plain"
        elif type(body) == dict:
            headers["Content-Type"] = "application/json"

        response = await self._request(
            "POST", f"{self.url}/{function_name}", headers=headers, json=body
        )
        is_relay_error = response.headers.get("x-relay-header")

        if is_relay_error and is_relay_error == "true":
            raise FunctionsRelayError(response.json().get("error"))

        if response_type == "json":
            data = response.json()
        else:
            data = response.content
        return data
