# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Type, TypeVar

import aiohttp
from google.protobuf.message import Message
from urllib.parse import urlparse

DEFAULT_PREFIX = "twirp"


class TwirpError(Exception):
    def __init__(self, code: str, msg: str) -> None:
        self._code = code
        self._msg = msg

    @property
    def code(self) -> str:
        return self._code

    @property
    def message(self) -> str:
        return self._msg


class TwirpErrorCode:
    CANCELED = "canceled"
    UNKNOWN = "unknown"
    INVALID_ARGUMENT = "invalid_argument"
    MALFORMED = "malformed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    NOT_FOUND = "not_found"
    BAD_ROUTE = "bad_route"
    ALREADY_EXISTS = "already_exists"
    PERMISSION_DENIED = "permission_denied"
    UNAUTHENTICATED = "unauthenticated"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    FAILED_PRECONDITION = "failed_precondition"
    ABORTED = "aborted"
    OUT_OF_RANGE = "out_of_range"
    UNIMPLEMENTED = "unimplemented"
    INTERNAL = "internal"
    UNAVAILABLE = "unavailable"
    DATA_LOSS = "dataloss"


T = TypeVar("T", bound=Message)


class TwirpClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        host: str,
        pkg: str,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        parse_res = urlparse(host)
        scheme = parse_res.scheme
        if scheme.startswith("ws"):
            scheme = scheme.replace("ws", "http")

        host = f"{scheme}://{parse_res.netloc}/{parse_res.path}"
        self.host = host.rstrip("/")
        self.pkg = pkg
        self.prefix = prefix
        self._session = session

    async def request(
        self,
        service: str,
        method: str,
        data: Message,
        headers: Dict[str, str],
        response_class: Type[T],
    ) -> T:
        url = f"{self.host}/{self.prefix}/{self.pkg}.{service}/{method}"
        headers["Content-Type"] = "application/protobuf"

        serialized_data = data.SerializeToString()
        async with self._session.post(
            url, headers=headers, data=serialized_data
        ) as resp:
            if resp.status == 200:
                return response_class.FromString(await resp.read())
            else:
                # when we have an error, Twirp always encode it in json
                error_data = await resp.json()
                raise TwirpError(error_data["code"], error_data["msg"])
