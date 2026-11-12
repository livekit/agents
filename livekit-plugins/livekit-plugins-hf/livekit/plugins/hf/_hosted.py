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

"""Hosted inference through Hugging Face Inference Providers or a dedicated endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, TypeVar

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError

T = TypeVar("T")

# a hosted call covers queueing and synthesis of a whole clip, not just a connection, so
# it gets more time than APIConnectOptions.timeout (10 s by default)
MIN_REQUEST_TIMEOUT = 60.0


@dataclass
class HostedOptions:
    provider: str | None
    base_url: str | None
    api_key: str | None
    extra_body: dict[str, Any]

    @property
    def enabled(self) -> bool:
        return self.provider is not None or self.base_url is not None


def make_client(opts: HostedOptions) -> Any:
    from huggingface_hub import AsyncInferenceClient

    # base_url is the endpoint itself, so the model id is only sent to providers
    return AsyncInferenceClient(
        provider=opts.provider if opts.base_url is None else None,  # type: ignore[arg-type]
        base_url=opts.base_url,
        api_key=opts.api_key,
    )


def request_timeout(conn_timeout: float) -> float:
    return max(conn_timeout, MIN_REQUEST_TIMEOUT)


def request_model(opts: HostedOptions, model: str) -> str | None:
    return None if opts.base_url is not None else model


async def call(awaitable: Awaitable[T], *, what: str) -> T:
    """Await a client call, mapping Hugging Face errors to LiveKit's retry semantics."""
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return await awaitable
    except (TimeoutError, asyncio.TimeoutError) as e:  # InferenceTimeoutError too
        raise APITimeoutError(f"{what} timed out") from e
    except HfHubHTTPError as e:
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            raise APIConnectionError(f"{what} failed: {e}") from e
        # APIStatusError keeps 4xx non-retryable except transient codes such as 429
        raise APIStatusError(f"{what} failed: {e}", status_code=int(status), body=str(e)) from e
    except (APIConnectionError, APIStatusError):
        raise
    except Exception as e:
        raise APIConnectionError(f"{what} failed: {type(e).__name__}: {e}") from e


def sniff_audio_mime(data: bytes) -> str:
    """Providers return encoded audio in different containers; name it for the decoder."""
    if data[:4] == b"RIFF":
        return "audio/wav"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "audio/mpeg"
    return "audio/wav"
