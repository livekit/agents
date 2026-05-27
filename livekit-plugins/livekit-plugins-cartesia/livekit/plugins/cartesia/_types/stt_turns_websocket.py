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

from __future__ import annotations

from typing import Literal

from typing_extensions import NotRequired, TypedDict


class STTConnectedEvent(TypedDict):
    """Fires once when the WebSocket connection is established.

    You do not need to wait for this event before sending audio.

    Attributes:
        type: Event discriminator.
        request_id: Unique identifier for this connection. Does not change between turns.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["connected"]
    request_id: str


class STTTurnStartEvent(TypedDict):
    """Model predicts the start of a user turn.

    Attributes:
        type: Event discriminator.
        request_id: Unique identifier for this connection. Does not change between turns.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["turn.start"]
    request_id: str


class STTTurnUpdateEvent(TypedDict):
    """Fires repeatedly as the model transcribes the current user turn.

    Used for interim transcript events.

    Attributes:
        type: Event discriminator.
        transcript: Cumulative text for the current turn, i.e. the full text transcribed
            so far in this turn, not a delta.
        request_id: Unique identifier for this connection. Does not change between turns.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["turn.update"]
    transcript: str
    request_id: str


class STTTurnEagerEndEvent(TypedDict):
    """Fires when the model predicts the user might be done speaking.

    Used for preflight transcript events.

    Attributes:
        type: Event discriminator.
        transcript: Cumulative text for the current turn, i.e. the full text transcribed
            so far in this turn, not a delta.
        request_id: Unique identifier for this connection. Does not change between turns.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["turn.eager_end"]
    transcript: str
    request_id: str


class STTTurnResumeEvent(TypedDict):
    """Fires after ``turn.eager_end`` if the user turn has not actually ended.

    Attributes:
        type: Event discriminator.
        request_id: Unique identifier for this connection. Does not change between turns.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["turn.resume"]
    request_id: str


class STTTurnEndEvent(TypedDict):
    """Marks the end of a user turn.

    Used for end-of-speech and final transcript events.

    Attributes:
        type: Event discriminator.
        transcript: Cumulative text for the current turn, i.e. the full text transcribed
            so far in this turn, not a delta.
        request_id: Unique identifier for this connection. Does not change between turns.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["turn.end"]
    transcript: str
    request_id: str


class STTErrorEvent(TypedDict):
    """Error event sent by the server.

    Attributes:
        type: Event discriminator.
        error_code: Stable code identifying the error.
        status_code: HTTP-style status code; values >= 500 are treated as retryable.
        title: Short human-readable error title.
        message: Detailed human-readable error message.
        doc_url: URL to documentation describing this error.
        request_id: Unique identifier for this connection. Does not change between turns.

    See also:
        https://docs.cartesia.ai/api-reference/stt/turns/websocket
    """

    type: Literal["error"]
    error_code: NotRequired[str]
    status_code: NotRequired[int]
    title: NotRequired[str]
    message: NotRequired[str]
    doc_url: NotRequired[str]
    request_id: NotRequired[str]


STTEventMessage = (
    STTConnectedEvent
    | STTTurnStartEvent
    | STTTurnUpdateEvent
    | STTTurnEagerEndEvent
    | STTTurnResumeEvent
    | STTTurnEndEvent
    | STTErrorEvent
)
"""Server-sent message on the ``/stt/turns/websocket`` endpoint.

See also:
    https://docs.cartesia.ai/api-reference/stt/turns/websocket
"""
