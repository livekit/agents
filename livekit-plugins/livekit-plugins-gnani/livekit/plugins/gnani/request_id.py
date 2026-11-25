# Copyright 2025 LiveKit, Inc.
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

"""Request IDs that let a Vachana API call be matched with the plugin call that made it."""

from __future__ import annotations

import uuid

REQUEST_ID_HEADER = "X-API-Request-ID"


def generate_request_id() -> str:
    """A new ID for one outbound API request (one REST call or one WebSocket session)."""
    return f"lk_req_{uuid.uuid4().hex[:12]}"
