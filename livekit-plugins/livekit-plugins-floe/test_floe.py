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

import pytest

from livekit.agents import Plugin
from livekit.plugins import floe

pytestmark = pytest.mark.unit


def test_exports() -> None:
    assert floe.LLM is not None
    assert floe.FloeUsageReconciler is not None
    assert set(floe.__all__) == {"LLM", "FloeUsageReconciler", "__version__"}


def test_plugin_registered() -> None:
    names = [type(p).__name__ for p in Plugin.registered_plugins]
    assert "FloePlugin" in names
