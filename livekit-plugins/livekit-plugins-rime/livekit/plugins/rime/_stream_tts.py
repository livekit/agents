# Copyright 2026 LiveKit, Inc.
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

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts
from livekit.agents.metrics import TTSMetrics


class StreamTTS(tts.TTS):
    """Keep one stream's metadata fixed and forward its events to the owning TTS.

    The transport adapter owns synthesis and connections. This object only supplies
    the TTS interface used by the base stream for metrics, tracing, and errors.
    """

    def __init__(self, parent: tts.TTS, *, model: str) -> None:
        super().__init__(
            capabilities=parent.capabilities,
            sample_rate=parent.sample_rate,
            num_channels=parent.num_channels,
        )
        self._model = model
        self._provider = parent.provider
        self._label = parent.label

        @self.on("metrics_collected")
        def on_metrics(metrics: TTSMetrics) -> None:
            parent.emit("metrics_collected", metrics)

        @self.on("error")
        def on_error(error: tts.TTSError) -> None:
            parent.emit("error", error)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        raise NotImplementedError("Synthesis is owned by the Rime transport adapter")
