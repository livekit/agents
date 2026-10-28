# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/telemetry/attributes.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Final

ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID: Final = "lk.segment_id"

ATTRIBUTE_TRANSCRIPTION_TRACK_ID: Final = "lk.transcribed_track_id"

ATTRIBUTE_TRANSCRIPTION_FINAL: Final = "lk.transcription_final"

ATTRIBUTE_TRANSCRIPTION_EXPRESSION: Final = "lk.expression"
"""The expression (delivery/emotion) the agent used for a transcription segment, surfaced so
the frontend can react to it, when expressive markup is stripped from the transcript. The
value is a JSON object ``{"value": ...}`` carrying the segment's leading expression -- the
``<expression>`` tag for Inworld/ElevenLabs v3 or the ``<emotion>`` tag for Cartesia, e.g.
``{"value": "speak happy"}``. A JSON object (rather than a bare string) so the shape can
gain fields later without breaking parsers.
"""

ATTRIBUTE_PUBLISH_ON_BEHALF: Final = "lk.publish_on_behalf"
"""The identity of the agent participant that an avatar worker is publishing on behalf of."""

ATTRIBUTE_AGENT_STATE: Final = "lk.agent.state"
"""The state of the agent, stored in the agent's attributes.
This can be retrieved on the client side by using `RemoteParticipant.attributes`.

With components-js, this can be easily retrieved using:

```js
const { state, ... } = useVoiceAssistant();
```
"""

ATTRIBUTE_AGENT_NAME: Final = "lk.agent.name"
"""The name of the agent, stored in the agent's attributes.
This is set when the agent joins a room and can be used to identify the agent type.
"""

ATTRIBUTE_SIMULATOR: Final = "lk.simulator"
"""Indicates that the participant is a simulator for testing purposes.
When set to "true", the agent will skip audio input/output processing.
"""

ATTRIBUTE_SIMULATOR_DISPATCH: Final = "lk.simulator.dispatch"
"""The job attribute carrying the run's protojson ``SimulationDispatch``,
delivered with the agent dispatch and read by
``JobContext.simulation_context()``.
"""

ATTRIBUTE_SIMULATION_ENABLED: Final = "lk.simulation.enabled"
"""Telemetry metadata key marking the session as a simulation."""

ATTRIBUTE_SIMULATION_RUN_ID: Final = "lk.simulation.run_id"
"""Telemetry metadata key naming the simulation run this session belongs to.

Sits alongside :data:`ATTRIBUTE_SIMULATION_ENABLED`: the boolean says a session is
synthetic, this says which run it came from, so a run's sessions can be aggregated
without resolving each one's room first.
"""

ATTRIBUTE_SIMULATION_JOB_ID: Final = "lk.simulation.job_id"
"""Telemetry metadata key naming the simulation run job (one scenario) behind the
session.
"""

ATTRIBUTE_REDACTION_ENABLED: Final = "lk.redaction.enabled"
"""Telemetry metadata key requesting PII redaction for the session."""

TOPIC_CHAT: Final = "lk.chat"
"""Text-stream topic for chat messages."""

TOPIC_TRANSCRIPTION: Final = "lk.transcription"
"""Text-stream topic for transcription segments."""

CLIENT_PROTOCOL_TRANSCRIPTION_STREAMS: Final = 3
"""Minimum ``ParticipantInfo.client_protocol`` of a client that rebuilds transcription
events from ``lk.transcription`` text streams and ignores the deprecated ``rtc.Transcription``
data packet. While any considered participant is below this, the legacy packet is still
published. The value is defined by the client SDKs (``client-sdk-js`` ``src/version.ts``).
"""

USERDATA_TIMED_TRANSCRIPT: Final = "lk.timed_transcripts"
"""The key for the timed transcripts in the audio frame userdata."""

USERDATA_TTS_STARTED_TIME: Final = "lk.tts_started_time"
"""The key for the time (``time.perf_counter()``) at which the synthesized text was first
sent to the TTS provider, attached to the audio frame userdata. Used to compute TTFB
without attributing upstream (e.g. LLM streaming) latency to the TTS.
"""
