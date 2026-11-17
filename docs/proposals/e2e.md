# Shared E2E framework

The framework lets contributors write a small application as an SDK user would, run it in
real processes, act as a caller over voice or text and assert what actually happened. Voice
input/output is one capability of the shared framework. Delegation, RoomIO, approvals,
session continuity and worker recovery use the same infrastructure.

Pytest remains the runner and assertion system. The framework composes existing fixtures and
boundary helpers. The implementation checklist below distinguishes completed foundations
from the remaining delegation stories.

## Authoring experience

Application modules under tests/e2e/apps/ use ordinary Agent, AgentHandler, AgentServer,
RemoteAgent, VoiceSession, TextSession, Agent.run and storage APIs. Tools are ordinary
registered functions. Applications do not inherit from a test agent/session or implement
framework routing. Test configuration chooses providers and service addresses.

A single e2e fixture owns shared infrastructure and cleanup. The proposed starting surface is:

| API | Responsibility |
| --- | --- |
| e2e.start_app(module, env=...) | Launch an importable application in a separate process and await readiness. |
| app.endpoint(name) | Return that running application's actual endpoint address. |
| app.stop(), app.kill(), app.restart() | Explicit process lifecycle controls; restart constructs a fresh owner using retained configuration/storage. |
| e2e.voice_client(app) | Join/dispatch through the real room path and own a voice caller. |
| caller.say(text) | Publish microphone audio using the explicitly selected speech-provider mode. |
| caller.hear() | Return an observed utterance with its received audio/transcription evidence. |
| Voice speech-start/stop observations | Allow a test to interrupt during actual playback and check what stopped at the receiver; exact helper spelling follows the first voice regression. |
| e2e.text_client(app, ...) | Own an existing public AgentClient or stock A2A SDK client. Reuse its methods. |

The e2e handle composes app/client lifetimes; it does not implement transport, conversation
policy or pass/fail decisions. Domain-specific service fixtures remain normal pytest fixtures
using the same resource ownership. Assertions stay next to actions in the test.

External peers use the same `start_app()` lifecycle and export an ordinary `create_app(url)`
ASGI factory. Their runner supplies the bound URL and imports no LiveKit SDK. The stock A2A
application proves direct Message replies and input-required continuation after frontend
restart over real HTTP/SSE. It uses the upstream executor, task store and routes, with no
LiveKit extension or shared backend database.

For example, a story starts frontend and backend apps, asks for a flight, waits for the actual
service request, asks a side question while that request is held, checks the caller's answer,
then releases the service response and checks the search result. The side-question answer must
arrive before the service is released. No arbitrary minute-long sleep is needed to establish
that the frontend remained available.

## Deterministic provider policy

Correctness E2E uses fake models by default. Script the frontend model, every remote model,
the input-routing model and any model-based classifier involved in a scenario. Deterministic
speech providers exercise audio transport and delivery without relying on recognition quality.
Provider errors, malformed output, streaming chunks and stalls are explicit test inputs.
Configure the routing fake separately, just as automatic routing requires a separate model in
the application. It does not consume the frontend model's reply script. Frontend-model reuse
is a later optimization and is not exercised as an implicit initial path.

The remote agent itself remains real: the ordinary application runs its native executors,
tools, task state, A2A client/server, storage and presentation logic. A fake model emits normal
model content/tool calls through the provider interface. In particular, a scripted detector
emits model output that still goes through production parsing, validation, input reservation
and dispatch. It does not directly modify native input-request state.

Reuse FakeLLM and existing provider peers. Prefer role-specific instances and responses keyed
by relevant public request content, offered tools and schemas. Use ordered reply lists only
where the order is itself contractual. Do not couple unrelated concurrent agents through one
global response queue, match entire private system prompts or script the native state machine.

A fake fails on an unexpected/unconfigured request, exhausted response script or malformed
test setup. Capture the model request and configured actor on failure. Let real stream/owner
completion delimit negative assertions. Timing and failures use observable gates, bounded
waits and injected faults rather than retries to obtain a passing result.

| Run | Providers | What it establishes |
| --- | --- | --- |
| Correctness E2E | Scripted inference, routing and speech; controlled business services; real app processes, rooms, HTTP/SSE and storage. | Framework behavior under known model outputs, including interruption, correlation, approval, recovery and cleanup. |
| Provider/peer conformance | Real provider adapters against local wire peers; stock A2A/ADK servers with deterministic model stubs. | Serialization, streaming, SDK compatibility and the assumptions made by boundary fakes. |
| Conversation evaluation | Explicitly configured real models and speech providers through the same application code. | Intent-recognition quality, wording, grouping, pacing and actual provider latency. |

Correctness CI does not require a live LLM to make the expected tool choice or wording.
Live evaluation is a separate, explicitly selected result with retained artifacts; it neither
silently replaces a fake nor falls back to one when credentials are absent. Model-quality
scores do not substitute for deterministic framework assertions. A passing fake-model test
does not establish that a real model selects good routes or sounds natural.

## Shared resources and real boundaries

Launch application modules with serializable configuration and service addresses. Construct
Agent/model objects in their owning process; never pickle native handles, Agent objects or
closures from the pytest process. Frontend and backend can be stopped independently. Restart
uses fresh runtime objects with retained storage, rather than keeping the old conversation in
memory or serializing Python stacks.

Reuse Child/process ownership, the isolated SFU, SQLite/AgentDB fixtures, A2A clients and the
RecordingService. Deterministic voice can use the existing tone/marker speech fixtures over
real RTC. Recorded or synthesized speech is used with real providers in conversation evaluation.
A voice helper publishes actual audio; a transcription alone does not prove playback. Text
uses actual SDK/HTTP/SSE boundaries. Tests do not inject recognized messages into private core
methods to bypass those transports.

Business effects are checked through service calls or authoritative business state. Framework
state is checked through public task/history/state views. Delivery is checked at the caller.
Logs and traces help diagnose failures but do not establish a business effect. A scripted reply
that says "booked" is insufficient without the service action.

Each test owns its temporary data, peers, streams and processes. Cleanup runs on failure and
cancellation, releases held requests and reaps descendants. Failure artifacts include the
configured scripts, observed model requests, caller transcript/audio/timings, communication
observations and service ledger as applicable. Keep secrets out of those artifacts.

## Planned test-only types

These are internal testing helpers, not SDK APIs. Constructor/method details remain a review
item; the domain scope is limited to these four target types, reusing existing observations:

| Type | Purpose and reuse |
| --- | --- |
| E2E | Fixture-owned facade composing the existing process, room, client, storage and artifact resources. |
| RunningApp | An app process handle with endpoint information and explicit restart controls; builds on existing Child ownership. |
| VoiceClient | A caller using real RTC microphone/output paths, including speech and interruption observations. |
| ReceivedSpeech | Evolve the existing Heard/audio-timing observations into received text/audio/timing for an utterance, including partial delivery; no verdict about business success or naturalness. |

Text clients reuse AgentClient/the stock SDK. FakeLLM, Hang, Reply, Child, RecordingService,
ServiceCall and current tone/marker providers are reused or extended. Response gates belong
to controlled services; no separate Person, Scenario, fake delegation controller or custom
assertion engine is planned. Transport-specific handles remain ordinary implementation detail
within these existing boundaries, rather than additional authoring concepts.

## Existing E2E consolidation

The framework is shared infrastructure for existing suites as well as delegation. Preserve
feature-owned test files and their contract-specific assertions while consolidating reusable
mechanics. The current source audit identifies these boundaries:

| Existing code | Consolidation direction |
| --- | --- |
| resources.py Child/child_process/JobWorker/serving and processes.py watchdog/diagnostics | Reuse common process ownership, startup readiness, artifacts and cleanup. Retain the distinct sync/async launch and shutdown behavior that CLI/worker regressions exercise. RunningApp composes this ownership. |
| sfu.py | Keep one isolated SFU/token/connection resource layer. Reuse it from normal voice, RoomIO, encryption, avatar and compatibility stories. |
| v1_clients.py ToneSTT/ToneTTS, tones, microphone publishing, playback and audio/video observation | Extract reusable deterministic media and receiver mechanics into the shared provider/client helpers. Evolve Heard for ReceivedSpeech instead of introducing a parallel recorder. Seen remains the existing video observation. |
| v1_clients.py starter_token/is_agent/state/transcription helpers | Share raw connection/observation mechanics while keeping v1 dispatch, participant-selection and stream-attribute contracts explicit in compatibility adapters. |
| v1_clients.py CHAT_DEFLATED/PRECONNECT_WIRE, V1AvatarWorker/V1AvatarAgent | Keep version-specific wire encoding, missing fields, chunking and plain playback RPC behavior in compatibility-owned code. A modern client helper must not fill in attributes or methods that a v1 peer did not send. |
| drivers.py A2ACaller, RPC/SSE and stock-client helpers | Reuse wire helpers for tests of the protocol itself; normal user stories use public clients. Current send helpers assume Task responses, so the general client path must preserve Message-or-Task responses and streaming content. |
| conftest.py and storages.py | Compose existing process, SFU, temporary data, thread and SQLite/AgentDB fixtures behind e2e. Preserve explicit live prerequisites and teardown ownership. |
| RoomIO/encryption tests' repeated stream/audio readers | Adopt common receiver observation where it preserves the feature's evidence; keep encryption, binding, participant and routing assertions in their own tests. |

The v1 compatibility files currently exercise cited client-family wire behavior through
plain RTC and emulated v1 peers; they do not run every language SDK implementation. Keep that
boundary explicit. Consolidation must preserve each client's old wire behavior independently
of the modern client path, including avatar compatibility in both directions.

Keep test_v1_client_compat.py, test_v1_client_compat_conversation.py and
test_v1_client_compat_avatar.py identifiable as their own suites. Compatibility-specific
adapters can move under tests/e2e/compat/ as v1_clients.py is split. Framework code must not
depend on compatibility modules; those adapters can depend on shared lifecycle/media helpers.

Before each extraction, record the selected test collection and baseline results. After it,
preserve the parametrized cases, prerequisites, observations, assertions and cleanup behavior.
Use small migrations proved by representative text and v1 audio/interruption cases, then
broaden to affected suites. Do not add a new runner per feature or merge all stories into an
opaque scenario matrix. Low-level fault/wire tests may continue using direct boundary helpers.

## Code organization and rollout

| Location | Owns |
| --- | --- |
| tests/e2e/framework/ | The shared fixture facade and app/client handles, added by responsibility as stories require them. |
| tests/e2e/apps/ | Ordinary importable SDK applications and independently launched external peers. |
| tests/e2e/compat/ | Existing version-specific wire peers/profiles as they are extracted; common mechanics come from the framework. |
| tests/e2e/conftest.py | Fixture registration and explicit provider-mode configuration. |
| Existing resources.py, drivers.py, sfu.py and process helpers | Reused lifecycle and wire mechanics; move shared implementations when useful rather than copying them into a parallel stack. |
| tests/e2e/test_<feature>.py | Feature-owned stories, actions and plain assertions. |
| tests/fakes.py and provider-peer fixtures | Deterministic model/provider behavior. |

1. Record the existing collection and baseline; extract a narrow common process/media/client
   slice from current helpers. Prove one existing text story and one v1 audio/interruption
   story with unchanged wire and assertion semantics.
2. Introduce the e2e facade, app runner and owned text client around that infrastructure.
   Prove failure cleanup and process reaping with an ordinary public-API application.
3. Add the voice caller from existing RTC/audio/transcription mechanics. Prove a heard reply
   and receiver-observed interruption over the isolated SFU, then migrate affected stories.
4. Add only the controls needed by the first delegation state/recovery stories, then the
   responsive-conversation and grouped-question stories. Keep each regression focused.
5. Run third-party SDK/ADK peers through the same app/client infrastructure and add an
   explicitly selected real-provider conversation evaluation configuration.

Native DST remains separate. It runs the production Rust executors with controlled scheduling,
ports and storage faults; E2E verifies the assembled public application and real boundaries.
They exercise the same product contracts without requiring a second execution engine or one
shared scenario language.
