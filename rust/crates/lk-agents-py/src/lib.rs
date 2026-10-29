//! `livekit.agents._core` -- the PyO3 binding.
//!
//! Pure glue: every class, function and exception `_core` holds is declared beside its Rust item
//! in the core crate that owns it (`#[host(..)]`, expanded in place behind that crate's `python`
//! feature, which only this crate turns on) and registers itself; `#[pymodule]` performs the
//! registrations.
//!
//! Importing this module must start nothing (§1.13): everything that starts lives behind
//! `init(role)`, and every entry point calls `check_pid()` first, so a fork-inherited core raises
//! instead of deadlocking.

use pyo3::prelude::*;

/// SDK room conversion for the declared avatar sink; runtime retains teardown ownership.
pub mod avatar_output;
/// The declared classes' pyo3 adapters, written by `lk-agents-codegen`.
pub mod generated;
/// Entering the host: a call in its operation's captured context, host streams and notices.
pub mod host;
/// `AgentNodes`, `tts_node` and `stt_node`: an agent's nodes over the core's `NodeFactory`.
pub mod nodes;

pub mod realtime_fallback;
/// `init(role)` and the registration table.
pub mod runtime;
/// `Streams` and `StreamReader`, and the record helpers of `turn.rs`.
pub mod storage;
/// `Thread`, `Turn` and `Boundary`: the run and the turn of `lk-agents-storage` over a Python host.
pub mod turn;

pub use lk_agents_runtime::{inference, process, worker};

/// The classes the generated adapters hand an implementation's own objects out as.
pub mod handles {
    pub use crate::storage::PyStreams as Streams;
    pub use crate::turn::PyThread as Thread;
}

/// The implementation of each declared class: a class without one, or an implementation
/// missing a declared method, does not compile.
pub struct Selected;

impl generated::Implementations for Selected {
    type BackgroundAudioPlayer =
        lk_agents_runtime::voice::background_audio::HostBackgroundAudioPlayer;
    type PlayState = lk_agents_runtime::voice::background_audio::PlayState;
    type TextFunctions = lk_agents_runtime::cores::TextAlgorithms;
    type TransportFunctions = lk_agents_runtime::net::strings::TransportAlgorithms;
    type ConnectionPoolState = lk_agents_runtime::net::connection_pool::ConnectionPool;
    type MediaTokenization = lk_agents_runtime::media::blingfire::MediaTokenization;
    type AudioDecoder =
        lk_agents_runtime::inference::codecs::StreamDecoder<lk_agents_runtime::process::Process>;
    type StreamBuffer = lk_agents_runtime::media::StreamBuffer;
    type AudioEmitter = lk_agents_runtime::inference::tts::emitter::AudioEmitter<
        lk_agents_runtime::process::Process,
    >;
    type DataStreamAudioReceiver = lk_agents_runtime::avatar::receiver::DataStreamAudioReceiver;
    type AvatarRunner = lk_agents_runtime::avatar::runner::AvatarRunner;
    type PlaybackReports = lk_agents_abi::voice::io::PlaybackTracker;
    type PlayoutLedger = lk_agents_runtime::rtc::playout::PlayoutLedger;
    type RoomRuntime = lk_agents_runtime::room_io::RoomFactory;
    type RtcRuntime = lk_agents_runtime::rtc::events::RoomEventBus;

    type ConsoleIo = lk_agents_runtime::voice::console::ConsoleIo;
    type VoiceSessionExecution = Selected;
    type WarmTransfer = lk_agents_runtime::voice::workflows::warm_transfer::WarmTransfer;
    type DtmfResult = lk_agents_runtime::voice::workflows::dtmf::DtmfResult;
    type DtmfIo = lk_agents_runtime::voice::workflows::dtmf::DtmfIo;
    type TtsTokenLog = lk_agents_runtime::inference::fallback::TokenLog;
    type FallbackTts =
        lk_agents_runtime::inference::fallback::FallbackTts<lk_agents_runtime::process::Process>;
    type InferenceTts =
        lk_agents_runtime::inference::tts::NativeTts<lk_agents_runtime::process::Process>;
    type FallbackTtsStream = lk_agents_runtime::inference::tts::stream::TtsStream;
    type InferenceTtsStream = lk_agents_runtime::inference::tts::stream::TtsStream;
    type StreamAdapterFunctions = lk_agents_runtime::inference::tts::adapter::AdapterStreams<
        lk_agents_runtime::process::Process,
    >;
    type TokenizeFunctions = lk_agents_runtime::cores::SpeechTokenizer;
    type HostTokenization = lk_agents_runtime::cores::HostTokenizer;
    type WordTokenizer = lk_agents_runtime::cores::BasicWordTokenizer;
    type SentenceTokenizer = lk_agents_runtime::cores::BasicSentenceTokenizer;
    type BufferedTokens = lk_agents_runtime::cores::BufferedTokenStream;
    type MarkupFunctions = lk_agents_runtime::cores::SpeechMarkup;
    type MarkupClass = lk_agents_runtime::cores::ProviderMarkup;
    type TranscriptStripper = lk_agents_runtime::cores::TranscriptMarkupStripper;
    type SearchDocument = lk_agents_runtime::cores::SearchDocument;
    type SearchIndex = lk_agents_runtime::cores::Bm25;
    type ToolSearch = lk_agents_runtime::cores::ToolDiscovery;

    type ExpFilter = lk_agents_runtime::media::filters::ExpFilter;
    type MovingAverage = lk_agents_runtime::media::filters::MovingAverage;
    type LlmAlgorithms = lk_agents_runtime::inference::llm::algorithms::LlmAlgorithms<
        lk_agents_runtime::process::Process,
    >;
    type ProviderAttempts = lk_agents_runtime::inference::provider::ProviderAttempts;
    type A2ADoor = lk_agents_runtime::a2a::door::Door;
    type A2AResponse = lk_agents_runtime::a2a::door::Response;
    type VoiceStore = lk_agents_runtime::storage::voice::VoiceStore;
    type NativeRealtimeModel = lk_agents_runtime::inference::realtime::native::NativeRealtime<
        lk_agents_runtime::process::Process,
    >;
    type RealtimeFallbackModel = crate::realtime_fallback::RealtimeFallback;
    type HostRealtimeModel = lk_agents_runtime::inference::realtime::host::HostRealtimeModel<
        lk_agents_runtime::process::Process,
    >;
    type AgentClient =
        lk_agents_runtime::net::a2a_client::AgentClient<lk_agents_runtime::process::Process>;
    type Reply = lk_agents_runtime::net::a2a_client::Reply;
    type RemoteInteraction = lk_agents_runtime::net::a2a_client::remote::RemoteInteraction;
    type RemoteConfiguration = lk_agents_runtime::net::a2a_client::remote::RemoteInteraction;
    type MessageContent = lk_agents_runtime::net::a2a_client::remote::RemoteInteraction;
    type ContextActivation = lk_agents_runtime::storage::context::store::ContextStore;
    type SessionRecorder = lk_agents_runtime::recording::SessionRecorder;
    type JudgeRunner =
        lk_agents_runtime::inference::evals::Judges<lk_agents_runtime::process::Process>;
    type Evaluations = lk_agents_runtime::inference::evals::Rules;
    type Credentials = lk_agents_runtime::net::a2a_auth::Credentials;
    type AuthPermissions = lk_agents_runtime::net::a2a_auth::Credentials;
    type EndpointCard = lk_agents_runtime::cores::CardBuilder;
    type McpSession = lk_agents_runtime::mcp::McpSession;
    type McpTools = lk_agents_runtime::mcp::McpTools;
    type WorkerProcSupervisor =
        lk_agents_runtime::worker::ipc::ProcessOwner<lk_agents_runtime::process::Process>;
    type WorkerIpcClient =
        lk_agents_runtime::worker::ipc::ChildChannel<lk_agents_runtime::process::Process>;
    type WorkerProcPool =
        lk_agents_runtime::worker::ipc::ProcessPool<lk_agents_runtime::process::Process>;
    type WorkerJobRun = lk_agents_runtime::worker::ipc::JobRun<lk_agents_runtime::process::Process>;
    type WorkerCli = lk_agents_runtime::worker::cli::Commands;
    type EndpointStream = lk_agents_runtime::worker::endpoint_host::NativeStream;
    type EndpointServer = lk_agents_runtime::worker::endpoint_host::Endpoints;
    type AsgiReceive = lk_agents_runtime::worker::endpoint_host::ReceiveCall;
    type AsgiSend = lk_agents_runtime::worker::endpoint_host::SendCall;
    type ServerConfig = lk_agents_runtime::worker::server_host::Configuration;
    type ServerRuntime =
        lk_agents_runtime::worker::server_host::AgentRuntime<lk_agents_runtime::process::Process>;
    type ServerEnvironment = lk_agents_runtime::worker::server::Environment;
    type JobTrace = lk_agents_runtime::worker::ipc::supervisor::JobTimeline;
    type InferenceCredentials = lk_agents_runtime::net::credentials::Credentials;
    type InferenceModels = lk_agents_runtime::inference::models::Models;
    type CliSignals = lk_agents_runtime::worker::cli_signals::CliSignals;
    type SessionReportProjection = lk_agents_abi::telemetry::report::projection::ReportProjection;
    type TelemetryPolicy = lk_agents_runtime::cores::DestinationPolicy;
    type EventRetention =
        lk_agents_runtime::net::session_report::EventRetention<lk_agents_runtime::process::Process>;
    type PreparedTranscript = lk_agents_runtime::net::session_report::TranscriptBuilder<
        lk_agents_runtime::process::Process,
    >;
    type MetricSeries = lk_agents_runtime::cores::MetricSeries;
    type SessionReportUpload = lk_agents_runtime::net::session_report::SessionReportUploader<
        lk_agents_runtime::process::Process,
    >;
    type LoopMonitor = crate::host::loop_monitor::LoopMonitor;
    type LoopMonitorThresholds = lk_agents_runtime::cores::LoopMonitorOptions;
    type TextSessionRunner = lk_agents_runtime::text_session::Runner;
    type AgentRunner = lk_agents_runtime::agent_run::Runner;
    type RunEvents = lk_agents_runtime::agent_run::RunEvents;
    type DataStreamAudioOutput = crate::avatar_output::DataStreamAudioOutput;
    type InferenceAvatar = lk_agents_runtime::inference::avatar::gateway::GatewayAvatar<
        lk_agents_runtime::process::Process,
    >;
    type FallbackLlm =
        lk_agents_runtime::inference::fallback::FallbackLlm<lk_agents_runtime::process::Process>;
    type InferenceLlm =
        lk_agents_runtime::inference::llm::gateway::GatewayLlm<lk_agents_runtime::process::Process>;
    type FallbackStt =
        lk_agents_runtime::inference::fallback::FallbackStt<lk_agents_runtime::process::Process>;
    type SttStreamAdapter = lk_agents_runtime::inference::stt::stream_adapter::StreamAdapter<
        lk_agents_runtime::process::Process,
    >;
    type InferenceVad =
        lk_agents_runtime::inference::vad::NativeVad<lk_agents_runtime::process::Process>;
    type InferenceEot =
        lk_agents_runtime::inference::eot::NativeEot<lk_agents_runtime::process::Process>;
    type EotStream = lk_agents_runtime::inference::eot::DetectorStream;
    type FixedGate = lk_agents_runtime::inference::duplex::GateClass<
        lk_agents_runtime::inference::duplex::FixedGate,
    >;
    type AdaptiveNoiseGate = lk_agents_runtime::inference::duplex::GateClass<
        lk_agents_runtime::inference::duplex::AdaptiveNoiseGate,
    >;
    type DuplexRealtimeModel =
        lk_agents_runtime::inference::duplex::NativeDuplexModel<lk_agents_runtime::process::Process>;
    type InterruptionDetector = lk_agents_runtime::inference::interruption::NativeInterruptionDetector<
        lk_agents_runtime::process::Process,
    >;
    type InferenceStt =
        lk_agents_runtime::inference::stt::NativeStt<lk_agents_runtime::process::Process>;
    type ChoiceParser = lk_agents_runtime::inference::llm::parse::ChoiceParser;
    type CompletionWire = lk_agents_runtime::inference::llm::parse::CompletionWire;
    type Storage =
        lk_agents_runtime::storage::host::HostStorage<lk_agents_runtime::process::Process>;
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Nothing here may spawn, connect, or allocate a runtime. The core's `log` lines reach the
    // Python logger of the same name from import on (installing a logger starts nothing).
    let _ = pyo3_log::try_init();
    runtime::register_all(m)?;
    generated::register(m)
}

impl lk_agents_abi::voice::console::VoiceSessionExecution for Selected {
    async fn start(
        session: lk_agents_abi::HostRef,
        capture_run: bool,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        use lk_agents_abi::python::{HostException as _, captured};
        let reply = Python::attach(|py| {
            let obj = (&session).into_pyobject(py)?;
            let api = obj.extract::<PyRef<'_, lk_agents_runtime::voice::SessionApi>>()?;
            api.start(
                capture_run,
                Box::new(lk_agents_runtime::voice::Executor::new(
                    api.time().clone(),
                    api.spawner(),
                )),
                Box::new(lk_agents_runtime::voice::remote_session::hosting),
                lk_agents_runtime::voice::amd::spawn,
            )
            .map_err(|e| e.to_py_err())
        })
        .map_err(captured)?;
        reply.await.map_err(|e| Python::attach(|_| captured(e.to_py_err())))
    }
}
