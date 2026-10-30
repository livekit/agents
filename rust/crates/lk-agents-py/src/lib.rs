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
/// The conversions of the `handwritten(.., py)` types, which the generated code calls.
pub mod conversions;
/// The declared classes' pyo3 adapters, written by `lk-agents-codegen`.
pub mod generated;
/// Entering the host: a call in its operation's captured context, host streams and notices.
pub mod host;

pub mod realtime_fallback;
/// `init(role)` and the registration table.
pub mod runtime;
/// The record helpers of `turn.rs`.
pub mod storage;
/// `Thread`, `Turn` and `Boundary`: the run and the turn of `lk-agents-storage` over a Python host.
pub mod turn;

pub use lk_agents_runtime::{inference, process, worker};

/// The classes the generated adapters hand an implementation's own objects out as.
pub mod handles {
    pub use crate::turn::PyThread as Thread;

    pub use crate::generated::voice::SpeechHandle;

    /// A session's speech, handed out as its `_core.SpeechHandle`.
    impl From<std::sync::Arc<lk_agents_runtime::voice::SpeechHandle>> for SpeechHandle {
        fn from(speech: std::sync::Arc<lk_agents_runtime::voice::SpeechHandle>) -> Self {
            SpeechHandle(speech)
        }
    }
}

/// The implementation of each declared class: a class without one, or an implementation
/// missing a declared method, does not compile.
pub struct Selected;

impl generated::Implementations for Selected {
    type BackgroundAudioPlayer = lk_agents_runtime::voice::background_audio::BackgroundAudio;
    type PlayState = lk_agents_runtime::voice::background_audio::PlayState;
    type LanguageFunctions = lk_agents_runtime::cores::Languages;
    type ApiErrorFunctions = lk_agents_runtime::net::http::HttpErrors;
    type ChatContextDiffs = lk_agents_runtime::cores::ChatDiffs;
    type ChatContextPlans = lk_agents_runtime::cores::ChatPlans;
    type ConnectOptionsFunctions = lk_agents_runtime::net::attempts::ConnectOptionsRules;
    type SentencePacer = lk_agents_runtime::inference::tts::pacer::SentencePacer;
    type SynthesisChecks = lk_agents_runtime::inference::tts::checks::SynthesisChecks;
    type ImageFunctions = lk_agents_runtime::inference::images::ImageFunctions;
    type FrameReaderInstall = lk_agents_runtime::inference::images::ImageFunctions;
    type Logging = crate::host::logging::Logging;
    type TraceParents = lk_agents_runtime::cores::TraceParents;
    type TelemetryAdmission = lk_agents_runtime::cores::Admission;
    type MetricsRecords = lk_agents_runtime::cores::MetricsRecords;
    type Tagger = lk_agents_runtime::net::session_report::records::Tagger;
    type UploadGate = lk_agents_runtime::net::session_report::records::UploadGate;
    type ReportRecords = lk_agents_runtime::net::session_report::records::ReportRecords;
    type ProviderMetrics = lk_agents_runtime::cores::ProviderMetrics;
    type LlmMetricState = lk_agents_runtime::cores::LlmMetricState;
    type TtsMetricState = lk_agents_runtime::cores::TtsMetricState;
    type ConnectionPoolState = lk_agents_runtime::net::connection_pool::ConnectionPool;
    type MediaTokenization = lk_agents_runtime::media::blingfire::MediaTokenization;
    type AudioDecoder = lk_agents_runtime::inference::codecs::StreamDecoder;
    type StreamBuffer = lk_agents_runtime::media::StreamBuffer;
    type AudioEmitter = lk_agents_runtime::inference::tts::emitter::AudioEmitter;
    type DataStreamAudioReceiver = lk_agents_runtime::avatar::receiver::DataStreamAudioReceiver;
    type AvatarRunner = lk_agents_runtime::avatar::runner::AvatarRunner;
    type PlaybackReports = lk_agents_abi::voice::io::PlaybackTracker;
    type Playback = lk_agents_abi::voice::io::PlaybackTracker;
    type PlayoutLedger = lk_agents_runtime::rtc::playout::PlayoutLedger;
    type RoomRuntime = lk_agents_runtime::room_io::RoomFactory;
    type RtcRuntime = lk_agents_runtime::rtc::events::RoomEventBus;

    type ConsoleIo = lk_agents_runtime::voice::console::ConsoleIo;
    type VoiceSession = lk_agents_runtime::voice::VoiceSession;
    type SpeechHandle = lk_agents_runtime::voice::SpeechHandle;
    type RoomAudioInput = lk_agents_runtime::room_io::RoomAudioInput;
    type RoomSession = lk_agents_runtime::room_io::FixedRoom;
    type RoomVideoInput = lk_agents_runtime::room_io::RoomVideoInput;
    type ByteStreamSender = lk_agents_runtime::rtc::events::ByteStreamSender;
    type TextStreamSender = lk_agents_runtime::rtc::events::TextStreamSender;
    type RpcHandler = lk_agents_runtime::rtc::events::RpcHandler;
    type AgentNodes = lk_agents_runtime::voice::nodes::AgentNodes;
    type NodeCalls = lk_agents_runtime::voice::nodes::NodeCalls;
    type TtsNodeFrames =
        lk_agents_runtime::voice::nodes::NodeOutput<lk_agents_abi::voice::nodes::NodeFrame>;
    type SttNodeEvents = lk_agents_runtime::voice::nodes::NodeOutput<lk_agents_abi::SpeechEvent>;
    type DefaultNodes = lk_agents_runtime::voice::nodes::DefaultNodes;
    type SessionOptions = lk_agents_runtime::voice::SessionOptions;
    type InterruptionGateFunctions = lk_agents_runtime::voice::InterruptionGateRules;
    type VoiceAgent = lk_agents_runtime::voice::VoiceAgentFrame;
    type SessionStart = lk_agents_runtime::voice::SessionStartPlans;
    type WorkflowChecks = lk_agents_runtime::voice::WorkflowRules;
    type AmdFunctions = lk_agents_runtime::voice::AmdRules;
    type KeytermFunctions = lk_agents_runtime::voice::KeytermRules;
    type WarmTransfer = lk_agents_runtime::voice::workflows::warm_transfer::WarmTransfer;
    type DtmfIo = lk_agents_runtime::voice::workflows::dtmf::DtmfIo;
    type TtsTokenLog = lk_agents_runtime::inference::fallback::TokenLog;
    type FallbackTts = lk_agents_runtime::inference::fallback::FallbackTts;
    type InferenceTts = lk_agents_runtime::inference::tts::GatewayTts;
    type TtsStream = lk_agents_runtime::inference::tts::stream::TtsStream;
    type StreamAdapterFunctions = lk_agents_runtime::inference::tts::adapter::AdapterStreams;
    type TokenizeFunctions = lk_agents_runtime::cores::SpeechTokenizer;
    type HostTokenization = lk_agents_runtime::cores::HostTokenizer;
    type WordTokenizer = lk_agents_runtime::cores::BasicWordTokenizer;
    type SentenceTokenizer = lk_agents_runtime::cores::BasicSentenceTokenizer;
    type BufferedTokens = lk_agents_runtime::cores::BufferedTokenStream;
    type ChatContextDict =
        lk_agents_runtime::cores::ChatDict<lk_agents_runtime::cores::SpeechMarkup>;
    type MarkupFunctions = lk_agents_runtime::cores::SpeechMarkup;
    type MarkupClass = lk_agents_runtime::cores::ProviderMarkup;
    type TranscriptStripper = lk_agents_runtime::cores::TranscriptMarkupStripper;
    type SearchDocument = lk_agents_runtime::cores::SearchDocument;
    type SearchIndex = lk_agents_runtime::cores::Bm25;
    type ToolSearch = lk_agents_runtime::cores::ToolDiscovery;

    type AudioByteStream = lk_agents_runtime::media::AudioByteStream;
    type ExpFilter = lk_agents_runtime::media::filters::ExpFilter;
    type MovingAverage = lk_agents_runtime::media::filters::MovingAverage;
    type LlmAlgorithms = lk_agents_runtime::inference::llm::algorithms::LlmAlgorithms;
    type ProviderAttempts = lk_agents_runtime::inference::provider::ProviderAttempts;
    type A2ADoor = lk_agents_runtime::a2a::door::Door;
    type A2AResponse = lk_agents_runtime::a2a::door::Response;
    type VoiceStore = lk_agents_runtime::storage::voice::VoiceStore;
    type OpenAIRealtimeModel = lk_agents_runtime::inference::realtime::openai::OpenAIRealtime;
    type RealtimeFallbackModel = crate::realtime_fallback::RealtimeFallback;
    type HostRealtimeModel = lk_agents_runtime::inference::realtime::host::HostRealtimeModel;
    type AgentClient = lk_agents_runtime::net::a2a_client::AgentClient;
    type Reply = lk_agents_runtime::net::a2a_client::Reply;
    type RemoteInteraction = lk_agents_runtime::net::a2a_client::remote::RemoteInteraction;
    type RemoteConfiguration = lk_agents_runtime::net::a2a_client::remote::RemoteInteraction;
    type MessageContent = lk_agents_runtime::net::a2a_client::remote::RemoteInteraction;
    type ContextActivation = lk_agents_runtime::storage::context::store::ContextStore;
    type AppTransaction = lk_agents_runtime::storage::context::transaction::Transaction;
    type SessionRecorder = lk_agents_runtime::recording::SessionRecorder;
    type JudgeRunner = lk_agents_runtime::inference::evals::Judges;
    type Evaluations = lk_agents_runtime::inference::evals::Rules;
    type Credentials = lk_agents_runtime::net::a2a_auth::Credentials;
    type AuthPermissions = lk_agents_runtime::net::a2a_auth::Credentials;
    type EndpointCard = lk_agents_runtime::cores::CardBuilder;
    type McpSession = lk_agents_runtime::mcp::McpSession;
    type McpTools = lk_agents_runtime::mcp::McpTools;
    type ProcessSupervisor = lk_agents_runtime::worker::ipc::ProcessOwner;
    type IpcClient = lk_agents_runtime::worker::ipc::ChildChannel;
    type ProcessPool = lk_agents_runtime::worker::ipc::ProcessPool;
    type JobRun = lk_agents_runtime::worker::ipc::JobRun;
    type WorkerCli = lk_agents_runtime::worker::cli::Commands;
    type EndpointStream = lk_agents_runtime::worker::endpoint::host::NativeStream;
    type EndpointServer = lk_agents_runtime::worker::endpoint::host::Endpoints;
    type AsgiReceive = lk_agents_runtime::worker::endpoint::host::ReceiveCall;
    type AsgiSend = lk_agents_runtime::worker::endpoint::host::SendCall;
    type ServerConfig = lk_agents_runtime::worker::agent_server::Configuration;
    type ServerRuntime = lk_agents_runtime::worker::agent_server::AgentServerRuntime;
    type ServerEnvironment = lk_agents_runtime::worker::server::Environment;
    type JobTrace = lk_agents_runtime::worker::ipc::supervisor::JobTimeline;
    type InferenceCredentials = lk_agents_runtime::net::credentials::Credentials;
    type JoinTokens = lk_agents_runtime::net::token::JoinTokens;
    type InferenceModels = lk_agents_runtime::inference::models::Models;
    type CliSignals = lk_agents_runtime::worker::cli_signals::CliSignals;
    type SessionReportProjection = lk_agents_runtime::cores::ReportProjection;
    type TelemetryPolicy = lk_agents_runtime::cores::DestinationPolicy;
    type EventRetention = lk_agents_runtime::net::session_report::EventRetention;
    type PreparedTranscript = lk_agents_runtime::net::session_report::TranscriptBuilder;
    type MetricSeries = lk_agents_runtime::cores::MetricSeries;
    type SessionReportUpload = lk_agents_runtime::net::session_report::SessionReportUploader;
    type TextSessionRunner = lk_agents_runtime::text_session::Runner;
    type AgentRunner = lk_agents_runtime::agent_run::Runner;
    type RunEvents = lk_agents_runtime::agent_run::RunEvents;
    type DataStreamAudioOutput = crate::avatar_output::DataStreamAudioOutput;
    type InferenceAvatar = lk_agents_runtime::inference::avatar::gateway::GatewayAvatar;
    type FallbackLlm = lk_agents_runtime::inference::fallback::FallbackLlm;
    type InferenceLlm = lk_agents_runtime::inference::llm::gateway::GatewayLlm;
    type FallbackStt = lk_agents_runtime::inference::fallback::FallbackStt;
    type SttStreamAdapter = lk_agents_runtime::inference::stt::stream_adapter::StreamAdapter;
    type InferenceVad = lk_agents_runtime::inference::vad::LocalVad;
    type InferenceEot = lk_agents_runtime::inference::eot::AudioEot;
    type EotStream = lk_agents_runtime::inference::eot::DetectorStream;
    type FixedGate = lk_agents_runtime::inference::duplex::GateClass<
        lk_agents_runtime::inference::duplex::FixedGate,
    >;
    type AdaptiveNoiseGate = lk_agents_runtime::inference::duplex::GateClass<
        lk_agents_runtime::inference::duplex::AdaptiveNoiseGate,
    >;
    type DuplexRealtimeModel = lk_agents_runtime::inference::duplex::DuplexAdapterModel;
    type InterruptionDetector =
        lk_agents_runtime::inference::interruption::NativeInterruptionDetector;
    type InferenceStt = lk_agents_runtime::inference::stt::GatewayStt;
    type ChoiceParser = lk_agents_runtime::inference::llm::parse::ChoiceParser;
    type CompletionWire = lk_agents_runtime::inference::llm::parse::CompletionWire;
    type Storage = lk_agents_runtime::storage::namespace::Namespace;
}

/// A contended Rust lock is waited for without the GIL: the Rust thread holding it may be making a
/// host call, which needs the GIL this thread would otherwise keep.
fn wait_without_the_gil(wait: &mut dyn FnMut()) {
    // SAFETY: the thread state is saved only by a thread holding the GIL, and restored by it.
    if crate::host::holds_gil() {
        let state = unsafe { pyo3::ffi::PyEval_SaveThread() };
        wait();
        unsafe { pyo3::ffi::PyEval_RestoreThread(state) };
    } else {
        wait();
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Nothing here may spawn, connect, or allocate a runtime. The core's `log` lines reach the
    // Python logger of the same name from import on (installing a logger starts nothing).
    lk_agents_runtime::net::lock::set_host_wait(wait_without_the_gil);
    runtime::register_all(m)?;
    generated::register(m)?;
    host::logging::install(m)
}

#[cfg(test)]
mod lock_tests {
    use lk_agents_runtime::net::lock::{Mutex, set_host_wait};
    use pyo3::prelude::*;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    // Catches: review13's deadlock. A thread holding the GIL blocks on a Rust lock whose holder
    // makes a host call; the waiter must wait without the GIL.
    #[test]
    fn a_host_call_under_a_lock_does_not_deadlock_against_a_gil_holder() {
        Python::initialize();
        set_host_wait(super::wait_without_the_gil);
        let lock = Arc::new(Mutex::new(0));
        let (locked, holding) = mpsc::channel();
        let (attached, attaching) = mpsc::channel();
        let (done, finished) = mpsc::channel();
        let holder = std::thread::spawn({
            let lock = lock.clone();
            move || {
                let mut guard = lock.lock();
                locked.send(()).unwrap();
                attaching.recv().unwrap();
                // The host call, under the lock.
                Python::attach(|py| py.run(c"x = 1", None, None).unwrap());
                *guard += 1;
            }
        });
        holding.recv().unwrap();
        let waiter = std::thread::spawn(move || {
            Python::attach(|_| {
                attached.send(()).unwrap();
                *lock.lock() += 1;
            });
            done.send(()).unwrap();
        });
        assert!(finished.recv_timeout(Duration::from_secs(10)).is_ok(), "deadlocked");
        holder.join().unwrap();
        waiter.join().unwrap();
    }
}

#[cfg(test)]
mod logging_tests {
    use pyo3::prelude::*;
    use std::sync::mpsc;
    use std::time::Duration;

    // Catches: a log line waiting for the GIL. The thread logging holds what a GIL holder waits
    // for (a lock, here a reply), as review14's storage_lost warn under the Requests lock did.
    #[test]
    fn a_line_logged_while_a_gil_holder_waits_on_the_logger_completes_and_arrives() {
        Python::initialize();
        crate::host::logging::install_logger();
        let seen = Python::attach(|py| {
            let code = c"import logging\nseen = []\nclass H(logging.Handler):\n    def emit(self, r): seen.append(r.getMessage())\nlogging.getLogger('livekit.agents.logtest').addHandler(H())\nlogging.getLogger('livekit.agents.logtest').setLevel(logging.INFO)\n";
            let globals = pyo3::types::PyDict::new(py);
            py.run(code, Some(&globals), None).unwrap();
            globals.get_item("seen").unwrap().unwrap().unbind()
        });
        let (attached, attaching) = mpsc::channel();
        let (logged, logging) = mpsc::channel::<()>();
        let waiter = std::thread::spawn(move || {
            Python::attach(|_| {
                attached.send(()).unwrap();
                logging.recv_timeout(Duration::from_secs(10)).is_ok()
            })
        });
        attaching.recv().unwrap();
        log::warn!(target: "livekit.agents.logtest", "under a lock");
        logged.send(()).unwrap();
        assert!(waiter.join().unwrap(), "the line waited for the GIL");
        crate::host::logging::flush();
        Python::attach(|py| {
            let seen: Vec<String> = seen.bind(py).extract().unwrap();
            assert_eq!(seen, ["under a lock"]);
        });
    }
}
