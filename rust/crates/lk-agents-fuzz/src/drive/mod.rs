//! A real session with controlled dependencies, and one function that applies an [`Input`] to it.
//!
//! The only place that touches the product, and only as a host does: the session is assembled
//! with `VoiceSession(...)` (`SessionApi::host_new`) over the fake room and the fakes, started,
//! and every call is the host method a Python `AgentSession` makes, with its argument shapes;
//! people join and leave through the fake room, which the real room io watches. What the host
//! cannot do it does not do: the test-only probe (the `dst` feature) only watches the journal and
//! the host events, and hands the session's cores to the run, which steps them as its own queue
//! while the real runtime owner performs their effects ([`executor`]).
//! It decides nothing: rules read [`Recording`] and nothing else.
//!
//! **Each resource is opened by whoever mints it.** Speeches, holds, fillers and hook calls come
//! from the driver's own journal, the one account of them that cannot drift from what it did. The
//! rest have no journal to read -- a chat exists because a fake was called, an answer because a
//! call returned a future -- and are opened here, from the side that knows.
//!
//! **Applying an input does not yield.** Only [`Input::Advance`] and the awaits reach the
//! executor, so a case can build a burst the driver must untangle in one cycle.
//! [`Recording::driver_ran_since`] is written against that, and breaking it would quietly weaken
//! every rule that allows the driver one input of grace.
//!
//! The module is split by what a run does, in order: [`start`] builds the session, [`apply`]
//! turns one input into what the product sees, [`collect`] writes down what became true, and
//! [`finish`] releases what a run owes and reports what it found. This file holds what they
//! share: the setup, the simulation, and the run itself.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use lk_agents_abi::{
    AudioFrame, BoxFuture, ChatItem, ChatMessage, HostRef, ImageContent, ImageSource, Maybe,
    MediaRef, Realtime, Spawner, Ts,
    errors::RefRegistry,
    metrics::base::AgentMetrics,
    turn::interruption::InterruptionSource,
    turn::keyterms::{DetectionLlm, KeytermDetectionConfig, SttContextConfig},
    voice::room_io::{AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextOutputConfig},
    voice::session_options::{
        EndpointingKwargs, HostValue, InterruptionKwargs, PreemptiveKwargs, SessionConnOptions,
        SessionKwargs, TurnDetectionKwarg, TurnHandlingKwargs, UserTurnLimitKwargs,
    },
};
use lk_agents_runtime::voice::background_audio::BackgroundAudioPlayer;
use lk_agents_test_support::rtc::{FakeRoom, ParticipantKind};
use lk_agents_voice::remote_session::{
    proto,
    proto::agent_session_message::{Message, console_io},
};
use lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM;
use lk_agents_voice::{
    agent_activity::options::{
        AgentSessionOptions, Deployment, HostOptions, InterruptionGate, InterruptionGateInputs,
        RealtimeGate, SttGate, TurnDetection, resolve_interruption_detector,
    },
    agent_activity::scheduling::SpeechId,
    agent_session::{
        Event, HoldId, HookId, HostEvent, IoKind, Origin, Probe, Pusher, SessionApi, SessionEvents,
        SpeechCommand, Timebase, TurnCommand, UpdateAgentArgs, api::NodeFactory, effects::Decision,
    },
    events::SessionEvent,
    filler_scheduler::FillerId,
    generation::Instructions,
    speech_handle::SpeechHandle,
    tool_executor::{HostToolRunner, ToolOutcome},
    trace::{
        CoreStep, DetectorStep, GenerationStep, HandoffStep, PlayoutStep, SpeechStep, Step,
        ToolStep, TurnStep,
    },
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use crate::{
    check::{Failure, Subject, check_all},
    fakes::{self, Control, Record, Registry, SAMPLE_RATE, SAMPLES_PER_CHUNK, Slot},
    input::{
        AgentText, AmdInput, Confidence, ConsoleAsk, Count, DtmfInput, HookAnswer, Input, IvrInput,
        Knob, Millis, Models, Ref, Role, RoomIn, RtEvent, SayInput, ToolAnswer, TurnMode, UserText,
    },
    record::step_name,
    record::{
        Answer, Detail, Handle, Host, Id, Kind, Observed, Phase, Recording, Sink, Stamps, Tag,
        TranscriptSetup, hook_of,
    },
};
use lk_agents_abi::voice::session_options::{EndpointingMode, InterruptionMode, TextTransformSet};

mod apply;
mod collect;
mod executor;
mod finish;
pub mod queue;
mod start;

pub use apply::BUDGET_MS;
pub use finish::{Ending, Outcome, PHASES, finish, guard, run};
pub use queue::Policy;

use self::apply::{Feeding, Pending};
use self::queue::Next;
// ==============================================================================================
// What a session is built with
// ==============================================================================================

/// What the session is built with: everything fixed for the life of a run. Everything a case can
/// change afterwards is an [`Input`] instead, so a case says what it did rather than how the
/// session was configured to let it.
#[derive(Clone, Debug, PartialEq)]
pub struct Setup {
    /// A recognition leg listening to the microphone, so recognition events can be delivered. The
    /// microphone itself is always there: a voice call has one whether or not anything transcribes
    /// it.
    pub stt: bool,
    /// A voice-activity detector on the microphone leg: the second observer of the person.
    pub vad: bool,
    /// An end-of-turn predictor on the microphone leg: the third observer of the person.
    pub eot: bool,
    /// The adaptive interruption detector: judges whether an overlap is a interruption.
    pub interruption: bool,
    /// Learn the endpointing delay from the conversation instead of a fixed one.
    pub dynamic_endpointing: bool,
    pub turn: TurnMode,
    /// Preemptive generation on a preflight transcript, with preemptive TTS.
    pub preemptive: bool,
    /// Tools advertised to the model; each is run by the controlled tool runner.
    pub tools: Vec<String>,
    pub max_tool_steps: Option<u32>,
    pub instructions: String,
    pub max_unrecoverable_errors: u32,
    /// Host hooks: the driver issues `HookCall`s on the event stream and waits up to this many
    /// milliseconds for the case to answer with `HookDone`. `None` runs without host hooks.
    pub hook_bound_ms: Option<u64>,
    /// A realtime model instead of the LLM+TTS pair: the socket is the voice, and the whole
    /// `Realtime` journal family exists only under one. `Some` describes the server the session
    /// is built against.
    pub realtime: Option<RealtimeSetup>,
    /// `ivr_detection`: `Some(true)` navigates from the session's start. The navigator itself
    /// runs in every session, as `start()` spawns it.
    pub ivr: Option<bool>,
    /// The answering-machine detector, attached as `AMD(session, ...)` attaches it.
    pub amd: Option<AmdSetup>,
    /// Static keyterms, which is also what makes the session have a keyterm pass at all.
    pub keyterms: Vec<String>,
    /// The agents override nodes (each as the default does): the opening agent these, every agent
    /// a handoff installs its own.
    pub nodes: Option<NodeSetup>,
    /// `stt_context_options["keyterm_detection"]` on, with the detection LLM the host resolves.
    pub keyterm_detection: Option<KeytermDetectionSetup>,
    /// What happens in the room: `None` is a room nothing happens in (a `Room` input is refused),
    /// `Some` a room people join and leave, with `RoomOptions`' two switches.
    pub room: Option<RoomSetup>,
    /// An audio output that can pause: what makes a false interruption resumable at all, and so
    /// the whole pause machine reachable. The fake pauses by recording that it was told to.
    pub pausable_output: bool,
    /// `min_consecutive_speech_delay`: the floor the scheduler holds between two of the agent's
    /// speeches. `None` is the default of none.
    pub speech_gap_ms: Option<u64>,
    /// `user_turn_limit`: the words and the duration after which a user turn is declared
    /// exceeded. `None` is no limit, the default.
    pub user_turn_max_words: Option<u32>,
    pub user_turn_max_ms: Option<u64>,
    /// `transcription_timeout`: how long after the detector's end of speech the driver waits for
    /// a transcript before giving up on one. `None` is the default of never.
    pub transcription_timeout_ms: Option<u64>,
    /// A TTS leg. Off, a pipeline session has an LLM and an audio output but no voice, which is
    /// the one configuration in which `say(words)` has nothing to speak with.
    pub tts: bool,
    /// How long every provider stream takes to open, on the session's clock. `None` is at once.
    pub connect_ms: Option<u64>,
    /// A camera attached from the start, for the case to feed frames into.
    pub video: bool,
    /// `RoomOptions(video_input=True)` on the room's legs: the person publishes a camera with
    /// their microphone, and the room's video input reads it.
    pub room_camera: bool,
    /// The transcript synchroniser between the two outputs: the agent's words are shown paced
    /// against its audio rather than as they are generated, which is the default with a room.
    pub sync_transcription: bool,
    /// An audio output that reports a cut a few scheduler hops late, as a host source does.
    pub lagging_output: bool,
    /// `tts_text_transforms`: `None` is not given (markdown and emoji filtered), `Some` the set
    /// written, `Some(None)` none.
    pub text_transforms: Option<Option<TextTransformSet>>,
    /// The room's own audio and transcription legs, which a host that hands none of its own
    /// runs on (`agent_session.py`'s default): the person is in the room from the start and
    /// speaks into a microphone track, and the agent's voice plays out on the room's clock.
    /// `None` is the host's own legs, the fakes.
    pub room_legs: Option<RoomLegs>,
    /// `lk agent console`: the CLI's socket is the session's microphone and speaker, and
    /// the session's remote session host on it forwards its events and answers the CLI's requests,
    /// as `voice/console.py` runs it. Never with the room's legs.
    pub console: bool,
    /// `VoiceActivityVideoSampler()`, which `agent_session.py` hands over by default: camera
    /// frames reach the model at one rate while the person speaks and another while silent.
    pub video_sampler: bool,
    /// `BackgroundAudioPlayer` on the room: an ambient sound from the start and a thinking sound
    /// while the agent thinks, mixed on its own published track. Only with the room's legs.
    pub background_audio: bool,
    /// An avatar worker in the room: the agent's voice streams to it (`DataStreamAudioOutput`,
    /// what an avatar plugin sets as `session.output.audio`) and it reports each segment played.
    /// Only with the room's legs.
    pub avatar: bool,
    /// The avatar reports when each segment starts playing (`wait_playback_start=True`).
    pub avatar_reports_start: bool,
    /// The avatar generator's frame rate and its latency per frame (under one frame period).
    pub avatar_fps: u64,
    pub avatar_latency_ms: u64,
    /// Which recorded recogniser the person's transcripts behave like, by its index in
    /// `corpus::STT_MODELS`: its timing, its cuts, its quirks.
    pub stt_model: usize,
    /// The voice takes as long to its first audio as the recorded one did, drawn from this seed;
    /// `None` answers at once.
    pub tts_latency: Option<u64>,
    /// `conn_options=SessionConnectOptions(..)`: each leg's retry budget. `None` is the defaults.
    pub conn: Option<ConnSetup>,
    /// A voice that does not stream: the session speaks it sentence by sentence through its
    /// stream adapter, each sentence a synthesis of its own, retried on its own.
    pub chunked_tts: bool,
    /// Whether the voice says it aligns its words to its audio, and whether it then does.
    pub tts_alignment: TtsAlignment,
    /// `VoiceSession(use_tts_aligned_transcript=)`; `None` is not given.
    pub aligned_transcript: Option<bool>,
    /// `Agent(use_tts_aligned_transcript=)` on the first agent and on every agent a handoff
    /// installs; `None` defers to the session's.
    pub agent_aligned_transcript: Option<bool>,
}

/// Each leg's `max_retry`, and the interval between retries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConnSetup {
    pub stt: u32,
    pub llm: u32,
    pub tts: u32,
    pub interval_ms: u64,
}

impl ConnSetup {
    /// `SessionConnectOptions()`: three retries on every leg, two seconds apart.
    pub const DEFAULT: ConnSetup = ConnSetup { stt: 3, llm: 3, tts: 3, interval_ms: 2_000 };
}

/// `TTSCapabilities.aligned_transcript`, and what the voice then sends.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TtsAlignment {
    /// Not advertised: the voice sends audio only.
    #[default]
    None,
    /// Advertised, and every word comes back stamped, as Cartesia's `word_timestamps` do.
    Words,
    /// Advertised, and nothing comes back: a model and language pair with no timings.
    Missing,
}

/// The room's legs, and whether the person's app sends the audio it buffered before the agent
/// joined (`pre_connect_audio`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomLegs {
    pub pre_connect: Option<PreConnect>,
}

/// What the person's app does with its pre-connect buffer: sends it, never sends it (the
/// session waits out its timeout), or sends one the session cannot read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreConnect {
    Sent,
    Never,
    Malformed,
}

/// The agent the session opens with; a handoff names another.
pub const OPENING_AGENT: &str = "agent";

/// Which nodes the opening agent overrides; every other agent flips some of them
/// (`start::overrides`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSetup {
    pub llm: bool,
    pub tts: bool,
    pub stt: bool,
    /// The `llm_node` answers this text itself (a node returning a `str`), never calling the model.
    pub llm_text: Option<&'static str>,
    /// `transcription_node` and `realtime_audio_output_node` (each as the default does), opened
    /// per segment and per realtime generation over the session's text and audio legs.
    pub transcription: bool,
    pub realtime_audio: bool,
}

/// Keyterm detection as `KeytermDetectionOptions` sets it up, and whether the session's
/// recogniser takes keyterms at all (`STTCapabilities.keyterms`): without it no pass runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeytermDetectionSetup {
    pub stt_takes_keyterms: bool,
    pub turn_interval: u32,
    pub max_keyterms: Option<usize>,
    pub timeout_ms: Option<u64>,
}

/// `AMD(session, ...)`: its switches, its recogniser, and what the room says once it is attached.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmdSetup {
    pub interrupt_on_machine: bool,
    pub ivr_detection: bool,
    pub wait_until_finished: bool,
    /// `AMD(stt=..)`: a recogniser of AMD's own; otherwise the session's finals.
    pub own_stt: bool,
    pub answer: AmdAnswer,
}

/// What the callee does in the room AMD awaits them in: joins with their audio (listening opens
/// at once), joins as a SIP call the case answers later (`Amd { Answer }` sets `sip.callStatus`
/// to `active`), or has not joined (`participant_missing` once the track wait runs out, unless
/// they join in time).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AmdAnswer {
    #[default]
    Track,
    Sip,
    Missing,
}

/// The realtime server a session is built against: what it can do, as its capabilities say.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealtimeSetup {
    /// The server detects the user's turns itself. The driver then has a choice the fake
    /// otherwise never offers it: to leave detection to the server or to switch it off.
    pub turn_detection: bool,
    /// Which server-side detector: `semantic_vad` (the default a session gets) or `server_vad`.
    pub semantic: bool,
    /// What the server's delays and choices are drawn from.
    pub seed: u64,
    /// How often the model calls a tool when it has some, in percent: the developer's prompt
    /// decides it. `None` is the corpus's rate.
    pub tool_calls: Option<u8>,
    /// `RealtimeModel(modalities=["text"])`: the server answers in text, which the session's TTS
    /// speaks when the setup has one.
    pub text_only: bool,
    /// `RealtimeModelFallbackAdapter` over two such servers: a fatal error swaps to the other.
    pub fallback: bool,
    /// A full-duplex speech model (`DuplexRealtimeAdapter`) instead of the server: it detects
    /// turns and speaks on its own; the server's switches above do not apply.
    pub duplex: bool,
}

impl Default for RealtimeSetup {
    fn default() -> Self {
        RealtimeSetup {
            turn_detection: true,
            semantic: true,
            seed: 0,
            tool_calls: None,
            text_only: false,
            fallback: false,
            duplex: false,
        }
    }
}

/// `RoomOptions`' two session-level switches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomSetup {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
}

/// One run, whole: how the session is built and what is sent to it.
///
/// The setup is not optional detail. Some of what a rule needs cannot be reached by any input,
/// because it is fixed when the session is constructed -- speculative generation, the turn mode,
/// whether host hooks are bound. A probe that carried only inputs could never reach those rules,
/// so a case carries both halves.
#[derive(Clone, Debug, Default)]
pub struct Case {
    pub setup: Setup,
    pub inputs: Vec<Input>,
    /// The scheduler's seed: which interleaving of the driver's concurrent work this case ran
    /// under. Part of the case because a finding may exist under one interleaving only. Zero is
    /// the schedule of a hand-written case.
    pub schedule: u64,
    /// Indices into `inputs` of the uniform moves: rows no actor had a reason to send. A failure
    /// in a case with none is one a real call produces with correct providers.
    pub random: Vec<usize>,
    /// How the run's queue picked the next pending event. Arrival order by default.
    pub policy: Policy,
}

impl Default for Setup {
    fn default() -> Self {
        Setup {
            stt: false,
            vad: false,
            eot: false,
            interruption: false,
            dynamic_endpointing: false,
            turn: TurnMode::Default,
            preemptive: false,
            tools: vec!["lookup".into()],
            max_tool_steps: None,
            instructions: "You are a test agent.".into(),
            max_unrecoverable_errors: 3,
            hook_bound_ms: None,
            realtime: None,
            pausable_output: false,
            speech_gap_ms: None,
            user_turn_max_words: None,
            user_turn_max_ms: None,
            transcription_timeout_ms: None,
            tts: true,
            connect_ms: None,
            video: false,
            room_camera: false,
            sync_transcription: false,
            lagging_output: false,
            ivr: None,
            amd: None,
            stt_model: 0,
            tts_latency: None,
            tts_alignment: TtsAlignment::None,
            aligned_transcript: None,
            agent_aligned_transcript: None,
            keyterms: Vec::new(),
            keyterm_detection: None,
            nodes: None,
            room: None,
            text_transforms: None,
            room_legs: None,
            console: false,
            video_sampler: false,
            background_audio: false,
            avatar: false,
            avatar_reports_start: false,
            avatar_fps: 25,
            avatar_latency_ms: 20,
            conn: None,
            chunked_tts: false,
        }
    }
}

// ==============================================================================================
// The simulation
// ==============================================================================================

/// The scheduler seed a run uses when its case names none. The bytes are the crate's former name;
/// changing them would change every interleaving the hand-written regression tests run under.
const SCHEDULER_SEED: &[u8] = b"lk-agents-voice-tests";

/// Run one simulation under the schedule every case ran with before schedules were a dimension.
/// Tests with no case go through here.
pub fn simulate<T>(body: impl std::future::Future<Output = T>) -> T {
    simulate_with(0, body)
}

/// Run one simulation: paused time, a single thread, and a scheduler whose choices are fixed by
/// `schedule`.
///
/// Tokio randomises unbiased `select!` branch order and task polling order per call, which is the
/// interleaving of the driver's concurrent work: two events in one instant, a verdict against an
/// overlap's close, a timer against a command. Pinning the seed is what makes a seed reproduce
/// its run; varying it across cases is what makes the campaign explore interleavings rather than
/// walk one of them a thousand times. Zero is the schedule of a hand-written case.
pub fn simulate_with<T>(schedule: u64, body: impl std::future::Future<Output = T>) -> T {
    let mut seed = SCHEDULER_SEED.to_vec();
    if schedule != 0 {
        seed.extend_from_slice(&schedule.to_le_bytes());
    }
    let _watchdog = watchdog(schedule, queue::Trail::install());
    tokio::runtime::Builder::new_current_thread()
        // Timers only: every leg is in memory, and an I/O driver would poll the OS on every turn
        // of the scheduler, which `event_interval(1)` makes every task poll.
        .enable_time()
        .start_paused(true)
        // The run's own future is polled after every task, so the queue owner can step the driver
        // between any two polls of the tasks it spawned.
        .event_interval(1)
        .rng_seed(tokio::runtime::RngSeed::from_bytes(&seed))
        .build()
        .expect("simulation runtime")
        .block_on(body)
}

/// Real seconds one simulation may take before the watchdog ends the process; `VOICE_HANG_S`
/// overrides it. A soak call, the longest run, takes about a second optimised and sixteen times
/// that unoptimised: this must never fire on a run that is merely slow.
const HANG_S: u64 = 600;

/// What no step count can see: a product task that never yields, or a lock that never comes back,
/// blocks the run's only thread, so nothing on it can notice. A thread beside it can, and the only
/// thing it can do about it is end the process loudly, naming the schedule, rather than leave a
/// campaign parked forever. Disarmed when the returned sender is dropped, with the simulation.
/// It dumps the run's trail: the last steps dispatched and what is still pending.
fn watchdog(schedule: u64, trail: Arc<Mutex<queue::Trail>>) -> std::sync::mpsc::Sender<()> {
    use std::io::Write;
    let limit = std::env::var("VOICE_HANG_S").ok().and_then(|s| s.parse().ok()).unwrap_or(HANG_S);
    let (disarm, armed) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
            armed.recv_timeout(Duration::from_secs(limit))
        {
            // Straight to the stream: the test harness captures `eprintln!` and loses it on abort.
            let _ = writeln!(
                std::io::stderr(),
                "HANG: the simulation of schedule {schedule} (the survey's seed) is still running \
                 after {limit}s of real time: a step or a task is blocked or never yields. \
                 Rerun it with VOICE_SEED={schedule} VOICE_TRACE=1 to see the input it stopped on."
            );
            // The run's thread may hold the trail's lock: never wait on it here.
            let dump = trail.try_lock().map_or_else(|_| "trail locked".into(), |t| t.dump());
            let _ = writeln!(std::io::stderr(), "{dump}");
            std::process::abort();
        }
    });
    disarm
}

/// Steps one virtual instant may hold, in the journal or in the run's own loop, before the run
/// counts as a livelock. Derived from the busiest instant measured with the microphone's frames
/// scheduled as steps (2026-10-28, `Outcome::busiest_instant`): 3,330 loop steps on the 1,024-seed
/// survey, 6,262 on the next 32,768 seeds (seed 30341). Twenty times the busiest is a session
/// feeding itself, and waiting longer only spends real time before the same verdict.
const NO_PROGRESS_STEPS: u32 = 20 * 6_262;

/// The run's spawner: every task of the run goes through it, so nothing bypasses the harness.
/// The product's tasks are retained as real Tokio joins, so a task the driver detached cannot
/// disappear from the test, each with when and from which line it was spawned, so a task that
/// never finishes can be placed in the run and in the source (`every_task_finishes`). What the
/// fakes run beside the session ([`World`]) is scheduled the same way and aborted with the run.
struct Tasks {
    stamps: Stamps,
    spawned: Mutex<Vec<Spawned>>,
    /// The fakes' tasks: the world's, not the product's, so never judged.
    world: Mutex<Vec<JoinHandle<()>>>,
    /// The task bridge, under `Policy::TaskBridge`.
    gate: Option<Arc<queue::Gate>>,
}

struct Spawned {
    at: Observed,
    site: &'static std::panic::Location<'static>,
    task: JoinHandle<()>,
}

impl Spawner for Tasks {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> lk_agents_abi::AbortHandle {
        let site = std::panic::Location::caller();
        let future = match &self.gate {
            Some(gate) => gate.wrap(future),
            None => future,
        };
        let task = tokio::spawn(future);
        let abort = task.abort_handle().into();
        self.spawned.lock().unwrap().push(Spawned { at: self.stamps.now(), site, task });
        abort
    }
}

impl Drop for Tasks {
    fn drop(&mut self) {
        for spawned in self.spawned.get_mut().unwrap().drain(..) {
            spawned.task.abort();
        }
        for task in self.world.get_mut().unwrap().drain(..) {
            task.abort();
        }
    }
}

/// The run's spawner as the fakes hold it: a peer's or provider's own loop (a CLI, an avatar
/// worker, a host source's late report) runs under the run's scheduling like the product's
/// tasks, and ends with the run rather than being judged by it.
pub struct World(Arc<Tasks>);

impl Spawner for World {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> lk_agents_abi::AbortHandle {
        let future = match &self.0.gate {
            Some(gate) => gate.wrap(future),
            None => future,
        };
        let task = tokio::spawn(future);
        let abort = task.abort_handle().into();
        self.0.world.lock().unwrap().push(task);
        abort
    }
}

/// Where a microphone's frames go: the host's own leg, or the person's track in the room.
enum Mic {
    Host(mpsc::UnboundedSender<AudioFrame>),
    /// The track, how many frames went into it at which instant, and, once the room io has read
    /// it, since when it is subscribed: past that instant the room io had its turn to read it
    /// again, so words spoken into it are owed to the session even when nothing reads them.
    Room {
        track: Arc<lk_agents_test_support::rtc::FakeTrackPublication>,
        burst: (u64, u32),
        read: Option<u64>,
    },
    /// The console's socket, as the CLI sends its microphone.
    Console(fakes::FakeCli),
}

/// The most frames pushed into the person's track in one instant: half of what each of the
/// track's two queues holds (`AUDIO_QUEUE_FRAMES`), which drain only as time moves.
const ROOM_BURST_FRAMES: u32 = (lk_agents_abi::AUDIO_QUEUE_FRAMES / 2) as u32;

/// The side table a host ref points into. Releases are no-ops: no host holds these objects.
struct NoopRefs;

impl RefRegistry for NoopRefs {
    fn release(&self, _id: u64) {}
}

// ==============================================================================================
// The run
// ==============================================================================================

/// A real session, the fakes behind it, and everything the run has learned.
pub struct Run {
    /// The session, as a host holds it.
    api: Arc<SessionApi>,
    /// The room it is in: people join and leave it as a case says.
    room: Arc<FakeRoom>,
    /// `aclose()` then `close_io()`, once a case asked for the close, until they resolve.
    closing: Option<BoxFuture<'static, ()>>,
    /// The real runtime owner, stepped over fake ports by the run's deterministic queue.
    /// `None` once they finished, or panicked.
    cores: Option<lk_agents_runtime::voice::Runtime>,
    queue: queue::Queue,
    time: Timebase,
    tasks: Arc<Tasks>,
    stamps: Stamps,
    pub registry: Arc<Registry>,
    /// What the fakes have done and not yet been read. Drained after every input; see [`Record`]
    /// for why reading it later still yields the order things happened in.
    records: mpsc::UnboundedReceiver<Record>,
    /// The way to reach a live fake, owned outright and shared with no one. A delivery is a
    /// lookup in this map and a call, so there is no lock on that path to re-enter.
    controls: BTreeMap<Slot, Control>,
    /// Which output each flushed segment sits on, kept after the segment is gone.
    segments: BTreeMap<u32, u32>,
    /// Journal steps the probe stamped, waiting to be absorbed. The probe runs under the
    /// driver's own lock and cannot touch the recording, so it parks them here.
    steps: Arc<Mutex<Vec<(Observed, Step)>>>,
    /// Host events the probe stamped as the session's drain handed them over.
    host: Arc<Mutex<Vec<(Observed, Host)>>>,
    /// Set by the observer when it stopped a step for journaling without bound at one instant.
    livelock: Arc<Mutex<Option<String>>>,
    pending: BTreeMap<u32, Pending>,
    /// The `VoiceTask.run()` answer each tool body awaits, by call id: answering the tool first
    /// drops the await, as a cancelled body does.
    task_awaits: BTreeMap<String, u32>,
    streams: BTreeMap<u32, Feeding>,
    handles: BTreeMap<u64, Arc<SpeechHandle>>,
    /// Which recognition stream each utterance is on: an `SttInterim` names the utterance, and the
    /// fake behind it is reached through its stream.
    utterances: BTreeMap<u32, u32>,
    /// Each attached microphone, and how many frames it has carried.
    mics: BTreeMap<u32, (Mic, u16)>,
    /// The host replaced the room's microphone: the person's track is no longer what the session
    /// hears, so a rejoin publishes one nobody listens to.
    host_mic: bool,
    /// How many times the person joined the room, which names their microphone track.
    joins: u32,
    /// The CLI on the other end of the console's socket.
    console: Option<fakes::FakeCli>,
    /// The next console request's id.
    next_request: u64,
    /// The background audio player, as `background_audio.py` runs it beside the session.
    background: Option<Arc<BackgroundAudioPlayer>>,
    /// The avatar worker the agent's voice streams to, in its own room.
    avatar: Option<Arc<fakes::FakeAvatarWorker>>,
    /// The agent's sink streaming to it, whose give-up timer after a clear is a deadline.
    avatar_sink: Option<Arc<lk_agents_runtime::avatar::output::DataStreamAudioOutput>>,
    /// The next inline task id to mint. Per-`Run` and not process-global, so two runs of one case
    /// journal the same `task_0001`, `task_0002`, ... -- the determinism suite reads those ids.
    refs: Arc<dyn RefRegistry>,
    /// The session's providers, given again by a handoff whose agent overrides nodes.
    legs: start::Legs,
    pub setup: Setup,
    pub record: Recording,
    /// The most steps the run took at one virtual instant: what `NO_PROGRESS_STEPS` is derived
    /// from.
    pub busiest_instant: u32,
    /// A task the run awaited could never finish: nothing was pending, no wake was armed and no
    /// task could run when its deadline passed. The evidence of a deadlock.
    deadlock: Option<String>,
}

impl Run {
    /// The order of every step dispatched so far, folded: a seed replays it exactly.
    pub fn dispatch_order(&self) -> u64 {
        self.queue.order()
    }
}

use lk_agents_runtime::room_io::FixedRoom;
