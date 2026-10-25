//! `voice/agent_session.py`: the session's state, one struct per concern (`AgentSession`), and the
//! [`SessionShared`] state its tasks read, with the host's [`SessionApi`], the commands, the host
//! events, the driver, its cycle and the clock beside it.

use crate::agent_activity::DriverConfig;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::{Effects, TurnTimer};
use crate::audio::LaunchGate;
use crate::generation::GenEvent;
use crate::io::RealtimeAudioNode;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::voice::io::{AudioOutput, TextOutput};
use lk_agents_abi::{Maybe, Spawner, Ts};
use std::sync::Arc;
use std::time::Duration;

pub(crate) mod agent_user_state;
pub mod api;
pub mod builder;
pub mod calls;
pub(crate) mod closing;
pub mod commands;
pub mod driver;
pub mod effects;
pub mod host_events;
pub(crate) mod idle;
pub(crate) mod outputs;
pub(crate) mod providers;
pub(crate) mod signals;
pub mod time;

// The paths the activity files, the fuzz harness and the napi binding name the session's items by.
pub use crate::filler_scheduler::FillerId;
pub(crate) use agent_user_state::AgentUserState;
pub use api::*;
pub use builder::*;
pub use calls::Calls;
pub(crate) use closing::{ClosePhase, Closing};
pub use commands::*;
pub use driver::*;
pub use host_events::*;
pub(crate) use idle::{IdleHolds, IdleWaiter};
pub use lk_agents_abi::voice::session_effects::{Span, SpanEnded, SpanStarted, StartPlan};
pub use time::*;

/// `_DEFAULT_AEC_WARMUP_DURATION`, as a `Duration`.
#[lk_agents_abi::host(export(name = "default_aec_warmup"))]
pub const DEFAULT_AEC_WARMUP: Duration = Duration::from_secs(3);

/// `voice/agent_session.py`'s `AgentSession`. The `Driver` orchestrates it together with the
/// [`AgentActivity`](crate::agent_activity::AgentActivity).
pub(crate) struct LiveSession {
    pub(crate) states: AgentUserState,
    pub(crate) aec: AecWarmup,
    /// Guard 1. Cleared when the adaptive detector takes interruption over.
    pub(crate) by_audio_activity: bool,
    pub(crate) errors: ErrorCounts,
    pub(crate) ids: IdMint,
    /// `_opts`, the layers and the leg facts they resolve against.
    pub(crate) config: DriverConfig,
    /// `_chat_ctx`: the one record of the session's items, each agent's view of them.
    pub(crate) chat_ctx: crate::agent::SessionChatCtx,
    /// Snapshotted into every reply at creation.
    pub(crate) llm: Option<lk_agents_abi::resources::LlmRef>,
    pub(crate) tts: Option<lk_agents_abi::resources::TtsRef>,
    pub(crate) stt: Option<lk_agents_abi::resources::SttRef>,
    /// Restored to a task's parent on exit.
    pub(crate) vad: Option<lk_agents_abi::resources::VadRef>,
    /// `session.output` as the core decides with it.
    pub(crate) outputs: outputs::Outputs,
    /// `AgentSession._tools`, and the hosts that run them.
    pub(crate) tools: crate::tool_executor::SessionTools,
    pub(crate) tool_hosts: Vec<lk_agents_abi::HostRef>,
    pub(crate) keyterms: Option<crate::keyterm_detection::KeytermSlot>,
    pub(crate) start_options: StartOptions,
    pub(crate) room: crate::room_io::link::RoomLink,
    /// `_closing_task`: `_aclose_impl` in flight.
    pub(crate) closing: Option<Closing>,
    /// `_global_run_state`.
    pub(crate) runs: crate::run_result::Runs,
    pub(crate) idle: IdleHolds,
    /// The providers held at the end of the last step, and those handed over since.
    pub(crate) providers: std::collections::BTreeSet<lk_agents_abi::resources::ResourceId>,
}

/// What the builder hands the session besides its options: the providers, the opening frame and
/// what `start()` asked for.
#[derive(Default)]
pub(crate) struct SessionParts {
    pub(crate) llm: Option<lk_agents_abi::resources::LlmRef>,
    pub(crate) tts: Option<lk_agents_abi::resources::TtsRef>,
    pub(crate) stt: Option<lk_agents_abi::resources::SttRef>,
    pub(crate) vad: Option<lk_agents_abi::resources::VadRef>,
    pub(crate) outputs: outputs::Outputs,
    pub(crate) max_unrecoverable_errors: u32,
    pub(crate) start_options: StartOptions,
    pub(crate) room_link: Option<crate::room_io::link::RoomLinkConfig>,
    /// The opening agent, whose view the constructor context becomes (not announced: the host
    /// has it).
    pub(crate) agent: String,
    pub(crate) chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
}

/// What `start()` asked for, taken once by the opening.
#[derive(Default)]
pub(crate) struct StartOptions {
    /// `start(record=..)`.
    pub(crate) recorder: Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>>,
    /// `start(capture_run=True)`.
    pub(crate) run: Option<crate::run_result::RunSpec>,
}

impl LiveSession {
    pub(crate) fn new(config: DriverConfig, parts: SessionParts) -> Self {
        let SessionParts {
            llm,
            tts,
            stt,
            vad,
            outputs,
            max_unrecoverable_errors,
            start_options,
            room_link,
            agent,
            chat_ctx: opening_items,
        } = parts;
        let mut chat_ctx = crate::agent::SessionChatCtx::new();
        chat_ctx.set_agent(&agent, false);
        if let Some(items) = opening_items {
            chat_ctx.update_agent_chat_ctx(items, &agent);
        }
        let cfg = &config.effective;
        LiveSession {
            states: AgentUserState::new(cfg.timers.user_away),
            aec: AecWarmup::new(cfg.timers.aec_warmup),
            by_audio_activity: cfg.turn.interruption.by_audio_activity,
            errors: ErrorCounts { max: max_unrecoverable_errors, ..Default::default() },
            ids: IdMint { next: 1, user_item: None },
            config,
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            outputs,
            tools: Default::default(),
            tool_hosts: Vec::new(),
            keyterms: None,
            start_options,
            room: crate::room_io::link::RoomLink::new(room_link),
            closing: None,
            runs: Default::default(),
            idle: IdleHolds::default(),
            providers: Default::default(),
        }
    }

    /// The previous state when it changed. A speaking edge means the pipeline produced audio, so
    /// the LLM and TTS recovered, and the first one with an audio output arms the AEC warmup.
    pub(crate) fn update_agent_state(
        &mut self,
        now: Ts,
        next: AgentState,
        audio_output_ready: bool,
    ) -> (Option<AgentState>, Effects) {
        let mut eff = Effects::new();
        let old = self.states.set_agent_state(next, &mut eff);
        if old.is_some() && next == AgentState::Speaking {
            self.errors.on_speaking();
            self.aec.on_speaking(now, audio_output_ready, &mut eff);
        }
        (old, eff)
    }

    /// The close's hard edge was performed (`closing_past_edge` holds from here on).
    pub(crate) fn on_close_edge(&mut self) {
        self.states.on_close_edge();
        // "always clear aec warmup when closing the session" (`AgentSession._aclose_impl`).
        self.aec.expire(true);
    }

    pub(crate) fn close_phase(&self) -> Option<ClosePhase> {
        self.closing.as_ref().map(|c| c.phase)
    }

    /// Past the hard edge no command may take effect any more.
    pub(crate) fn closing_past_edge(&self) -> bool {
        self.close_phase().is_some_and(ClosePhase::past_edge)
    }

    /// `AgentSession._tools.extend(tools)`: new names join every request from this command on (no
    /// `AgentConfigUpdate`, as in the reference). The names added.
    pub(crate) fn extend_tools(
        &mut self,
        tools: crate::tool_executor::SessionTools,
    ) -> Vec<String> {
        let mut added = Vec::new();
        for spec in tools.specs {
            if !self.tools.specs.iter().any(|o| o.name == spec.name) {
                added.push(spec.name.clone());
                self.tools.specs.push(spec);
            }
        }
        if !added.is_empty() {
            self.tool_hosts.extend(tools.host);
        }
        added
    }

    /// `NotGiven` keeps the provider; `Given(None)` clears it. A new one is prewarmed; in-flight
    /// replies keep their snapshot.
    pub(crate) fn apply_providers(
        &mut self,
        llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
        tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
        out: &mut crate::agent_session::effects::Out,
    ) {
        use crate::agent_session::effects::{Effect, ProviderCall};
        if let Maybe::Given(llm) = llm {
            if let Some(l) = &llm {
                out.machine().push(Effect::Provider(ProviderCall::Prewarm(l.id)));
            }
            self.llm = llm;
        }
        if let Maybe::Given(tts) = tts {
            if let Some(t) = &tts {
                out.machine().push(Effect::Provider(ProviderCall::Prewarm(t.id)));
            }
            self.tts = tts;
        }
    }
}

/// `_aec_warmup_remaining`: interruptions stay off for a window armed once, on the first speaking
/// edge with an audio output, so the agent's own echo cannot cut it.
pub(crate) struct AecWarmup {
    remaining: Duration,
    armed: bool,
}

impl AecWarmup {
    fn new(duration: Option<Duration>) -> Self {
        AecWarmup { remaining: duration.unwrap_or(Duration::ZERO), armed: false }
    }

    pub(crate) fn remaining(&self) -> Duration {
        self.remaining
    }

    pub(crate) fn armed(&self) -> bool {
        self.armed
    }

    /// `_aec_warmup_timer` is `None` until the first `speaking` edge, so warmup cannot mute STT
    /// before the agent has said anything.
    fn on_speaking(&mut self, now: Ts, audio_output_ready: bool, eff: &mut Effects) {
        if !self.remaining.is_zero() && !self.armed && audio_output_ready {
            self.armed = true;
            eff.arm(TurnTimer::AecWarmup, now.saturating_add(self.remaining));
        }
    }

    /// `_on_aec_warmup_expired`. Idempotent. Whether interruptions were re-enabled by this call.
    /// The timer itself is the loop's to cancel.
    pub(crate) fn expire(&mut self, closing: bool) -> bool {
        let re_enabled = !self.remaining.is_zero() && !closing;
        self.remaining = Duration::ZERO;
        self.armed = false;
        if re_enabled {
            log::debug!("aec warmup expired, re-enabling interruptions");
        }
        re_enabled
    }

    /// A new `aec_warmup_duration` reached the running driver; a running window is disarmed when
    /// warmup is switched off.
    pub(crate) fn set(&mut self, duration: Option<Duration>) -> Effects {
        let mut eff = Effects::new();
        self.remaining = duration.unwrap_or(Duration::ZERO);
        if duration.is_none() && self.armed {
            self.armed = false;
            eff.cancel(TurnTimer::AecWarmup);
        }
        eff
    }
}

/// Consecutive unrecoverable errors per leg. STT resets on a non-empty transcript, LLM and TTS on
/// the edge into `speaking`.
#[derive(Default)]
pub(crate) struct ErrorCounts {
    stt: u32,
    llm: u32,
    tts: u32,
    max: u32,
}

impl ErrorCounts {
    /// `AgentSession._on_error`, for an error the attempt loop gave up on: whether it closes the
    /// session. `<= max`: with `max_unrecoverable_errors = 3` the fourth closes. Realtime closes
    /// at once.
    pub(crate) fn exhausted_by(&mut self, source: crate::events::ErrorSource) -> bool {
        use crate::events::ErrorSource as S;
        let count = match source {
            S::Stt => &mut self.stt,
            S::Llm => &mut self.llm,
            S::Tts => &mut self.tts,
            S::Realtime => return true,
        };
        *count += 1;
        *count > self.max
    }

    /// Any non-empty transcript, interim or final, means the STT recovered.
    pub(crate) fn on_transcript(&mut self, text: &str) {
        if !text.is_empty() {
            self.stt = 0;
        }
    }

    fn on_speaking(&mut self) {
        self.llm = 0;
        self.tts = 0;
    }
}

/// The session's own ids: speeches and items share one counter.
pub(crate) struct IdMint {
    next: u64,
    /// The user item an interim transcript opened, reused until its final.
    user_item: Option<String>,
}

impl IdMint {
    pub(crate) fn mint(&mut self, prefix: &str) -> String {
        format!("{prefix}_{:04}", self.next_speech_id().0)
    }

    pub(crate) fn next_speech_id(&mut self) -> SpeechId {
        self.next += 1;
        SpeechId(self.next - 1)
    }

    pub(crate) fn user_transcript_item(&mut self, is_final: bool) -> String {
        let id = match self.user_item.take() {
            Some(id) => id,
            None => self.mint("item"),
        };
        if !is_final {
            self.user_item = Some(id.clone());
        }
        id
    }
}

pub struct SessionShared {
    pub time: Timebase,
    /// `session.output`.
    pub output: AgentOutput,
    pub recorder: Arc<std::sync::RwLock<Option<crate::recorder_io::RecorderTap>>>,
    pub recorder_observer:
        std::sync::Mutex<Option<Arc<dyn lk_agents_abi::voice::io::PlaybackObserver>>>,
    /// Published by the loop every iteration, read by `perform_audio_forwarding` before its first
    /// push.
    pub launch: LaunchGate,
    /// A pipeline is built per segment, so a half-open `**` cannot leak into the next utterance.
    pub tts_transforms: Vec<lk_agents_voice::transcription::text_transforms::TransformSpec>,
    /// A push/emit pair because the node may hold frames back or emit several for one.
    pub realtime_audio: Option<Arc<dyn RealtimeAudioNode>>,
    /// Unbounded: a task must never block on the loop draining it, and there are only a handful per
    /// turn.
    pub events: crate::agent_session::Tx<GenEvent>,
    /// Flips once, when the legs close or the runtime is dropped: every forwarder, drain, pump and
    /// reader selects on it, or one parked on an idle source would outlive the session.
    pub closing: tokio::sync::watch::Sender<bool>,
    /// A drain close while tools run: outputs are committed and no follow-up step is planned.
    pub draining: tokio::sync::watch::Sender<bool>,
    /// The host runs hooks (`hook_bound`): a step waits for `function_tools_executed`'s handlers.
    pub hook_host: bool,
    /// Never `tokio::spawn`.
    pub spawner: Arc<dyn Spawner>,
    /// The task enters its own frame; it never reads the driver's `current`.
    pub frames: Arc<lk_agents_abi::utils::frame::FrameTable>,
    /// A task cannot reach `AgentSession::mint`; the `g` prefix keeps the id spaces apart.
    pub gen_ids: std::sync::atomic::AtomicU64,
    pub activity: tokio::sync::watch::Sender<ActivitySnapshot>,
    /// The providers by id, until the runtime owns them.
    pub resources: lk_agents_abi::resources::Resources,
}

impl SessionShared {
    pub fn mint_item_id(&self) -> String {
        let n = self.gen_ids.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("item_g{n:04}")
    }

    /// A closed channel means the session is ending; not an error.
    pub fn report(&self, ev: GenEvent) {
        let _ = self.events.send(ev);
    }

    pub fn trace(&self, step: impl Into<crate::trace::Step>) {
        self.report(GenEvent::Trace(step.into()));
    }

    pub fn recorder_tap(&self) -> Option<crate::recorder_io::RecorderTap> {
        self.recorder.read().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

/// `session.output`: the agent's audio and transcription outputs, each swappable while the session
/// runs (a reply keeps the one it started with) and each with its switch.
pub struct AgentOutput {
    audio: std::sync::RwLock<Option<Arc<dyn AudioOutput>>>,
    audio_enabled: std::sync::atomic::AtomicBool,
    transcription: std::sync::RwLock<Option<Arc<dyn TextOutput>>>,
    transcription_enabled: std::sync::atomic::AtomicBool,
}

impl AgentOutput {
    pub fn new(
        audio: Option<Arc<dyn AudioOutput>>,
        transcription: Option<Arc<dyn TextOutput>>,
    ) -> Self {
        AgentOutput {
            audio: std::sync::RwLock::new(audio),
            audio_enabled: std::sync::atomic::AtomicBool::new(true),
            transcription: std::sync::RwLock::new(transcription),
            transcription_enabled: std::sync::atomic::AtomicBool::new(true),
        }
    }

    pub fn audio(&self) -> Option<Arc<dyn AudioOutput>> {
        self.audio.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn transcription(&self) -> Option<Arc<dyn TextOutput>> {
        self.transcription.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// `output.audio_enabled`: the switch, which a speech freezes at its voice leg.
    pub fn audio_enabled(&self) -> bool {
        self.audio_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `output.transcription_enabled`: the switch.
    pub fn transcription_enabled(&self) -> bool {
        self.transcription_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Audio plays: a sink, switched on.
    pub fn plays_audio(&self) -> bool {
        self.audio().is_some() && self.audio_enabled()
    }

    /// Transcription shows: a sink, switched on.
    pub fn shows_transcription(&self) -> bool {
        self.transcription().is_some() && self.transcription_enabled()
    }

    /// Whether it changed.
    pub fn set_audio_enabled(&self, enabled: bool) -> bool {
        self.audio_enabled.swap(enabled, std::sync::atomic::Ordering::Relaxed) != enabled
    }

    /// Whether it changed.
    pub fn set_transcription_enabled(&self, enabled: bool) -> bool {
        self.transcription_enabled.swap(enabled, std::sync::atomic::Ordering::Relaxed) != enabled
    }

    /// The old sink, for the driver to detach.
    pub fn set_audio(&self, out: Option<Arc<dyn AudioOutput>>) -> Option<Arc<dyn AudioOutput>> {
        std::mem::replace(&mut *self.audio.write().unwrap_or_else(|p| p.into_inner()), out)
    }

    /// The old sink.
    pub fn set_transcription(
        &self,
        out: Option<Arc<dyn TextOutput>>,
    ) -> Option<Arc<dyn TextOutput>> {
        std::mem::replace(&mut *self.transcription.write().unwrap_or_else(|p| p.into_inner()), out)
    }
}

/// **As they stand now**, published after every commit and instruction change (a tool may run an
/// inline task or call `update_instructions`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ActivitySnapshot {
    pub chat_ctx: lk_agents_abi::ChatContext,
    pub instructions: crate::generation::Instructions,
    pub finished: std::collections::BTreeSet<SpeechId>,
    /// What a step's follow-up waits for, and whether the activity that made the speech drains
    /// (`self.scheduling_paused` when the step's tools were done).
    pub tools_committed: std::collections::BTreeMap<(SpeechId, u32), bool>,
}

#[cfg(test)]
mod label_tests {
    use super::*;
    use crate::agent_session::effects::{Effect, TimerOp};
    use crate::events::SessionEvent;
    use lk_agents_abi::Ts;

    fn state(aec: Option<Duration>) -> LiveSession {
        let legs = crate::agent_activity::options::LegSet::none();
        let mut config = DriverConfig::new(Default::default(), legs);
        config.effective.timers.aec_warmup = aec;
        LiveSession::new(config, SessionParts::default())
    }

    fn arms(eff: &Effects) -> Vec<Ts> {
        eff.as_slice()
            .iter()
            .filter_map(|e| match e {
                Effect::Timer(TimerOp::Arm(TurnTimer::AecWarmup, at)) => Some(*at),
                _ => None,
            })
            .collect()
    }

    // Catches: the warmup armed on a `speaking` edge with no audio output, armed twice, or the
    // span edge missing from the label's announcement.
    #[test]
    fn the_warmup_arms_once_and_only_with_an_audio_output() {
        let mut s = state(Some(Duration::from_secs(3)));
        let t0 = Ts::from_nanos(1_000_000_000);
        let (_, eff) = s.update_agent_state(t0, AgentState::Speaking, false);
        assert!(arms(&eff).is_empty(), "armed with no audio output: {eff:?}");
        assert!(!s.aec.armed());
        let events: Vec<_> = eff
            .as_slice()
            .iter()
            .filter_map(|e| match e {
                Effect::Emit(ev) => Some(ev.clone()),
                _ => None,
            })
            .collect();
        assert!(
            matches!(
                events.as_slice(),
                [
                    SessionEvent::SpanStarted(sp),
                    SessionEvent::AgentStateChanged { new: AgentState::Speaking, .. },
                ] if sp.span == Span::AgentSpeaking
            ),
            "the span opens right before the edge it belongs to: {events:?}"
        );

        let _ = s.update_agent_state(t0, AgentState::Listening, true);
        let t1 = Ts::from_nanos(2_000_000_000);
        let (_, eff) = s.update_agent_state(t1, AgentState::Speaking, true);
        assert_eq!(arms(&eff), vec![t1.saturating_add(Duration::from_secs(3))]);
        assert!(s.aec.armed());

        let (_, mut eff) = s.update_agent_state(t1, AgentState::Listening, true);
        eff.append(s.update_agent_state(t1, AgentState::Speaking, true).1);
        assert!(arms(&eff).is_empty(), "armed a second time: {eff:?}");
    }

    // Catches: a new `aec_warmup_duration` reaching the options and not the running window.
    #[test]
    fn switching_warmup_off_disarms_the_window_and_arms_no_other() {
        let mut s = state(Some(Duration::from_secs(3)));
        let t0 = Ts::from_nanos(1_000_000_000);
        let _ = s.update_agent_state(t0, AgentState::Speaking, true);
        assert!(s.aec.armed());

        let eff = s.aec.set(None);
        assert_eq!(eff.as_slice(), &[Effect::cancel(TurnTimer::AecWarmup)]);
        assert!(!s.aec.armed());
        assert_eq!(s.aec.remaining(), Duration::ZERO);

        let mut s = state(Some(Duration::from_secs(3)));
        let mut eff = s.aec.set(None);
        eff.append(s.update_agent_state(t0, AgentState::Speaking, true).1);
        assert!(arms(&eff).is_empty() && !s.aec.armed(), "{eff:?}");
    }
}
