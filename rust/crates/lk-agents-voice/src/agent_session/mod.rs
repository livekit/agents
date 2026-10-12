//! `voice/agent_session.py`: the session's record, states, counters and close sequence
//! (`AgentSession`) and the [`SessionShared`] state its tasks read, with the host's [`SessionApi`],
//! the commands, the host events, the driver, its cycle and the clock beside it.

use crate::agent_activity::options::SessionConfig;
use crate::agent_activity::speech::IdleWaiter;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::cycle::{Effects, TurnTimer};
use crate::agent_activity::DriverConfig;
use crate::audio::LaunchGate;
use crate::events::SessionEvent;
use crate::generation::GenEvent;
use crate::io::RealtimeAudioNode;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::{AgentState, CloseReason, UserState};
use lk_agents_abi::{Llm, Maybe, Spawner, Stt, Ts, Tts, Vad};
use lk_agents_voice::io::{AudioOutput, TextOutput};
use std::sync::Arc;
use std::time::Duration;

pub mod api;
pub mod commands;
pub mod cycle;
pub mod driver;
pub mod host_events;
pub mod time;

// The paths the activity files, the fuzz harness and the napi binding name the session's items by.
pub use api::*;
pub use commands::*;
pub use driver::*;
pub use crate::filler_scheduler::FillerId;
pub use host_events::*;
pub use time::*;
pub use lk_agents_abi::voice::session_effects::{Span, SpanEnded, SpanStarted, StartPlan};

/// `_DEFAULT_AEC_WARMUP_DURATION`, as a `Duration`.
#[lk_agents_abi::host(export(name = "default_aec_warmup"))]
pub const DEFAULT_AEC_WARMUP: Duration = Duration::from_secs(3);

/// `voice/agent_session.py`'s `AgentSession`: the options, the providers, the record, the agent and
/// user states with their timers and error counters, the run state, and the close sequence
/// (`_aclose_impl`). The `Driver` orchestrates it together with the
/// [`AgentActivity`](crate::agent_activity::AgentActivity).
pub(crate) struct AgentSession {
    /// `AgentSession.agent_state`, moved only by [`AgentSession::update_agent_state`].
    agent_state: AgentState,
    agent_speaking_span_open: bool,
    /// `_aec_warmup_remaining`; armed once, on the first speaking edge with an audio output.
    aec_warmup_remaining: Duration,
    aec_warmup_armed: bool,
    /// `AgentSession.user_state`, moved only by [`AgentSession::update_user_state`].
    user_state: UserState,
    /// `timers.user_away`: `None` is no away timer at all.
    user_away_timeout: Option<Duration>,
    /// Whether `TurnTimer::UserAway` is armed, so a restart cancels only a timer that exists.
    away_armed: bool,
    /// While one runs the away timer stays off (#6883).
    running_tool_calls: u32,
    /// Guard 1. Cleared when the adaptive detector takes barge-in over.
    by_audio_activity_enabled: bool,
    /// `_user_turn_claims`: while > 0 the user state is pinned to `Speaking`.
    user_turn_claims: u32,
    /// Consecutive unrecoverable errors per leg. STT resets on a non-empty transcript, LLM and
    /// TTS on the edge into `speaking`.
    stt_error_counts: u32,
    llm_error_counts: u32,
    tts_error_counts: u32,
    next_id: u64,
    user_item_id: Option<String>,
    /// `_opts`, the layers and the leg facts they resolve against.
    config: DriverConfig,
    /// `_chat_ctx`: the one record of the session's items, each agent's view of them. Written only
    /// through [`AgentSession::commit_item`] and the view edits beside it.
    chat_ctx: crate::agent::SessionChatCtx,
    /// Bumped by every record write; a preemptive generation planned against an older version is
    /// discarded.
    record_version: u64,
    /// Snapshotted into every reply at creation.
    llm: Option<Arc<dyn Llm>>,
    tts: Option<Arc<dyn Tts>>,
    stt: Option<Arc<dyn Stt>>,
    /// Restored to a task's parent on exit.
    vad: Option<Arc<dyn Vad>>,
    max_unrecoverable_errors: u32,
    /// `AgentSession._tools`.
    session_tools: crate::tool_executor::SessionTools,
    session_tools_host: Vec<lk_agents_abi::HostRef>,
    keyterms: Option<crate::keyterm_detection::KeytermSlot>,
    opening_recorder: Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>>,
    room_link: crate::agent_activity::lifecycle::RoomLinkConfig,
    /// A later disconnect closes nothing, a second close deletes nothing.
    room_deletion_requested: bool,
    /// `_closing_task`: `_aclose_impl` in flight.
    closing_task: Option<CloseState>,
    /// `_global_run_state`.
    runs: crate::run_result::Runs,
    opening_run: Option<crate::run_result::RunSpec>,
    idle_waiters: Vec<IdleWaiter>,
    /// While non-empty, only a waiter naming one of them passes.
    holds: std::collections::BTreeSet<HoldId>,
    next_hold: u64,
}

/// What the builder hands the session besides its options: the providers, the opening frame and
/// what `start()` asked for.
#[derive(Default)]
pub(crate) struct SessionParts {
    pub(crate) llm: Option<Arc<dyn Llm>>,
    pub(crate) tts: Option<Arc<dyn Tts>>,
    pub(crate) stt: Option<Arc<dyn Stt>>,
    pub(crate) vad: Option<Arc<dyn Vad>>,
    pub(crate) max_unrecoverable_errors: u32,
    pub(crate) opening_recorder: Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>>,
    pub(crate) opening_run: Option<crate::run_result::RunSpec>,
    pub(crate) room_link: Option<crate::agent_activity::lifecycle::RoomLinkConfig>,
    /// The opening agent, whose view the constructor context becomes (not announced: the host
    /// has it).
    pub(crate) agent: String,
    pub(crate) chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
}

impl AgentSession {
    pub(crate) fn new(config: DriverConfig, parts: SessionParts) -> Self {
        let SessionParts {
            llm,
            tts,
            stt,
            vad,
            max_unrecoverable_errors,
            opening_recorder,
            opening_run,
            room_link,
            agent,
            chat_ctx,
        } = parts;
        let mut session_chat_ctx = crate::agent::SessionChatCtx::new();
        session_chat_ctx.set_frame(&agent, false);
        if let Some(items) = chat_ctx {
            session_chat_ctx.update_agent_chat_ctx(items, &agent);
        }
        let cfg = &config.effective;
        AgentSession {
            // `_aec_warmup_timer` is `None` until the first `speaking` edge, so warmup cannot
            // mute STT before the agent has said anything.
            agent_state: AgentState::Initializing,
            agent_speaking_span_open: false,
            aec_warmup_remaining: cfg.timers.aec_warmup.unwrap_or(Duration::ZERO),
            aec_warmup_armed: false,
            user_state: UserState::Listening,
            user_away_timeout: cfg.timers.user_away,
            away_armed: false,
            running_tool_calls: 0,
            by_audio_activity_enabled: cfg.turn.interruption.by_audio_activity,
            user_turn_claims: 0,
            stt_error_counts: 0,
            llm_error_counts: 0,
            tts_error_counts: 0,
            next_id: 1,
            user_item_id: None,
            config,
            chat_ctx: session_chat_ctx,
            record_version: 0,
            llm,
            tts,
            stt,
            vad,
            max_unrecoverable_errors,
            session_tools: Default::default(),
            session_tools_host: Vec::new(),
            keyterms: None,
            opening_recorder,
            room_link: room_link.unwrap_or(crate::agent_activity::lifecycle::RoomLinkConfig {
                close_on_disconnect: true,
                delete_room_on_close: false,
                room: String::new(),
            }),
            room_deletion_requested: false,
            closing_task: None,
            runs: Default::default(),
            opening_run,
            idle_waiters: Vec::new(),
            holds: std::collections::BTreeSet::new(),
            next_hold: 0,
        }
    }

    pub fn agent_state(&self) -> AgentState {
        self.agent_state
    }

    /// Returns the previous state when it changed, and **writes the events itself** so no call site
    /// can change the state without telling the host.
    pub fn update_agent_state(
        &mut self,
        now: lk_agents_abi::Ts,
        next: AgentState,
        audio_output_ready: bool,
    ) -> (Option<AgentState>, Effects) {
        let mut eff = Effects::new();
        if self.agent_state == next {
            return (None, eff);
        }
        let speaking = next == AgentState::Speaking;
        if speaking {
            // The pipeline produced audio, so whatever was failing is not failing now.
            self.llm_error_counts = 0;
            self.tts_error_counts = 0;
        }
        if speaking != self.agent_speaking_span_open {
            self.agent_speaking_span_open = speaking;
            let span = Span::AgentSpeaking;
            eff.emit(if speaking {
                SessionEvent::SpanStarted(SpanStarted { span, start_time: None })
            } else {
                SessionEvent::SpanEnded(SpanEnded { span, end_time: None })
            });
        }
        // One-shot wall-clock warmup, on the first speaking turn only.
        if speaking
            && self.aec_warmup_remaining > Duration::ZERO
            && !self.aec_warmup_armed
            && audio_output_ready
        {
            self.aec_warmup_armed = true;
            eff.arm(TurnTimer::AecWarmup, now.saturating_add(self.aec_warmup_remaining));
        }
        let old = std::mem::replace(&mut self.agent_state, next);
        eff.emit(SessionEvent::AgentStateChanged { old, new: next });
        (Some(old), eff)
    }

    /// `AgentSession._on_error`, for an error the attempt loop gave up on. Already closing wins.
    /// `<= max`: with `max_unrecoverable_errors = 3` the session closes on the fourth. Realtime
    /// closes at once.
    pub fn on_unrecoverable_error(&mut self, source: crate::events::ErrorSource) -> bool {
        use crate::events::ErrorSource as S;
        if self.closing_past_edge() {
            return false;
        }
        let count = match source {
            S::Stt => &mut self.stt_error_counts,
            S::Llm => &mut self.llm_error_counts,
            S::Tts => &mut self.tts_error_counts,
            S::Realtime => return true,
        };
        *count += 1;
        *count > self.max_unrecoverable_errors
    }

    /// Any non-empty transcript, interim or final, means the STT recovered.
    pub fn on_transcript(&mut self, text: &str) {
        if !text.is_empty() {
            self.stt_error_counts = 0;
        }
    }

    pub fn user_state(&self) -> UserState {
        self.user_state
    }

    /// The event, when the state changed. The away timer is reconciled by the loop, not here.
    pub fn update_user_state(&mut self, at: lk_agents_abi::Ts, next: UserState) -> Effects {
        let mut eff = Effects::new();
        // Pinned to `Speaking` while a claim is held: a VAD `Listening` mid-claim would end the
        // turn under the holder (`AgentSession._update_user_state`).
        if self.user_turn_claims > 0 && next != UserState::Speaking {
            return eff;
        }
        if self.user_state == next {
            return eff;
        }
        let old = std::mem::replace(&mut self.user_state, next);
        eff.emit(SessionEvent::UserStateChanged { old, new: next, at });
        eff
    }

    /// `_claim_user_turn`'s enter: the depth after. The first claim writes `Speaking`.
    pub fn claim_user_turn(&mut self, at: lk_agents_abi::Ts) -> (u32, Effects) {
        self.user_turn_claims += 1;
        if self.user_turn_claims != 1 {
            return (self.user_turn_claims, Effects::new());
        }
        let eff = self.update_user_state(at, UserState::Speaking);
        (self.user_turn_claims, eff)
    }

    /// `_claim_user_turn`'s exit. Saturating: wrapping would pin the user to `Speaking`. The last
    /// release re-derives the state from `user_speaking`.
    pub fn release_user_turn(
        &mut self,
        at: lk_agents_abi::Ts,
        user_speaking: bool,
    ) -> (u32, Effects) {
        self.user_turn_claims = self.user_turn_claims.saturating_sub(1);
        if self.user_turn_claims != 0 {
            return (self.user_turn_claims, Effects::new());
        }
        let next = if user_speaking { UserState::Speaking } else { UserState::Listening };
        (self.user_turn_claims, self.update_user_state(at, next))
    }

    pub fn user_turn_claims(&self) -> u32 {
        self.user_turn_claims
    }

    /// Both participants silent: the only state in which the away deadline runs. `Initializing`
    /// counts as the agent's silence.
    fn both_listening(&self) -> bool {
        matches!(self.agent_state(), AgentState::Listening | AgentState::Initializing)
            && self.user_state == UserState::Listening
    }

    /// `_set_user_away_timer`: cancel (only if armed), then re-arm unless a guard says not to.
    /// Called on every state edge, a final transcript while both listen, and the last tool end.
    pub fn reconcile_away_timer(&mut self, now: lk_agents_abi::Ts) -> Effects {
        let mut eff = Effects::new();
        if self.closing_past_edge() {
            return eff;
        }
        if self.away_armed {
            self.away_armed = false;
            eff.cancel(TurnTimer::UserAway);
        }
        let Some(timeout) = self.user_away_timeout else { return eff };
        if self.closing_past_edge() || !self.both_listening() || self.running_tool_calls > 0 {
            return eff;
        }
        self.away_armed = true;
        eff.arm(TurnTimer::UserAway, now.saturating_add(timeout));
        eff
    }

    pub fn on_user_away(&mut self, now: lk_agents_abi::Ts) -> Effects {
        self.away_armed = false;
        self.update_user_state(now, UserState::Away)
    }

    /// `_user_input_transcribed`'s timer half. `away` returns to `listening`; `listening` under a
    /// listening agent pushes the deadline out (writing `listening` would be a no-op edge and the
    /// away timeout would fire mid-conversation).
    pub fn on_final_transcript(&mut self, now: lk_agents_abi::Ts) -> Effects {
        match self.user_state {
            UserState::Away => self.update_user_state(now, UserState::Listening),
            UserState::Listening if self.both_listening() => self.reconcile_away_timer(now),
            UserState::Listening | UserState::Speaking => Effects::new(),
        }
    }

    /// A tool call started: the away window is held off (and cancelled) until the last one lands.
    pub fn on_tool_call_started(&mut self, now: lk_agents_abi::Ts) -> Effects {
        self.running_tool_calls += 1;
        if !self.away_armed {
            return Effects::new();
        }
        self.reconcile_away_timer(now)
    }

    /// A tool call ended; the last one to land while both listen restarts the away window
    /// (#6883). Saturating.
    pub fn on_tool_call_ended(&mut self, now: lk_agents_abi::Ts) -> Effects {
        self.running_tool_calls = self.running_tool_calls.saturating_sub(1);
        if self.running_tool_calls != 0 || !self.both_listening() {
            return Effects::new();
        }
        self.reconcile_away_timer(now)
    }

    /// `_on_aec_warmup_expired`. Idempotent. Returns whether interruptions were re-enabled by this
    /// call. The timer itself is the loop's to cancel.
    pub fn expire_aec_warmup(&mut self) -> bool {
        let re_enabled = self.aec_warmup_remaining > Duration::ZERO && !self.closing_past_edge();
        self.aec_warmup_remaining = Duration::ZERO;
        self.aec_warmup_armed = false;
        if re_enabled {
            log::debug!("aec warmup expired, re-enabling interruptions");
        }
        re_enabled
    }

    /// A new `aec_warmup_duration` reached the running driver; a running window is disarmed when
    /// warmup is switched off.
    pub fn set_aec_warmup(&mut self, duration: Option<Duration>) -> Effects {
        let mut eff = Effects::new();
        self.aec_warmup_remaining = duration.unwrap_or(Duration::ZERO);
        if duration.is_none() && self.aec_warmup_armed {
            self.aec_warmup_armed = false;
            eff.cancel(TurnTimer::AecWarmup);
        }
        eff
    }

    pub fn user_transcript_item_id(&mut self, is_final: bool) -> String {
        let id = match self.user_item_id.take() {
            Some(id) => id,
            None => self.mint("item"),
        };
        if !is_final {
            self.user_item_id = Some(id.clone());
        }
        id
    }

    pub fn aec_warmup_remaining(&self) -> Duration {
        self.aec_warmup_remaining
    }

    pub fn aec_warmup_armed(&self) -> bool {
        self.aec_warmup_armed
    }

    pub fn by_audio_activity_enabled(&self) -> bool {
        self.by_audio_activity_enabled
    }

    pub fn set_by_audio_activity_enabled(&mut self, on: bool) {
        self.by_audio_activity_enabled = on;
    }

    /// The close's hard edge was performed (`closing_past_edge` holds from here on).
    pub fn on_close_edge(&mut self) {
        // The loop cancels every armed timer at `Closing`, the away timer among them.
        self.away_armed = false;
        // "always clear aec warmup when closing the session" (`agent_session.AgentSession._aclose_impl`).
        self.expire_aec_warmup();
    }

    pub fn mint(&mut self, prefix: &str) -> String {
        format!("{prefix}_{:04}", self.next_speech_id().0)
    }

    pub fn next_speech_id(&mut self) -> crate::agent_activity::scheduling::SpeechId {
        self.next_id += 1;
        crate::agent_activity::scheduling::SpeechId(self.next_id - 1)
    }
}

/// `AgentSession._aclose_impl` in flight: the step it is at, and the `drain()` callers it answers.
pub(crate) struct CloseState {
    reason: CloseReason,
    error: Option<ProviderError>,
    phase: ClosePhase,
    settled: Option<tokio::sync::oneshot::Receiver<()>>,
    /// Host `drain()`s answered by the close's own drain (the reference's drain waits on the
    /// activity lock `_aclose_impl` holds). Answered `Ok` once the edge has settled.
    drains: Vec<tokio::sync::oneshot::Sender<Result<(), crate::agent_session::DrainRefused>>>,
    /// `drains` was answered: a later caller is answered at once.
    drains_answered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClosePhase {
    /// An inline task chain is being unwound, innermost first, before the remaining frame's
    /// `on_exit`. `deadline` is `None` once the bound passed and the rest was abandoned.
    Unwinding {
        deadline: Option<Ts>,
    },
    Exiting {
        hook: HookId,
    },
    Draining {
        deadline: Ts,
    },
    /// The hard edge was performed; `until` bounds the cut speeches' reports.
    Edge {
        until: Ts,
    },
    Committing {
        until: Ts,
    },
    ClosingLegs,
    Closed,
}

impl ClosePhase {
    pub(crate) fn deadline(self) -> Option<Ts> {
        match self {
            ClosePhase::Unwinding { deadline } => deadline,
            ClosePhase::Draining { deadline } => Some(deadline),
            ClosePhase::Edge { until } | ClosePhase::Committing { until } => Some(until),
            ClosePhase::Exiting { .. } | ClosePhase::ClosingLegs | ClosePhase::Closed => None,
        }
    }
}

/// `DrainRefused`'s waiters on the close's own drain.
type DrainWaiter = tokio::sync::oneshot::Sender<Result<(), DrainRefused>>;

impl AgentSession {
    /// `AgentSession._tools.extend(tools)`: new names join every request from this command on (no
    /// `AgentConfigUpdate`, as in the reference). The names added.
    pub(crate) fn extend_tools(
        &mut self,
        tools: crate::tool_executor::SessionTools,
    ) -> Vec<String> {
        let mut added = Vec::new();
        for spec in tools.specs {
            if !self.session_tools.specs.iter().any(|o| o.name == spec.name) {
                added.push(spec.name.clone());
                self.session_tools.specs.push(spec);
            }
        }
        if !added.is_empty() {
            self.session_tools_host.extend(tools.host);
        }
        added
    }

    /// `NotGiven` keeps the provider; `Given(None)` clears it. In-flight replies keep their
    /// snapshot.
    pub(crate) fn apply_providers(
        &mut self,
        llm: Maybe<Option<Arc<dyn Llm>>>,
        tts: Maybe<Option<Arc<dyn Tts>>>,
    ) {
        if let Maybe::Given(llm) = llm {
            if let Some(l) = &llm {
                l.prewarm();
            }
            self.llm = llm;
        }
        if let Maybe::Given(tts) = tts {
            if let Some(t) = &tts {
                t.prewarm();
            }
            self.tts = tts;
        }
    }

    /// Does this disconnect end the session? A deletion in flight is the end of the call already
    /// (`not self._delete_room_task`).
    pub(crate) fn disconnect_closes(
        &self,
        reason: lk_agents_abi::voice::rtc::DisconnectReason,
    ) -> bool {
        !self.room_deletion_requested
            && crate::agent_activity::lifecycle::closes_session(
                self.room_link.close_on_disconnect,
                reason,
            )
    }

    /// `delete_room_on_close`, once: the room to delete.
    pub(crate) fn request_room_deletion(&mut self) -> Option<String> {
        if self.room_deletion_requested || !self.room_link.delete_room_on_close {
            return None;
        }
        self.room_deletion_requested = true;
        Some(self.room_link.room.clone())
    }

    /// A floor hold's id, minted at the ask.
    pub(crate) fn mint_hold(&mut self) -> HoldId {
        self.next_hold += 1;
        HoldId(self.next_hold)
    }
}

/// The options, the providers, the record and the waiters, as the driver reads and writes them.
impl AgentSession {
    /// `_opts`, resolved.
    pub(crate) fn options(&self) -> &crate::agent_activity::options::SessionConfig {
        &self.config.effective
    }

    /// The leg facts the options resolve against.
    pub(crate) fn legs(&self) -> &crate::agent_activity::options::LegSet {
        &self.config.legs
    }

    /// The running agent's layer (empty when it set nothing).
    pub(crate) fn agent_options(&self) -> crate::agent_activity::options::SessionPatch {
        self.config.stack.agent.unwrap_or_default()
    }

    /// `update_options(..)`: merged into the session layer, re-resolved.
    pub(crate) fn configure(
        &mut self,
        patch: &crate::agent_activity::options::SessionPatch,
    ) -> crate::agent_activity::ConfigDiff {
        self.config.configure(patch)
    }

    /// A new agent's layer, replaced whole, resolved over the STT and VAD the update left.
    pub(crate) fn set_agent_options(
        &mut self,
        stt: bool,
        vad: bool,
        options: crate::agent_activity::options::SessionPatch,
    ) -> crate::agent_activity::ConfigDiff {
        self.config.legs.stt = stt;
        self.config.legs.vad = vad;
        self.config.set_agent(options)
    }

    /// The turn detection mode a realtime session refined the resolved one to.
    pub(crate) fn set_turn_detection_mode(
        &mut self,
        mode: crate::audio_recognition::TurnDetectionMode,
    ) {
        self.config.effective.turn.owner = mode;
    }

    pub(crate) fn llm(&self) -> Option<&Arc<dyn Llm>> {
        self.llm.as_ref()
    }

    pub(crate) fn tts(&self) -> Option<&Arc<dyn Tts>> {
        self.tts.as_ref()
    }

    pub(crate) fn stt(&self) -> Option<&Arc<dyn Stt>> {
        self.stt.as_ref()
    }

    pub(crate) fn vad(&self) -> Option<&Arc<dyn Vad>> {
        self.vad.as_ref()
    }

    /// The STT an agent change reopened, or closed.
    pub(crate) fn set_stt(&mut self, stt: Option<Arc<dyn Stt>>) {
        self.stt = stt;
    }

    /// The VAD an agent change reopened, or closed.
    pub(crate) fn set_vad(&mut self, vad: Option<Arc<dyn Vad>>) {
        self.vad = vad;
    }

    /// `AgentSession._tools`.
    pub(crate) fn session_tools(&self) -> &crate::tool_executor::SessionTools {
        &self.session_tools
    }

    pub(crate) fn session_tools_host(&self) -> &[lk_agents_abi::HostRef] {
        &self.session_tools_host
    }

    /// The recorder `start(record=..)` attached, once.
    pub(crate) fn take_opening_recorder(
        &mut self,
    ) -> Option<Arc<dyn lk_agents_media::recorder::AudioRecorder>> {
        self.opening_recorder.take()
    }

    /// The run `start(capture_run=True)` opened, once.
    pub(crate) fn take_opening_run(&mut self) -> Option<crate::run_result::RunSpec> {
        self.opening_run.take()
    }

    /// The session's record: every frame's items, each agent's view of them.
    pub(crate) fn chat_ctx(&self) -> &crate::agent::SessionChatCtx {
        &self.chat_ctx
    }

    /// Bumped by every record write; a preemptive generation planned against an older version is
    /// discarded.
    pub(crate) fn record_version(&self) -> u64 {
        self.record_version
    }

    /// The one write into the record. An item added or replaced moves the record's version.
    pub(crate) fn commit_item(
        &mut self,
        item: lk_agents_abi::ChatItem,
        source: crate::agent::Source,
        visibility: crate::agent::Visibility,
        agent: &str,
    ) -> (crate::agent::Committed, crate::agent::CommitOutcome) {
        let commit = self.chat_ctx.commit(item, source, visibility, agent);
        match commit.1 {
            crate::agent::CommitOutcome::Added | crate::agent::CommitOutcome::Replaced => {
                self.record_version += 1;
            }
            crate::agent::CommitOutcome::Filtered => {}
        }
        commit
    }

    /// `Agent.update_chat_ctx`: `agent`'s view becomes exactly `items` (the version is kept).
    pub(crate) fn update_agent_chat_ctx(
        &mut self,
        items: Vec<lk_agents_abi::ChatItem>,
        agent: &str,
    ) {
        self.chat_ctx.update_agent_chat_ctx(items, agent);
    }

    /// The driving frame is `agent`; `carry` makes the previous frame's view its own. The ids
    /// carried.
    pub(crate) fn set_chat_ctx_frame(&mut self, agent: &str, carry: bool) -> Vec<String> {
        self.chat_ctx.set_frame(agent, carry)
    }

    /// `exclude_function_call`: an inline task's calls leave every view.
    pub(crate) fn exclude_task_calls(&mut self, task_agent: &str) {
        self.chat_ctx.exclude_task_calls(task_agent);
    }

    pub(crate) fn keyterms(&self) -> Option<&crate::keyterm_detection::KeytermSlot> {
        self.keyterms.as_ref()
    }

    pub(crate) fn keyterms_mut(&mut self) -> Option<&mut crate::keyterm_detection::KeytermSlot> {
        self.keyterms.as_mut()
    }

    /// `KeytermDetector.start`.
    pub(crate) fn set_keyterms(&mut self, slot: crate::keyterm_detection::KeytermSlot) {
        self.keyterms = Some(slot);
    }

    /// `_global_run_state`.
    pub(crate) fn runs(&self) -> &crate::run_result::Runs {
        &self.runs
    }

    pub(crate) fn runs_mut(&mut self) -> &mut crate::run_result::Runs {
        &mut self.runs
    }

    /// A `wait_for_idle()` or `hold_floor()` ask, answered at the first idle instant it passes.
    pub(crate) fn ask_idle(&mut self, waiter: IdleWaiter) {
        self.idle_waiters.push(waiter);
    }

    pub(crate) fn has_idle_waiters(&self) -> bool {
        !self.idle_waiters.is_empty()
    }

    /// The asks the holds let through, in ask order, their holds granted. While a hold is held
    /// only an ask inside it passes (`_idle_holds > 0 and not _IdleHoldContextVar.get()`).
    pub(crate) fn admit_idle_waiters(&mut self) -> Vec<IdleWaiter> {
        let mut admitted = Vec::new();
        let mut kept = Vec::new();
        for waiter in std::mem::take(&mut self.idle_waiters) {
            let passes =
                self.holds.is_empty() || waiter.hold.is_some_and(|h| self.holds.contains(&h));
            if !passes {
                kept.push(waiter);
                continue;
            }
            if let Some(hold) = waiter.done.hold() {
                self.holds.insert(hold);
            }
            admitted.push(waiter);
        }
        self.idle_waiters = kept;
        admitted
    }

    /// The asks `speech`'s tools made, withdrawn with it.
    pub(crate) fn withdraw_idle_asks(
        &mut self,
        speech: crate::agent_activity::scheduling::SpeechId,
    ) -> Vec<IdleWaiter> {
        let (gone, kept) = std::mem::take(&mut self.idle_waiters)
            .into_iter()
            .partition(|w| w.speech == Some(speech));
        self.idle_waiters = kept;
        gone
    }

    /// Every ask still waiting, taken.
    pub(crate) fn take_idle_waiters(&mut self) -> Vec<IdleWaiter> {
        std::mem::take(&mut self.idle_waiters)
    }

    /// `release_floor(hold)`: whether it was held.
    pub(crate) fn release_hold(&mut self, hold: HoldId) -> bool {
        self.holds.remove(&hold)
    }

    /// `hold` is still asked for or held: its `foreground()` scope has not exited.
    pub(crate) fn hold_live(&self, hold: HoldId) -> bool {
        self.holds.contains(&hold) || self.idle_waiters.iter().any(|w| w.done.hold() == Some(hold))
    }

    /// `_idle_holds > 0`.
    pub(crate) fn has_idle_holds(&self) -> bool {
        !self.holds.is_empty()
    }
}

/// The close sequence, as the session holds it: the component's own reads and writes of
/// `_closing_task`; the sequence itself is the `Driver`'s (`agent_activity::lifecycle`).
impl AgentSession {
    /// `_closing`: a close was requested (the reference sets it before the drain).
    pub(crate) fn is_closing(&self) -> bool {
        self.closing_task.is_some()
    }

    /// Past the hard edge no command may take effect any more.
    pub(crate) fn closing_past_edge(&self) -> bool {
        self.closing_task.as_ref().is_some_and(|c| {
            !matches!(
                c.phase,
                ClosePhase::Unwinding { .. }
                    | ClosePhase::Exiting { .. }
                    | ClosePhase::Draining { .. }
            )
        })
    }

    pub(crate) fn close_phase(&self) -> Option<ClosePhase> {
        self.closing_task.as_ref().map(|c| c.phase)
    }

    pub(crate) fn close_reason(&self) -> Option<CloseReason> {
        self.closing_task.as_ref().map(|c| c.reason)
    }

    pub(crate) fn close_deadline(&self) -> Option<Ts> {
        self.closing_task.as_ref().and_then(|c| c.phase.deadline())
    }

    pub(crate) fn set_close_phase(&mut self, phase: ClosePhase) {
        if let Some(state) = &mut self.closing_task {
            state.phase = phase;
        }
    }

    /// A second request joins the close in progress: a drain asked for again restarts its bound.
    /// `false` when no close is running.
    pub(crate) fn rejoin_close(&mut self, drain: bool, deadline: Ts) -> bool {
        let Some(state) = &mut self.closing_task else { return false };
        if let (true, ClosePhase::Draining { deadline: bound }) = (drain, &mut state.phase) {
            *bound = deadline;
        }
        true
    }

    /// `_closing_task` begins, unwinding until `deadline`, answering `drains` with its drain.
    pub(crate) fn start_close(
        &mut self,
        reason: CloseReason,
        error: Option<ProviderError>,
        deadline: Ts,
        drains: Vec<DrainWaiter>,
    ) {
        let phase = ClosePhase::Unwinding { deadline: Some(deadline) };
        let (settled, drains_answered) = (None, false);
        self.closing_task =
            Some(CloseState { reason, error, phase, settled, drains, drains_answered });
    }

    /// The close's drain runs until `deadline`, settled when `settled` answers.
    pub(crate) fn close_draining(
        &mut self,
        deadline: Ts,
        settled: tokio::sync::oneshot::Receiver<()>,
    ) {
        self.set_close_phase(ClosePhase::Draining { deadline });
        if let Some(state) = &mut self.closing_task {
            state.settled = Some(settled);
        }
    }

    /// Whether a caller of `drain()` can still wait on the close's own drain.
    pub(crate) fn close_drains_open(&self) -> bool {
        self.closing_task.as_ref().is_some_and(|c| !c.drains_answered)
    }

    pub(crate) fn join_close_drains(&mut self, waiters: Vec<DrainWaiter>) {
        if let Some(state) = &mut self.closing_task {
            state.drains.extend(waiters);
        }
    }

    /// The close's `drain()` callers, once: empty when answered already or nobody waits.
    pub(crate) fn take_close_drains(&mut self) -> Vec<DrainWaiter> {
        let Some(state) = &mut self.closing_task else { return Vec::new() };
        if std::mem::replace(&mut state.drains_answered, true) {
            return Vec::new();
        }
        std::mem::take(&mut state.drains)
    }

    /// Whoever still waits on the close's drain when the loop ends, answered or not.
    pub(crate) fn remaining_close_drains(&mut self) -> Vec<DrainWaiter> {
        match &mut self.closing_task {
            Some(state) => std::mem::take(&mut state.drains),
            None => Vec::new(),
        }
    }

    /// The scheduling task answered or refused the close's pause: `true` when that ends the drain
    /// now.
    pub(crate) fn close_drain_settled(&mut self) -> bool {
        let Some(state) = &mut self.closing_task else { return false };
        let Some(settled) = &mut state.settled else { return false };
        if let Err(tokio::sync::oneshot::error::TryRecvError::Empty) = settled.try_recv() {
            return false;
        }
        state.settled = None;
        matches!(state.phase, ClosePhase::Draining { .. })
    }

    /// The `close` event's reason and error.
    pub(crate) fn close_outcome(&self) -> Option<(CloseReason, Option<ProviderError>)> {
        self.closing_task.as_ref().map(|state| (state.reason, state.error.clone()))
    }
}

pub struct SessionShared {
    pub time: Timebase,
    /// As of the session's start; the live copy is `Driver::config.effective`.
    pub opts: SessionConfig,
    /// Swappable: a reply keeps the one it started with; the loop re-reads per decision.
    pub audio_out: std::sync::RwLock<Option<Arc<dyn AudioOutput>>>,
    /// Read per plan.
    pub audio_out_enabled: std::sync::atomic::AtomicBool,
    pub text_out_enabled: std::sync::atomic::AtomicBool,
    pub recorder: Arc<std::sync::RwLock<Option<crate::recorder_io::RecorderTap>>>,
    pub recorder_observer: std::sync::Mutex<Option<Arc<dyn crate::io::PlaybackObserver>>>,
    /// Published by the loop every iteration, read by `perform_audio_forwarding` before its first
    /// push.
    pub launch: LaunchGate,
    /// A pipeline is built per segment, so a half-open `**` cannot leak into the next utterance.
    pub tts_transforms: Vec<lk_agents_voice::transcription::text_transforms::TransformSpec>,
    /// A push/emit pair because the node may hold frames back or emit several for one.
    pub realtime_audio: Option<Arc<dyn RealtimeAudioNode>>,
    /// `None` is the text leg off. Swappable like [`audio_out`](Self::audio_out).
    pub text_out: std::sync::RwLock<Option<Arc<dyn TextOutput>>>,
    /// Unbounded: a task must never block on the loop draining it, and there are only a handful per
    /// turn.
    pub events: crate::agent_session::Tx<GenEvent>,
    /// Flips once, in `begin_closing`. Provider drains select on it, or one parked on an idle STT
    /// or VAD would never call `aclose()`.
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
    pub frame: tokio::sync::watch::Sender<FrameView>,
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

    pub fn frame_view(&self) -> FrameView {
        self.frame.borrow().clone()
    }

    /// `session.output.audio_enabled and session.output.audio is not None`.
    pub fn audio_enabled(&self) -> bool {
        self.audio_out().is_some()
            && self.audio_out_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn recorder_tap(&self) -> Option<crate::recorder_io::RecorderTap> {
        self.recorder.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// What a speech freezes at its voice leg, while the legs stay live per segment.
    pub fn audio_out_enabled_flag(&self) -> bool {
        self.audio_out_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn text_out_enabled_flag(&self) -> bool {
        self.text_out_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns whether it changed.
    pub fn set_audio_out_enabled(&self, enabled: bool) -> bool {
        self.audio_out_enabled.swap(enabled, std::sync::atomic::Ordering::Relaxed) != enabled
    }

    /// Returns whether it changed.
    pub fn set_text_out_enabled(&self, enabled: bool) -> bool {
        self.text_out_enabled.swap(enabled, std::sync::atomic::Ordering::Relaxed) != enabled
    }

    pub fn audio_out(&self) -> Option<Arc<dyn AudioOutput>> {
        self.audio_out.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn text_out(&self) -> Option<Arc<dyn TextOutput>> {
        self.text_out.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Returns the old sink for the caller to detach. Driver loop only.
    pub fn set_audio_out(&self, out: Option<Arc<dyn AudioOutput>>) -> Option<Arc<dyn AudioOutput>> {
        std::mem::replace(&mut *self.audio_out.write().unwrap_or_else(|p| p.into_inner()), out)
    }

    /// Returns the old leg.
    pub fn set_text_out(&self, out: Option<Arc<dyn TextOutput>>) -> Option<Arc<dyn TextOutput>> {
        std::mem::replace(&mut *self.text_out.write().unwrap_or_else(|p| p.into_inner()), out)
    }

    /// `session.output.transcription_enabled`: is there a leg.
    pub fn text_enabled(&self) -> bool {
        self.text_out().is_some()
            && self.text_out_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// **As they stand now**, published after every commit and instruction change (a tool may run an
/// inline task or call `update_instructions`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameView {
    pub record: lk_agents_abi::ChatContext,
    pub instructions: crate::generation::Instructions,
    pub finished: std::collections::BTreeSet<SpeechId>,
    /// What a step's follow-up waits for, and whether the activity that made the speech drains
    /// (`self.scheduling_paused` when the step's tools were done, agent_activity.py:3659).
    pub tools_committed: std::collections::BTreeMap<(SpeechId, u32), bool>,
}

#[cfg(test)]
mod label_tests {
    use super::*;
    use crate::agent_session::cycle::{Effect, TimerOp};
    use lk_agents_abi::Ts;

    fn state(aec: Option<Duration>) -> AgentSession {
        let legs = crate::agent_activity::options::LegSet::none();
        let mut config = DriverConfig::new(Default::default(), legs);
        config.effective.timers.aec_warmup = aec;
        AgentSession::new(config, SessionParts::default())
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
        assert!(!s.aec_warmup_armed());
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
        assert!(s.aec_warmup_armed());

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
        assert!(s.aec_warmup_armed());

        let eff = s.set_aec_warmup(None);
        assert_eq!(eff.as_slice(), &[Effect::cancel(TurnTimer::AecWarmup)]);
        assert!(!s.aec_warmup_armed());
        assert_eq!(s.aec_warmup_remaining(), Duration::ZERO);

        let mut s = state(Some(Duration::from_secs(3)));
        let mut eff = s.set_aec_warmup(None);
        eff.append(s.update_agent_state(t0, AgentState::Speaking, true).1);
        assert!(arms(&eff).is_empty() && !s.aec_warmup_armed(), "{eff:?}");
    }
}
