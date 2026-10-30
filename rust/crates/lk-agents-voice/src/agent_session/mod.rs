//! `voice/agent_session.py`: the session's state, one struct per concern (`AgentSession`), with the
//! commands, the host events, the driver and the step clock beside it.

use crate::agent_activity::DriverConfig;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::{Effect, Effects, Out, Timer};
use lk_agents_abi::resources::{AudioOutputRef, ResourceId, TextOutputRef};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::voice::io::AudioOutputCapabilities;
use lk_agents_abi::{Maybe, Ts};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

pub(crate) mod agent_user_state;
pub(crate) mod closing;
pub mod commands;
pub mod driver;
pub mod effects;
pub mod host_events;
pub mod initial;
pub mod lifecycle;
pub mod nodes;
pub mod output;
pub(crate) mod storage;
pub mod time;

// The paths the activity files, the fuzz harness and the napi binding name the session's items by.
pub(crate) use agent_user_state::AgentUserState;
pub(crate) use closing::{ClosePhase, Closing};
pub use commands::*;
pub use driver::*;
pub use host_events::*;
pub use initial::InitialSession;
pub use time::*;

/// `voice/agent_session.py`'s `AgentSession`. The `Driver` orchestrates it together with the
/// [`AgentActivity`](crate::agent_activity::AgentActivity).
pub(crate) struct VoiceSession {
    /// `session.usage`: provider metrics records and tool bodies started (usage.md "Voice
    /// sessions"), published with the view.
    pub(crate) usage: lk_agents_core::usage::Ledger,
    /// The model and synthesis requests each speech started and the ledger still holds
    /// unmeasured, oldest first: a metrics record of the speech measures the oldest of its kind;
    /// the speech's release leaves the rest unmeasured.
    pub(crate) requests: std::collections::BTreeMap<
        (SpeechId, lk_agents_core::usage::ModelKind),
        std::collections::VecDeque<u64>,
    >,
    /// The realtime responses created and not yet measured, by response id.
    pub(crate) realtime_requests: std::collections::BTreeMap<String, u64>,
    /// The request of each recognition stream opened and not yet ended, by stream.
    pub(crate) stt_requests: std::collections::BTreeMap<lk_agents_abi::resources::ResourceId, u64>,
    pub(crate) next_request: u64,
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
    pub(crate) vad: Option<lk_agents_abi::resources::VadRef>,
    /// `session.output` as the core decides with it.
    pub(crate) outputs: Outputs,
    pub(crate) output_bindings: output::OutputBindings,
    /// `AgentSession._tools`, and the hosts that run them.
    pub(crate) tools: crate::tool_executor::Toolset,
    pub(crate) tool_hosts: Vec<lk_agents_abi::HostRef>,
    pub(crate) keyterms: Option<crate::keyterm_detection::KeytermSlot>,
    /// `stt_context_options["forward_chat_context"]` (default on).
    pub(crate) forward_chat_context: bool,
    /// `start(record=..)`, attached by the opening.
    pub(crate) start_recorder: Option<Arc<dyn lk_agents_abi::recorder::AudioRecorder>>,
    pub(crate) recorder_attached: bool,
    pub(crate) room: crate::room_io::RoomLink,
    /// `_closing_task`: `_aclose_impl` in flight.
    pub(crate) closing: Option<Closing>,
    /// `_global_run_state`.
    pub(crate) runs: crate::run_result::Runs,
    pub(crate) idle: IdleHolds,
    /// The providers held at the end of the last step, and those handed over since.
    pub(crate) providers: std::collections::BTreeSet<lk_agents_abi::resources::ResourceId>,
    /// `expressive=`, the session's and the active agent's.
    pub(crate) expressive: crate::generation::ExpressiveLayers,
    /// `session._amd._on_end_of_turn(info)` would answer `True`: the AMD's (`SkipReplies`).
    pub(crate) amd_skips_replies: bool,
    /// The session's OTel span tree.
    pub(crate) trace: crate::telemetry::SessionTrace,
    pub(crate) storage: storage::StorageTurns,
    /// The frontend expert event queue, kept across agent replacements.
    pub(crate) experts: crate::experts::Experts,
}

/// What the builder hands the session besides its options: the providers, the opening agent and
/// what `start()` asked for.
#[derive(Default)]
pub struct SessionParts {
    pub llm: Option<lk_agents_abi::resources::LlmRef>,
    pub tts: Option<lk_agents_abi::resources::TtsRef>,
    pub stt: Option<lk_agents_abi::resources::SttRef>,
    pub vad: Option<lk_agents_abi::resources::VadRef>,
    pub outputs: Outputs,
    pub max_unrecoverable_errors: u32,
    pub start_recorder: Option<Arc<dyn lk_agents_abi::recorder::AudioRecorder>>,
    pub room_link: Option<crate::room_io::RoomLinkConfig>,
    /// The opening agent, whose view the constructor context becomes (not announced: the host
    /// has it).
    pub agent: String,
    pub chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    pub session_id: String,
    pub storage_enabled: bool,
}

impl VoiceSession {
    pub(crate) fn new(config: DriverConfig, parts: SessionParts, next_id: NextId) -> Self {
        let SessionParts {
            llm,
            tts,
            stt,
            vad,
            outputs,
            max_unrecoverable_errors,
            start_recorder,
            room_link,
            agent,
            chat_ctx: opening_items,
            session_id,
            storage_enabled,
        } = parts;
        let mut chat_ctx = crate::agent::SessionChatCtx::new(&agent);
        if let Some(items) = opening_items {
            chat_ctx.update_agent_chat_ctx(items, &agent);
        }
        let cfg = &config.effective;
        VoiceSession {
            usage: Default::default(),
            requests: Default::default(),
            realtime_requests: Default::default(),
            stt_requests: Default::default(),
            next_request: 0,
            states: AgentUserState::new(cfg.timers.user_away()),
            aec: AecWarmup::new(cfg.timers.aec_warmup()),
            by_audio_activity: cfg.turn.interruption.by_audio_activity(),
            errors: ErrorCounts { max: max_unrecoverable_errors, ..Default::default() },
            ids: IdMint { session: session_id, next: next_id, user_item: None },
            config,
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            outputs,
            output_bindings: Default::default(),
            tools: Default::default(),
            tool_hosts: Vec::new(),
            keyterms: None,
            forward_chat_context: true,
            start_recorder,
            recorder_attached: false,
            room: crate::room_io::RoomLink::new(room_link),
            closing: None,
            runs: Default::default(),
            idle: IdleHolds::default(),
            providers: Default::default(),
            expressive: Default::default(),
            amd_skips_replies: false,
            trace: Default::default(),
            storage: storage::StorageTurns::new(storage_enabled),
            experts: Default::default(),
        }
    }

    /// This turn's markup guide over the session's TTS. A reply that runs with it off scrubs the
    /// markup from the agent's history (`_pipeline_reply_task`).
    /// A provider request started (usage.md "Counting"); returns its ledger id.
    pub(crate) fn start_request(
        &mut self,
        kind: lk_agents_core::usage::ModelKind,
        provider: &str,
        model: &str,
    ) -> u64 {
        self.next_request += 1;
        self.usage.start(self.next_request, kind, provider, model);
        self.next_request
    }

    pub(crate) fn expressive_guide(&mut self, reply: bool) -> Option<String> {
        let guide = self.expressive.guide(self.tts.as_ref(), lk_agents_abi::Modality::Audio);
        if guide.is_none() && reply {
            self.chat_ctx.strip_assistant_markup();
        }
        guide
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

    /// The close drains or went past: no reply plans a follow-up.
    pub(crate) fn closing_drained(&self) -> bool {
        let before =
            |p: ClosePhase| matches!(p, ClosePhase::Requested | ClosePhase::Exiting { .. });
        self.close_phase().is_some_and(|p| !before(p))
    }

    /// Past the hard edge no command may take effect any more.
    pub(crate) fn closing_past_edge(&self) -> bool {
        self.close_phase().is_some_and(ClosePhase::past_edge)
    }

    /// `AgentSession._tools.extend(tools)`: new names join every request from this command on (no
    /// `AgentConfigUpdate`, as in the reference). The names added.
    pub(crate) fn extend_tools(&mut self, tools: crate::tool_executor::Toolset) -> Vec<String> {
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
        use crate::agent_session::effects::Effect;
        if let Maybe::Given(llm) = llm {
            if let Some(l) = &llm {
                out.machine().push(Effect::ProviderPrewarm(l.id));
            }
            self.llm = llm;
        }
        if let Maybe::Given(tts) = tts {
            if let Some(t) = &tts {
                out.machine().push(Effect::ProviderPrewarm(t.id));
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
            eff.arm(Timer::AecWarmup, now.saturating_add(self.remaining));
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
            eff.cancel(Timer::AecWarmup);
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

/// The session's id sequence: the runtime owns it (the host's `say()` / `generate_reply()` take
/// their speech's id from it as they return the handle) and supplies the core its next value.
pub type NextId = Box<dyn FnMut() -> u64 + Send>;

/// The session's own ids: speeches and items share one sequence, items carry the session's id.
pub struct IdMint {
    session: String,
    next: NextId,
    /// The user item an interim transcript opened, reused until its final.
    user_item: Option<String>,
}

impl IdMint {
    pub(crate) fn session(&self) -> &str {
        &self.session
    }

    pub(crate) fn mint(&mut self, prefix: &str) -> String {
        let n = self.next_speech_id().0;
        format!("{}_{prefix}_{n:04}", self.session)
    }

    pub(crate) fn next_speech_id(&mut self) -> SpeechId {
        SpeechId((self.next)())
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

/// Runtime-published values accepted by the session's decision core.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionView {
    pub launch_holds: std::collections::BTreeSet<SpeechId>,
    pub config: DriverConfig,
    pub agent: String,
    pub vad: Option<lk_agents_abi::resources::VadRef>,
    /// `session.usage`'s latest snapshot, and the ledger revision it was taken at.
    pub usage: lk_agents_abi::usage::Usage,
    pub usage_revision: u64,
}

/// `session.output`: the outputs by id and their switches, decided by the core alone and published
/// to the data plane, which reads its sinks from the core's word.
#[derive(Debug, Clone, PartialEq)]
pub struct Outputs {
    pub audio: Option<AudioOutputRef>,
    /// `output.audio_enabled`: the switch, which a speech freezes at its voice leg.
    pub audio_enabled: bool,
    pub transcription: Option<TextOutputRef>,
    /// `output.transcription_enabled`.
    pub transcription_enabled: bool,
}

/// No output, both switches on.
impl Default for Outputs {
    fn default() -> Self {
        Outputs::new(None, None)
    }
}

impl Outputs {
    pub fn new(audio: Option<AudioOutputRef>, transcription: Option<TextOutputRef>) -> Self {
        Outputs { audio, audio_enabled: true, transcription, transcription_enabled: true }
    }

    /// Audio plays: a sink, switched on.
    pub(crate) fn plays_audio(&self) -> bool {
        self.audio.is_some() && self.audio_enabled
    }

    /// Transcription shows: a sink, switched on.
    pub(crate) fn shows_transcription(&self) -> bool {
        self.transcription.is_some() && self.transcription_enabled
    }

    /// The resources the selection names.
    pub(crate) fn ids(&self) -> impl Iterator<Item = ResourceId> + use<> {
        [self.audio.as_ref().map(|r| r.id), self.transcription.as_ref().map(|r| r.id)]
            .into_iter()
            .flatten()
    }

    /// The pause facts: what the sink can do, and whether audio plays.
    pub(crate) fn pause_facts(&self) -> (AudioOutputCapabilities, bool) {
        let capabilities = self.audio.as_ref().map(|a| a.capabilities).unwrap_or_default();
        (capabilities, self.plays_audio())
    }
}

impl Driver {
    /// The one write of `session.output`, published to the data plane at once.
    pub(crate) fn set_outputs<R>(&mut self, f: impl FnOnce(&mut Outputs) -> R) -> R {
        let r = f(&mut self.session.outputs);
        self.outbox.push(Effect::Outputs(self.session.outputs.clone()));
        r
    }
}

/// The providers the core holds by id: received in commands, kept by the session or a switch. One
/// no longer held anywhere at the end of a step is released.
impl Driver {
    /// What the core holds now.
    pub(crate) fn providers_held(&self) -> BTreeSet<ResourceId> {
        let s = &self.session;
        let slots = [
            s.llm.as_ref().map(|r| r.id),
            s.tts.as_ref().map(|r| r.id),
            s.stt.as_ref().map(|r| r.id),
            s.vad.as_ref().map(|r| r.id),
            s.keyterms.as_ref().and_then(crate::keyterm_detection::KeytermSlot::resource),
            self.activity.rt_session.as_ref().map(|r| r.id),
            s.outputs.audio.as_ref().map(|r| r.id),
            s.outputs.transcription.as_ref().map(|r| r.id),
        ];
        let updates = self.activity.handoff.updates().chain(self.activity.voice_tasks.parents());
        slots
            .into_iter()
            .flatten()
            .chain(s.output_bindings.held())
            .chain(updates.flat_map(|u| u.resources()))
            .collect()
    }

    /// Every provider the core held or was handed since the last step and holds no more.
    pub(crate) fn release_providers(&mut self, out: &mut Out) {
        let held = self.providers_held();
        for &id in self.session.providers.difference(&held) {
            out.machine().push(Effect::ProviderRelease(id));
        }
        self.session.providers = held;
    }

    /// The loop ended: nothing is held.
    pub fn release_all_providers(&mut self) {
        self.session.providers.extend(self.providers_held());
        for id in std::mem::take(&mut self.session.providers) {
            self.outbox.push(Effect::ProviderRelease(id));
        }
    }
}

/// A `wait_for_idle()` or `wait_for_idle_and_hold()` ask, waiting for an idle instant.
pub(crate) struct IdleWaiter {
    /// The hold the caller is inside, which it may pass through.
    pub(crate) hold: Option<HoldId>,
    /// The speech whose tool asked; its interruption withdraws the ask. `None` for the host.
    pub(crate) speech: Option<SpeechId>,
    /// `wait_for_idle_and_hold()`'s hold: minted at the ask, granted when idle answers it.
    pub(crate) grant: Option<HoldId>,
    pub(crate) call: Option<RequestId>,
}

impl IdleWaiter {
    pub(crate) fn answer(&self, result: Result<(), IdleRefused>) -> Answer {
        match self.grant {
            None => Answer::Idle(result),
            Some(id) => Answer::Hold(result.map(|()| id)),
        }
    }
}

/// The idle asks and the floor holds that only their own asks pass (`_idle_holds`).
#[derive(Default)]
pub(crate) struct IdleHolds {
    waiters: Vec<IdleWaiter>,
    /// While non-empty, only a waiter naming one of them passes.
    holds: BTreeSet<HoldId>,
    next_hold: u64,
}

impl IdleHolds {
    /// A floor hold's id, minted at the ask.
    pub(crate) fn mint_hold(&mut self) -> HoldId {
        self.next_hold += 1;
        HoldId(self.next_hold)
    }

    /// Answered at the first idle instant the ask passes.
    pub(crate) fn ask(&mut self, waiter: IdleWaiter) {
        self.waiters.push(waiter);
    }

    pub(crate) fn has_waiters(&self) -> bool {
        !self.waiters.is_empty()
    }

    /// `_idle_holds > 0`.
    pub(crate) fn held(&self) -> bool {
        !self.holds.is_empty()
    }

    /// The asks the holds let through, in ask order, their holds granted as they pass: a hold
    /// granted holds off the asks after it. While a hold is held only an ask inside it passes
    /// (`_idle_holds > 0 and not _IdleHoldContextVar.get()`).
    pub(crate) fn admit(&mut self) -> Vec<IdleWaiter> {
        let mut admitted = Vec::new();
        for waiter in std::mem::take(&mut self.waiters) {
            if !self.holds.is_empty() && !waiter.hold.is_some_and(|h| self.holds.contains(&h)) {
                self.waiters.push(waiter);
                continue;
            }
            self.holds.extend(waiter.grant);
            admitted.push(waiter);
        }
        admitted
    }

    /// The asks `speech`'s tools made, withdrawn with it.
    pub(crate) fn withdraw(&mut self, speech: SpeechId) -> Vec<IdleWaiter> {
        let (gone, kept) =
            std::mem::take(&mut self.waiters).into_iter().partition(|w| w.speech == Some(speech));
        self.waiters = kept;
        gone
    }

    /// Every ask still waiting.
    pub(crate) fn take_waiters(&mut self) -> Vec<IdleWaiter> {
        std::mem::take(&mut self.waiters)
    }

    /// `release_idle_hold(hold)`: whether it was held.
    pub(crate) fn release(&mut self, hold: HoldId) -> bool {
        self.holds.remove(&hold)
    }

    /// Every hold still held.
    pub(crate) fn take_holds(&mut self) -> BTreeSet<HoldId> {
        std::mem::take(&mut self.holds)
    }
}

#[cfg(test)]
mod idle_tests {
    use super::*;

    fn ask(gate: &mut IdleHolds, hold: Option<HoldId>, grant: Option<HoldId>) {
        gate.ask(IdleWaiter { hold, speech: None, grant, call: None });
    }

    // Catches: holds checked against the state before the pass, so a wait asked after a
    // `wait_for_idle_and_hold()` slips past the hold granted in the same idle instant.
    #[test]
    fn a_hold_granted_in_a_pass_holds_off_the_asks_after_it() {
        let mut gate = IdleHolds::default();
        let floor = gate.mint_hold();
        ask(&mut gate, None, Some(floor));
        ask(&mut gate, None, None);
        ask(&mut gate, Some(floor), None);
        let admitted: Vec<_> = gate.admit().iter().map(|w| (w.hold, w.grant)).collect();
        assert_eq!(admitted, [(None, Some(floor)), (Some(floor), None)]);
        assert!(gate.held() && gate.has_waiters());
        assert!(gate.release(floor));
        assert_eq!(gate.admit().len(), 1, "released, the outside ask passes");
    }
}

#[cfg(test)]
mod label_tests {
    use super::*;
    use crate::agent_session::effects::StepOp;
    use crate::events::SessionEvent;
    use lk_agents_abi::Ts;

    fn state(aec: Option<Duration>) -> VoiceSession {
        let legs = crate::agent_activity::options::LegSet::none();
        let mut config = DriverConfig::new(Default::default(), legs);
        config.effective.timers.aec_warmup = lk_agents_abi::Maybe::Given(aec);
        VoiceSession::new(config, SessionParts::default(), Box::new(|| 1))
    }

    fn arms(eff: &Effects) -> Vec<Ts> {
        eff.transitions
            .iter()
            .filter_map(|e| match e {
                crate::agent_session::effects::Transition::Op(StepOp::Arm(
                    Timer::AecWarmup,
                    at,
                )) => Some(*at),
                _ => None,
            })
            .collect()
    }

    // Catches: the warmup armed on a `speaking` edge with no audio output, armed twice, or the
    // edge missing from the label's announcement.
    #[test]
    fn the_warmup_arms_once_and_only_with_an_audio_output() {
        let mut s = state(Some(Duration::from_secs(3)));
        let t0 = Ts::from_nanos(1_000_000_000);
        let (_, eff) = s.update_agent_state(t0, AgentState::Speaking, false);
        assert!(arms(&eff).is_empty(), "armed with no audio output: {eff:?}");
        assert!(!s.aec.armed());
        let events: Vec<_> = eff
            .transitions
            .iter()
            .filter_map(|signal| match signal {
                crate::agent_session::effects::Transition::Signal(
                    crate::agent_session::effects::Signal::SessionEvent(event),
                ) => Some(event.clone()),
                _ => None,
            })
            .collect();
        assert!(
            matches!(
                events.as_slice(),
                [SessionEvent::AgentStateChanged { new: AgentState::Speaking, .. }]
            ),
            "{events:?}"
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
        assert!(
            matches!(
                eff.transitions.as_slice(),
                [crate::agent_session::effects::Transition::Op(StepOp::Cancel(Timer::AecWarmup))]
            ),
            "{eff:?}"
        );
        assert!(!s.aec.armed());
        assert_eq!(s.aec.remaining(), Duration::ZERO);

        let mut s = state(Some(Duration::from_secs(3)));
        let mut eff = s.aec.set(None);
        eff.append(s.update_agent_state(t0, AgentState::Speaking, true).1);
        assert!(arms(&eff).is_empty() && !s.aec.armed(), "{eff:?}");
    }
}

#[cfg(test)]
mod tests;
