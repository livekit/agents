//! `voice/agent_activity.py`'s `AgentActivity`: the running agent's frame and components, the
//! driver's configuration, and the frame commands (prompt, tools, context, IO) the driver applies.

pub mod events;
pub mod handoff;
pub mod hooks;
pub mod io;
pub mod lifecycle;
pub mod options;
pub mod pause;
pub mod realtime;
pub mod scheduling;
pub mod speech;
pub mod tasks;
pub mod tools;

use crate::agent::{CommitOutcome, Committed, Source, Visibility};
use crate::agent_session::cycle::Cycle;
use crate::agent_activity::options::{ConfigStack, LegSet, SessionConfig};
use crate::agent_activity::pause::{InterruptOutcome, PausedSpeech};
use crate::agent_activity::scheduling::{SchedulingTask, SpeechId};
use crate::agent_session::{Driver, FrameCommand, HostAudioOut, HostTextOut, HostVideo};
use crate::audio::{GateInputs, audio_gate};
use crate::audio_recognition::AudioRecognition;
use crate::trace::{CoreStep, GenerationStep, InputStep, InterruptCause, PlayoutStep, ToolStep};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{ChatItem, Modality, Ts};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

/// What the STT and VAD legs are fed: every model in the tree wants 16 kHz mono.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

pub(crate) use activity::{ActivityParts, AgentActivity};

/// The activity's state, behind its methods: the driver reaches none of its fields.
mod activity {
    use super::*;

    /// `voice/agent_activity.py`'s `AgentActivity`: the running agent's frame (its name, prompt and
    /// tools), its audio recognition, the speech queue and the current speech, the
    /// false-interruption pause, the hooks and inline tasks, and the realtime session. The `Driver`
    /// orchestrates it together with the [`AgentSession`](crate::agent_session::AgentSession): what
    /// spans two of its components is the driver's to answer.
    pub(crate) struct AgentActivity {
        /// The running agent's id (`AgentActivity.agent.id`).
        agent: String,
        instructions: crate::generation::Instructions,
        toolset: Arc<crate::tool_executor::Toolset>,
        tools: crate::tool_executor::Tools,
        /// `AgentActivity._audio_recognition`.
        audio_recognition: AudioRecognition,
        /// `_speech_q`, `_current_speech`, `_scheduling_paused`, `_preemptive_generation`, and every
        /// live speech's handle.
        scheduling_task: SchedulingTask,
        /// `_paused_speech` and the false-interruption timer.
        paused_speech: PausedSpeech,
        /// Kept for the session: late tool items belong to the speech's frame.
        speech_frames: BTreeMap<SpeechId, String>,
        finished: BTreeSet<SpeechId>,
        tools_committed: BTreeMap<(SpeechId, u32), bool>,
        /// Keeps the agent `thinking` across a filler `say` from inside a tool.
        tools_in_flight: BTreeSet<SpeechId>,
        /// Only the first segment's first frame transitions. Cleared when the step is spoken.
        spoke: BTreeSet<SpeechId>,
        /// `_current_speech` as last announced (`FloorChanged`); compared at the end of every cycle.
        announced_current_speech: Option<SpeechId>,
        interrupt_waiters: Vec<crate::agent_activity::pause::InterruptWaiter>,
        hook_bound: Option<Duration>,
        hooks: crate::agent_activity::hooks::Hooks,
        tasks: crate::agent_activity::tasks::Tasks,
        /// The user turn parked on `on_user_turn_completed` (`_user_turn_completed_atask`).
        pending_turn: Option<crate::audio_recognition::PendingTurn>,
        /// So a preemptive generation can be keyed to the turn after it.
        last_turn: Option<crate::audio_recognition::TurnId>,
        /// `generate_reply` matches on it.
        preemptive_turn: Option<crate::audio_recognition::TurnId>,
        /// The record version a preemptive generation was planned against.
        preemptive_record_version: u64,
        /// `_rt_session`. Exclusive with a pipeline.
        rt_session: Option<Arc<dyn lk_agents_abi::RealtimeSession>>,
        realtime_turns: crate::agent_activity::realtime::RealtimeTurns,
        /// Dropping a sender ends its generation.
        realtime_players:
            BTreeMap<SpeechId, tokio::sync::mpsc::UnboundedSender<lk_agents_abi::llm::realtime::Event>>,
        realtime_turn_detection_disabled: bool,
        /// Before the model's refinement, so a live change is one that moved this.
        realtime_resolved_mode: crate::audio_recognition::TurnDetectionMode,
        /// The `turn_detection` the options last said, so a live change is one that moved it.
        realtime_requested_detection: crate::agent_activity::options::TurnDetection,
    }

    /// What the builder hands the activity: the opening frame, its machines, and the realtime session
    /// it opened.
    pub(crate) struct ActivityParts {
        pub(crate) agent: String,
        pub(crate) instructions: crate::generation::Instructions,
        pub(crate) toolset: crate::tool_executor::Toolset,
        pub(crate) audio_recognition: AudioRecognition,
        pub(crate) scheduling_task: SchedulingTask,
        pub(crate) paused_speech: PausedSpeech,
        pub(crate) hook_bound: Option<Duration>,
        pub(crate) rt_session: Option<Arc<dyn lk_agents_abi::RealtimeSession>>,
        pub(crate) realtime_caps: lk_agents_abi::llm::realtime::RealtimeCapabilities,
        pub(crate) realtime_turn_detection_disabled: bool,
        pub(crate) realtime_resolved_mode: crate::audio_recognition::TurnDetectionMode,
        pub(crate) realtime_requested_detection: crate::agent_activity::options::TurnDetection,
    }

    impl AgentActivity {
        pub(crate) fn new(parts: ActivityParts) -> Self {
            let ActivityParts {
                agent,
                instructions,
                toolset,
                audio_recognition,
                scheduling_task,
                paused_speech,
                hook_bound,
                rt_session,
                realtime_caps,
                realtime_turn_detection_disabled,
                realtime_resolved_mode,
                realtime_requested_detection,
            } = parts;
            AgentActivity {
                agent,
                instructions,
                toolset: Arc::new(toolset),
                tools: Default::default(),
                audio_recognition,
                scheduling_task,
                paused_speech,
                speech_frames: BTreeMap::new(),
                finished: BTreeSet::new(),
                tools_committed: BTreeMap::new(),
                tools_in_flight: BTreeSet::new(),
                spoke: BTreeSet::new(),
                announced_current_speech: None,
                interrupt_waiters: Vec::new(),
                hook_bound,
                hooks: Default::default(),
                tasks: Default::default(),
                pending_turn: None,
                last_turn: None,
                preemptive_turn: None,
                preemptive_record_version: 0,
                rt_session,
                realtime_turns: crate::agent_activity::realtime::RealtimeTurns::new(realtime_caps),
                realtime_players: BTreeMap::new(),
                realtime_turn_detection_disabled,
                realtime_resolved_mode,
                realtime_requested_detection,
            }
        }
    }

    /// The activity's components, each with its own methods: the driver orchestrates across them.
    impl AgentActivity {
        /// `_audio_recognition`.
        pub(crate) fn audio_recognition(&self) -> &AudioRecognition {
            &self.audio_recognition
        }

        pub(crate) fn audio_recognition_mut(&mut self) -> &mut AudioRecognition {
            &mut self.audio_recognition
        }

        /// `_scheduling_task` with the state it reads: `_speech_q`, `_current_speech`,
        /// `_scheduling_paused`, `_preemptive_generation`.
        pub(crate) fn scheduling_task(&self) -> &SchedulingTask {
            &self.scheduling_task
        }

        pub(crate) fn scheduling_task_mut(&mut self) -> &mut SchedulingTask {
            &mut self.scheduling_task
        }

        /// `_paused_speech` and the false-interruption timer.
        pub(crate) fn paused_speech(&self) -> &PausedSpeech {
            &self.paused_speech
        }

        pub(crate) fn paused_speech_mut(&mut self) -> &mut PausedSpeech {
            &mut self.paused_speech
        }

        pub(crate) fn hooks(&self) -> &crate::agent_activity::hooks::Hooks {
            &self.hooks
        }

        pub(crate) fn hooks_mut(&mut self) -> &mut crate::agent_activity::hooks::Hooks {
            &mut self.hooks
        }

        /// The inline `AgentTask` stack.
        pub(crate) fn tasks(&self) -> &crate::agent_activity::tasks::Tasks {
            &self.tasks
        }

        pub(crate) fn tasks_mut(&mut self) -> &mut crate::agent_activity::tasks::Tasks {
            &mut self.tasks
        }

        /// `_tool_executor`.
        pub(crate) fn tools(&self) -> &crate::tool_executor::Tools {
            &self.tools
        }

        pub(crate) fn tools_mut(&mut self) -> &mut crate::tool_executor::Tools {
            &mut self.tools
        }

        pub(crate) fn realtime_turns(&self) -> &crate::agent_activity::realtime::RealtimeTurns {
            &self.realtime_turns
        }

        pub(crate) fn realtime_turns_mut(&mut self) -> &mut crate::agent_activity::realtime::RealtimeTurns {
            &mut self.realtime_turns
        }
    }

    /// The running frame: `AgentActivity.agent`, its instructions and tools.
    impl AgentActivity {
        pub(crate) fn agent(&self) -> &str {
            &self.agent
        }

        pub(crate) fn instructions(&self) -> &crate::generation::Instructions {
            &self.instructions
        }

        pub(crate) fn toolset(&self) -> &Arc<crate::tool_executor::Toolset> {
            &self.toolset
        }

        /// A frame change lands: the new agent, prompt and tools. The previous agent's id.
        pub(crate) fn set_frame(
            &mut self,
            agent: String,
            instructions: crate::generation::Instructions,
            toolset: crate::tool_executor::Toolset,
        ) -> String {
            self.instructions = instructions;
            self.toolset = Arc::new(toolset);
            std::mem::replace(&mut self.agent, agent)
        }

        /// `update_instructions`: replies in flight keep the prompt they snapshotted.
        pub(crate) fn set_instructions(&mut self, instructions: crate::generation::Instructions) {
            self.instructions = instructions;
        }

        /// `update_tools`: replies in flight keep the tools they snapshotted.
        pub(crate) fn set_toolset(&mut self, toolset: crate::tool_executor::Toolset) {
            self.toolset = Arc::new(toolset);
        }

        /// What a request planned now is offered: the frame's toolset, then session additions whose
        /// names the frame lacks (a provider refuses duplicates).
        pub(crate) fn advertised_toolset(
            &self,
            session_tools: &crate::tool_executor::SessionTools,
        ) -> Arc<crate::tool_executor::Toolset> {
            let extra: Vec<_> = session_tools
                .specs
                .iter()
                .filter(|s| !self.toolset.specs.iter().any(|o| o.name == s.name))
                .cloned()
                .collect();
            if extra.is_empty() {
                return Arc::clone(&self.toolset);
            }
            let mut toolset = (*self.toolset).clone();
            toolset.specs.extend(extra);
            Arc::new(toolset)
        }

        /// The speech is the running frame's: its late items are attributed to it.
        pub(crate) fn note_speech_frame(&mut self, speech: SpeechId) {
            self.speech_frames.insert(speech, self.agent.clone());
        }

        /// The frame `speech` was created in.
        pub(crate) fn speech_frame(&self, speech: SpeechId) -> Option<&str> {
            self.speech_frames.get(&speech).map(String::as_str)
        }

        /// How long a hook whose answer is needed is waited on; `None` when no host runs hooks.
        pub(crate) fn hook_bound(&self) -> Option<Duration> {
            self.hook_bound
        }
    }

    /// Per-speech facts the frame view and the tool steps read.
    impl AgentActivity {
        pub(crate) fn mark_finished(&mut self, speech: SpeechId) {
            self.finished.insert(speech);
        }

        pub(crate) fn is_finished(&self, speech: SpeechId) -> bool {
            self.finished.contains(&speech)
        }

        pub(crate) fn finished(&self) -> &BTreeSet<SpeechId> {
            &self.finished
        }

        /// `draining`: the activity that made the speech drains, so the step's reply offers no tools.
        pub(crate) fn mark_tools_committed(&mut self, speech: SpeechId, step: u32, draining: bool) {
            self.tools_committed.insert((speech, step), draining);
        }

        pub(crate) fn tools_committed(&self) -> &BTreeMap<(SpeechId, u32), bool> {
            &self.tools_committed
        }

        pub(crate) fn set_tools_in_flight(&mut self, speech: SpeechId, running: bool) {
            if running {
                self.tools_in_flight.insert(speech);
            } else {
                self.tools_in_flight.remove(&speech);
            }
        }

        pub(crate) fn tools_in_flight(&self, speech: SpeechId) -> bool {
            self.tools_in_flight.contains(&speech)
        }

        /// The speeches whose tools run.
        pub(crate) fn speeches_in_tools(&self) -> impl Iterator<Item = SpeechId> + '_ {
            self.tools_in_flight.iter().copied()
        }

        /// The step's first frame: `true` only the first time.
        pub(crate) fn first_frame(&mut self, speech: SpeechId) -> bool {
            self.spoke.insert(speech)
        }

        /// The step was spoken: the next step's first frame transitions again.
        pub(crate) fn step_ended(&mut self, speech: SpeechId) {
            self.spoke.remove(&speech);
        }

        /// `_current_speech` as last announced (`FloorChanged`): the new one when it moved.
        pub(crate) fn announce_current_speech(
            &mut self,
            current: Option<SpeechId>,
        ) -> Option<Option<SpeechId>> {
            if current == self.announced_current_speech {
                return None;
            }
            self.announced_current_speech = current;
            Some(current)
        }

        /// An `interrupt()` waiting for its speeches' release.
        pub(crate) fn add_interrupt_waiter(&mut self, waiter: crate::agent_activity::pause::InterruptWaiter) {
            self.interrupt_waiters.push(waiter);
        }

        pub(crate) fn take_interrupt_waiters(
            &mut self,
        ) -> Vec<crate::agent_activity::pause::InterruptWaiter> {
            std::mem::take(&mut self.interrupt_waiters)
        }

        /// The waiters still waiting, put back.
        pub(crate) fn keep_interrupt_waiters(
            &mut self,
            waiters: Vec<crate::agent_activity::pause::InterruptWaiter>,
        ) {
            self.interrupt_waiters = waiters;
        }
    }

    /// The user turn: the one parked on its hook, and the turn a preemptive generation is keyed to.
    impl AgentActivity {
        /// `_user_turn_completed_atask`: the decided turn waits on its hook.
        pub(crate) fn park_turn(&mut self, turn: crate::audio_recognition::PendingTurn) {
            self.pending_turn = Some(turn);
        }

        pub(crate) fn pending_turn(&self) -> Option<&crate::audio_recognition::PendingTurn> {
            self.pending_turn.as_ref()
        }

        pub(crate) fn take_pending_turn(&mut self) -> Option<crate::audio_recognition::PendingTurn> {
            self.pending_turn.take()
        }

        /// The parked turn, taken only when `hook` is its hook.
        pub(crate) fn take_pending_turn_of(
            &mut self,
            hook: crate::agent_session::HookId,
        ) -> Option<crate::audio_recognition::PendingTurn> {
            self.pending_turn.take_if(|p| p.hook == Some(hook))
        }

        /// The last turn decided.
        pub(crate) fn set_last_turn(&mut self, turn: crate::audio_recognition::TurnId) {
            self.last_turn = Some(turn);
        }

        /// A preemptive generation starts: it belongs to the turn after the last one, planned against
        /// `record_version`.
        pub(crate) fn key_preemptive_generation(&mut self, record_version: u64) {
            self.preemptive_turn =
                Some(crate::audio_recognition::TurnId(self.last_turn.map_or(1, |t| t.0 + 1)));
            self.preemptive_record_version = record_version;
        }

        /// The turn the preemptive generation was keyed to, taken.
        pub(crate) fn take_preemptive_turn(&mut self) -> Option<crate::audio_recognition::TurnId> {
            self.preemptive_turn.take()
        }

        pub(crate) fn preemptive_record_version(&self) -> u64 {
            self.preemptive_record_version
        }
    }

    /// `_rt_session` and the speeches playing its generations.
    impl AgentActivity {
        pub(crate) fn rt_session(&self) -> Option<&Arc<dyn lk_agents_abi::RealtimeSession>> {
            self.rt_session.as_ref()
        }

        /// The model's turn detection was switched off at the open.
        pub(crate) fn realtime_turn_detection_disabled(&self) -> bool {
            self.realtime_turn_detection_disabled
        }

        /// The turn detection mode the options resolved to, before the model's refinement: whether it
        /// moved.
        /// The turn detection the options said, and the mode they resolved to before the model's
        /// refinement: whether either moved.
        pub(crate) fn note_resolved_mode(
            &mut self,
            resolved: crate::audio_recognition::TurnDetectionMode,
            detection: crate::agent_activity::options::TurnDetection,
        ) -> bool {
            let moved =
                resolved != self.realtime_resolved_mode || detection != self.realtime_requested_detection;
            self.realtime_resolved_mode = resolved;
            self.realtime_requested_detection = detection;
            moved
        }

        pub(crate) fn add_realtime_player(
            &mut self,
            speech: SpeechId,
            player: tokio::sync::mpsc::UnboundedSender<lk_agents_abi::llm::realtime::Event>,
        ) {
            self.realtime_players.insert(speech, player);
        }

        pub(crate) fn has_realtime_player(&self, speech: SpeechId) -> bool {
            self.realtime_players.contains_key(&speech)
        }

        /// The event reaches `speech`'s player; a gone player (interrupted) is forgotten.
        pub(crate) fn play_realtime(
            &mut self,
            speech: SpeechId,
            ev: lk_agents_abi::llm::realtime::Event,
        ) {
            if self.realtime_players.get(&speech).is_some_and(|tx| tx.send(ev).is_err()) {
                self.realtime_players.remove(&speech);
            }
        }

        /// Its player's channel closes: the speech finishes with what the user heard.
        pub(crate) fn end_realtime_player(&mut self, speech: SpeechId) {
            self.realtime_players.remove(&speech);
        }

        pub(crate) fn end_realtime_players(&mut self) {
            self.realtime_players.clear();
        }
    }
}

/// The driver's configuration: the layers, the facts, and the resolved answer.
///
/// `effective` is only written by `resolve`; a change reports what moved as a [`ConfigDiff`] so
/// only the affected machines are told.
#[derive(Debug, Clone, PartialEq)]
pub struct DriverConfig {
    pub stack: ConfigStack,
    pub legs: LegSet,
    pub effective: SessionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfigDiff {
    /// The endpointing leaves changed: replace the model.
    pub endpointing: Option<crate::turn::Endpointing>,
    /// The resolved mode changed: `AudioRecognition::set_turn_detection_mode`.
    pub turn_detection_mode: Option<crate::audio_recognition::TurnDetectionMode>,
    /// `timers.aec_warmup` changed (the host's outbound-SIP re-evaluation): the labels' window.
    pub aec_warmup: Option<Option<Duration>>,
}

impl DriverConfig {
    pub fn new(stack: ConfigStack, legs: LegSet) -> Self {
        let effective = stack.resolve(&legs);
        DriverConfig { stack, legs, effective }
    }

    /// Before `start()`: no machine exists yet to tell.
    pub fn edit(
        &mut self,
        f: impl FnOnce(&mut crate::agent_activity::options::SessionPatch, &mut LegSet),
    ) {
        f(&mut self.stack.session, &mut self.legs);
        self.effective = self.stack.resolve(&self.legs);
    }

    /// Merged into the session layer, re-resolved.
    pub fn configure(&mut self, patch: &crate::agent_activity::options::SessionPatch) -> ConfigDiff {
        self.stack.session.merge(patch);
        self.resolve()
    }

    /// The agent layer, replaced whole, re-resolved.
    pub fn set_agent(&mut self, agent: crate::agent_activity::options::SessionPatch) -> ConfigDiff {
        self.stack.agent = Some(agent);
        self.resolve()
    }

    fn resolve(&mut self) -> ConfigDiff {
        let next = self.stack.resolve(&self.legs);
        let diff = ConfigDiff {
            endpointing: (next.turn.endpointing != self.effective.turn.endpointing)
                .then_some(next.turn.endpointing),
            turn_detection_mode: (next.turn.owner != self.effective.turn.owner)
                .then_some(next.turn.owner),
            aec_warmup: (next.timers.aec_warmup != self.effective.timers.aec_warmup)
                .then_some(next.timers.aec_warmup),
        };
        self.effective = next;
        diff
    }
}

/// Neither verdict, so the first publish is always an edge.
const VERDICT_UNPUBLISHED: u8 = u8::MAX;

/// The pump runs at 50 fps outside the driver loop: staleness is one loop iteration, and the
/// barge-in leg is never substituted.
#[derive(Debug)]
pub struct GateSnapshot {
    /// [`VERDICT_PASS`], [`VERDICT_DISCARD`], or [`VERDICT_UNPUBLISHED`] until the first `publish`.
    verdict: AtomicU8,
}

const VERDICT_PASS: u8 = 0;
const VERDICT_DISCARD: u8 = 1;

impl GateSnapshot {
    pub fn new() -> Arc<Self> {
        Arc::new(GateSnapshot { verdict: AtomicU8::new(VERDICT_UNPUBLISHED) })
    }

    /// Run the gate and return whether it discards **only if that changed** (the
    /// `GatePublished` edge).
    #[must_use]
    pub fn publish(&self, i: &GateInputs) -> Option<bool> {
        let discard = audio_gate(i);
        let encoded = if discard { VERDICT_DISCARD } else { VERDICT_PASS };
        let previous = self.verdict.swap(encoded, Ordering::AcqRel);
        (previous != encoded).then_some(discard)
    }

    /// Whether the substituted legs get silence for the next frame; `false` until the first
    /// publish.
    pub fn discard(&self) -> bool {
        self.verdict.load(Ordering::Acquire) == VERDICT_DISCARD
    }
}

impl Driver {
    pub(crate) fn on_frame(&mut self, now: Ts, cmd: FrameCommand, cycle: &mut Cycle) {
        match cmd {
            FrameCommand::Update(_) => {
                // Routed in `run`, which can await; traced rather than panicking on a tokio worker.
                cycle.trace(CoreStep::PortFailure("UpdateAgent reached on_api"));
            }
            FrameCommand::UpdateInstructions(next) => {
                self.activity.set_instructions(next.clone());
                // "Record the configuration change" (`AgentActivity.update_instructions`), on the
                // driver's clock.
                self.commit_config(Some(next.render(Some(Modality::Audio))), vec![], vec![], cycle);
                // Ungated: the reference tells any realtime session and lets the plugin refuse.
                if self.activity.rt_session().is_some() {
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::UpdateInstructions(
                            lk_agents_abi::UpdateInstructions {
                                instructions: next.render(Some(Modality::Audio)),
                            },
                        ),
                        cycle,
                    );
                }
                self.publish_frame();
            }
            FrameCommand::UpdateTools(toolset) => {
                let tools = toolset.specs.len();
                // "Record the configuration change (skip if no visible diff)", in spec order so
                // the record is deterministic.
                let tools_added: Vec<String> = toolset
                    .specs
                    .iter()
                    .filter(|t| !self.activity.toolset().specs.iter().any(|o| o.name == t.name))
                    .map(|t| t.name.clone())
                    .collect();
                let tools_removed: Vec<String> = self
                    .activity
                    .toolset()
                    .specs
                    .iter()
                    .filter(|o| !toolset.specs.iter().any(|t| t.name == o.name))
                    .map(|o| o.name.clone())
                    .collect();
                self.activity.set_toolset(toolset);
                if !tools_added.is_empty() || !tools_removed.is_empty() {
                    self.commit_config(None, tools_added, tools_removed, cycle);
                }
                // Ungated, like instructions: a `mutable_tools: false` server must still learn a
                // handoff's new toolset.
                if self.activity.rt_session().is_some() {
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools {
                            tools: self.activity.toolset().specs.clone(),
                        }),
                        cycle,
                    );
                }
                cycle.trace(ToolStep::ToolsetUpdated { tools });
            }
            FrameCommand::SetAudioInput(_) => {
                // Routed in `run`, like `UpdateModels`: a re-attach reopens the STT leg.
                cycle.trace(CoreStep::PortFailure("SetAudioInput reached on_api"));
            }
            FrameCommand::SetVideoInput(input) => {
                let input = input.map(|HostVideo(input)| input);
                let attached = input.is_some();
                // The leg already read, handed back: nothing to swap.
                if !self.input.replace_video_in(input) {
                    return;
                }
                cycle.machine().push(crate::agent_session::cycle::Effect::Audio(
                    crate::agent_session::cycle::AudioOp::SetVideoInput,
                ));
                cycle.trace(InputStep::VideoInputSet { attached });
            }
            FrameCommand::SetIoEnabled { leg, enabled } => {
                self.set_io_enabled(now, leg, enabled, cycle)
            }
            FrameCommand::Drain { done } => self.drain(now, done.into_iter().collect(), cycle),
            FrameCommand::SetAudioOutput(out) => {
                let audio = out.is_some();
                // Dropped here: a segment in flight holds its own `Arc`; an unstarted speech reads
                // the new sink at its next segment.
                let _previous = self.shared.set_audio_out(out.map(|HostAudioOut(out)| out));
                let now = self.shared.time.now();
                let playing = self.session.agent_state() == AgentState::Speaking;
                if !audio || playing {
                    // A segment playing into the old sink can't move: force-interrupt it (and the
                    // realtime socket, once), rather than report a playout nobody received.
                    let socket_told = match self.activity.scheduling_task().current_speech() {
                        Some(speech) => {
                            let realtime = self.is_realtime_speech(speech);
                            match self.interrupt_speech(
                                now,
                                speech,
                                InterruptCause::Host,
                                true,
                                cycle,
                            ) {
                                Ok(InterruptOutcome::Now { .. }) => realtime,
                                Ok(InterruptOutcome::Already) | Err(_) => false,
                            }
                        }
                        None => false,
                    };
                    if !socket_told {
                        self.realtime_interrupt(cycle);
                    }
                }
                // `can_pause` and "audio enabled" are the new sink's; release a pause it can't hold.
                self.refresh_pause_facts(now, cycle);
                self.observe_recorder(_previous.as_deref());
                cycle.trace(PlayoutStep::OutputReplaced {
                    audio,
                    transcription: self.shared.text_enabled(),
                });
            }
            FrameCommand::SetSpeakerFilter(config) => {
                // The filter follows the STT the host named.
                self.input.set_speaker_filter(config);
            }
            FrameCommand::SetTranscriptionOutput(out) => {
                let transcription = out.is_some();
                let _previous = self.shared.set_text_out(out.map(|HostTextOut(out)| out));
                cycle.trace(PlayoutStep::OutputReplaced {
                    audio: self.shared.audio_enabled(),
                    transcription,
                });
            }
            FrameCommand::UpdateModels { .. } => {
                // Routed in `run`, like `Update`: reopening an audio leg opens a provider stream.
                cycle.trace(CoreStep::PortFailure("UpdateModels reached on_api"));
            }
            FrameCommand::UpdateChatCtx(items) => {
                let count = items.len();
                // Host-supplied, so not announced: the host already holds these items.
                let agent = self.activity.agent().to_owned();
                self.session.update_agent_chat_ctx(items, &agent);
                // Ungated: the reference guards only on there being a session.
                if self.activity.rt_session().is_some() {
                    let ctx = self.session.chat_ctx().for_realtime();
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::UpdateChatCtx(
                            lk_agents_abi::UpdateChatCtx {
                                req_id: self.shared.mint_item_id(),
                                chat_ctx: ctx,
                            },
                        ),
                        cycle,
                    );
                }
                self.publish_frame();
                cycle.trace(GenerationStep::ChatCtxUpdated { items: count });
            }
        }
    }

    pub(crate) fn publish_frame(&self) {
        self.shared.frame.send_replace(crate::agent_session::FrameView {
            record: self.session.chat_ctx().agent_chat_ctx(),
            instructions: self.activity.instructions().clone(),
            finished: self.activity.finished().clone(),
            tools_committed: self.activity.tools_committed().clone(),
        });
    }

    /// The history records a frame leaves when it takes over: `AgentHandoff`, then
    /// `AgentConfigUpdate` (skipped with neither instructions nor tools), in the reference's order.
    pub(crate) fn record_frame(&mut self, previous_agent: Option<String>, cycle: &mut Cycle) {
        let handoff = lk_agents_abi::AgentHandoffItem {
            id: self.session.mint("item"),
            old_agent_id: previous_agent,
            new_agent_id: self.activity.agent().to_owned(),
            created_at: self.shared.time.wall(),
        };
        self.commit_record(ChatItem::AgentHandoff(handoff), cycle);
        let rendered = self.activity.instructions().render(Some(Modality::Audio));
        let tools_added: Vec<String> =
            self.activity.toolset().specs.iter().map(|t| t.name.clone()).collect();
        if !rendered.is_empty() || !tools_added.is_empty() {
            self.commit_config(
                (!rendered.is_empty()).then_some(rendered),
                tools_added,
                vec![],
                cycle,
            );
        }
    }

    /// An `AgentConfigUpdate` record; an empty tool list is absent. The live callables (`tools`)
    /// are host objects the host attaches.
    fn commit_config(
        &mut self,
        instructions: Option<String>,
        tools_added: Vec<String>,
        tools_removed: Vec<String>,
        cycle: &mut Cycle,
    ) {
        let config = lk_agents_abi::AgentConfigUpdate {
            id: self.session.mint("item"),
            instructions,
            tools_added: (!tools_added.is_empty()).then_some(tools_added),
            tools_removed: (!tools_removed.is_empty()).then_some(tools_removed),
            created_at: self.shared.time.wall(),
            tools: None,
        };
        self.commit_record(ChatItem::AgentConfigUpdate(config), cycle);
    }

    /// Tell the realtime session which frame drives: instructions (audio rendering), conversation,
    /// tools, in the reference's order. No-op on a pipeline session.
    pub(crate) fn push_frame(
        &self,
        chat_ctx: Option<lk_agents_abi::ChatContext>,
        cycle: &mut Cycle,
    ) {
        if self.activity.rt_session().is_none() {
            return;
        }
        self.rt_send(
            lk_agents_abi::RealtimeCommand::UpdateInstructions(lk_agents_abi::UpdateInstructions {
                instructions: self.activity.instructions().render(Some(Modality::Audio)),
            }),
            cycle,
        );
        if let Some(chat_ctx) = chat_ctx {
            self.rt_send(
                lk_agents_abi::RealtimeCommand::UpdateChatCtx(lk_agents_abi::UpdateChatCtx {
                    req_id: self.shared.mint_item_id(),
                    chat_ctx,
                }),
                cycle,
            );
        }
        self.rt_send(
            lk_agents_abi::RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools {
                tools: self.activity.toolset().specs.clone(),
            }),
            cycle,
        );
    }

    /// Put a frame record on the record. `AgentHandoff` is history-only; `AgentConfigUpdate` goes to
    /// the frame's context and history (only the realtime view strips it).
    pub(crate) fn commit_record(&mut self, item: ChatItem, cycle: &mut Cycle) {
        let (source, visibility) = match &item {
            ChatItem::AgentHandoff(_) => (Source::Handoff, Visibility::History),
            _ => (Source::ConfigUpdate, Visibility::Both),
        };
        self.commit_item(item, source, visibility, cycle);
    }

    /// The one write to the record: inserted by its acoustic `created_at`, the speech told, the
    /// host told via `ItemAdded{source}`, and the frame republished.
    pub(crate) fn commit_item(
        &mut self,
        item: ChatItem,
        source: Source,
        visibility: Visibility,
        cycle: &mut Cycle,
    ) {
        // Attributed to the frame that created the speech, not the driving one (a task's tool
        // commits after the parent is back); an item with no speech is the driving frame's.
        let agent = source
            .speech()
            .and_then(|s| self.activity.speech_frame(s).map(str::to_owned))
            .unwrap_or_else(|| self.activity.agent().to_owned());
        let commit = self.session.commit_item(item, source, visibility, &agent);
        self.perform_commit(commit, cycle);
    }

    /// A session instant on the wall clock, for an item stamped at the moment it was heard.
    pub(crate) fn wall_at(&self, at: Ts) -> std::time::SystemTime {
        let now = self.shared.time.now();
        let wall = self.shared.time.wall();
        wall.checked_sub(now.saturating_duration_since(at)).unwrap_or(wall)
    }

    /// The speech is told, the host learns of a new item, the frame is republished.
    fn perform_commit(&mut self, (entry, outcome): (Committed, CommitOutcome), cycle: &mut Cycle) {
        // A call was pushed to the handle when the model made it; everything else is new here.
        let tell_speech =
            entry.source.speech().filter(|_| !matches!(entry.source, Source::ToolCall { .. }));
        if let Some(h) = tell_speech.and_then(|id| self.activity.scheduling_task().handle(id)) {
            h.item_added(&entry.item);
        }
        match outcome {
            CommitOutcome::Added => {
                let run = self.run_of_item(&entry.source);
                cycle.machine().emit(crate::events::SessionEvent::ItemAdded {
                    item: Box::new(entry.item),
                    agent: entry.agent,
                    source: entry.source,
                    visibility: entry.visibility,
                    run,
                });
            }
            CommitOutcome::Replaced | CommitOutcome::Filtered => {}
        }
        self.publish_frame();
    }
}
