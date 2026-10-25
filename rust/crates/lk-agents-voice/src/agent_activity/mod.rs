//! `voice/agent_activity.py`'s `AgentActivity`: the running agent's frame and components, the
//! driver's configuration, and the frame commands (prompt, tools, context, IO) the driver applies.

pub mod drain;
pub mod dtmf;
pub mod false_interruption;
pub mod handoff;
pub mod hooks;
pub mod io;
pub(crate) mod on_exit;
pub mod options;
pub mod realtime;
pub mod recognition;
pub mod scheduling;
pub mod speech;
pub mod tasks;
pub mod tools;

use crate::agent::{CommitOutcome, Committed, Source, Visibility};
use crate::agent_activity::false_interruption::{InterruptOutcome, PausedSpeech};
use crate::agent_activity::options::{ConfigStack, LegSet, SessionConfig};
use crate::agent_activity::scheduling::{SpeechId, SpeechScheduler};
use crate::agent_session::effects::Out;
use crate::agent_session::{
    ActivityCommand, Driver, HostAudio, HostAudioOut, HostTextOut, HostVideo,
};
use crate::audio::{GateInputs, audio_gate};
use crate::audio_recognition::AudioRecognition;
use crate::trace::{GenerationStep, InputStep, InterruptCause, PlayoutStep, ToolStep};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{ChatItem, Modality, Ts};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

/// What the STT and VAD legs are fed: every model in the tree wants 16 kHz mono.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// `voice/agent_activity.py`'s `AgentActivity`: the running agent's frame (its name, prompt and
/// tools), its audio recognition, the speech queue and the current speech, the false-interruption
/// pause, the hooks and inline tasks, and the realtime session. The `Driver` orchestrates it
/// together with the [`LiveSession`](crate::agent_session::LiveSession): what spans two of its
/// components is the driver's to answer.
pub(crate) struct AgentActivity {
    /// The running agent's id (`AgentActivity.agent.id`).
    pub(crate) agent: String,
    /// Replies in flight keep the prompt and tools they snapshotted.
    pub(crate) instructions: crate::generation::Instructions,
    pub(crate) toolset: Arc<crate::tool_executor::Toolset>,
    /// `_tool_executor`.
    pub(crate) tools: crate::tool_executor::Tools,
    pub(crate) audio_recognition: AudioRecognition,
    /// `_speech_q`, `_current_speech`, `_scheduling_paused`, `_preemptive_generation`, and every
    /// live speech's handle.
    pub(crate) speech_scheduler: SpeechScheduler,
    /// `_paused_speech` and the false-interruption timer.
    pub(crate) paused_speech: PausedSpeech,
    /// The frame each speech was created in: late tool items belong to it.
    speech_agents: BTreeMap<SpeechId, String>,
    pub(crate) finished: BTreeSet<SpeechId>,
    /// Per tool step: whether the activity that made the speech drains, so the step's reply
    /// offers no tools.
    pub(crate) tools_committed: BTreeMap<(SpeechId, u32), bool>,
    /// Keeps the agent `thinking` across a filler `say` from inside a tool.
    pub(crate) tools_in_flight: BTreeSet<SpeechId>,
    /// The step's first frame transitions, once. Cleared when the step is spoken.
    pub(crate) spoke: BTreeSet<SpeechId>,
    /// `_current_speech` as last announced (`FloorChanged`); compared at the end of every cycle.
    announced_current_speech: Option<SpeechId>,
    /// `interrupt()`s waiting for their speeches' release.
    pub(crate) interrupt_waiters: Vec<crate::agent_activity::false_interruption::InterruptWaiter>,
    /// How long a hook whose answer is needed is waited on; `None` when no host runs hooks.
    pub(crate) hook_bound: Option<Duration>,
    /// The developer's callbacks in flight.
    pub(crate) hooks: crate::agent_activity::hooks::Hooks,
    /// `update_agent()`: the switches asked for and in progress.
    pub(crate) handoff: crate::agent_activity::handoff::Handoff,
    /// The `on_exit` each agent owes.
    pub(crate) on_exit: crate::agent_activity::on_exit::OnExit,
    /// The host's `drain()`.
    pub(crate) drain: crate::agent_activity::drain::Drain,
    /// `GetDtmfTask` while it runs.
    pub(crate) dtmf: crate::agent_activity::dtmf::Dtmf,
    /// The inline `AgentTask` stack.
    pub(crate) tasks: crate::agent_activity::tasks::Tasks,
    pub(crate) drain_blocked: crate::agent_activity::tasks::DrainBlocked,
    /// The user turn parked on `on_user_turn_completed` (`_user_turn_completed_atask`).
    pub(crate) pending_turn: Option<crate::audio_recognition::PendingTurn>,
    /// So a preemptive generation can be keyed to the turn after it.
    pub(crate) last_turn: Option<crate::audio_recognition::TurnId>,
    /// `generate_reply` matches on it.
    pub(crate) preemptive_turn: Option<crate::audio_recognition::TurnId>,
    /// The record version a preemptive generation was planned against.
    pub(crate) preemptive_record_version: u64,
    /// `_rt_session`. Exclusive with a pipeline.
    pub(crate) rt_session: Option<lk_agents_abi::resources::RealtimeRef>,
    pub(crate) realtime_state: crate::agent_activity::realtime::RealtimeSessionState,
    /// Dropping a sender ends its generation.
    pub(crate) realtime_players:
        BTreeMap<SpeechId, tokio::sync::mpsc::UnboundedSender<lk_agents_abi::llm::realtime::Event>>,
    /// The model's turn detection was switched off at the open.
    pub(crate) realtime_turn_detection_disabled: bool,
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
    pub(crate) speech_scheduler: SpeechScheduler,
    pub(crate) paused_speech: PausedSpeech,
    pub(crate) hook_bound: Option<Duration>,
    pub(crate) rt_session: Option<lk_agents_abi::resources::RealtimeRef>,
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
            speech_scheduler,
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
            speech_scheduler,
            paused_speech,
            speech_agents: BTreeMap::new(),
            finished: BTreeSet::new(),
            tools_committed: BTreeMap::new(),
            tools_in_flight: BTreeSet::new(),
            spoke: BTreeSet::new(),
            announced_current_speech: None,
            interrupt_waiters: Vec::new(),
            hook_bound,
            hooks: Default::default(),
            handoff: Default::default(),
            on_exit: Default::default(),
            drain: Default::default(),
            dtmf: Default::default(),
            tasks: Default::default(),
            drain_blocked: Default::default(),
            pending_turn: None,
            last_turn: None,
            preemptive_turn: None,
            preemptive_record_version: 0,
            rt_session,
            realtime_state: crate::agent_activity::realtime::RealtimeSessionState::new(
                realtime_caps,
            ),
            realtime_players: BTreeMap::new(),
            realtime_turn_detection_disabled,
            realtime_resolved_mode,
            realtime_requested_detection,
        }
    }

    /// A frame change lands: the new agent, prompt and tools. The previous agent's id.
    pub(crate) fn set_agent(
        &mut self,
        agent: String,
        instructions: crate::generation::Instructions,
        toolset: crate::tool_executor::Toolset,
    ) -> String {
        self.instructions = instructions;
        self.toolset = Arc::new(toolset);
        std::mem::replace(&mut self.agent, agent)
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
    pub(crate) fn note_speech_agent(&mut self, speech: SpeechId) {
        self.speech_agents.insert(speech, self.agent.clone());
    }

    /// The frame `speech` was created in.
    pub(crate) fn speech_agent(&self, speech: SpeechId) -> Option<&str> {
        self.speech_agents.get(&speech).map(String::as_str)
    }

    pub(crate) fn set_tools_in_flight(&mut self, speech: SpeechId, running: bool) {
        if running {
            self.tools_in_flight.insert(speech);
        } else {
            self.tools_in_flight.remove(&speech);
        }
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

    /// A preemptive generation starts: it belongs to the turn after the last one, planned against
    /// `record_version`.
    pub(crate) fn key_preemptive_generation(&mut self, record_version: u64) {
        self.preemptive_turn =
            Some(crate::audio_recognition::TurnId(self.last_turn.map_or(1, |t| t.0 + 1)));
        self.preemptive_record_version = record_version;
    }

    /// The turn detection the options said, and the mode they resolved to before the model's
    /// refinement: whether either moved.
    pub(crate) fn note_resolved_mode(
        &mut self,
        resolved: crate::audio_recognition::TurnDetectionMode,
        detection: crate::agent_activity::options::TurnDetection,
    ) -> bool {
        let moved = resolved != self.realtime_resolved_mode
            || detection != self.realtime_requested_detection;
        self.realtime_resolved_mode = resolved;
        self.realtime_requested_detection = detection;
        moved
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
    pub endpointing: Option<crate::turn::EndpointingOptions>,
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
    pub fn configure(
        &mut self,
        patch: &crate::agent_activity::options::SessionPatch,
    ) -> ConfigDiff {
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
/// interruption leg is never substituted.
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
    pub(crate) fn on_frame(&mut self, now: Ts, cmd: ActivityCommand, out: &mut Out) {
        match cmd {
            // A sequence: `on_exit`, the drain, then the swap.
            ActivityCommand::Update(update) => {
                self.handoff_step(crate::agent_activity::handoff::HandoffEvent::Asked(update), out)
            }
            ActivityCommand::UpdateInstructions(next) => {
                self.activity.instructions = next.clone();
                // "Record the configuration change" (`AgentActivity.update_instructions`), on the
                // driver's clock.
                self.commit_config(Some(next.render(Some(Modality::Audio))), vec![], vec![], out);
                // Ungated: the reference tells any realtime session and lets the plugin refuse.
                if self.activity.rt_session.is_some() {
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::UpdateInstructions(
                            lk_agents_abi::UpdateInstructions {
                                instructions: next.render(Some(Modality::Audio)),
                            },
                        ),
                        out,
                    );
                }
                self.publish_activity();
            }
            ActivityCommand::UpdateTools(toolset) => {
                let tools = toolset.specs.len();
                // "Record the configuration change (skip if no visible diff)", in spec order so
                // the record is deterministic.
                let tools_added: Vec<String> = toolset
                    .specs
                    .iter()
                    .filter(|t| !self.activity.toolset.specs.iter().any(|o| o.name == t.name))
                    .map(|t| t.name.clone())
                    .collect();
                let tools_removed: Vec<String> = self
                    .activity
                    .toolset
                    .specs
                    .iter()
                    .filter(|o| !toolset.specs.iter().any(|t| t.name == o.name))
                    .map(|o| o.name.clone())
                    .collect();
                self.activity.toolset = Arc::new(toolset);
                if !tools_added.is_empty() || !tools_removed.is_empty() {
                    self.commit_config(None, tools_added, tools_removed, out);
                }
                // Ungated, like instructions: a `mutable_tools: false` server must still learn a
                // handoff's new toolset.
                if self.activity.rt_session.is_some() {
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools {
                            tools: self.activity.toolset.specs.clone(),
                        }),
                        out,
                    );
                }
                out.trace(ToolStep::ToolsetUpdated { tools });
            }
            // A re-attached microphone opens a fresh STT stream.
            ActivityCommand::SetAudioInput(input) => {
                self.set_audio_input(now, input.map(|HostAudio(input)| input), out)
            }
            ActivityCommand::SetVideoInput(input) => {
                let input = input.map(|HostVideo(input)| input);
                let attached = input.is_some();
                // The leg already read, handed back: nothing to swap.
                if !self.input.replace_video_in(input) {
                    return;
                }
                out.machine().push(crate::agent_session::effects::Effect::Audio(
                    crate::agent_session::effects::AudioOp::SetVideoInput,
                ));
                out.trace(InputStep::VideoInputSet { attached });
            }
            ActivityCommand::SetIoEnabled { leg, enabled } => {
                self.set_io_enabled(now, leg, enabled, out)
            }
            ActivityCommand::Drain { call } => self.drain(call.into_iter().collect(), out),
            ActivityCommand::SetAudioOutput(output) => {
                let audio = output.is_some();
                // Dropped here: a segment in flight holds its own `Arc`; an unstarted speech reads
                // the new sink at its next segment.
                let _previous = self.shared.output.set_audio(output.map(|HostAudioOut(sink)| sink));
                let now = self.step.now;
                let playing = self.session.states.agent_state() == AgentState::Speaking;
                if !audio || playing {
                    // A segment playing into the old sink can't move: force-interrupt it (and the
                    // realtime socket, once), rather than report a playout nobody received.
                    let socket_told = match self.activity.speech_scheduler.current_speech() {
                        Some(speech) => {
                            let realtime = self.is_realtime_speech(speech);
                            match self.interrupt_speech(
                                now,
                                speech,
                                InterruptCause::Host,
                                true,
                                out,
                            ) {
                                Ok(InterruptOutcome::Now { .. }) => realtime,
                                Ok(InterruptOutcome::Already) | Err(_) => false,
                            }
                        }
                        None => false,
                    };
                    if !socket_told {
                        self.realtime_interrupt(out);
                    }
                }
                // `can_pause` and "audio enabled" are the new sink's; release a pause it can't hold.
                self.refresh_pause_facts(now, out);
                self.observe_recorder(_previous.as_deref());
                out.trace(PlayoutStep::OutputReplaced {
                    audio,
                    transcription: self.shared.output.shows_transcription(),
                });
            }
            ActivityCommand::SetSpeakerFilter(config) => {
                // The filter follows the STT the host named.
                self.input.set_speaker_filter(config);
            }
            ActivityCommand::SetTranscriptionOutput(output) => {
                let transcription = output.is_some();
                let _previous =
                    self.shared.output.set_transcription(output.map(|HostTextOut(sink)| sink));
                out.trace(PlayoutStep::OutputReplaced {
                    audio: self.shared.output.plays_audio(),
                    transcription,
                });
            }
            ActivityCommand::UpdateModels { stt, vad, llm, tts } => {
                self.update_models(stt, vad, llm, tts, out)
            }
            ActivityCommand::UpdateChatCtx(items) => {
                let count = items.len();
                // Host-supplied, so not announced: the host already holds these items.
                let agent = self.activity.agent.clone();
                self.session.chat_ctx.update_agent_chat_ctx(items, &agent);
                // Ungated: the reference guards only on there being a session.
                if self.activity.rt_session.is_some() {
                    let ctx = self.session.chat_ctx.for_realtime();
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::UpdateChatCtx(
                            lk_agents_abi::UpdateChatCtx {
                                req_id: self.shared.mint_item_id(),
                                chat_ctx: ctx,
                            },
                        ),
                        out,
                    );
                }
                self.publish_activity();
                out.trace(GenerationStep::ChatCtxUpdated { items: count });
            }
        }
    }

    pub(crate) fn publish_activity(&self) {
        self.shared.activity.send_replace(crate::agent_session::ActivitySnapshot {
            chat_ctx: self.session.chat_ctx.agent_chat_ctx(),
            instructions: self.activity.instructions.clone(),
            finished: self.activity.finished.clone(),
            tools_committed: self.activity.tools_committed.clone(),
        });
    }

    /// The history records a frame leaves when it takes over: `AgentHandoff`, then
    /// `AgentConfigUpdate` (skipped with neither instructions nor tools), in the reference's order.
    pub(crate) fn record_handoff(&mut self, previous_agent: Option<String>, out: &mut Out) {
        let handoff = lk_agents_abi::AgentHandoffItem {
            id: self.session.ids.mint("item"),
            old_agent_id: previous_agent,
            new_agent_id: self.activity.agent.clone(),
            created_at: self.step.wall(),
        };
        self.commit_record(ChatItem::AgentHandoff(handoff), out);
        let rendered = self.activity.instructions.render(Some(Modality::Audio));
        let tools_added: Vec<String> =
            self.activity.toolset.specs.iter().map(|t| t.name.clone()).collect();
        if !rendered.is_empty() || !tools_added.is_empty() {
            self.commit_config(
                (!rendered.is_empty()).then_some(rendered),
                tools_added,
                vec![],
                out,
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
        out: &mut Out,
    ) {
        let config = lk_agents_abi::AgentConfigUpdate {
            id: self.session.ids.mint("item"),
            instructions,
            tools_added: (!tools_added.is_empty()).then_some(tools_added),
            tools_removed: (!tools_removed.is_empty()).then_some(tools_removed),
            created_at: self.step.wall(),
            tools: None,
        };
        self.commit_record(ChatItem::AgentConfigUpdate(config), out);
    }

    /// Tell the realtime session which frame drives: instructions (audio rendering), conversation,
    /// tools, in the reference's order. No-op on a pipeline session.
    pub(crate) fn push_activity(
        &self,
        chat_ctx: Option<lk_agents_abi::ChatContext>,
        out: &mut Out,
    ) {
        if self.activity.rt_session.is_none() {
            return;
        }
        self.rt_send(
            lk_agents_abi::RealtimeCommand::UpdateInstructions(lk_agents_abi::UpdateInstructions {
                instructions: self.activity.instructions.render(Some(Modality::Audio)),
            }),
            out,
        );
        if let Some(chat_ctx) = chat_ctx {
            self.rt_send(
                lk_agents_abi::RealtimeCommand::UpdateChatCtx(lk_agents_abi::UpdateChatCtx {
                    req_id: self.shared.mint_item_id(),
                    chat_ctx,
                }),
                out,
            );
        }
        self.rt_send(
            lk_agents_abi::RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools {
                tools: self.activity.toolset.specs.clone(),
            }),
            out,
        );
    }

    /// Put a frame record on the record. `AgentHandoff` is history-only; `AgentConfigUpdate` goes to
    /// the frame's context and history (only the realtime view strips it).
    pub(crate) fn commit_record(&mut self, item: ChatItem, out: &mut Out) {
        let (source, visibility) = match &item {
            ChatItem::AgentHandoff(_) => (Source::Handoff, Visibility::History),
            _ => (Source::ConfigUpdate, Visibility::Both),
        };
        self.commit_item(item, source, visibility, out);
    }

    /// The one write to the record: inserted by its acoustic `created_at`, the speech told, the
    /// host told via `ItemAdded{source}`, and the frame republished.
    pub(crate) fn commit_item(
        &mut self,
        item: ChatItem,
        source: Source,
        visibility: Visibility,
        out: &mut Out,
    ) {
        // Attributed to the frame that created the speech, not the driving one (a task's tool
        // commits after the parent is back); an item with no speech is the driving frame's.
        let agent = source
            .speech()
            .and_then(|s| self.activity.speech_agent(s).map(str::to_owned))
            .unwrap_or_else(|| self.activity.agent.clone());
        let commit = self.session.chat_ctx.commit(item, source, visibility, &agent);
        self.perform_commit(commit, out);
    }

    /// The speech is told, the host learns of a new item, the frame is republished.
    fn perform_commit(&mut self, (entry, outcome): (Committed, CommitOutcome), out: &mut Out) {
        // A call was pushed to the handle when the model made it; everything else is new here.
        let tell_speech =
            entry.source.speech().filter(|_| !matches!(entry.source, Source::ToolCall { .. }));
        if let Some(h) = tell_speech.and_then(|id| self.activity.speech_scheduler.handle(id)) {
            h.item_added(&entry.item);
        }
        match outcome {
            CommitOutcome::Added => {
                let run = self.run_of_item(&entry.source);
                out.machine().emit(crate::events::SessionEvent::ConversationItemAdded {
                    item: Box::new(entry.item),
                    agent: entry.agent,
                    source: entry.source,
                    visibility: entry.visibility,
                    run,
                });
            }
            CommitOutcome::Replaced | CommitOutcome::Filtered => {}
        }
        self.publish_activity();
    }
}
