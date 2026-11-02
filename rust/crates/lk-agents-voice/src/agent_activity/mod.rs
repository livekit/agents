//! `voice/agent_activity.py`'s `AgentActivity`: the running agent and its components, the driver's
//! configuration, and the agent commands (prompt, tools, context, IO) the driver applies.

use lk_agents_core::tts::markup::ChatMessageExt;
pub mod drain;
mod experts;
pub mod false_interruption;
pub mod handoff;
pub mod hooks;
mod idle;
pub mod io;
pub mod options;
pub mod realtime;
pub mod recognition;
pub(crate) mod reply_run;
pub mod scheduling;
pub mod speech;
mod tool_results;
pub mod tools;
pub mod video_sampler;
pub mod voice_task;

use crate::agent::{CommitOutcome, Committed, Source, Visibility};
use crate::agent_activity::false_interruption::PausedSpeech;
use crate::agent_activity::options::{
    AgentSessionOptions, ConfigStack, OverrideLayer, SessionLegs,
};
use crate::agent_activity::scheduling::{SpeechId, SpeechKind, SpeechScheduler};
use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::{ActivityCommand, AudioInputHandle, VoiceDriver};
use crate::audio_recognition::AudioRecognition;
use crate::trace::{GenerationStep, InputStep, PlayoutStep, ToolStep};
use lk_agents_abi::{ChatItem, Modality, Ts};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

/// How many of the latest speeches the driver remembers once settled (their agent, end, tool
/// steps): a late item or a reply planned behind one is never that far behind.
const KEPT_SPEECHES: usize = 128;

/// Facts retained together after a speech leaves the scheduler. An unknown late completion
/// carries no invented origin; attribution becomes available only when the speech is registered.
#[derive(Default)]
struct SpeechHistory {
    records: BTreeMap<SpeechId, SpeechRecord>,
    facts: usize,
}

#[derive(Default)]
struct SpeechRecord {
    origin: Option<(String, SpeechKind)>,
    finished: bool,
    tools_committed: bool,
}

impl SpeechRecord {
    fn facts(&self) -> usize {
        usize::from(self.origin.is_some())
            + usize::from(self.finished)
            + usize::from(self.tools_committed)
    }
}

impl SpeechHistory {
    fn remember(&mut self, speech: SpeechId, agent: String, kind: SpeechKind) {
        let record = self.records.entry(speech).or_default();
        self.facts += usize::from(record.origin.is_none());
        record.origin = Some((agent, kind));
    }

    fn finish(&mut self, speech: SpeechId) {
        let record = self.records.entry(speech).or_default();
        self.facts += usize::from(!record.finished);
        record.finished = true;
    }

    fn commit_tools(&mut self, speech: SpeechId) {
        let record = self.records.entry(speech).or_default();
        self.facts += usize::from(!record.tools_committed);
        record.tools_committed = true;
    }

    fn finished(&self, speech: SpeechId) -> bool {
        self.records.get(&speech).is_some_and(|record| record.finished)
    }

    fn settled(&self, speech: SpeechId) -> bool {
        self.records.get(&speech).is_some_and(|record| record.finished || record.tools_committed)
    }

    fn prune(&mut self, live: impl Fn(SpeechId) -> bool) {
        // Preserve the original retention trigger: each recorded fact counts, not each id.
        if self.facts < 2 * KEPT_SPEECHES {
            return;
        }
        let Some(newest) =
            self.records.iter().rev().find_map(|(id, record)| record.origin.as_ref().map(|_| *id))
        else {
            return;
        };
        let facts = &mut self.facts;
        self.records.retain(|id, record| {
            let keep = newest.0.saturating_sub(id.0) < KEPT_SPEECHES as u64 || live(*id);
            if !keep {
                *facts -= record.facts();
            }
            keep
        });
    }
}

/// What the STT and VAD legs are fed: every model in the tree wants 16 kHz mono.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// `voice/agent_activity.py`'s `AgentActivity`: the running agent (its name, prompt and
/// tools), its audio recognition, the speech queue and the current speech, the false-interruption
/// pause, the hooks, and the realtime session. The `VoiceDriver` orchestrates it
/// together with the [`VoiceSession`](crate::agent_session::VoiceSessionState): what spans two of its
/// components is the driver's to answer.
pub(crate) struct AgentActivity {
    /// The running agent's id (`AgentActivity.agent.id`).
    agent: String,
    /// Replies in flight keep the prompt and tools they snapshotted.
    instructions: crate::generation::Instructions,
    dynamic_instructions: Option<lk_agents_abi::resources::ResourceId>,
    /// The agent's `transcription_node`.
    output_nodes: crate::agent_session::nodes::OutputNodes,
    toolset: Arc<crate::tool_executor::Toolset>,
    /// `_tool_executor`.
    tools: crate::tool_executor::Tools,
    audio_recognition: AudioRecognition,
    /// `_speech_q`, `_current_speech`, `_scheduling_paused`, `_preemptive_generation`, and every
    /// live speech's handle.
    speech_scheduler: SpeechScheduler,
    /// `_paused_speech` and the false-interruption timer.
    paused_speech: PausedSpeech,
    /// Attribution and completion facts share one lifetime, including after the close's edge.
    speech_history: SpeechHistory,
    /// Replies waiting for their predecessor's commit before their first step is planned.
    reply_plans:
        std::collections::VecDeque<(crate::agent_session::RequestId, crate::generation::ReplyPlan)>,
    /// Keeps the agent `thinking` across a filler `say` from inside a tool.
    tools_in_flight: BTreeSet<SpeechId>,
    /// The step's first frame transitions, once. Cleared when the step is spoken.
    spoke: BTreeSet<SpeechId>,
    /// `_current_speech` as last announced (`FloorChanged`); compared at the end of every cycle.
    announced_current_speech: Option<SpeechId>,
    /// `interrupt()`s waiting for their speeches' release.
    interrupt_waiters: Vec<crate::agent_activity::false_interruption::InterruptWaiter>,
    /// How long a hook whose answer is needed is waited on; `None` when no host runs hooks.
    hook_bound: Option<Duration>,
    /// The developer's callbacks in flight.
    hooks: crate::agent_activity::hooks::Hooks,
    /// `update_agent()`: the switches asked for and in progress.
    handoff: crate::agent_activity::handoff::Handoff,
    /// The `on_exit` each agent owes.
    on_exit: crate::agent_activity::hooks::OnExit,
    /// The host's `drain()`.
    drain: crate::agent_activity::drain::Drain,
    /// `VoiceTask.run()`: the tasks holding the floor and the parents they suspended.
    voice_tasks: crate::agent_activity::voice_task::VoiceTasks,
    /// DTMF collection while it runs.
    dtmf: crate::workflows::dtmf::Dtmf,
    /// `EndCallTool`: the speech whose goodbye the session closes after.
    end_call: Option<crate::workflows::end_call::EndCall>,
    /// `on_user_turn_exceeded`, waited for or running.
    turn_limit: crate::agent_activity::hooks::TurnLimit,
    /// The user turn parked on `on_user_turn_completed` (`_user_turn_completed_atask`).
    pending_turn: Option<crate::audio_recognition::PendingTurn>,
    /// Decided turns waiting for `pending_turn`'s hook to return, in order.
    waiting_turns: std::collections::VecDeque<crate::audio_recognition::EndOfTurnInfo>,
    /// So a preemptive generation can be keyed to the turn after it.
    last_turn: Option<crate::audio_recognition::TurnId>,
    /// A final arrived while the current speech was still uninterrupted: the preemptive
    /// generation that final asks for this cycle sees it so, as the reference's does (its final
    /// cuts a paused speech in a task, after `on_preemptive_generation` ran).
    live_speech_at_final: bool,
    /// `generate_reply` matches on it.
    preemptive_turn: Option<crate::audio_recognition::TurnId>,
    /// The record version a preemptive generation was planned against.
    preemptive_record_version: u64,
    /// `_rt_session`. Exclusive with a pipeline.
    rt_session: Option<lk_agents_abi::resources::RealtimeRef>,
    realtime_state: crate::agent_activity::realtime::RealtimeSessionState,
    /// Speech identities whose runtime player is still open.
    realtime_players: BTreeSet<SpeechId>,
    /// Authorized speeches awaiting their first frame.
    launching: BTreeSet<SpeechId>,
    /// The model's turn detection was switched off at the open.
    realtime_turn_detection_disabled: bool,
    /// Before the model's refinement, so a live change is one that moved this.
    realtime_resolved_mode: crate::audio_recognition::TurnDetectionMode,
    /// The `turn_detection` the options last said, so a live change is one that moved it.
    realtime_requested_detection: crate::agent_activity::options::TurnDetection,
}

impl AgentActivity {
    pub(crate) fn scope(&self) -> u64 {
        self.toolset.agent.as_ref().map_or(0, |agent| agent.toolset_scope())
    }

    /// Replace the activity's advertised tools and report the visible diff in deterministic order.
    pub(crate) fn update_tools(
        &mut self,
        toolset: crate::tool_executor::Toolset,
    ) -> (Vec<String>, Vec<String>) {
        // "Record the configuration change (skip if no visible diff)", in spec order so
        // the record is deterministic.
        let tools_added: Vec<String> = toolset
            .specs
            .iter()
            .filter(|t| !self.toolset.specs.iter().any(|o| o.name == t.name))
            .map(|t| t.name.clone())
            .collect();
        let tools_removed: Vec<String> = self
            .toolset
            .specs
            .iter()
            .filter(|o| !toolset.specs.iter().any(|t| t.name == o.name))
            .map(|o| o.name.clone())
            .collect();
        self.toolset = Arc::new(toolset);
        (tools_added, tools_removed)
    }

    /// A handoff lands: the new agent, prompt, nodes and tools. The previous agent's id.
    pub(crate) fn set_agent(
        &mut self,
        agent: String,
        instructions: crate::generation::Instructions,
        dynamic_instructions: Option<lk_agents_abi::resources::ResourceId>,
        output_nodes: crate::agent_session::nodes::OutputNodes,
        toolset: crate::tool_executor::Toolset,
    ) -> String {
        self.instructions = instructions;
        self.dynamic_instructions = dynamic_instructions;
        self.output_nodes = output_nodes;
        self.toolset = Arc::new(toolset);
        std::mem::replace(&mut self.agent, agent)
    }

    /// The running agent's retry budgets (`Agent(retries=)`).
    pub(crate) fn retries(&self) -> lk_agents_abi::agent::Retries {
        self.toolset.agent.as_ref().map(|a| a.definition().retries).unwrap_or_default()
    }

    /// What a request planned now is offered: the agent's toolset, then session additions whose
    /// names the agent lacks (a provider refuses duplicates).
    pub(crate) fn advertised_toolset(
        &self,
        session_tools: &crate::tool_executor::Toolset,
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

    /// The speech is the running agent's: its late items are attributed to it.
    pub(crate) fn speech_belongs_to_run(
        &self,
        speech: SpeechId,
        run: crate::events::RunId,
    ) -> bool {
        self.speech_scheduler.belongs_to_run(speech, run)
    }

    /// Ask the lifecycle owners, rather than maintaining a duplicate pending-work ledger.
    pub(crate) fn has_run_work(&self, run: crate::events::RunId) -> bool {
        self.speech_scheduler.has_run_work(run)
            || self.hooks.has_run_work(run)
            || self.handoff.has_run_work(run)
    }

    pub(crate) fn note_speech_agent(&mut self, speech: SpeechId, kind: SpeechKind) {
        self.speech_history.remember(speech, self.agent.clone(), kind);
    }

    /// The agent `speech` was created under.
    pub(crate) fn speech_agent(&self, speech: SpeechId) -> Option<&str> {
        self.speech_history.records.get(&speech)?.origin.as_ref().map(|(agent, _)| agent.as_str())
    }

    /// What `speech` was created as, after its release too.
    pub(crate) fn speech_kind(&self, speech: SpeechId) -> Option<SpeechKind> {
        self.speech_history.records.get(&speech)?.origin.as_ref().map(|(_, kind)| *kind)
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
}

/// The driver's configuration: the layers, the facts, and the resolved answer.
///
/// `effective` is only written by `resolve`; a change reports what moved as a [`ConfigDiff`] so
/// only the affected machines are told.
#[derive(Debug, Clone, PartialEq)]
pub struct DriverConfig {
    pub stack: ConfigStack,
    pub legs: SessionLegs,
    pub effective: AgentSessionOptions,
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
    pub fn new(stack: ConfigStack, legs: SessionLegs) -> Self {
        let effective = stack.resolve(&legs);
        DriverConfig { stack, legs, effective }
    }

    /// Before `start()`: no machine exists yet to tell.
    pub fn edit(&mut self, f: impl FnOnce(&mut AgentSessionOptions, &mut SessionLegs)) {
        f(&mut self.stack.session, &mut self.legs);
        self.effective = self.stack.resolve(&self.legs);
    }

    /// Whether the application set `aec_warmup_duration` (on the session or the agent).
    pub fn aec_warmup_given(&self) -> bool {
        self.stack.session.timers.aec_warmup.is_given()
            || self.stack.agent.as_ref().is_some_and(|a| a.timers.aec_warmup.is_given())
    }

    /// Merged into the session layer, re-resolved.
    pub fn configure(&mut self, update: &AgentSessionOptions) -> ConfigDiff {
        self.stack.session.merge(update);
        self.resolve()
    }

    /// An `override_options` layer laid on top, re-resolved.
    pub fn push_override(&mut self, layer: OverrideLayer) -> ConfigDiff {
        self.stack.overrides.push(layer);
        self.resolve()
    }

    /// The layer `id` taken out wherever it sits, re-resolved; an unknown id changes nothing.
    pub fn release_override(&mut self, id: u64) -> ConfigDiff {
        self.stack.overrides.retain(|layer| layer.id != id);
        self.resolve()
    }

    /// The agent layer, replaced whole, re-resolved.
    pub fn set_agent(&mut self, agent: AgentSessionOptions) -> ConfigDiff {
        self.stack.agent = Some(agent);
        self.resolve()
    }

    fn resolve(&mut self) -> ConfigDiff {
        let next = self.stack.resolve(&self.legs);
        let diff = ConfigDiff {
            endpointing: (next.turn.endpointing != self.effective.turn.endpointing)
                .then_some(next.turn.endpointing),
            turn_detection_mode: (next.turn.owner() != self.effective.turn.owner())
                .then_some(next.turn.owner()),
            aec_warmup: (next.timers.aec_warmup() != self.effective.timers.aec_warmup())
                .then_some(next.timers.aec_warmup()),
        };
        self.effective = next;
        diff
    }
}

impl VoiceDriver {
    pub(crate) fn on_frame(&mut self, now: Ts, cmd: ActivityCommand) {
        match cmd {
            // A sequence: `on_exit`, the drain, then the swap.
            // A VoiceTask holds the floor: its parent waits on it, so no agent replaces it.
            ActivityCommand::Update(update) if self.activity.voice_tasks.running() => {
                let refused = crate::agent_session::HandoffRefused::TaskRunning;
                self.refuse_handoff(None, update, refused)
            }
            ActivityCommand::Update(update) => {
                let run = self.session.runs.current().map(|run| run.id);
                if run.is_some() {
                    self.session.runs.participated();
                }
                self.handoff_step(crate::agent_activity::handoff::HandoffEvent::Asked {
                    update,
                    run,
                })
            }
            ActivityCommand::UpdateInstructions(next) => {
                self.activity.instructions = next.clone();
                // "Record the configuration change" (`AgentActivity.update_instructions`), on the
                // driver's clock.
                self.commit_config(Some(next.render(Some(Modality::Audio))), vec![], vec![]);
                // Ungated: the reference tells any realtime session and lets the plugin refuse.
                if self.activity.rt_session.is_some() {
                    self.rt_send(lk_agents_abi::RealtimeCommand::UpdateInstructions(
                        lk_agents_abi::UpdateInstructions {
                            instructions: next.render(Some(Modality::Audio)),
                        },
                    ));
                }
                self.publish_activity();
            }
            ActivityCommand::UpdateExpressive(expressive) => {
                self.session.expressive.session = expressive;
            }
            ActivityCommand::UpdateTools(toolset) => {
                let tools = toolset.specs.len();
                let (tools_added, tools_removed) = self.activity.update_tools(toolset);
                if !tools_added.is_empty() || !tools_removed.is_empty() {
                    self.commit_config(None, tools_added, tools_removed);
                }
                // Ungated, like instructions: a `mutable_tools: false` server must still learn a
                // handoff's new toolset.
                if self.activity.rt_session.is_some() {
                    self.rt_send(lk_agents_abi::RealtimeCommand::UpdateTools(
                        lk_agents_abi::UpdateTools { tools: self.activity.toolset.specs.clone() },
                    ));
                }
                self.record_trace(ToolStep::ToolsetUpdated { tools });
            }
            // A re-attached microphone opens a fresh STT stream.
            ActivityCommand::SetAudioInput(input) => {
                self.set_audio_input(now, input.map(|AudioInputHandle(input)| input))
            }
            ActivityCommand::SetVideoInput(input) => {
                let attached = input.is_some();
                // The leg already read, handed back: nothing to swap, and its new id is released.
                if self.input.video_instance.map(|v| v.instance) == input.map(|v| v.instance) {
                    return;
                }
                self.input.video_instance = input;
                self.record_effect(crate::agent_session::effects::VoiceEffect::Audio(
                    crate::agent_session::effects::AudioOp::SetVideoInput(input.map(|i| i.id)),
                ));
                self.record_trace(InputStep::VideoInputSet { attached });
            }
            ActivityCommand::SetIoEnabled { leg, enabled } => {
                self.set_io_enabled(now, leg, enabled)
            }
            ActivityCommand::Drain { call } => self.drain(call.into_iter().collect()),
            ActivityCommand::SetAudioOutput(output) => {
                let audio = output.is_some();
                self.set_outputs(|o| o.audio = output);
                // Selection changes future admissions. Existing segments keep their binding,
                // including its pause and cleanup controls, until their release arrives.
                self.retire_outputs();
                self.record_trace(PlayoutStep::OutputReplaced {
                    audio,
                    transcription: self.session.outputs.shows_transcription(),
                });
            }
            ActivityCommand::SetTranscriptionOutput(output) => {
                let transcription = output.is_some();
                self.set_outputs(|o| o.transcription = output);
                self.retire_outputs();
                self.record_trace(PlayoutStep::OutputReplaced {
                    audio: self.session.outputs.plays_audio(),
                    transcription,
                });
            }
            ActivityCommand::UpdateModels(models) => {
                let crate::agent_session::ModelUpdate { stt, vad, llm, tts } = *models;
                self.update_models(stt, vad, llm, tts)
            }
            ActivityCommand::UpdateChatCtx(items) => {
                let count = items.len();
                // Host-supplied, so not announced: the host already holds these items.
                let agent = self.activity.agent.clone();
                self.session.chat_ctx.update_agent_chat_ctx(items, &agent);
                // Ungated: the reference guards only on there being a session.
                if self.activity.rt_session.is_some() {
                    let ctx = self.session.chat_ctx.for_realtime();
                    self.rt_send(lk_agents_abi::RealtimeCommand::UpdateChatCtx(
                        lk_agents_abi::UpdateChatCtx { chat_ctx: ctx },
                    ));
                }
                self.publish_activity();
                self.record_trace(GenerationStep::ChatCtxUpdated { items: count });
            }
        }
    }

    /// The activity changed: the snapshot the spawned work reads is rebuilt once, as the cycle's
    /// effects are done ([`VoiceDriver::flush_activity`]), not per commit (a copy of the whole agent
    /// context each time grew a long call's cost with its length squared).
    pub(crate) fn publish_activity(&self) {
        self.activity_changed.set(true);
    }

    pub(crate) fn flush_activity(&mut self) {
        if !self.activity_changed.replace(false) {
            return;
        }
        self.activity.prune_settled_speeches();
        self.record_effect(VoiceEffect::ChatContext(Arc::new(
            self.session.chat_ctx.agent_chat_ctx(),
        )));
    }

    /// The history records an agent leaves when it takes over: `AgentHandoff`, then
    /// `AgentConfigUpdate` (skipped with neither instructions nor tools), in the reference's order.
    pub(crate) fn record_handoff(&mut self, previous_agent: Option<String>) {
        let handoff = lk_agents_abi::AgentHandoffItem {
            id: self.session.ids.mint("item"),
            old_agent_id: previous_agent,
            new_agent_id: self.activity.agent.clone(),
            created_at: self.step.wall(),
        };
        self.commit_record(ChatItem::AgentHandoff(handoff));
        let rendered = self.activity.instructions.render(Some(Modality::Audio));
        let tools_added: Vec<String> =
            self.activity.toolset.specs.iter().map(|t| t.name.clone()).collect();
        if !rendered.is_empty() || !tools_added.is_empty() {
            self.commit_config((!rendered.is_empty()).then_some(rendered), tools_added, vec![]);
        }
    }

    /// An `AgentConfigUpdate` record; an empty tool list is absent. The live callables (`tools`)
    /// are host objects the host attaches.
    fn commit_config(
        &mut self,
        instructions: Option<String>,
        tools_added: Vec<String>,
        tools_removed: Vec<String>,
    ) {
        let config = lk_agents_abi::AgentConfigUpdate {
            id: self.session.ids.mint("item"),
            instructions,
            tools_added: (!tools_added.is_empty()).then_some(tools_added),
            tools_removed: (!tools_removed.is_empty()).then_some(tools_removed),
            created_at: self.step.wall(),
            tools: None,
        };
        self.commit_record(ChatItem::AgentConfigUpdate(config));
    }

    /// Tell the realtime session which frame drives: instructions (audio rendering), conversation,
    /// tools, in the reference's order. No-op on a pipeline session.
    pub(crate) fn push_activity(&mut self, chat_ctx: Option<lk_agents_abi::ChatContext>) {
        if self.activity.rt_session.is_none() {
            return;
        }
        self.rt_send(lk_agents_abi::RealtimeCommand::UpdateInstructions(
            lk_agents_abi::UpdateInstructions {
                instructions: self.activity.instructions.render(Some(Modality::Audio)),
            },
        ));
        if let Some(chat_ctx) = chat_ctx {
            self.rt_send(lk_agents_abi::RealtimeCommand::UpdateChatCtx(
                lk_agents_abi::UpdateChatCtx { chat_ctx },
            ));
        }
        self.rt_send(lk_agents_abi::RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools {
            tools: self.activity.toolset.specs.clone(),
        }));
    }

    /// Put a handoff or config record on the record. `AgentHandoff` is history-only;
    /// `AgentConfigUpdate` goes to the agent's context and history (only the realtime view strips
    /// it).
    pub(crate) fn commit_record(&mut self, item: ChatItem) {
        let (source, visibility) = match &item {
            ChatItem::AgentHandoff(_) => (Source::Handoff, Visibility::History),
            _ => (Source::ConfigUpdate, Visibility::Both),
        };
        self.commit_item(item, source, visibility);
    }

    /// The one write to the record: inserted by its acoustic `created_at`, the speech told, the
    /// host told via `ItemAdded{source}`, and the activity republished.
    pub(crate) fn commit_item(&mut self, item: ChatItem, source: Source, visibility: Visibility) {
        // Attributed to the agent that created the speech, not the driving one (a tool may commit
        // after its agent was left); an item with no speech is the driving agent's.
        let agent = source
            .speech()
            .and_then(|s| self.activity.speech_agent(s).map(str::to_owned))
            .unwrap_or_else(|| self.activity.agent.clone());
        self.reply_recorded(&item, &source);
        // A delegated call's pending output: its expert's outcome may now replace it.
        if let ChatItem::FunctionCallOutput(output) = &item {
            self.session.experts.recorded(&output.call_id);
        }
        let commit = self.session.chat_ctx.commit(item, source, visibility, &agent);
        self.perform_commit(commit);
    }

    /// Items on the record changed in place (answered `InputRequired` outputs, an expert's
    /// outcome): the host is told again of each, under the item's own id, source and agent, and a
    /// realtime server's copy follows once.
    pub(crate) fn replace_items(&mut self, items: impl IntoIterator<Item = ChatItem>) {
        let mut replaced = false;
        for item in items {
            let Some(entry) = self.session.chat_ctx.entry(item.id()) else { continue };
            let (source, visibility, agent) =
                (entry.source.clone(), entry.visibility, entry.agent.clone());
            let (entry, _) = self.session.chat_ctx.commit(item, source, visibility, &agent);
            replaced = true;
            if entry.visibility.history() {
                let run = self.run_of_item(&entry.source);
                self.publish_event(crate::events::SessionEvent::ConversationItemAdded {
                    item: Box::new(entry.item),
                    agent: entry.agent,
                    source: entry.source,
                    visibility: entry.visibility,
                    run,
                });
            }
        }
        if !replaced {
            return;
        }
        if self.activity.rt_session.is_some() {
            let chat_ctx = self.session.chat_ctx.for_realtime();
            let update = lk_agents_abi::UpdateChatCtx { chat_ctx };
            self.rt_send(lk_agents_abi::RealtimeCommand::UpdateChatCtx(update));
        }
        self.publish_activity();
    }

    /// The speech is told, the host learns of a new item, the activity is republished.
    fn perform_commit(&mut self, (entry, outcome): (Committed, CommitOutcome)) {
        // A call was pushed to the handle when the model made it; everything else is new here.
        let tell_speech =
            entry.source.speech().filter(|_| !matches!(entry.source, Source::ToolCall { .. }));
        if let Some(state) =
            tell_speech.and_then(|id| self.activity.speech_scheduler.speech_mut(id))
        {
            state.item_added(&entry.item);
        }
        match outcome {
            // An item only in the agent's view is the model's: the host's history never sees it.
            CommitOutcome::Added if entry.visibility.history() => {
                // The person's message answers the call an `InputRequired` left open.
                if let ChatItem::Message(m) = &entry.item
                    && m.role == lk_agents_abi::Role::User
                {
                    self.answer_pending_input(&m.text_content().unwrap_or_default());
                }
                let run = self.run_of_item(&entry.source);
                self.publish_event(crate::events::SessionEvent::ConversationItemAdded {
                    item: Box::new(entry.item),
                    agent: entry.agent,
                    source: entry.source,
                    visibility: entry.visibility,
                    run,
                });
            }
            CommitOutcome::Added | CommitOutcome::Replaced | CommitOutcome::Filtered => {}
        }
        self.publish_activity();
    }
}

impl AgentActivity {
    /// What the driver keeps per speech outlives it only while something may still ask: the
    /// speech is live, a tool of it runs, or it is among the latest [`KEPT_SPEECHES`].
    fn prune_settled_speeches(&mut self) {
        let scheduler = &self.speech_scheduler;
        let tools = &self.tools;
        self.speech_history.prune(|speech| {
            scheduler.contains(speech) || tools.running.iter().any(|tool| tool.speech == speech)
        });
    }
}

impl AgentActivity {
    /// Construct the opening activity from the session assembly, before any transitions run.
    pub(crate) fn from_initial(
        initial: &mut crate::agent_session::InitialSession,
        paused_speech: PausedSpeech,
    ) -> Self {
        let effective = &initial.config.effective;
        let mut audio_recognition = AudioRecognition::new(effective, &initial.config.legs);
        audio_recognition.set_thresholds(initial.eot_thresholds.take());
        audio_recognition.set_interruption_active(initial.config.legs.adaptive_detector);
        audio_recognition.set_eot_provider(initial.eot_provider.take());
        Self {
            agent: initial.parts.agent.clone(),
            instructions: std::mem::take(&mut initial.instructions),
            dynamic_instructions: initial.dynamic_instructions.take(),
            output_nodes: std::mem::take(&mut initial.output_nodes),
            toolset: Arc::new(std::mem::take(&mut initial.toolset)),
            tools: Default::default(),
            audio_recognition,
            speech_scheduler: SpeechScheduler::new(),
            paused_speech,
            speech_history: Default::default(),
            reply_plans: Default::default(),
            tools_in_flight: Default::default(),
            spoke: Default::default(),
            announced_current_speech: None,
            interrupt_waiters: Vec::new(),
            hook_bound: initial.hook_bound,
            hooks: Default::default(),
            handoff: Default::default(),
            on_exit: Default::default(),
            drain: Default::default(),
            voice_tasks: Default::default(),
            dtmf: Default::default(),
            end_call: None,
            pending_turn: None,
            turn_limit: Default::default(),
            waiting_turns: Default::default(),
            last_turn: None,
            live_speech_at_final: false,
            preemptive_turn: None,
            preemptive_record_version: 0,
            rt_session: initial.rt_session.take(),
            realtime_state: crate::agent_activity::realtime::RealtimeSessionState::new(
                initial.rt_caps,
            ),
            realtime_players: Default::default(),
            launching: Default::default(),
            realtime_turn_detection_disabled: initial.rt_disabled,
            realtime_resolved_mode: initial.rt_resolved,
            realtime_requested_detection: initial.config.effective.turn.detection(),
        }
    }
}

impl AgentActivity {
    pub(crate) fn agent_name(&self) -> &str {
        &self.agent
    }
    pub(crate) fn realtime(&self) -> Option<&lk_agents_abi::resources::RealtimeRef> {
        self.rt_session.as_ref()
    }
    pub(crate) fn speeches(&self) -> impl Iterator<Item = SpeechId> + '_ {
        self.speech_scheduler.speech_ids()
    }
    pub(crate) fn speech_queue_empty(&self) -> bool {
        self.speech_scheduler.is_empty()
    }
    pub(crate) fn speech_can_bind_output(&self, speech: SpeechId) -> bool {
        self.speech_scheduler.speech(speech).is_some_and(|h| !h.interrupted() && !h.done())
    }
    pub(crate) fn speech_can_activate_output(&self, speech: SpeechId) -> bool {
        self.speech_scheduler.current_speech() == Some(speech)
            && self.speech_can_bind_output(speech)
    }
    pub(crate) fn retained_resources(
        &self,
    ) -> impl Iterator<Item = lk_agents_abi::resources::ResourceId> + '_ {
        [
            self.rt_session.as_ref().map(|r| r.id),
            self.dynamic_instructions,
            self.output_nodes.transcription,
        ]
        .into_iter()
        .flatten()
        .chain(self.handoff.updates().chain(self.voice_tasks.parents()).flat_map(|u| u.resources()))
    }
    pub(crate) fn finish_step(&mut self) {
        self.live_speech_at_final = false;
    }
    pub(crate) fn retire_speech_later(&mut self, speech: SpeechId) {
        self.speech_scheduler.retire_later(speech);
    }
    pub(crate) fn take_speech_updates(
        &mut self,
    ) -> Vec<(SpeechId, crate::speech_handle::SpeechUpdate)> {
        self.speech_scheduler.take_updates()
    }
    pub(crate) fn retire_settled_speeches(
        &mut self,
        retain: impl Fn(SpeechId) -> bool,
    ) -> Vec<SpeechId> {
        let tools = &self.tools;
        self.speech_scheduler
            .take_retired(|speech| retain(speech) || tools.live(speech).next().is_some())
    }
    pub(crate) fn current_speech_changed(&mut self) -> Option<Option<SpeechId>> {
        self.announce_current_speech(self.speech_scheduler.current_speech())
    }
    #[cfg(test)]
    pub(crate) fn realtime_players(&self) -> &BTreeSet<SpeechId> {
        &self.realtime_players
    }
}

impl AgentActivity {
    /// Interruptible speeches only (an uninterruptible greeting plays over the user by design); a
    /// speech whose first frame is out is never held again.
    pub(crate) fn launch_hold(&self) -> std::collections::BTreeSet<SpeechId> {
        // A block or supersession can move the floor away from a speech still at launch.
        if !self.audio_recognition.user_speaking() {
            return std::collections::BTreeSet::new();
        }
        self.launching
            .iter()
            .copied()
            .filter(|speech| {
                self.speech_scheduler.speech(*speech).is_some_and(|h| h.interruptible())
            })
            .collect()
    }
}

/// Facts and exit admission shared by the activity's handoff and drain protocols.
struct Cx<'a> {
    quiet: bool,
    closing: bool,
    switching: bool,
    leave: hooks::Leave<'a>,
}

impl AgentActivity {
    #[must_use]
    pub(crate) fn handoff_transition(
        &mut self,
        event: handoff::HandoffEvent,
        closing: bool,
    ) -> Vec<crate::agent_session::machine::coordination::VoiceCallback> {
        let (cx, handoff, _) = self.machines(closing);
        handoff.handle(event, cx)
    }
    #[must_use]
    pub(crate) fn drain_transition(
        &mut self,
        event: drain::DrainEvent,
        closing: bool,
    ) -> Vec<drain::DrainOutcome> {
        let (cx, _, drain) = self.machines(closing);
        drain.handle(event, cx)
    }
    fn machines(&mut self, closing: bool) -> (Cx<'_>, &mut handoff::Handoff, &mut drain::Drain) {
        let scope = self.scope();
        let cx = Cx {
            quiet: self.speech_scheduler.is_empty(),
            closing,
            switching: self.handoff.switching(),
            leave: hooks::Leave {
                on_exit: &mut self.on_exit,
                hooks: &mut self.hooks,
                agent: &self.agent,
                scope,
            },
        };
        (cx, &mut self.handoff, &mut self.drain)
    }
    pub(crate) fn handoff_exit_finished(
        &mut self,
        hook: crate::agent_session::HookId,
    ) -> Option<crate::agent_session::machine::coordination::VoiceCallback> {
        self.handoff.exit_hook_finished(hook)
    }
    #[must_use]
    pub(crate) fn drain_exit_finished(
        &mut self,
        hook: crate::agent_session::HookId,
        closing: bool,
    ) -> Vec<drain::DrainOutcome> {
        let (cx, _, drain) = self.machines(closing);
        drain.exit_hook_finished(hook, cx.quiet)
    }
    pub(crate) fn claim_exit(&mut self) -> hooks::Exit {
        let scope = self.scope();
        hooks::Leave {
            on_exit: &mut self.on_exit,
            hooks: &mut self.hooks,
            agent: &self.agent,
            scope,
        }
        .claim()
    }
    pub(crate) fn exit_ended(&mut self, hook: crate::agent_session::HookId) {
        self.on_exit.ended(hook);
    }
    pub(crate) fn exiting_hook(&self) -> Option<crate::agent_session::HookId> {
        self.on_exit.exiting()
    }
    pub(crate) fn take_claimed_exits(
        &mut self,
    ) -> Vec<(crate::agent_session::HookId, String, u64)> {
        self.on_exit.take_claimed()
    }
    pub(crate) fn has_voice_task(&self) -> bool {
        self.voice_tasks.running()
    }
    pub(crate) fn take_drain_waiters(&mut self) -> Vec<crate::agent_session::RequestId> {
        self.drain.take_waiters()
    }
}

impl AgentActivity {
    pub(crate) fn realtime_generating(&self) -> bool {
        self.realtime_state.generating()
    }
    pub(crate) fn has_pending_commit(&self) -> bool {
        self.audio_recognition.commit_pending()
    }
    pub(crate) fn transcript_pending(&self, now: Ts) -> bool {
        self.audio_recognition.transcript_pending(now)
    }
    pub(crate) fn close_recognition(
        &mut self,
        now: Ts,
        commit: bool,
    ) -> Vec<crate::audio_recognition::effects::AudioRecognitionEffect> {
        self.audio_recognition.on_closing(now, commit)
    }
    pub(crate) fn recognition_timer(
        &mut self,
        now: Ts,
        timer: crate::agent_session::effects::Timer,
    ) -> Vec<crate::audio_recognition::effects::AudioRecognitionEffect> {
        self.audio_recognition.on_timer(now, timer)
    }
    pub(crate) fn open_overlap(&self) -> Option<lk_agents_abi::turn::detector::ReqId> {
        self.audio_recognition.open_overlap()
    }
    pub(crate) fn end_agent_speech_after_overlap(
        &mut self,
        now: Ts,
        ignore_until: Ts,
    ) -> Vec<crate::audio_recognition::effects::AudioRecognitionEffect> {
        if self.paused_speech.paused_handle().is_none() {
            self.audio_recognition.on_end_of_agent_speech(now, ignore_until)
        } else {
            Vec::new()
        }
    }
    pub(crate) fn finish_interrupts(&mut self) -> Vec<VoiceEffect> {
        std::mem::take(&mut self.interrupt_waiters)
            .into_iter()
            .filter_map(|w| {
                w.call.map(|call| {
                    VoiceEffect::Answer(call, crate::agent_session::Answer::Interrupt(Ok(())))
                })
            })
            .collect()
    }
    pub(crate) fn finish_pending_commit(&mut self) -> Option<VoiceEffect> {
        self.audio_recognition.take_pending_commit().and_then(|p| p.call).map(|call| {
            VoiceEffect::Answer(call, crate::agent_session::Answer::Commit(Ok(String::new())))
        })
    }
    pub(crate) fn finish_drain_and_dtmf(&mut self) -> Vec<VoiceEffect> {
        let mut effects: Vec<_> = self
            .drain
            .take_waiters()
            .into_iter()
            .map(|call| {
                VoiceEffect::Answer(
                    call,
                    crate::agent_session::Answer::Drain(Err(
                        crate::agent_session::DrainRefused::Closing,
                    )),
                )
            })
            .collect();
        if let Some(call) = self.dtmf.take_call() {
            effects.push(VoiceEffect::Answer(
                call,
                crate::agent_session::Answer::Dtmf(crate::workflows::dtmf::DtmfOutcome::Stopped),
            ));
        }
        effects
    }
    #[must_use]
    pub(crate) fn dtmf_transition(
        &mut self,
        event: crate::workflows::dtmf::DtmfTaskEvent,
        now: Ts,
    ) -> Vec<crate::workflows::dtmf::DtmfTaskOutcome> {
        self.dtmf.handle(event, now)
    }
    pub(crate) fn wait_for_interrupts(
        &mut self,
        speeches: Vec<SpeechId>,
        call: Option<crate::agent_session::RequestId>,
    ) {
        self.interrupt_waiters.push(false_interruption::InterruptWaiter { speeches, call });
    }
    pub(crate) fn speech_is_held(&self, speech: SpeechId) -> bool {
        self.speech_scheduler.contains(speech)
    }
}

impl AgentActivity {
    pub(crate) fn request_end_call(&mut self, speech: SpeechId) {
        self.end_call = Some(crate::workflows::end_call::EndCall::AfterSpeech(speech));
    }
    pub(crate) fn end_call_event(
        &mut self,
        now: Ts,
        event: &crate::events::SessionEvent,
    ) -> Option<crate::workflows::end_call::EndCallUpdate> {
        use crate::events::SessionEvent;
        use crate::workflows::end_call::EndCallUpdate;
        use crate::workflows::end_call::{EndCall, REPLY_WAIT};
        let state = self.end_call?;
        match (state, event) {
            (EndCall::AfterSpeech(id), SessionEvent::SpeechDone { speech, .. })
                if *speech == id =>
            {
                if self.rt_session.is_some() && self.realtime_state.auto_tool_reply() {
                    self.end_call = Some(EndCall::AwaitingReply);
                    return Some(EndCallUpdate::ReplyDeadline(Some(
                        now.saturating_add(REPLY_WAIT),
                    )));
                } else {
                    self.end_call = None;
                    return Some(EndCallUpdate::Close);
                }
            }
            (EndCall::AwaitingReply, SessionEvent::SpeechCreated { speech, .. }) => {
                self.end_call = Some(EndCall::AfterReply(*speech));
                return Some(EndCallUpdate::ReplyDeadline(None));
            }
            (EndCall::AfterReply(id), SessionEvent::SpeechDone { speech, .. }) if *speech == id => {
                self.end_call = None;
                return Some(EndCallUpdate::Close);
            }
            _ => {}
        }
        None
    }
    pub(crate) fn end_call_timed_out(&mut self) -> bool {
        if self.end_call == Some(crate::workflows::end_call::EndCall::AwaitingReply) {
            self.end_call = None;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod speech_history_tests {
    use super::{SpeechHistory, SpeechId, SpeechKind};

    #[test]
    fn retention_keeps_live_speeches_and_does_not_invent_late_origins() {
        let mut history = SpeechHistory::default();
        for id in 1..=140 {
            history.remember(SpeechId(id), "original-agent".into(), SpeechKind::Say);
            history.finish(SpeechId(id));
        }
        // A completion can arrive before its origin. It must not move the retention window.
        history.commit_tools(SpeechId(500));
        history.prune(|id| id == SpeechId(1));
        assert!(history.finished(SpeechId(1)));
        assert!(!history.finished(SpeechId(12)));
        assert!(history.finished(SpeechId(13)));
        assert!(history.settled(SpeechId(500)));
        assert!(history.records[&SpeechId(500)].origin.is_none());
        // Registering that late speech preserves its already committed tool step.
        history.remember(SpeechId(500), "next-agent".into(), SpeechKind::TurnReply);
        assert!(history.settled(SpeechId(500)));
        assert_eq!(history.records[&SpeechId(1)].origin.as_ref().unwrap().0, "original-agent");
    }

    #[test]
    fn repeated_completion_does_not_trigger_retention_early() {
        let mut history = SpeechHistory::default();
        history.remember(SpeechId(1), "agent".into(), SpeechKind::Say);
        history.remember(SpeechId(500), "agent".into(), SpeechKind::Say);
        for _ in 0..300 {
            history.finish(SpeechId(1));
            history.commit_tools(SpeechId(1));
        }
        history.prune(|_| false);
        assert!(history.finished(SpeechId(1)));
    }

    #[test]
    fn retention_handles_the_last_speech_ids_without_overflow() {
        let mut history = SpeechHistory::default();
        for offset in 0..140 {
            let speech = SpeechId(u64::MAX - offset);
            history.remember(speech, "agent".into(), SpeechKind::Say);
            history.finish(speech);
        }
        history.prune(|_| false);
        assert!(history.finished(SpeechId(u64::MAX)));
        assert!(history.finished(SpeechId(u64::MAX - 127)));
        assert!(!history.finished(SpeechId(u64::MAX - 128)));
    }
}
