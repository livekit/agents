use crate::agent::{CommitOutcome, Committed, Source, Visibility};
use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::machine::activity::false_interruption::PausedSpeech;
use crate::agent_session::machine::activity::scheduling::{SpeechId, SpeechKind, SpeechScheduler};
use crate::agent_session::options::{AgentSessionOptions, ConfigStack, OverrideLayer, SessionLegs};
use crate::agent_session::{ActivityCommand, AudioInputHandle, VoiceDriver};
use crate::audio_recognition::AudioRecognition;
use crate::trace::{GenerationStep, InputStep, PlayoutStep, ToolStep};
use lk_agents_abi::llm::toolsets::SESSION_TOOLSET_SCOPE;
use lk_agents_abi::{ChatItem, Modality, Ts};
use lk_agents_core::markup::ChatMessageExt;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

pub(crate) mod drain;
mod experts;
pub mod false_interruption;
pub mod handoff;
pub mod hooks;
mod idle;
pub(crate) mod io;
pub mod realtime;
pub(crate) mod recognition;
pub(crate) mod reply_run;
pub mod scheduling;
mod spans;
pub(crate) mod speech;
mod tool_results;
pub(crate) mod tools;
pub(crate) mod user_turn;
pub mod voice_task;

/// How many of the latest speeches the driver remembers once settled (their agent, end, tool
/// steps): a late item or a reply planned behind one is never that far behind.
const KEPT_SPEECHES: usize = 128;

/// Facts retained together after a speech leaves the scheduler. An unknown late completion
/// carries no invented origin; attribution becomes available only when the speech is registered.
#[derive(Default)]
struct SpeechHistory {
    records: BTreeMap<SpeechId, SpeechRecord>,
}

#[derive(Default)]
struct SpeechRecord {
    origin: Option<(String, SpeechKind)>,
    finished: bool,
    tools_committed: bool,
}

impl SpeechHistory {
    fn remember(&mut self, speech: SpeechId, agent: String, kind: SpeechKind) {
        let record = self.records.entry(speech).or_default();
        record.origin = Some((agent, kind));
    }

    fn finish(&mut self, speech: SpeechId) {
        let record = self.records.entry(speech).or_default();
        record.finished = true;
    }

    fn commit_tools(&mut self, speech: SpeechId) {
        let record = self.records.entry(speech).or_default();
        record.tools_committed = true;
    }

    fn finished(&self, speech: SpeechId) -> bool {
        self.records.get(&speech).is_some_and(|record| record.finished)
    }

    fn settled(&self, speech: SpeechId) -> bool {
        self.records.get(&speech).is_some_and(|record| record.finished || record.tools_committed)
    }

    fn prune(&mut self, live: impl Fn(SpeechId) -> bool) {
        if self.records.len() <= KEPT_SPEECHES {
            return;
        }
        let Some(newest) =
            self.records.iter().rev().find_map(|(id, record)| record.origin.as_ref().map(|_| *id))
        else {
            return;
        };
        self.records
            .retain(|id, _| newest.0.saturating_sub(id.0) < KEPT_SPEECHES as u64 || live(*id));
    }
}

/// What the STT and VAD legs are fed: every model in the tree wants 16 kHz mono.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// The running agent's state. What spans two of its components is the `VoiceDriver`'s to answer.
pub(crate) struct AgentActivity {
    agent: String,
    /// Replies in flight keep the prompt and tools they snapshotted.
    instructions: crate::generation::Instructions,
    dynamic_instructions: Option<lk_agents_abi::resources::ResourceId>,
    output_nodes: crate::agent_session::OutputNodes,
    toolset: Arc<crate::tool_executor::Toolset>,
    tools: crate::tool_executor::Tools,
    audio_recognition: AudioRecognition,
    speech_scheduler: SpeechScheduler,
    paused_speech: PausedSpeech,
    /// Attribution and completion facts share one lifetime, including after the close's edge.
    speech_history: SpeechHistory,
    /// Replies waiting for their predecessor's commit before their first step is planned.
    reply_plans:
        std::collections::VecDeque<(crate::agent_session::RequestId, crate::generation::ReplyPlan)>,
    /// Keeps the agent `thinking` across a filler `say` from inside a tool.
    pub(in crate::agent_session) tools_in_flight: BTreeSet<SpeechId>,
    /// Replies the person cut (their interruption or the host's `interrupt()`), released once
    /// in their tools; never a cut the framework makes, such as a VoiceTask taking the floor.
    pub(in crate::agent_session) cut_by_person: BTreeSet<SpeechId>,
    /// Interrupted replies released while their tools run: each keeps its reply run until its
    /// own end or its interruption watchdog.
    pub(in crate::agent_session) outliving_tools: BTreeSet<SpeechId>,
    /// The step's first frame transitions, once. Cleared when the step is spoken.
    spoke: BTreeSet<SpeechId>,
    /// Compared at the end of every cycle to announce floor changes.
    announced_current_speech: Option<SpeechId>,
    interrupt_waiters:
        Vec<crate::agent_session::machine::activity::false_interruption::InterruptWaiter>,
    /// How long a hook whose answer is needed is waited on; `None` when no host runs hooks.
    hook_bound: Option<Duration>,
    hooks: crate::agent_session::machine::activity::hooks::Hooks,
    handoff: crate::agent_session::machine::activity::handoff::Handoff,
    on_exit: crate::agent_session::machine::activity::hooks::OnExit,
    drain: crate::agent_session::machine::activity::drain::Drain,
    voice_tasks: crate::agent_session::machine::activity::voice_task::VoiceTasks,
    dtmf: crate::workflows::machine::dtmf::Dtmf,
    /// The speech whose goodbye the session closes after.
    end_call: Option<crate::workflows::end_call::EndCall>,
    turn_limit: crate::agent_session::machine::activity::hooks::TurnLimit,
    /// The user turn parked on `on_user_turn_completed`.
    pending_turn: Option<recognition::PendingTurn>,
    /// Decided turns waiting for `pending_turn`'s hook to return, in order.
    waiting_turns: std::collections::VecDeque<recognition::DecidedTurn>,
    pub(crate) user_turn: Option<crate::telemetry::UserTurnSpan>,
    /// From a handoff's `on_exit` or a VoiceTask's suspension to the arriving agent's `on_enter`.
    update: Option<crate::telemetry::Holding<crate::telemetry::UpdateAgent>>,
    pub(crate) arriving: Option<crate::telemetry::Arriving>,
    /// What the activity's input streams keep running under.
    running_activity: Option<crate::telemetry::SpanContext>,
    last_turn: Option<crate::audio_recognition::TurnId>,
    /// A final arrived while the current speech was uninterrupted: this cycle's preemptive
    /// generation must still see the speech as live.
    live_speech_at_final: bool,
    preemptive: Option<Preemptive>,
    model: ActivityModel,
    launching: BTreeSet<SpeechId>,
}

enum ActivityModel {
    Pipeline,
    Realtime(Box<RealtimeActivity>),
}

struct RealtimeActivity {
    session: lk_agents_abi::resources::RealtimeRef,
    state: realtime::RealtimeSessionState,
    players: BTreeSet<SpeechId>,
    heard: realtime::HeardInput,
    turn_detection_disabled: bool,
    resolved_mode: crate::audio_recognition::TurnDetectionMode,
    requested_detection: Option<crate::agent_session::options::TurnDetection>,
}

struct Preemptive {
    turn: crate::audio_recognition::TurnId,
    record_version: u64,
}

impl AgentActivity {
    pub(crate) fn scope(&self) -> u64 {
        self.toolset.agent.as_ref().map_or(SESSION_TOOLSET_SCOPE, |agent| agent.toolset_scope())
    }

    /// Replace the activity's advertised tools and report the visible diff in deterministic order.
    pub(crate) fn update_tools(
        &mut self,
        toolset: crate::tool_executor::Toolset,
    ) -> (Vec<String>, Vec<String>) {
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

    /// A handoff lands; returns the previous agent's id.
    pub(crate) fn set_agent(
        &mut self,
        agent: String,
        instructions: crate::generation::Instructions,
        dynamic_instructions: Option<lk_agents_abi::resources::ResourceId>,
        output_nodes: crate::agent_session::OutputNodes,
        toolset: crate::tool_executor::Toolset,
    ) -> String {
        self.instructions = instructions;
        self.dynamic_instructions = dynamic_instructions;
        self.output_nodes = output_nodes;
        self.toolset = Arc::new(toolset);
        std::mem::replace(&mut self.agent, agent)
    }

    pub(crate) fn retries(&self) -> lk_agents_abi::agent::Retries {
        self.toolset.agent.as_ref().map(|a| a.definition().retries).unwrap_or_default()
    }

    /// `generate_reply(tools=)`: the agent's toolset offering only `names` and every
    /// provider-owned tool, in declaration order.
    pub(crate) fn reply_toolset(&self, names: &[String]) -> Arc<crate::tool_executor::Toolset> {
        Arc::new(self.toolset.narrowed(names))
    }

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

    pub(crate) fn has_run_work(&self, run: crate::events::RunId) -> bool {
        self.speech_scheduler.has_run_work(run)
            || self.hooks.has_run_work(run)
            || self.handoff.has_run_work(run)
    }

    pub(crate) fn note_speech_agent(&mut self, speech: SpeechId, kind: SpeechKind) {
        self.speech_history.remember(speech, self.agent.clone(), kind);
    }

    pub(crate) fn speech_agent(&self, speech: SpeechId) -> Option<&str> {
        self.speech_history.records.get(&speech)?.origin.as_ref().map(|(agent, _)| agent.as_str())
    }

    /// What `speech` was created as, after its release too.
    pub(crate) fn speech_kind(&self, speech: SpeechId) -> Option<SpeechKind> {
        self.speech_history.records.get(&speech)?.origin.as_ref().map(|(_, kind)| *kind)
    }

    /// The new current speech when it moved since last announced.
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
        self.preemptive = Some(Preemptive {
            turn: crate::audio_recognition::TurnId(self.last_turn.map_or(1, |t| t.0 + 1)),
            record_version,
        });
    }

    /// The turn detection the options said, and the mode they resolved to before the model's
    /// refinement: whether either moved.
    pub(crate) fn note_resolved_mode(
        &mut self,
        resolved: crate::audio_recognition::TurnDetectionMode,
        detection: Option<crate::agent_session::options::TurnDetection>,
    ) -> bool {
        let Some(rt) = self.realtime_activity_mut() else { return false };
        let moved = resolved != rt.resolved_mode || detection != rt.requested_detection;
        rt.resolved_mode = resolved;
        rt.requested_detection = detection;
        moved
    }
}

/// The driver's configuration layers and their resolved answer. `effective` is only written by
/// `resolve`; a change reports what moved as a [`ConfigDiff`].
#[derive(Debug, Clone, PartialEq)]
pub struct DriverConfig {
    pub stack: ConfigStack,
    pub legs: SessionLegs,
    effective: AgentSessionOptions,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfigDiff {
    /// The endpointing leaves changed: replace the model.
    pub endpointing: Option<crate::turn::EndpointingOptions>,
    pub turn_detection_mode: Option<crate::audio_recognition::TurnDetectionMode>,
    /// `timers.aec_warmup` changed.
    pub aec_warmup: Option<Option<Duration>>,
    pub push_to_talk: Option<bool>,
}

impl DriverConfig {
    pub fn effective(&self) -> &AgentSessionOptions {
        &self.effective
    }

    pub fn refine_turn_detection(&mut self, owner: crate::audio_recognition::TurnDetectionMode) {
        self.effective.turn.owner = Some(owner);
    }

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
            push_to_talk: (next.turn.push_to_talk() != self.effective.turn.push_to_talk())
                .then_some(next.turn.push_to_talk()),
        };
        self.effective = next;
        diff
    }
}

impl VoiceDriver {
    pub(crate) fn on_frame(&mut self, now: Ts, cmd: ActivityCommand) {
        match cmd {
            // A VoiceTask's parent waits on it, so no agent replaces it.
            ActivityCommand::Update(update) if self.activity.voice_tasks.running() => {
                let refused = crate::agent_session::HandoffRefused::TaskRunning;
                self.refuse_handoff(update, refused)
            }
            ActivityCommand::Update(update) => {
                self.arriving(&update.agent, update.activation);
                let run = self.session.runs.current().map(|run| run.id);
                if run.is_some() {
                    self.session.runs.participated();
                }
                self.handoff_step(
                    crate::agent_session::machine::activity::handoff::HandoffEvent::Asked {
                        update,
                        run,
                    },
                )
            }
            ActivityCommand::UpdateInstructions(next) => {
                self.activity.instructions = next.clone();
                self.commit_config(Some(next.render(Some(Modality::Audio))), vec![], vec![]);
                // Ungated: the plugin may refuse.
                if self.activity.realtime().is_some() {
                    self.rt_send(lk_agents_abi::RealtimeCommand::UpdateInstructions(
                        lk_agents_abi::UpdateInstructions {
                            instructions: next.render(Some(Modality::Audio)),
                        },
                    ));
                }
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
                if self.activity.realtime().is_some() {
                    self.rt_send(lk_agents_abi::RealtimeCommand::UpdateTools(
                        lk_agents_abi::UpdateTools { tools: self.activity.toolset.specs.clone() },
                    ));
                }
                self.record_trace(ToolStep::ToolsetUpdated { tools });
            }
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
                // Existing segments keep their binding until their release arrives.
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
                if self.activity.realtime().is_some() {
                    let ctx = self.session.chat_ctx.for_realtime();
                    self.rt_send(lk_agents_abi::RealtimeCommand::UpdateChatCtx(
                        lk_agents_abi::UpdateChatCtx { chat_ctx: ctx },
                    ));
                }
                self.record_trace(GenerationStep::ChatCtxUpdated { items: count });
            }
        }
    }

    pub(crate) fn flush_activity(&mut self) {
        self.activity.prune_settled_speeches();
        let revision = self.session.chat_ctx.view_revision();
        if self.published_chat_revision == Some(revision) {
            return;
        }
        self.published_chat_revision = Some(revision);
        self.record_effect(VoiceEffect::ChatContext(Arc::new(
            self.session.chat_ctx.agent_chat_ctx(),
        )));
    }

    /// `AgentHandoff`, then `AgentConfigUpdate` (skipped with neither instructions nor tools).
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

    /// Tell the realtime session the instructions, conversation and tools, in that order.
    pub(crate) fn push_activity(&mut self, chat_ctx: Option<lk_agents_abi::ChatContext>) {
        if self.activity.realtime().is_none() {
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

    /// `AgentHandoff` is history-only; `AgentConfigUpdate` goes to the agent's context and history.
    pub(crate) fn commit_record(&mut self, item: ChatItem) {
        let (source, visibility) = match &item {
            ChatItem::AgentHandoff(_) => (Source::Handoff, Visibility::History),
            _ => (Source::ConfigUpdate, Visibility::Both),
        };
        self.commit_item(item, source, visibility);
    }

    /// The one write to the record, inserted by its acoustic `created_at`.
    pub(crate) fn commit_item(&mut self, item: ChatItem, source: Source, visibility: Visibility) {
        // Attributed to the speech's agent: a tool may commit after its agent was left.
        let agent = source
            .speech()
            .and_then(|s| self.activity.speech_agent(s).map(str::to_owned))
            .unwrap_or_else(|| self.activity.agent.clone());
        self.reply_recorded(&item, &source);
        // A delegated call's pending output: its expert's outcome may now replace it.
        if let ChatItem::FunctionCallOutput(output) = &item {
            self.session.experts.recorded(&output.call_id);
        }
        let user_turn = match (&item, &source) {
            (ChatItem::Message(message), Source::UserTurn { turn }) => {
                Some((*turn, message.clone()))
            }
            _ => None,
        };
        let commit = self.session.chat_ctx.commit(item, source, visibility, &agent);
        self.perform_commit(commit);
        if let Some((turn, message)) = user_turn {
            self.user_turn_message_added(turn, &message);
        }
    }

    /// Items on the record changed in place: the host is told again of each under its own id,
    /// and a realtime server's copy follows once.
    pub(crate) fn replace_items(&mut self, items: impl IntoIterator<Item = ChatItem>) {
        let mut replaced = false;
        for item in items {
            let Some(entry) = self.session.chat_ctx.entry(item.id()) else { continue };
            let (source, visibility, agent) =
                (entry.source.clone(), entry.visibility, entry.agent.clone());
            let (entry, _) = self.session.chat_ctx.commit(item, source, visibility, &agent);
            replaced = true;
            if entry.visibility.history() {
                if self.run_owns_item(&entry.source) {
                    self.session.runs.record(&entry.item);
                }
                self.item_turn(&entry.item, &entry.source);
                self.publish_event(crate::events::SessionEvent::ConversationItemAdded {
                    item: Box::new(entry.item),
                    agent: entry.agent,
                    source: entry.source,
                    visibility: entry.visibility,
                });
            }
        }
        if !replaced {
            return;
        }
        if self.activity.realtime().is_some() {
            let chat_ctx = self.session.chat_ctx.for_realtime();
            let update = lk_agents_abi::UpdateChatCtx { chat_ctx };
            self.rt_send(lk_agents_abi::RealtimeCommand::UpdateChatCtx(update));
        }
    }

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
                if self.run_owns_item(&entry.source) {
                    self.session.runs.record(&entry.item);
                }
                self.item_turn(&entry.item, &entry.source);
                self.publish_event(crate::events::SessionEvent::ConversationItemAdded {
                    item: Box::new(entry.item),
                    agent: entry.agent,
                    source: entry.source,
                    visibility: entry.visibility,
                });
            }
            CommitOutcome::Added | CommitOutcome::Replaced | CommitOutcome::Filtered => {}
        }
    }
}

impl AgentActivity {
    /// Keeps per-speech facts while the speech is live, a tool of it runs, or it is among the
    /// latest [`KEPT_SPEECHES`].
    fn prune_settled_speeches(&mut self) {
        let scheduler = &self.speech_scheduler;
        let tools = &self.tools;
        self.speech_history.prune(|speech| {
            scheduler.contains(speech) || tools.running.iter().any(|tool| tool.speech == speech)
        });
    }
}

impl AgentActivity {
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
            cut_by_person: Default::default(),
            outliving_tools: Default::default(),
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
            user_turn: None,
            update: None,
            arriving: None,
            running_activity: None,
            last_turn: None,
            live_speech_at_final: false,
            preemptive: None,
            model: match initial.rt_session.take() {
                Some(session) => ActivityModel::Realtime(Box::new(RealtimeActivity {
                    session,
                    state: realtime::RealtimeSessionState::new(initial.rt_caps),
                    players: Default::default(),
                    heard: Default::default(),
                    turn_detection_disabled: initial.rt_disabled,
                    resolved_mode: initial.rt_resolved,
                    requested_detection: initial.config.effective.turn.detection(),
                })),
                None => ActivityModel::Pipeline,
            },
            launching: Default::default(),
        }
    }
}

impl AgentActivity {
    pub(crate) fn agent_name(&self) -> &str {
        &self.agent
    }
    pub(crate) fn realtime(&self) -> Option<&lk_agents_abi::resources::RealtimeRef> {
        self.realtime_activity().map(|rt| &rt.session)
    }
    fn realtime_activity(&self) -> Option<&RealtimeActivity> {
        match &self.model {
            ActivityModel::Pipeline => None,
            ActivityModel::Realtime(rt) => Some(rt),
        }
    }
    fn realtime_activity_mut(&mut self) -> Option<&mut RealtimeActivity> {
        match &mut self.model {
            ActivityModel::Pipeline => None,
            ActivityModel::Realtime(rt) => Some(rt),
        }
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
        [self.realtime().map(|r| r.id), self.dynamic_instructions, self.output_nodes.transcription]
            .into_iter()
            .flatten()
            .chain(
                self.handoff
                    .updates()
                    .chain(self.voice_tasks.parents())
                    .flat_map(|u| u.resources()),
            )
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
    pub(crate) fn realtime_players(&self) -> impl Iterator<Item = &SpeechId> {
        self.realtime_activity().into_iter().flat_map(|rt| rt.players.iter())
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
        self.realtime_activity().is_some_and(|rt| rt.state.generating())
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
                crate::agent_session::Answer::Dtmf(
                    crate::workflows::machine::dtmf::DtmfOutcome::Stopped,
                ),
            ));
        }
        effects
    }
    #[must_use]
    pub(crate) fn dtmf_transition(
        &mut self,
        event: crate::workflows::machine::dtmf::DtmfTaskEvent,
        now: Ts,
    ) -> Vec<crate::workflows::machine::dtmf::DtmfTaskOutcome> {
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
                if self.realtime_activity().is_some_and(|rt| rt.state.auto_tool_reply()) {
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
    fn retention_counts_speeches_instead_of_their_facts() {
        let mut history = SpeechHistory::default();
        for id in 1..=100 {
            let speech = SpeechId(id * 1000);
            history.remember(speech, "agent".into(), SpeechKind::Say);
            history.finish(speech);
            history.commit_tools(speech);
        }
        history.prune(|_| false);
        assert_eq!(history.records.len(), 100);
        assert!(history.finished(SpeechId(1000)));
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
