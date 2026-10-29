//! `voice/agent_activity.py`'s `AgentActivity`: the running agent and its components, the driver's
//! configuration, and the agent commands (prompt, tools, context, IO) the driver applies.

use crate::tts::markup::ChatMessageExt;
pub mod drain;
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
pub mod tools;
pub mod voice_task;

use crate::agent::{CommitOutcome, Committed, Source, Visibility};
use crate::agent_activity::false_interruption::PausedSpeech;
use crate::agent_activity::options::{AgentSessionOptions, ConfigStack, LegSet};
use crate::agent_activity::scheduling::{SpeechId, SpeechKind, SpeechScheduler};
use crate::agent_session::effects::{Effect, Out};
use crate::agent_session::{ActivityCommand, Driver, HostAudio, HostVideo};
use crate::audio_recognition::AudioRecognition;
use crate::trace::{GenerationStep, InputStep, PlayoutStep, ToolStep};
use lk_agents_abi::{ChatItem, Modality, Ts};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

/// How many of the latest speeches the driver remembers once settled (their agent, end, tool
/// steps): a late item or a reply planned behind one is never that far behind.
const KEPT_SPEECHES: usize = 128;

/// What the STT and VAD legs are fed: every model in the tree wants 16 kHz mono.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// `voice/agent_activity.py`'s `AgentActivity`: the running agent (its name, prompt and
/// tools), its audio recognition, the speech queue and the current speech, the false-interruption
/// pause, the hooks, and the realtime session. The `Driver` orchestrates it
/// together with the [`VoiceSession`](crate::agent_session::VoiceSession): what spans two of its
/// components is the driver's to answer.
pub(crate) struct AgentActivity {
    /// The running agent's id (`AgentActivity.agent.id`).
    pub(crate) agent: String,
    /// Replies in flight keep the prompt and tools they snapshotted.
    pub(crate) instructions: crate::generation::Instructions,
    pub(crate) dynamic_instructions: Option<Arc<dyn crate::generation::HostInstructions>>,
    /// The agent's `transcription_node` and `realtime_audio_output_node`.
    pub(crate) output_nodes: crate::agent_session::nodes::OutputNodes,
    pub(crate) toolset: Arc<crate::tool_executor::Toolset>,
    /// `_tool_executor`.
    pub(crate) tools: crate::tool_executor::Tools,
    pub(crate) audio_recognition: AudioRecognition,
    /// `_speech_q`, `_current_speech`, `_scheduling_paused`, `_preemptive_generation`, and every
    /// live speech's handle.
    pub(crate) speech_scheduler: SpeechScheduler,
    /// `_paused_speech` and the false-interruption timer.
    pub(crate) paused_speech: PausedSpeech,
    /// The agent each speech was created under, and what it was created as: late items belong to
    /// it, and keep their source once the close's edge released the speech.
    pub(crate) speech_agents: BTreeMap<SpeechId, (String, SpeechKind)>,
    pub(crate) finished: BTreeSet<SpeechId>,
    /// The speeches whose tools a step committed: what they said is on the record.
    pub(crate) tools_committed: BTreeSet<SpeechId>,
    /// Replies waiting for their predecessor's commit before their first step is planned.
    pub(crate) reply_plans:
        std::collections::VecDeque<(crate::agent_session::RequestId, crate::generation::ReplyPlan)>,
    /// Keeps the agent `thinking` across a filler `say` from inside a tool.
    pub(crate) tools_in_flight: BTreeSet<SpeechId>,
    /// The step's first frame transitions, once. Cleared when the step is spoken.
    pub(crate) spoke: BTreeSet<SpeechId>,
    /// `_current_speech` as last announced (`FloorChanged`); compared at the end of every cycle.
    pub(crate) announced_current_speech: Option<SpeechId>,
    /// `interrupt()`s waiting for their speeches' release.
    pub(crate) interrupt_waiters: Vec<crate::agent_activity::false_interruption::InterruptWaiter>,
    /// How long a hook whose answer is needed is waited on; `None` when no host runs hooks.
    pub(crate) hook_bound: Option<Duration>,
    /// The developer's callbacks in flight.
    pub(crate) hooks: crate::agent_activity::hooks::Hooks,
    /// `update_agent()`: the switches asked for and in progress.
    pub(crate) handoff: crate::agent_activity::handoff::Handoff,
    /// The `on_exit` each agent owes.
    pub(crate) on_exit: crate::agent_activity::hooks::OnExit,
    /// The host's `drain()`.
    pub(crate) drain: crate::agent_activity::drain::Drain,
    /// `VoiceTask.run()`: the tasks holding the floor and the parents they suspended.
    pub(crate) voice_tasks: crate::agent_activity::voice_task::VoiceTasks,
    /// DTMF collection while it runs.
    pub(crate) dtmf: crate::workflows::dtmf::Dtmf,
    /// `EndCallTool`: the speech whose goodbye the session closes after.
    pub(crate) end_call: Option<crate::workflows::end_call::EndCall>,
    /// `on_user_turn_exceeded`, waited for or running.
    pub(crate) turn_limit: crate::agent_activity::hooks::TurnLimit,
    /// The user turn parked on `on_user_turn_completed` (`_user_turn_completed_atask`).
    pub(crate) pending_turn: Option<crate::audio_recognition::PendingTurn>,
    /// Decided turns waiting for `pending_turn`'s hook to return, in order.
    pub(crate) waiting_turns: std::collections::VecDeque<crate::audio_recognition::EndOfTurnInfo>,
    /// So a preemptive generation can be keyed to the turn after it.
    pub(crate) last_turn: Option<crate::audio_recognition::TurnId>,
    /// A final arrived while the current speech was still uninterrupted: the preemptive
    /// generation that final asks for this cycle sees it so, as the reference's does (its final
    /// cuts a paused speech in a task, after `on_preemptive_generation` ran).
    pub(crate) live_speech_at_final: bool,
    /// `generate_reply` matches on it.
    pub(crate) preemptive_turn: Option<crate::audio_recognition::TurnId>,
    /// The record version a preemptive generation was planned against.
    pub(crate) preemptive_record_version: u64,
    /// `_rt_session`. Exclusive with a pipeline.
    pub(crate) rt_session: Option<lk_agents_abi::resources::RealtimeRef>,
    pub(crate) realtime_state: crate::agent_activity::realtime::RealtimeSessionState,
    /// Speech identities whose runtime player is still open.
    pub(crate) realtime_players: BTreeSet<SpeechId>,
    /// Authorized speeches awaiting their first frame.
    pub(crate) launching: BTreeSet<SpeechId>,
    /// The model's turn detection was switched off at the open.
    pub(crate) realtime_turn_detection_disabled: bool,
    /// Before the model's refinement, so a live change is one that moved this.
    pub(crate) realtime_resolved_mode: crate::audio_recognition::TurnDetectionMode,
    /// The `turn_detection` the options last said, so a live change is one that moved it.
    pub(crate) realtime_requested_detection: crate::agent_activity::options::TurnDetection,
}

impl AgentActivity {
    /// A handoff lands: the new agent, prompt, nodes and tools. The previous agent's id.
    pub(crate) fn set_agent(
        &mut self,
        agent: String,
        instructions: crate::generation::Instructions,
        dynamic_instructions: Option<Arc<dyn crate::generation::HostInstructions>>,
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
    pub(crate) fn note_speech_agent(&mut self, speech: SpeechId, kind: SpeechKind) {
        self.speech_agents.insert(speech, (self.agent.clone(), kind));
    }

    /// The agent `speech` was created under.
    pub(crate) fn speech_agent(&self, speech: SpeechId) -> Option<&str> {
        self.speech_agents.get(&speech).map(|(agent, _)| agent.as_str())
    }

    /// What `speech` was created as, after its release too.
    pub(crate) fn speech_kind(&self, speech: SpeechId) -> Option<SpeechKind> {
        self.speech_agents.get(&speech).map(|(_, kind)| *kind)
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
    pub legs: LegSet,
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
    pub fn new(stack: ConfigStack, legs: LegSet) -> Self {
        let effective = stack.resolve(&legs);
        DriverConfig { stack, legs, effective }
    }

    /// Before `start()`: no machine exists yet to tell.
    pub fn edit(&mut self, f: impl FnOnce(&mut AgentSessionOptions, &mut LegSet)) {
        f(&mut self.stack.session, &mut self.legs);
        self.effective = self.stack.resolve(&self.legs);
    }

    /// Merged into the session layer, re-resolved.
    pub fn configure(&mut self, update: &AgentSessionOptions) -> ConfigDiff {
        self.stack.session.merge(update);
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

impl Driver {
    pub(crate) fn on_frame(&mut self, now: Ts, cmd: ActivityCommand, out: &mut Out) {
        match cmd {
            // A sequence: `on_exit`, the drain, then the swap.
            // A VoiceTask holds the floor: its parent waits on it, so no agent replaces it.
            ActivityCommand::Update(update) if self.activity.voice_tasks.running() => {
                let refused = crate::agent_session::HandoffRefused::TaskRunning;
                self.refuse_handoff(None, update, refused, out)
            }
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
            ActivityCommand::UpdateExpressive(expressive) => {
                self.session.expressive.session = expressive;
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
                let instance = input.as_ref().map(|input| Arc::as_ptr(input).cast::<()>() as usize);
                if self.input.video_instance == instance {
                    return;
                }
                self.input.video_instance = instance;
                out.machine().push(crate::agent_session::effects::Effect::Audio(
                    crate::agent_session::effects::AudioOp::SetVideoInput(input.map(HostVideo)),
                ));
                out.trace(InputStep::VideoInputSet { attached });
            }
            ActivityCommand::SetIoEnabled { leg, enabled } => {
                self.set_io_enabled(now, leg, enabled, out)
            }
            ActivityCommand::Drain { call } => self.drain(call.into_iter().collect(), out),
            ActivityCommand::SetAudioOutput(output) => {
                let audio = output.is_some();
                self.set_outputs(|o| o.audio = output);
                // Selection changes future admissions. Existing segments keep their binding,
                // including its pause and cleanup controls, until their release arrives.
                self.retire_outputs(out);
                out.trace(PlayoutStep::OutputReplaced {
                    audio,
                    transcription: self.session.outputs.shows_transcription(),
                });
            }
            ActivityCommand::SetTranscriptionOutput(output) => {
                let transcription = output.is_some();
                self.set_outputs(|o| o.transcription = output);
                self.retire_outputs(out);
                out.trace(PlayoutStep::OutputReplaced {
                    audio: self.session.outputs.plays_audio(),
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
                            lk_agents_abi::UpdateChatCtx { chat_ctx: ctx },
                        ),
                        out,
                    );
                }
                self.publish_activity();
                out.trace(GenerationStep::ChatCtxUpdated { items: count });
            }
        }
    }

    /// The activity changed: the snapshot the spawned work reads is rebuilt once, as the cycle's
    /// effects are done ([`Driver::flush_activity`]), not per commit (a copy of the whole agent
    /// context each time grew a long call's cost with its length squared).
    pub(crate) fn publish_activity(&self) {
        self.activity_changed.set(true);
    }

    pub(crate) fn flush_activity(&mut self) {
        if !self.activity_changed.replace(false) {
            return;
        }
        self.prune_settled_speeches();
        self.outbox.push(Effect::ChatContext(Arc::new(self.session.chat_ctx.agent_chat_ctx())));
    }

    /// What the driver keeps per speech outlives it only while something may still ask: the
    /// speech is live, a tool of it runs, or it is among the latest [`KEPT_SPEECHES`].
    fn prune_settled_speeches(&mut self) {
        let a = &mut self.activity;
        if a.speech_agents.len() + a.finished.len() + a.tools_committed.len() < 2 * KEPT_SPEECHES {
            return;
        }
        let Some(newest) = a.speech_agents.keys().next_back().copied() else { return };
        let scheduler = &a.speech_scheduler;
        let running: BTreeSet<SpeechId> = a.tools.running.iter().map(|r| r.speech).collect();
        let keep = |s: &SpeechId| {
            s.0 + KEPT_SPEECHES as u64 > newest.0 || scheduler.contains(*s) || running.contains(s)
        };
        a.speech_agents.retain(|s, _| keep(s));
        a.finished.retain(|s| keep(s));
        a.tools_committed.retain(|s| keep(s));
    }

    /// The history records an agent leaves when it takes over: `AgentHandoff`, then
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

    /// Put a handoff or config record on the record. `AgentHandoff` is history-only;
    /// `AgentConfigUpdate` goes to the agent's context and history (only the realtime view strips
    /// it).
    pub(crate) fn commit_record(&mut self, item: ChatItem, out: &mut Out) {
        let (source, visibility) = match &item {
            ChatItem::AgentHandoff(_) => (Source::Handoff, Visibility::History),
            _ => (Source::ConfigUpdate, Visibility::Both),
        };
        self.commit_item(item, source, visibility, out);
    }

    /// The one write to the record: inserted by its acoustic `created_at`, the speech told, the
    /// host told via `ItemAdded{source}`, and the activity republished.
    pub(crate) fn commit_item(
        &mut self,
        item: ChatItem,
        source: Source,
        visibility: Visibility,
        out: &mut Out,
    ) {
        // Attributed to the agent that created the speech, not the driving one (a tool may commit
        // after its agent was left); an item with no speech is the driving agent's.
        let agent = source
            .speech()
            .and_then(|s| self.activity.speech_agent(s).map(str::to_owned))
            .unwrap_or_else(|| self.activity.agent.clone());
        self.reply_recorded(&item, &source, out);
        let commit = self.session.chat_ctx.commit(item, source, visibility, &agent);
        self.perform_commit(commit, out);
    }

    /// An item on the record changed in place (an answered `InputRequired`'s output): the host
    /// is told again, under the item's own id, source and agent.
    pub(crate) fn replace_item(&mut self, item: ChatItem, out: &mut Out) {
        let Some(entry) = self.session.chat_ctx.entry(item.id()) else { return };
        let (source, visibility, agent) =
            (entry.source.clone(), entry.visibility, entry.agent.clone());
        let (entry, _) = self.session.chat_ctx.commit(item, source, visibility, &agent);
        if entry.visibility.history() {
            let run = self.run_of_item(&entry.source);
            out.machine().emit(crate::events::SessionEvent::ConversationItemAdded {
                item: Box::new(entry.item),
                agent: entry.agent,
                source: entry.source,
                visibility: entry.visibility,
                run,
            });
        }
        self.publish_activity();
    }

    /// The speech is told, the host learns of a new item, the activity is republished.
    fn perform_commit(&mut self, (entry, outcome): (Committed, CommitOutcome), out: &mut Out) {
        // A call was pushed to the handle when the model made it; everything else is new here.
        let tell_speech =
            entry.source.speech().filter(|_| !matches!(entry.source, Source::ToolCall { .. }));
        if let Some(h) = tell_speech.and_then(|id| self.activity.speech_scheduler.handle(id)) {
            h.item_added(&entry.item);
        }
        match outcome {
            // An item only in the agent's view is the model's: the host's history never sees it.
            CommitOutcome::Added if entry.visibility.history() => {
                // The person's message answers the call an `InputRequired` left open.
                if let ChatItem::Message(m) = &entry.item
                    && m.role == lk_agents_abi::Role::User
                {
                    self.answer_pending_input(&m.text_content().unwrap_or_default(), out);
                }
                let run = self.run_of_item(&entry.source);
                out.machine().emit(crate::events::SessionEvent::ConversationItemAdded {
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
