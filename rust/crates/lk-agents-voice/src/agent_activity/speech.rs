//! Speech from request to release: `say`, `generate_reply` and the turn's reply through
//! `start_speech`, the generation's reports, the agent state, and `wait_for_idle`.

use crate::agent::{Source, Visibility};
use crate::agent_activity::realtime::RealtimeReplyRole;
use crate::agent_activity::scheduling::{PreemptiveDiscard, SpeechId, SpeechKind};
use crate::agent_session::effects::{GateCause, GateOp, Timer};
use crate::agent_session::{
    Answer, ClosePhase, IdleWaiter, SayText, SpeechCall, SpeechCommand, VoiceCommandOrigin,
    VoiceDriver,
};
use crate::audio_recognition::{EndOfTurnInfo, TurnEdits};
use crate::events::SessionEvent;
use crate::generation::{EndOfSpeech, TurnOutcome as GenTurnOutcome, end_of_speech};
use crate::generation::{GenEvent, Providers, ReplyContext, ReplySource};
use crate::tool_executor::Toolset;
use crate::trace::{GenerationStep, InterruptStep, PlayoutStep, SpeechStep, ToolStep};
use lk_agents_abi::ChatItem;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{ChatContext, ChatMessage, Modality, Role, Ts};
use lk_agents_core::tts::markup::ChatMessageExt;
use std::sync::Arc;

/// A `say(stream)` the realtime model would speak itself (no TTS): the wire carries one text item.
pub(crate) const SAY_SHAPE_REFUSED_ON_REALTIME: &str =
    "say(stream) on a realtime model with no TTS: the wire takes one text item";

/// A text `say()` with no TTS while audio output is attached and enabled, refused before a handle
/// exists (`AgentActivity.say`'s first `RuntimeError`).
pub(crate) const SAY_WITHOUT_TTS_REFUSED: &str = "trying to generate speech from text without a TTS model or a RealtimeSession that supports say(); add a TTS model to AgentSession to enable say()";

pub(crate) struct SpeechSpec {
    /// Only causally participating speech extends the explicit voice run. Internal expert and
    /// filler presentation must opt out unless its parent speech belongs to that run.
    pub(crate) participates_in_voice_run: bool,
    pub(crate) new_message: Option<ChatMessage>,
    pub(crate) turn_instructions: Option<String>,
    pub(crate) source: ReplySource,
    /// Already resolved against the session default: the loop does not re-resolve options.
    pub(crate) allow_interruptions: bool,
    pub(crate) add_to_chat_ctx: bool,
    /// `generate_reply(tool_choice=)`. `None` is `NOT_GIVEN`, which is **not** `"none"`.
    pub(crate) tool_choice: Option<lk_agents_abi::ToolChoice>,
    pub(crate) chat_ctx: Option<ChatContext>,
    /// `None` is `generate_reply(user_input=)` (`Source::UserInput`).
    pub(crate) turn: Option<crate::audio_recognition::TurnId>,
    /// A hook's `UpdateInstructions`: the prompt for this one request, already rendered.
    pub(crate) instructions_override: Option<String>,
    pub(crate) toolset: Option<Arc<Toolset>>,
    /// Its methods alone decide scheduling, priority and draining-queue entry.
    pub(crate) kind: SpeechKind,
    /// Issued from inside the agent's `on_enter`, so `IGNORE_ON_ENTER` tools are withheld. Decided
    /// by the host (an ambient context var the core may not read).
    pub(crate) on_enter: bool,
    /// The host asked by name (`say`/`generate_reply`) although the kind says otherwise: a
    /// realtime generation for a host request is `user_initiated` to the host.
    pub(crate) host_asked: bool,
    /// The decided turn this speech replies to: its `EOUMetrics` go out between the speech's
    /// announcement and the user item, as the reference's `on_end_of_turn` emits them.
    pub(crate) end_of_turn: Option<EndOfTurnInfo>,
    /// The id a host's `say()` / `generate_reply()` answered with: the speech is created under it.
    pub(crate) host_speech: Option<SpeechId>,
    /// The turn's reply answers an `InputRequired(tool=)`.
    pub(crate) answering: Option<crate::tool_executor::Answering>,
}

impl SpeechSpec {
    /// Added to the context, planned against the record and the agent's prompt and tools.
    pub(crate) fn new(kind: SpeechKind, source: ReplySource, allow_interruptions: bool) -> Self {
        SpeechSpec {
            participates_in_voice_run: kind.user_initiated(),
            new_message: None,
            turn_instructions: None,
            source,
            allow_interruptions,
            add_to_chat_ctx: true,
            tool_choice: None,
            chat_ctx: None,
            turn: None,
            instructions_override: None,
            toolset: None,
            kind,
            on_enter: false,
            host_asked: false,
            end_of_turn: None,
            host_speech: None,
            answering: None,
        }
    }
}

impl super::AgentActivity {
    fn speech_was_released(&self, speech: SpeechId) -> bool {
        self.speech_history.finished(speech) && !self.speech_scheduler.contains(speech)
    }

    fn reply_predecessor_settled(&self, speech: SpeechId) -> bool {
        self.speech_history.settled(speech)
            || self.speech_scheduler.speech(speech).is_none_or(|handle| handle.done())
    }
}

impl VoiceDriver {
    pub(crate) fn on_speech(&mut self, now: Ts, cmd: SpeechCommand) {
        // The host's speech during the drain is the arriving agent's (`_next_activity`); the
        // leaving agent's own tools and hooks are refused by the paused scheduler, as before.
        if matches!(cmd, SpeechCommand::Say { .. } | SpeechCommand::GenerateReply { .. })
            && self.command_origin == VoiceCommandOrigin::Host
            && self.session.closing.is_none()
            && self.activity.handoff.draining()
        {
            return self.activity.handoff.held_speech.push(cmd);
        }
        match cmd {
            // A realtime session answers on its own socket. These arms must come *before* the
            // pipeline arms: a guarded arm after its unguarded twin is dead code.
            SpeechCommand::GenerateReply { chat_ctx, toolset, tool_names, call, .. }
                if self.activity.rt_session.is_some()
                    && (chat_ctx.is_some() || toolset.is_some() || tool_names.is_some()) =>
            {
                self.refuse_realtime_reply_overrides(call)
            }
            SpeechCommand::GenerateReply {
                user_input,
                instructions,
                allow_interruptions,
                tool_choice,
                call,
                ..
            } if self.activity.rt_session.is_some() => {
                self.realtime_reply(
                    now,
                    user_input,
                    instructions,
                    allow_interruptions,
                    tool_choice,
                    call,
                    RealtimeReplyRole::VoiceRun,
                );
            }
            // The model speaks a `say` only with no TTS to (`AgentActivity.say`); with one, a
            // realtime session's `say` is synthesised like a pipeline's, audio and streams too.
            SpeechCommand::Say {
                text: SayText::Text(text),
                allow_interruptions,
                add_to_chat_ctx,
                call,
            } if self.realtime_says() => {
                self.realtime_say(now, text, allow_interruptions, add_to_chat_ctx, call)
            }
            SpeechCommand::Say { text: SayText::Stream, call, .. } if self.realtime_says() => {
                self.refuse_realtime_say_stream(call)
            }
            SpeechCommand::CommitUserTurn {
                transcript_timeout,
                stt_flush_duration,
                skip_reply,
                call,
            } => {
                self.commit_user_turn(
                    now,
                    transcript_timeout,
                    stt_flush_duration,
                    skip_reply,
                    call,
                );
            }
            SpeechCommand::WaitForIdle { hold, call } => {
                let speech = self.tool_speech(&self.command_origin);
                self.session.idle.ask(IdleWaiter { hold, speech, grant: None, call, run: None });
            }
            SpeechCommand::WaitForIdleAndHold { call } => self.wait_for_idle_and_hold(call),
            SpeechCommand::ReleaseIdleHold { hold } => {
                if self.session.idle.release(hold) {
                    self.record_trace(SpeechStep::IdleHoldReleased { hold });
                }
            }
            SpeechCommand::EndCall { speech } => self.end_call(speech),
            SpeechCommand::ClaimUserTurn => self.claim_user_turn(now),
            SpeechCommand::ReleaseUserTurn => self.release_user_turn(now),
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call } => {
                self.say(now, text, allow_interruptions, add_to_chat_ctx, call)
            }
            SpeechCommand::GenerateReply {
                user_input,
                instructions,
                allow_interruptions,
                tool_choice,
                chat_ctx,
                toolset,
                tool_names,
                on_enter,
                call,
            } => {
                let toolset =
                    toolset.or_else(|| tool_names.map(|names| self.activity.reply_toolset(&names)));
                self.generate_reply_speech(
                    now,
                    user_input,
                    instructions,
                    allow_interruptions,
                    tool_choice,
                    chat_ctx,
                    toolset,
                    on_enter,
                    call.as_ref().map(|c| c.speech),
                );
                self.answer_speech(call);
            }
            // The AMD system's authorization hold: the scheduling task keeps its head.
            SpeechCommand::PauseAuthorization => {
                self.record_trace(SpeechStep::AuthorizationPaused);
                self.activity.speech_scheduler.authorization_paused = true;
            }
            // Its release: the head is authorized in this same cycle.
            SpeechCommand::ResumeAuthorization => {
                self.record_trace(SpeechStep::AuthorizationResumed);
                self.activity.speech_scheduler.authorization_paused = false;
            }
            SpeechCommand::Interrupt { force, call } => self.host_interrupt_all(now, force, call),
            SpeechCommand::InterruptSpeech { speech, force } => {
                self.host_interrupt_speech(now, speech, force)
            }
            SpeechCommand::HoldInterruptions { speech, hold } => {
                self.hold_interruptions(speech, hold)
            }
            SpeechCommand::SetAllowInterruptions { speech, allow } => {
                self.set_allow_interruptions(speech, allow)
            }
            // A background tool's later update or return: the pair, the event, the coalesced reply.
            SpeechCommand::ToolUpdate { invocation_id, operation, call_id, update } => {
                self.on_tool_update(now, invocation_id, operation, call_id, update)
            }
            SpeechCommand::FillerStart {
                id,
                call_id,
                speech,
                delay,
                interval,
                max_steps,
                source,
            } => self.on_filler_start(now, id, call_id, speech, delay, interval, max_steps, source),
            SpeechCommand::FillerStop { id } => {
                self.end_filler(id, crate::trace::FillerEnd::Stopped)
            }
        }
    }

    fn say(
        &mut self,
        now: Ts,
        text: SayText,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        call: Option<SpeechCall>,
    ) {
        // `say(text)` with no TTS and an enabled audio output: refused before a handle
        // exists. `say(audio=)` is exempt; a detached or disabled output speaks as text.
        if !matches!(text, SayText::Audio { .. })
            && self.session.tts.is_none()
            && self.session.outputs.plays_audio()
        {
            self.record_trace(SpeechStep::SayRefused { why: SAY_WITHOUT_TTS_REFUSED });
            self.fail_waiter(call, SpeechKind::Say, SAY_WITHOUT_TTS_REFUSED);
            return;
        }
        let interruptible = self.session.config.effective.turn.interruption.enabled();
        let source = match text {
            SayText::Text(text) => ReplySource::Text(text),
            SayText::Stream => ReplySource::TextStream,
            SayText::Audio { text } => ReplySource::Audio { text },
        };
        let allow_interruptions = allow_interruptions.unwrap_or(interruptible);
        self.start_speech(
            now,
            SpeechSpec {
                add_to_chat_ctx,
                host_speech: call.as_ref().map(|c| c.speech),
                ..SpeechSpec::new(SpeechKind::Say, source, allow_interruptions)
            },
        );
        self.answer_speech(call);
    }

    /// Shared with the run's own replies.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_reply_speech(
        &mut self,
        now: Ts,
        user_input: Option<String>,
        instructions: Option<String>,
        allow_interruptions: Option<bool>,
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        chat_ctx: Option<ChatContext>,
        toolset: Option<Arc<Toolset>>,
        on_enter: bool,
        host_speech: Option<SpeechId>,
    ) -> SpeechId {
        let wall = self.step.wall();
        let new_message = user_input
            .map(|text| ChatMessage::text(self.session.ids.mint("item"), Role::User, &text, wall));
        let interruptible = self.session.config.effective.turn.interruption.enabled();
        self.start_speech(
            now,
            SpeechSpec {
                new_message,
                turn_instructions: instructions,
                on_enter,
                tool_choice,
                chat_ctx,
                toolset,
                host_speech,
                ..SpeechSpec::new(
                    SpeechKind::GenerateReply,
                    ReplySource::Llm,
                    allow_interruptions.unwrap_or(interruptible),
                )
            },
        )
    }

    /// The reply to a decided user turn (`on_end_of_turn`'s `_generate_reply`). A preemptive
    /// generation for another turn is discarded; adopted, the user item enters the conversation
    /// stamped at the turn's onset.
    pub(crate) fn reply_to_turn(&mut self, now: Ts, info: &EndOfTurnInfo, edits: TurnEdits) {
        self.activity.last_turn = Some(info.turn_id);
        // A realtime session answers on its own socket, from the audio it heard: the buffer is
        // committed when the server did not segment and the reply is asked for. The STT's words
        // are neither sent nor kept ("ignore stt transcription for realtime model"); the server's
        // own transcript fills the user item.
        if self.activity.rt_session.is_some() {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned);
            // The commit of the server's buffer and the ask, as one request; its speech is
            // announced now, so the metrics name it.
            let speech = self.realtime_user_turn(now);
            self.emit_eou_metrics(info, speech);
            return;
        }
        // A preemptive generation for another turn, against a record since changed, or whose
        // request the hook edited, answers a question this turn is not asking.
        let edited =
            edits.instructions.is_some() || edits.chat_ctx.is_some() || edits.user_text.is_some();
        let stale = self.activity.preemptive_record_version != self.session.chat_ctx.version();
        if let Some(preemptive_turn) = self.activity.preemptive_turn.take()
            && (preemptive_turn != info.turn_id || edited || stale)
            && self.activity.speech_scheduler.preemptive_generation().is_some()
        {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned);
        }
        // The person answers the expert question that owns their reply: forwarded, not replied to.
        if self.forward_expert_reply(now, info, edits.user_text.as_deref()) {
            return;
        }
        // A tool claimed this turn (`InputRequired(tool=)`): the reply calls it, a different
        // request from any guessed one.
        let claimed = self.activity.consume_claim();
        if claimed.is_some() && self.activity.speech_scheduler.preemptive_generation().is_some() {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned);
        }
        let adopted = self.adopt_preemptive_generation(now, &info.new_transcript);
        let TurnEdits { instructions, chat_ctx, user_text } = edits;
        let user = self.user_turn_item(info, user_text.as_deref());
        let answering = claimed.map(|tool| crate::tool_executor::Answering {
            tool,
            said: user.text_content().unwrap_or_default(),
        });
        if let Some(preemptive) = adopted {
            if let Some(state) = self.activity.speech_scheduler.speech_mut(preemptive) {
                state.set_user_metrics(user.metrics.clone());
            }
            self.commit_item(
                ChatItem::Message(user),
                Source::UserTurn { turn: info.turn_id },
                Visibility::Both,
            );
            self.set_agent_state(self.step.now, AgentState::Thinking);
            // The adopted preemptive generation is the turn's reply: the record names it.
            self.emit_eou_metrics(info, Some(preemptive));
            return;
        }
        self.start_speech(
            now,
            SpeechSpec {
                new_message: Some(user),
                chat_ctx,
                turn: Some(info.turn_id),
                instructions_override: instructions,
                end_of_turn: Some(info.clone()),
                tool_choice: answering
                    .as_ref()
                    .map(|a| lk_agents_abi::ToolChoice::Named(a.tool.clone())),
                answering,
                ..SpeechSpec::new(
                    SpeechKind::TurnReply,
                    ReplySource::Llm,
                    self.session.config.effective.turn.interruption.enabled(),
                )
            },
        );
    }

    /// Emitted once the reply speech exists so it carries its id; never for a turn with no reply.
    /// `metadata` names the detector (model, or mode with `model_name="unknown"`).
    pub(crate) fn emit_eou_metrics(&mut self, info: &EndOfTurnInfo, speech: Option<SpeechId>) {
        use crate::agent_activity::options::TurnDetection;
        let detection = self.session.config.effective.turn.detection();
        let metadata = match (self.activity.audio_recognition.eot_provider(), detection) {
            (Some(eot), _) => Some(eot.metadata()),
            (None, TurnDetection::Vad) => Some(mode_metadata("vad")),
            (None, TurnDetection::Stt) => Some(mode_metadata("stt")),
            (None, TurnDetection::Manual) => Some(mode_metadata("manual")),
            (None, TurnDetection::RealtimeLlm) => Some(mode_metadata("realtime_llm")),
            (None, TurnDetection::Auto) => None,
        };
        let secs = |d: Option<std::time::Duration>| d.map(|d| d.as_secs_f64());
        let record = lk_agents_abi::metrics::base::EouMetrics {
            timestamp: self.step.wall_secs(),
            end_of_utterance_delay: secs(info.metrics.end_of_turn_delay),
            transcription_delay: secs(info.metrics.transcription_delay),
            on_user_turn_completed_delay: secs(info.metrics.on_user_turn_completed_delay),
            speech_id: speech.map(|s| s.to_string()),
            metadata,
        };
        self.publish_event(SessionEvent::MetricsCollected {
            metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::Eou(record)),
            speech,
            counted: false,
        });
    }

    /// Stamped at the turn's onset when known.
    pub(crate) fn user_turn_item(
        &mut self,
        info: &EndOfTurnInfo,
        text: Option<&str>,
    ) -> ChatMessage {
        let created_at = match info.metrics.started_speaking_at {
            Some(at) => self.step.wall_at(at),
            None => self.step.wall(),
        };
        let text = text.unwrap_or(&info.new_transcript);
        let mut user =
            ChatMessage::text(self.session.ids.mint("item"), Role::User, text, created_at);
        user.transcript_confidence = Some(info.transcript_confidence);
        // `_init_metrics_from_end_of_turn`: the reply's `e2e_latency` is measured from here.
        let m = &info.metrics;
        let secs = |d: Option<std::time::Duration>| d.map(|d| d.as_secs_f64());
        user.metrics = lk_agents_abi::MetricsReport {
            started_speaking_at: m.started_speaking_at.map(|at| self.step.wall_secs_at(at)),
            stopped_speaking_at: m.stopped_speaking_at.map(|at| self.step.wall_secs_at(at)),
            transcription_delay: secs(m.transcription_delay),
            end_of_turn_delay: secs(m.end_of_turn_delay),
            on_user_turn_completed_delay: secs(m.on_user_turn_completed_delay),
            stt_metadata: self.stt_identity().map(|id| id.metrics_metadata()),
            ..Default::default()
        };
        user
    }

    /// Through the one write, in order with `thinking`.
    /// The single admission point for every speech. The runtime launches accepted work after
    /// this step settles.
    pub(crate) fn start_speech(&mut self, now: Ts, spec: SpeechSpec) -> SpeechId {
        let wall = self.step.wall();
        let user_metrics = spec.new_message.as_ref().map(|m| m.metrics.clone());
        let is_reply = matches!(spec.source, ReplySource::Llm);
        let kind = spec.kind;

        // `chat_ctx=` and `ReplaceItems` replace what this request is planned against; the record
        // is not written.
        let record =
            spec.chat_ctx.clone().unwrap_or_else(|| self.session.chat_ctx.agent_chat_ctx());
        // The agent's prompt for the modality, or a hook's edit for this turn.
        let instructions = spec
            .instructions_override
            .clone()
            .unwrap_or_else(|| self.activity.instructions.render(Some(Modality::Audio)));
        // Read once, so the request's specs and host objects cannot disagree.
        let advertised = self.activity.advertised_toolset(&self.session.tools);
        let turn_toolset = spec.toolset.clone().unwrap_or_else(|| Arc::clone(&advertised));
        let expressive = match &spec.source {
            ReplySource::Realtime => None,
            source => self.session.expressive_guide(matches!(source, ReplySource::Llm)),
        };
        let mut plan = crate::generation::PlanSpec {
            record: spec.chat_ctx.clone(),
            new_message: spec.new_message.clone(),
            instructions: spec.instructions_override.clone(),
            turn_instructions: spec.turn_instructions.clone(),
            // Snapshotted for this request: handoffs replace the list, replies overlap.
            tools: turn_toolset.specs.clone(),
            on_enter: spec.on_enter,
            running_tool_calls: self.activity.tools.running_tool_calls(),
            audio_output_enabled: self.session.outputs.plays_audio(),
            text_output_enabled: self.session.outputs.shows_transcription(),
            expressive,
        };

        // A reply created while a speech is being cut is requested once that speech's item is on
        // the record (the cut settles within the interruption watchdog). One created while a
        // speech plays is requested now, on the record as it stands, as the reference's reply
        // task copies the context at creation: the playing speech's item may be seconds away.
        let after = self
            .activity
            .current_speech()
            .filter(|(_, s)| !s.done() && s.interrupted())
            .map(|(id, _)| id);

        let (id, state) = self.new_speech(spec.host_speech, spec.allow_interruptions);

        self.activity.note_speech_agent(id, kind);

        // A model reply runs as one `AgentRun`, which names its first step's message and adds
        // the tools its output takes.
        let realtime = matches!(spec.source, ReplySource::Realtime);
        let opened = if is_reply || realtime {
            let budgets = self.activity.retries();
            let invocation_id = id.to_string();
            let kind = (realtime, self.activity.voice_tasks.running());
            self.open_reply_run(id, invocation_id, &turn_toolset, budgets, &plan, kind)
        } else {
            None
        };
        let message_id = match opened {
            Some((message_id, tools)) => {
                plan.tools = tools;
                message_id
            }
            None => self.session.ids.mint("item"),
        };
        let request = match &spec.source {
            ReplySource::Llm
            | ReplySource::Text(_)
            | ReplySource::TextStream
            | ReplySource::Audio { .. } => plan.plan(&record, &instructions, wall),
            // No plan: the server decides what to say. The step needs the record and tools; there
            // is no TTS leg, but the words still go on the text stream.
            ReplySource::Realtime => crate::generation::Request {
                chat_ctx: record,
                tools: plan.tools.clone(),
                tts_enabled: false,
                text_output_enabled: self.session.outputs.shows_transcription(),
                instructions_hash: crate::generation::stable_hash(&instructions),
                expressive: None,
            },
        };

        let replan = match (after, &spec.source) {
            (Some(after), ReplySource::Llm) => {
                Some(crate::generation::Replan { after, spec: plan })
            }
            (
                Some(_),
                ReplySource::Text(_)
                | ReplySource::TextStream
                | ReplySource::Audio { .. }
                | ReplySource::Realtime,
            )
            | (None, _) => None,
        };

        let ctx = ReplyContext {
            preemptive_tts: self.session.config.effective.turn.preemptive.enabled()
                && self.session.config.effective.turn.preemptive.preemptive_tts(),
            request,
            replan,
            source: spec.source,
            tool_choice: spec.tool_choice,
            toolset: spec.toolset,
            providers: Providers {
                dynamic_instructions: self.activity.dynamic_instructions,
                toolset: advertised,
                tools_host_extra: self.session.tool_hosts.clone(),
                output_nodes: self.activity.output_nodes.clone(),
            },
            user_metrics,
            message_id,
            speech: self.session.config.effective.speech,
            answering: spec.answering,
            task: self.activity.voice_tasks.running(),
            step: 1,
        };

        // Past the close's hard edge no generation is started.
        let past_edge = self.session.closing_past_edge();
        // Announced, recorded and scheduled in `request`, the one place every speech passes.
        let admitted = self.schedule_speech(
            now,
            id,
            state,
            kind,
            spec.add_to_chat_ctx,
            spec.host_asked,
            spec.participates_in_voice_run,
        );
        if admitted
            && kind.scheduled()
            && let Some(state) = self.activity.speech_scheduler.speech_mut(id)
        {
            state.mark_scheduled();
        }
        if admitted && !past_edge {
            self.storage_started(id);
            let reply = crate::telemetry::Reply {
                agent: self.activity.agent_name().to_owned(),
                user: ctx.user_metrics.clone(),
                realtime,
                instructions: spec.turn_instructions.clone().filter(|_| is_reply),
                user_input: spec.new_message.as_ref().and_then(|m| m.raw_text_content()),
                llm: (self.session.llm.as_ref().map(|l| &l.provider))
                    .or_else(|| self.activity.realtime().map(|r| &r.provider))
                    .cloned(),
            };
            let span = self.agent_turn_started(id, reply);
            if kind.scheduled()
                && let Some(turn) = self.activity.speech_scheduler.agent_turn(id)
            {
                turn.scheduled(now);
            }
            self.record_effect(crate::agent_session::effects::VoiceEffect::StartSpeech {
                speech: id,
                context: Box::new(ctx),
                llm: self.session.llm.as_ref().map(|l| l.id),
                tts: self.session.tts.as_ref().map(|t| t.id),
                stored: self.session.storage.enabled,
                span,
            });
        }
        if !admitted && self.activity.speech_scheduler.contains(id) {
            self.release_speech(now, id);
        }
        if let Some(info) = &spec.end_of_turn {
            self.emit_eou_metrics(info, Some(id));
        }
        // The reference's reply task adds the user item once its speech is scheduled: after the
        // announcement, never for a refused speech. A preemptive generation plans against the
        // message but does not commit it; `generate_reply` commits on adoption.
        if let (Some(user), true) = (spec.new_message, admitted && kind.scheduled()) {
            let source = match spec.turn {
                Some(turn) => Source::UserTurn { turn },
                None => Source::UserInput { speech: id },
            };
            self.commit_item(ChatItem::Message(user), source, Visibility::Both);
        }
        // `thinking` when the generation exists, only for an admitted reply (`say()` goes straight
        // to `speaking`; a refused or discarded preemptive generation never showed thinking).
        if admitted && is_reply && kind.scheduled() && !past_edge {
            self.set_agent_state(self.step.now, AgentState::Thinking);
        }
        id
    }

    pub(crate) fn on_generation(&mut self, now: Ts, ev: GenEvent) {
        match ev {
            GenEvent::OutputBinding(event) => self.on_output_binding(event),
            GenEvent::Plan { call, plan } => self.activity.reply_plans.push_back((call, *plan)),
            // A text-only turn's first frame; with audio output the edge is `PlaybackStarted`'s.
            GenEvent::FirstFrame { speech } => self.agent_started_speaking(now, speech),

            GenEvent::Done { speech, outcome, interrupted } => {
                if self.session.storage.enabled {
                    self.storage_settle(speech, *outcome, interrupted);
                } else {
                    self.activity.set_tools_in_flight(speech, false);
                    self.finish_reply(now, speech, *outcome, interrupted);
                }
            }

            // One step's words are done, before its tools run; the reply's run reads them once
            // they are on the record.
            GenEvent::StepSpoken {
                speech,
                step,
                interrupted,
                has_tool_output,
                tools_running,
                item,
                cut_at,
            } => {
                if let Some(position) = cut_at
                    && let Some(turn) = self.agent_turn(speech)
                {
                    turn.played(position, true);
                }
                self.activity.set_tools_in_flight(speech, tools_running);
                let text = item.as_ref().and_then(|m| m.raw_text_content());
                self.step_spoken(
                    now,
                    speech,
                    step,
                    interrupted,
                    has_tool_output,
                    tools_running,
                    item,
                );
                self.reply_spoken(speech, text);
            }
            GenEvent::Trace(step) => self.record_trace(step),
            GenEvent::ProviderFailed { speech, leg, recoverable, error } => {
                let source = match leg {
                    crate::trace::ProviderKind::Llm => crate::events::ErrorSource::Llm,
                    crate::trace::ProviderKind::Tts => crate::events::ErrorSource::Tts,
                };
                self.report_error(now, source, recoverable, *error, Some(speech));
            }
            GenEvent::KeytermAnswer { req, arguments, failed, metrics } => {
                self.on_keyterm_answer(now, req, arguments, failed, metrics);
            }

            // Reported by a task; said by the loop, which is the only place with the bus.
            GenEvent::Emit(ev) => self.publish_event(*ev),

            GenEvent::ToolsFinished { speech, step, end, answer } => {
                self.on_tools_finished(now, (speech, step), *end, answer);
            }

            // The reply's tools are done and it has a next step: behind whatever was queued.
            GenEvent::ToolStep { speech, step } => {
                // `speech_handle._num_steps += 1`: the reply counts its own steps and reports each.
                if let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
                    let advanced = state.advance_step();
                    debug_assert_eq!(advanced, step, "the reply and the core disagree on its step");
                }
                self.record_trace(ToolStep::StepStarted { speech, step });
                if let Some(turn) = self.activity.agent_turn(speech) {
                    turn.step(&mut self.telemetry, step);
                }
                self.requeue_speech(now, speech);
            }

            GenEvent::Failed { speech, error } => {
                if self.session.storage.enabled {
                    self.storage_abandon(speech, error);
                } else {
                    self.generation_failed(now, speech, error);
                }
            }

            GenEvent::PlaybackFinished { speech, position, interrupted } => {
                if let Some(turn) = self.agent_turn(speech) {
                    turn.played(position, interrupted);
                }
            }
            GenEvent::PlaybackStarted { speech } => {
                // The launch hold is over for this speech: its first frame is out (I-06, I-07).
                self.activity.launching.remove(&speech);
                self.record_trace(PlayoutStep::PlaybackStarted { speech });
                // `_on_first_frame` listens on the sink's `playback_started`: the edge is the sink's
                // report, not the words half's (which waits for its segment).
                self.agent_started_speaking(now, speech);
            }
            GenEvent::LaunchReleased { speech } => {
                // Released because the user stopped, or cancelled because the speech became
                // uninterruptible while parked.
                let cause = match self
                    .activity
                    .speech_scheduler
                    .speech(speech)
                    .map(|s| s.allow_interruptions())
                {
                    Some(true) | None => GateCause::Launch,
                    Some(false) => GateCause::Cancelled,
                };
                self.record_trace(InterruptStep::PlayoutGate { speech, op: GateOp::Resume, cause })
            }
            GenEvent::RequestStarted { speech, kind, provider, model } => {
                let request = self.session.start_request(kind, &provider, &model);
                self.session.requests.entry((speech, kind)).or_default().push_back(request);
            }
            GenEvent::LlmRequestFinished { speech, step, error } => {
                let failed = error.is_some();
                self.record_trace(GenerationStep::LlmRequestFinished { speech, step, error });
                // A failed preemptive generation answers nothing; the commit starts a fresh reply.
                if failed && self.activity.speech_scheduler.preemptive_generation() == Some(speech)
                {
                    self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned);
                }
            }
            GenEvent::ToolCallMade { speech, step, made, go } => {
                self.on_tool_made(now, (speech, step), *made, go);
            }
            GenEvent::ToolPrepared { speech, call, identity, go } => {
                self.on_tool_prepared(speech, call, identity, go);
            }
            GenEvent::ReadBackRendered { speech, call, answer, go } => {
                self.on_read_back(speech, call, answer, go);
            }
            GenEvent::ToolCallEnded { speech, call, call_id, end, message, outcome } => {
                self.on_tool_ended(now, (speech, call), call_id, end, message, outcome);
            }
            GenEvent::OutputValidated { speech, validation, verdict, value } => {
                self.on_output_validated(speech, validation, verdict, value);
            }
            GenEvent::ToolHandlingTemplateRendered { rendering, text } => {
                self.on_template_rendered(now, rendering, text);
            }
        }
    }

    /// Answers each waiting [`ReplyPlan`](crate::generation::ReplyPlan) whose record is ready, or
    /// with `None` once its speech is gone or the close passed its edge. `true` when one was.
    pub(crate) fn settle_reply_plans(&mut self) -> bool {
        let mut ready = false;
        let mut waiting = std::collections::VecDeque::new();
        while let Some((call, plan)) = self.activity.reply_plans.pop_front() {
            let cancelled = self.session.closing_past_edge()
                || self
                    .activity
                    .speech_scheduler
                    .speech(plan.speech)
                    .is_none_or(|s| s.interrupted() || s.done());
            let committed = self.activity.reply_predecessor_settled(plan.after);
            if !cancelled && !committed {
                waiting.push_back((call, plan));
                continue;
            }
            ready = true;
            let reply = if cancelled {
                None
            } else {
                let instructions = self.activity.instructions.render(Some(Modality::Audio));
                let record = self.session.chat_ctx.agent_chat_ctx();
                let request = plan.spec.plan(&record, &instructions, self.step.wall());
                Some(Box::new(crate::generation::PlannedReply {
                    request,
                    tool_choice: plan.tool_choice,
                }))
            };
            self.answer(Some(call), Answer::ReplyPlan(reply));
        }
        self.activity.reply_plans = waiting;
        ready
    }

    /// `_on_first_frame`: `speaking` and `_on_start_of_agent_speech` (which enables interruption) at
    /// the same instant, once per step.
    fn agent_started_speaking(&mut self, now: Ts, speech: SpeechId) {
        if !self.activity.spoke.insert(speech) {
            return;
        }
        let wall = self.step.wall_secs_at(now);
        self.agent_turn_speaking(speech, wall);
        // The labels arm the AEC warmup at the session's first `speaking` edge with audio output.
        self.set_agent_state(now, AgentState::Speaking);
        let eff = self.activity.audio_recognition.on_start_of_agent_speech(now, now);
        self.apply_recognition_effects(eff);
        self.record_trace(SpeechStep::AgentStartedSpeaking { speech });
    }

    pub(crate) fn generation_failed(
        &mut self,
        now: Ts,
        speech: SpeechId,
        error: Box<lk_agents_abi::errors::ProviderError>,
    ) {
        self.activity.set_tools_in_flight(speech, false);
        // Released by the close's edge already: the edge was its end (`CloseAct::Released`).
        if self.activity.speech_was_released(speech) {
            return;
        }
        self.record_trace(SpeechStep::ReplyFailed {
            speech,
            class: crate::trace::FailureClass::of(&error),
        });
        if let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
            // Moved, not formatted: the binding releases a `ProviderError::Host`'s entry.
            state.mark_done(Some(*error));
        }
        self.release_speech(now, speech);
    }

    /// Releases the current speech and moves the state (`thinking` while tools run, else
    /// `listening`). `add_to_chat_ctx=False` keeps the item out of the conversation only.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_spoken(
        &mut self,
        now: Ts,
        speech: SpeechId,
        step: u32,
        interrupted: bool,
        has_tool_output: bool,
        tools_running: bool,
        item: Option<Box<ChatMessage>>,
    ) {
        if let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
            let _ = state.mark_generation_done();
        }
        self.record_trace(SpeechStep::StepSpoken { speech, step, interrupted });
        // The step's edge is spent: the next step's first frame is a new one.
        self.activity.spoke.remove(&speech);
        // Read off the record kept past the release: a `say` cut by the close's edge reports its
        // step after the edge released it, and is still a `say`'s item.
        let kind =
            self.activity.speech_kind(speech).or(self.activity.speech_scheduler.kind(speech));
        let mut item = item.map(|msg| {
            let source = match kind {
                Some(SpeechKind::Say) => Source::Say { speech },
                Some(
                    SpeechKind::TurnReply
                    | SpeechKind::GenerateReply
                    | SpeechKind::Preemptive
                    | SpeechKind::Realtime,
                )
                | None => Source::Reply { speech, step },
            };
            (ChatItem::Message(*msg), source)
        });
        // The reference's realtime generation moves the agent's state before it adds the
        // messages; the pipeline adds its message first.
        let realtime = kind == Some(SpeechKind::Realtime);
        if !realtime {
            self.commit_spoken(speech, item.take());
        }
        self.activity.speech_scheduler.spoken(speech, now);
        // A blocked current speech's step ending is what a pause from inside its tool waits on.
        self.settle_scheduling_pause();
        // The state moves on the speech's interruption, not the step's; a filler ending inside
        // another speech's tool span goes back to `thinking` (listening would arm the away timer).
        let speech_interrupted =
            self.activity.speech_scheduler.speech(speech).is_some_and(|s| s.interrupted());
        let eos = end_of_speech(
            speech_interrupted,
            has_tool_output,
            tools_running || self.tools_running_elsewhere(speech),
            self.session.states.agent_state() == AgentState::Speaking,
            self.session.config.effective.turn.interruption.detection_enabled(),
        );
        self.apply_end_of_speech(now, eos);
        self.commit_spoken(speech, item);
    }

    /// A step's words: into the record, or only told to the handle under `add_to_chat_ctx=False`.
    fn commit_spoken(&mut self, speech: SpeechId, item: Option<(ChatItem, Source)>) {
        let Some((item, source)) = item else { return };
        if self.activity.speech_scheduler.add_to_chat_ctx(speech) {
            self.commit_item(item, source, Visibility::Both);
        } else if let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
            state.item_added(&item);
        }
    }

    fn apply_end_of_speech(&mut self, now: Ts, eos: EndOfSpeech) {
        if let Some(state) = eos.state {
            self.set_agent_state(now, state);
        }
        if eos.end_of_agent_speech {
            let eff = self.activity.audio_recognition.on_end_of_agent_speech(now, now);
            self.apply_recognition_effects(eff);
        }
        if eos.restore_interruption_by_audio_activity {
            self.session.by_audio_activity = true;
        }
    }

    pub(crate) fn finish_reply(
        &mut self,
        now: Ts,
        speech: SpeechId,
        outcome: GenTurnOutcome,
        interrupted: bool,
    ) {
        // Already released by the close's hard edge: a second terminal would be a second end.
        if self.activity.speech_was_released(speech) {
            return;
        }
        self.record_trace(SpeechStep::ReplyFinished {
            speech,
            text: outcome.forwarded_text.clone(),
            interrupted,
        });
        if let Some(state) = self.activity.speech_scheduler.speech(speech) {
            self.record_trace(SpeechStep::ItemsRecorded { speech, ids: state.item_ids().to_vec() });
        }
        // **`speaking` belongs to the current speech**: releasing a refused `Say` or a discarded
        // preemptive generation must not drop the agent out of `speaking` while another speech
        // plays.
        let was_current = self.activity.speech_scheduler.current_speech() == Some(speech);
        let eos = end_of_speech(
            interrupted,
            false,
            false,
            was_current && self.session.states.agent_state() == AgentState::Speaking,
            self.session.config.effective.turn.interruption.detection_enabled(),
        );
        self.apply_end_of_speech(now, eos);

        self.release_speech(now, speech);
    }

    pub(crate) fn release_speech(&mut self, now: Ts, speech: SpeechId) {
        let (requests, usage) = (&mut self.session.requests, &mut self.session.usage);
        requests.retain(|(of, _), unmeasured| {
            if *of == speech {
                unmeasured.drain(..).for_each(|request| usage.finish(request, false));
            }
            *of != speech
        });
        self.activity.spoke.remove(&speech);
        self.activity.launching.remove(&speech);
        // `_scheduling_task`'s post-await release: a finished speech that is the paused one drops
        // the pause and resumes the sink, current or not (the floor left it at its last step).
        // The release at `Authorized` can miss a pause taken in the same instant, which would
        // leave the sink paused until the close.
        self.release_paused_speech(speech);
        for (request, answer) in self.activity.reply_released(speech) {
            self.answer(Some(request), answer);
        }
        self.activity.tools.dropped.retain(|(s, _), _| *s != speech);
        // A close ending a speech with blocking tools still running: they are answered with the
        // closing error before the speech's own `SpeechDone`; nothing is journalled after `close`.
        // Their bodies are cancelled, as the reference's close cancels its tool tasks
        // (`_ToolExecutor.aclose`): answered already, nothing would ever hear their end.
        if self.session.closing.is_some()
            && self.activity.speech_scheduler.contains(speech)
            && self.activity.tools.live(speech).next().is_some()
            && self.session.close_phase() != Some(ClosePhase::Closed)
        {
            self.end_live_tools(now, speech);
            if self.activity.speech_scheduler.contains(speech) {
                self.record_effect(crate::agent_session::effects::VoiceEffect::CancelSpeech(
                    speech,
                ));
            }
        }
        if let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
            // The host learns of the end only here, with the speech already `done()` (idempotent).
            // A host `interrupt()` queued after the input that ended it arrives too late.
            state.mark_done(None);
            let view = state.view();
            let (outcome, failed) = (view.outcome(), view.error.is_some());
            let error = view.error.clone();
            self.agent_turn_done(speech, outcome, error);
            self.publish_event(SessionEvent::SpeechDone { speech, outcome });
            // The run that owned it lets it go, and its verdict moves to this speech.
            self.run_speech_ended(speech, failed);
        }
        // The deferred tool reply's fate is read off the handle; the tools' fillers go with it.
        self.tool_reply_done(speech);
        self.fillers_speech_ended(speech);
        // The handle leaves with the queue's entry (`speech_finished`, below); `speech_frames`
        // keeps its entry: a call and output can land after the speech is done.
        self.activity.speech_history.finish(speech);
        self.storage_speech_released(now, speech);
        self.publish_activity();
        self.speech_finished(now, speech);
        self.voice_task_speech_released(now, speech);
        // A realtime session swapped meanwhile gets its frame once what it cut is released.
        self.realtime_replay(now, Some(speech));
    }

    /// `_update_agent_state(next)`: the one door for agent-state writes, reading the audio-output
    /// fact the warmup edge gates on at the instant of the edge.
    pub(crate) fn set_agent_state(&mut self, now: Ts, next: AgentState) -> Option<AgentState> {
        let audio_output_ready = self
            .activity
            .speech_scheduler
            .current_speech()
            .map(|speech| self.speech_outputs(speech))
            .unwrap_or(&self.session.outputs)
            .plays_audio();
        let (old, deadline) = self.session.update_agent_state(now, next, audio_output_ready);
        if let Some(old) = old {
            self.agent_state_span_event(old, next);
            self.publish_event(SessionEvent::AgentStateChanged { old, new: next });
        }
        if let Some(at) = deadline {
            self.set_deadline(Timer::AecWarmup, at);
        }
        old
    }
}

/// `Metadata(model_name="unknown", model_provider=<the turn_detection string>)`.
fn mode_metadata(mode: &str) -> lk_agents_abi::metrics::base::Metadata {
    lk_agents_abi::metrics::base::Metadata {
        model_name: Some("unknown".to_owned()),
        model_provider: Some(mode.to_owned()),
    }
}
