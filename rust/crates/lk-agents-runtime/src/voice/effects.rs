use super::legs::PendingLeg;
use super::{HostCallbackId, VoiceRuntime};
use crate::telemetry;
use crate::voice::events::{Opened, StreamOpened};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::resources::ResourceId;
use lk_agents_voice::agent_session::effects::{SpeechHistory, StreamKind, VoiceEffect};
use lk_agents_voice::agent_session::machine::activity::realtime::RealtimeWake;
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::{SystemTap, VoiceEvent, VoiceHostEvent};
use lk_agents_voice::audio::AudioChannel;
use lk_agents_voice::audio_recognition::InterruptionSignal;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::{GenEvent, ReplySource};
use lk_agents_voice::tool_executor::{HostedTool, Rendering};
use std::sync::Arc;

#[cfg(test)]
#[path = "effects_tests.rs"]
mod tests;

/// The overlap request a detector is told of when the core named none.
const UNNAMED_OVERLAP: lk_agents_abi::turn::detector::ReqId =
    lk_agents_abi::turn::detector::ReqId(0);

/// Dropping an awaited host operation must cancel its captured invocation and report its
/// terminal receipt. Otherwise the shared run can wait forever after the Rust future ends.
struct PendingHostCallback {
    shared: Arc<crate::voice::session::SessionShared>,
    hosted: Option<HostedTool>,
    on_cancel: Option<GenEvent>,
}

impl PendingHostCallback {
    /// A callback for `hosted` under its speech's live invocation.
    fn bind(
        shared: &Arc<crate::voice::session::SessionShared>,
        mut hosted: HostedTool,
        on_cancel: GenEvent,
    ) -> Self {
        hosted.invocation = shared.speeches.invocation(&hosted);
        PendingHostCallback {
            shared: Arc::clone(shared),
            hosted: Some(hosted),
            on_cancel: Some(on_cancel),
        }
    }

    fn completed(&mut self) {
        self.hosted = None;
        self.on_cancel = None;
    }
}

impl Drop for PendingHostCallback {
    fn drop(&mut self) {
        if let Some(hosted) = self.hosted.take() {
            hosted.host.cancel(&hosted.invocation);
        }
        if let Some(event) = self.on_cancel.take() {
            self.shared.report(event);
        }
    }
}

impl VoiceRuntime {
    pub(super) fn prepare_leg(&mut self, opened: StreamOpened) -> VoiceEvent {
        let StreamOpened { open, leg, outcome } = opened;
        let outcome = outcome.map(|opened| {
            let sink = opened.resource();
            self.pending_legs.insert(sink, opened);
            sink
        });
        VoiceEvent::StreamOpened { open, leg, outcome }
    }

    pub(crate) fn perform_voice(&mut self, effects: Vec<VoiceEffect>) {
        let operations = self.publish_views(effects);
        self.operate(operations);
    }

    /// Views first: a receipt or event performed in the same batch must read what the core
    /// accepted, not the state before it. Returns the rest.
    pub(super) fn publish_views(&mut self, effects: Vec<VoiceEffect>) -> Vec<VoiceEffect> {
        let mut operations = Vec::with_capacity(effects.len());
        for effect in effects {
            match effect {
                VoiceEffect::SessionView(_)
                | VoiceEffect::ChatContext(_)
                | VoiceEffect::Outputs(_)
                | VoiceEffect::Speech(..) => self.perform_voice_effect(effect),
                effect => operations.push(effect),
            }
        }
        operations
    }

    pub(super) fn operate(&mut self, operations: Vec<VoiceEffect>) {
        for effect in operations {
            self.perform_voice_effect(effect);
        }
    }

    pub(super) fn deliver_session_event(&mut self, event: SessionEvent) {
        if let SessionEvent::MetricsCollected { metrics, .. } = &event {
            crate::telemetry::record_usage(&self.shared.trace, metrics);
            if let lk_agents_abi::metrics::base::AgentMetrics::RealtimeModel(m) = &**metrics {
                self.shared.realtime_spans.metrics(&self.shared.trace, m);
            }
        }
        self.peer.publish(SystemTap::Session(event.clone()));
        let _ = self.deliver(VoiceHostEvent::Session(event));
    }

    fn perform_voice_effect(&mut self, effect: VoiceEffect) {
        use lk_agents_abi::storage::voice::VoiceStorageCommand;
        use lk_agents_voice::agent_session::effects::AudioOp;
        use lk_agents_voice::agent_session::effects::VoiceEffect as E;
        // VoiceDriver journalled each effect when it was produced.
        match effect {
            E::RealtimePlayer { speech, event } => self.forward_to_realtime_player(speech, *event),
            E::CloseRealtimePlayer(speech) => {
                self.realtime_players.remove(&speech);
            }
            E::CloseRealtimePlayers => self.realtime_players.clear(),
            E::CancelSpeech(speech) => self.speeches.cancel(speech),
            E::Speech(speech, update) => {
                // Detached tool bodies and deferred tool-reply rendering outlive the release.
                if matches!(&update, lk_agents_voice::speech_handle::SpeechUpdate::Released) {
                    self.speeches.release(speech);
                }
                let producers = &mut self.speech_signals;
                self.shared.speeches.apply(producers, speech, update, &self.reports);
            }
            E::AttachRecorder { recorder, output: _ } => self.attach_recorder(recorder),
            E::DetachRecorder(_) => {
                self.recorder_observers.detach(&self.shared, None);
                self.publications.recorder.set(None);
                for output in self.driver.active_audio_outputs() {
                    self.recorder_observers.attach(&self.shared, output);
                }
            }
            E::FlushRecorder(at) => self.flush_recorder(at),
            E::PushKeyterms { stt, terms, warn_unsupported } => {
                self.push_keyterms(stt, terms, warn_unsupported)
            }
            // The request's `llm_request` opens under the pass's span.
            E::RequestKeyterms { llm, req, request, span } => {
                self.request_keyterms(llm, req, *request, span)
            }
            E::CancelKeyterms(req) => self.cancel_keyterms(req),
            E::AdoptOpenedInput { channel, sink } => {
                if let Some(opened) = self.pending_legs.remove(&sink) {
                    let (_, sink, stream) = opened.adopt();
                    self.drain(channel, sink, stream);
                }
            }
            E::DiscardOpenedInput(sink) => match self.pending_legs.remove(&sink) {
                Some(opened) => drop(opened),
                None => self.shared.resources.release(sink),
            },
            E::Journal(step) => self.shared.journal.record(*step),
            // A late or unknown call's answer is dropped.
            E::Answer(call, answer) => match self.calls.take(call) {
                Some(crate::voice::session::ReplyTo::Speech(reply)) => {
                    let created = matches!(answer, lk_agents_voice::agent_session::Answer::Speech);
                    let id = reply.reserved();
                    reply.answer(id.and_then(|id| self.speech_signals.get(id)), created);
                    // Lost: its writers go, which ends every wait on it.
                    if let (false, Some(id)) = (created, id) {
                        self.speech_signals.retire(id);
                    }
                }
                Some(reply_to) => reply_to.settle(answer),
                None => {}
            },
            E::Host(event) => self.deliver_host_event(event),
            E::StorageCheckpoint { checkpoint, speech, agent, history } => {
                let storage = self.storage.clone();
                let queue = self.reports.clone();
                let lost = queue.clone();
                self.speeches.spawn_feeding(
                    speech,
                    "voice conversation checkpoint",
                    move || {
                        let _ = lost.send(VoiceEvent::StorageCheckpointed {
                            checkpoint,
                            result: Err(lk_agents_abi::storage::StorageError::Unavailable(
                                "checkpoint task ended".into(),
                            )),
                        });
                    },
                    async move {
                        let result = match storage {
                            Some(storage) => storage.checkpoint(checkpoint, agent, history).await,
                            None => Ok(()),
                        };
                        let _ = queue.send(VoiceEvent::StorageCheckpointed { checkpoint, result });
                    },
                );
            }
            E::StorageAbort(speech) => {
                self.storage_command(VoiceStorageCommand::Abort { speech: speech.0 })
            }
            E::StorageCommit { speech, history } => {
                self.storage_command(VoiceStorageCommand::Settle { speech: speech.0, history })
            }
            E::ChatContext(snapshot) => {
                self.publications.chat_ctx.set(snapshot);
            }
            E::Outputs(outputs) => {
                self.publications.output.set(outputs);
            }
            E::Tap(tap) => self.peer.publish(tap),
            E::SessionView(view) => {
                self.publications.view.set(view);
            }
            E::Closing => {
                self.publications.closing.token().cancel();
            }
            E::SttConversationItem { stt, role, text } => {
                if let Some(stt) = self.shared.resources.stt(stt)
                    && stt.capabilities().chat_context
                {
                    stt.push_conversation_item(&role, &text);
                }
            }
            E::StartSpeech { speech, context, llm, tts, history, span } => {
                self.start_speech(speech, *context, (llm, tts), history, span)
            }
            // The session's ended spans and measurements go to the exporter, never the host.
            E::Emit(SessionEvent::Telemetry(records)) => {
                for record in records.iter() {
                    self.shared.tracer.record(record.clone());
                }
            }
            // Not traced here: VoiceDriver journalled it where it was decided.
            E::Emit(ev) => self.deliver_session_event(ev),
            E::FlushStream(sink) => {
                if let Some(sink) = self.shared.resources.audio_sink(sink) {
                    let _ = sink.flush();
                }
            }
            E::EotFlush(reason) => {
                if let Some(d) = self.eot.as_ref() {
                    d.flush(Some(reason));
                }
            }
            E::EotPredict { req, timeout } => self.predict_end_of_turn(req, timeout),
            E::EotCancel { req: _, timed_out } => {
                if let Some(d) = self.eot.as_ref() {
                    d.cancel_inference(timed_out);
                }
            }
            E::InterruptionSignal { overlap, signal } => self.signal_interruption(overlap, signal),
            // Opens off the loop; the stream comes back as `VoiceEvent::StreamOpened`.
            E::OpenSttStream { open, stt, span } => {
                let stt = self.shared.resources.stt(stt);
                let cx = self.shared.trace.clone().continuing(span);
                self.open_stream(open, StreamKind::Stt, stt.map(|stt| PendingOpen::Stt(stt, cx)));
            }
            E::OpenVadStream { open, vad } => {
                let vad = self.shared.resources.vad(vad);
                let cx = self.shared.trace.clone();
                self.open_stream(open, StreamKind::Vad, vad.map(|vad| PendingOpen::Vad(vad, cx)));
            }
            E::ProviderPrewarm(id) => self.shared.resources.prewarm(id, &self.shared.trace),
            E::ProviderRelease(id) => self.shared.resources.release(id),
            E::CloseRealtimeSession(id) => self.close_realtime_session(id),
            E::EotClose => self.predictions.clear(),
            E::Audio(AudioOp::Gate { speech: _, op, cause: _, output }) => {
                self.gate_audio(op, output)
            }
            E::Audio(AudioOp::SetAudioOutput { previous, next, enabled }) => {
                self.set_audio_output(previous, next, enabled)
            }
            E::Audio(AudioOp::SetTranscriptionOutput { previous, next, enabled }) => {
                self.set_transcription_output(previous, next, enabled)
            }
            E::Audio(AudioOp::SettleUnreported { output, segment }) => {
                self.settle_unreported(output, segment)
            }
            E::Audio(AudioOp::SetOutputEnabled { output, sink, enabled }) => {
                self.set_output_enabled(output, sink, enabled)
            }
            E::UpdateAudioInputRouting(command) => self.command(command),
            E::Audio(AudioOp::SetAudioInput(input)) => self.set_audio_input(input),
            E::Audio(AudioOp::SetVideoInput(input)) => self.set_video_input(input),
            E::Realtime { session, command } => {
                if let Some(session) = self.shared.resources.realtime(session) {
                    session.send(command);
                }
            }
            E::CancelTool { hosted, .. } => self.cancel_tool(hosted),
            E::AnswerExpert { hosted, answer } => {
                hosted.host.answer_expert(&self.shared.speeches.invocation(&hosted), answer)
            }
            E::ValidateOutput { speech, validation, candidate, hosted, turn } => {
                let cx = self.shared.turn_cx(turn);
                self.validate_output(speech, validation, candidate, hosted, cx)
            }
            E::CancelOutputValidation { speech, validation } => {
                self.speeches.end_callback(speech, HostCallbackId::OutputValidation(validation))
            }
            E::RenderToolHandlingTemplate { rendering, template, hosted, turn } => {
                let cx = self.shared.turn_cx(turn);
                self.render_tool_handling_template(rendering, template, hosted, cx)
            }
            E::CancelToolRefusal { speech, call } => {
                self.speeches.end_callback(speech, HostCallbackId::ToolRefusal(call))
            }
        }
    }

    /// A realtime speech's event, handed to its player. A player that can't take it is ended.
    fn forward_to_realtime_player(
        &mut self,
        speech: SpeechId,
        event: lk_agents_abi::llm::realtime::Event,
    ) {
        let Some(player) = self.realtime_players.get(&speech) else { return };
        let Err(error) = player.try_send(event) else { return };
        if matches!(error, tokio::sync::mpsc::error::TrySendError::Full(_)) {
            log::error!(target: lk_agents_abi::LOG_TARGET, speech = speech.0,
                capacity = crate::voice::generation::realtime_generation::PLAYER_CAPACITY;
                "a realtime speech fell too far behind its reply; it ends on what it holds");
        }
        self.realtime_players.remove(&speech);
        let _ = self.reports.send(RealtimeWake::PlayerEnded(speech));
    }

    fn attach_recorder(&mut self, recorder: lk_agents_voice::agent_session::RecorderHandle) {
        self.recorder_observers.detach(&self.shared, None);
        let Some(recorder) = self.shared.resources.recorder(recorder.0) else { return };
        recorder.observe(&self.peer.session);
        let tap = crate::voice::recorder::RecorderTap::new(recorder, self.shared.time.clock());
        let outputs: Vec<_> = self.driver.active_audio_outputs().collect();
        crate::voice::recorder::join(&self.shared, &tap, &outputs);
        self.publications.recorder.set(Some(tap));
        for output in outputs {
            self.recorder_observers.attach(&self.shared, output);
        }
    }

    fn flush_recorder(&self, at: f64) {
        let Some(tap) = self.shared.recorder_tap() else { return };
        let ok = tap.recorder.flush(at).is_ok();
        self.shared.journal.record(lk_agents_voice::trace::PlayoutStep::RecorderFlushed { ok });
    }

    fn push_keyterms(&self, stt: Option<ResourceId>, terms: Vec<String>, warn_unsupported: bool) {
        if let Some(stt) = stt.and_then(|id| self.shared.resources.stt(id)) {
            if warn_unsupported {
                log::warn!(target: lk_agents_abi::LOG_TARGET, stt = stt.label().as_str();
                    "keyterms are not supported by this STT, ignoring keyterms update");
            }
            stt.update_keyterms(&terms);
        }
        self.publications.keyterms.set(Some(terms));
    }

    fn request_keyterms(
        &mut self,
        llm: ResourceId,
        req: u64,
        request: lk_agents_abi::ChatRequest,
        span: Option<telemetry::SpanContext>,
    ) {
        let shared = Arc::clone(&self.shared);
        let Some(llm) = shared.resources.llm(llm) else {
            shared.report(lk_agents_voice::keyterm_detection::DetectionPass::opening_failed(req));
            return;
        };
        let task = crate::voice::keyterm_detection::detection_pass(
            Arc::clone(&shared),
            llm,
            request,
            req,
            span,
        );
        let handle = self.scoped("keyterm detection", task);
        self.keyterms = Some((req, handle));
    }

    fn cancel_keyterms(&mut self, req: u64) {
        if self.keyterms.as_ref().is_some_and(|(pending, _)| *pending == req) {
            self.keyterms = None;
        }
    }

    /// To the host's drain: a `speech_done` carries the speech's `reported` writer, which the
    /// drain finishes once the host handled it; a retirement lets the speech's writers go.
    fn deliver(&mut self, event: VoiceHostEvent) -> Result<(), Box<VoiceHostEvent>> {
        let reported = match &event {
            VoiceHostEvent::Session(SessionEvent::SpeechDone { speech, .. }) => {
                self.speech_signals.take_reported(*speech)
            }
            VoiceHostEvent::SpeechRetired(speech) => {
                self.speech_signals.retire(*speech);
                None
            }
            _ => None,
        };
        self.peer.deliver(event, reported)
    }

    /// A hook the host can't take is answered as failed, so its turn does not wait for it.
    fn deliver_host_event(&mut self, event: VoiceHostEvent) {
        if let Err(event) = self.deliver(event)
            && let VoiceHostEvent::Hook(call) = *event
        {
            let _ = self.reports.send(lk_agents_voice::agent_session::Command::HookDone {
                id: call.id(),
                outcome: lk_agents_voice::agent_session::HookOutcome::Failed,
            });
        }
    }

    /// Binds the speech's reply to its providers and spawns it. A stored speech first reloads its
    /// history from the voice partition; a realtime one plays what the socket sends at once while its
    /// partition reloads beside it.
    fn start_speech(
        &mut self,
        speech: SpeechId,
        context: lk_agents_voice::generation::ReplyContext,
        (llm, tts): (Option<ResourceId>, Option<ResourceId>),
        history: SpeechHistory,
        span: Option<telemetry::SpanContext>,
    ) {
        let Some(handle) = self.shared.speeches.get(speech) else { return };
        // A realtime speech plays what the socket sends it, through its own player.
        let player = matches!(context.source, ReplySource::Realtime).then(|| {
            let capacity = crate::voice::generation::realtime_generation::PLAYER_CAPACITY;
            let (tx, rx) = tokio::sync::mpsc::channel(capacity);
            self.realtime_players.insert(handle.id(), tx);
            rx
        });
        let bound =
            super::generation::ReplyExecution::bind(context, llm, tts, &self.shared.resources);
        let mut context = match bound {
            Ok(context) => super::generation::ReplyExecution { span, ..context },
            Err(error) => {
                let error = Box::new(lk_agents_voice::generation::fatal(error.to_string()));
                self.shared.report(GenEvent::Failed { speech, error });
                return;
            }
        };
        let shared = Arc::clone(&self.shared);
        let task = Arc::clone(&handle);
        let mut storage = self.storage.clone().filter(|_| history == SpeechHistory::VoicePartition);
        let queue = self.reports.clone();
        if player.is_some()
            && let Some(storage) = storage.take()
        {
            let local = context.plan.request.chat_ctx.items.clone();
            let queue = queue.clone();
            let speech = Arc::clone(&handle);
            let reload = async move {
                match reload_history(&speech, storage, local, &queue).await {
                    // Reported to the core either way, which hands it to the playing speech.
                    Ok(_) | Err(_) => {}
                }
            };
            self.speeches.spawn(handle.id(), "speech partition_key reload", reload);
        }
        let speeches = self.speeches.clone();
        // A speech task that panics never reports `Done` or `Failed`: its death is the input.
        let reports = Arc::clone(&self.shared);
        let lost = move || {
            let error = Box::new(lk_agents_voice::generation::fatal("the speech's task panicked"));
            reports.report(GenEvent::Failed { speech, error });
        };
        self.speeches.spawn_feeding(handle.id(), "speech", lost, async move {
            if let Some(storage) = storage {
                let local = context.plan.request.chat_ctx.items.clone();
                match reload_history(&task, storage, local, &queue).await {
                    Ok(Some(history)) => context.plan.request.chat_ctx.items = history,
                    Ok(None) => {} // An interrupted speech settles through the normal core path.
                    Err(_) => return, // StorageFailed owns the call's error/cleanup path.
                }
            }
            crate::voice::generation::pipeline_reply::pipeline_reply_task(
                shared, task, context, player, speeches,
            )
            .await;
        });
    }

    fn predict_end_of_turn(
        &mut self,
        req: lk_agents_abi::ReqId,
        timeout: Option<std::time::Duration>,
    ) {
        if self.eot.is_none() {
            return;
        }
        self.shared.journal.record(lk_agents_voice::trace::InputStep::LegRequested {
            leg: lk_agents_voice::trace::Leg::Eot,
            req: req.0,
        });
        self.predict(req, timeout);
    }

    /// Not awaited, like the reference's `send_nowait` on `_interruption_ch`. A session with no
    /// detector raises no signals.
    fn signal_interruption(
        &self,
        overlap: Option<lk_agents_abi::ReqId>,
        signal: InterruptionSignal,
    ) {
        let Some(d) = self.interruption.clone() else { return };
        match signal {
            InterruptionSignal::OverlapStarted { .. } => {
                let overlap = overlap.unwrap_or(UNNAMED_OVERLAP);
                d.signal_overlap(overlap, signal);
            }
            InterruptionSignal::AgentSpeechStarted
            | InterruptionSignal::AgentSpeechEnded
            | InterruptionSignal::OverlapEnded { .. } => d.signal(signal),
        }
    }

    fn close_realtime_session(&mut self, id: ResourceId) {
        if self.realtime.is_none() {
            let Some(session) = self.shared.resources.realtime(id) else { return };
            self.realtime =
                Some(super::legs::RealtimeConnection::new(session, &self.shared.background()));
        }
        if let Some(connection) = &mut self.realtime {
            connection.close(self.reports.clone());
        }
    }

    fn gate_audio(&self, op: lk_agents_voice::agent_session::effects::GateOp, output: ResourceId) {
        use lk_agents_voice::agent_session::effects::GateOp;
        let Some(sink) = self.shared.resources.audio_output(output) else { return };
        match op {
            GateOp::Pause => sink.pause(),
            GateOp::Resume => sink.resume(),
            // A bare clear: the cut speech's playout closes its own segment. A flush would end a
            // synchronised segment whose audio never reached the sink as if it had been said.
            GateOp::ClearBuffer => sink.clear_buffer(),
        }
    }

    /// A cut segment its sink never reported, settled interrupted where the output last reported
    /// it played, leaf first so each link's report reaches the one above it as a real report would.
    fn settle_unreported(
        &self,
        output: ResourceId,
        segment: lk_agents_abi::voice::io::PlaybackSegment,
    ) {
        let Some(top) = self.shared.resources.audio_output(output) else { return };
        let tracker = top.tracker();
        let playback_position =
            if tracker.accepts_report(segment) { tracker.position() } else { Default::default() };
        let finished = lk_agents_abi::voice::io::PlaybackFinished {
            playback_position,
            interrupted: true,
            ..Default::default()
        };
        let mut links = vec![top];
        while let Some(next) = links.last().and_then(|link| link.next_in_chain()) {
            links.push(next);
        }
        for link in links.iter().rev() {
            link.tracker().playback_finished(segment, finished.clone());
        }
    }

    /// The recorder stops observing the previous output before it is detached.
    fn set_audio_output(
        &mut self,
        previous: Option<ResourceId>,
        next: Option<ResourceId>,
        enabled: bool,
    ) {
        let resources = &self.shared.resources;
        if let Some(id) = previous {
            self.recorder_observers.detach(&self.shared, Some(id));
            if let Some(old) = resources.audio_output(id) {
                old.on_detached();
            }
        }
        if let Some(id) = next {
            if let Some(new) = resources.audio_output(id) {
                if enabled { new.on_attached() } else { new.on_detached() }
            }
            self.recorder_observers.attach(&self.shared, id);
        }
    }

    fn set_transcription_output(
        &self,
        previous: Option<ResourceId>,
        next: Option<ResourceId>,
        enabled: bool,
    ) {
        let resources = &self.shared.resources;
        if let Some(old) = previous.and_then(|id| resources.transcription_output(id)) {
            old.on_detached();
        }
        if let Some(new) = next.and_then(|id| resources.transcription_output(id)) {
            if enabled { new.on_attached() } else { new.on_detached() }
        }
    }

    fn set_output_enabled(
        &self,
        output: lk_agents_voice::trace::OutputKind,
        sink: Option<ResourceId>,
        enabled: bool,
    ) {
        let resources = &self.shared.resources;
        match output {
            lk_agents_voice::trace::OutputKind::Audio => {
                if let Some(sink) = sink.and_then(|id| resources.audio_output(id)) {
                    if enabled { sink.on_attached() } else { sink.on_detached() }
                }
            }
            lk_agents_voice::trace::OutputKind::Transcription => {
                if let Some(sink) = sink.and_then(|id| resources.transcription_output(id)) {
                    if enabled { sink.on_attached() } else { sink.on_detached() }
                }
            }
        }
    }

    /// A source the router can no longer take is released.
    fn set_audio_input(&self, input: Option<lk_agents_voice::agent_session::AudioInputHandle>) {
        let source = input.map(|input| input.0);
        if self
            .reports
            .send(lk_agents_voice::data_plane::AudioInputRouterCommand::SetSource(source))
            .is_err()
            && let Some(source) = source
        {
            self.shared.resources.release(source);
        }
    }

    fn cancel_tool(&self, hosted: HostedTool) {
        let invocation = self.shared.speeches.invocation(&hosted);
        if self.shared.tool_bodies.claim_cancellation(&invocation) {
            hosted.host.cancel(&invocation);
        }
    }

    /// The host validates the candidate output; a validation of the same key still running is
    /// cancelled.
    fn validate_output(
        &mut self,
        speech: SpeechId,
        validation: lk_agents_abi::agent::run::ValidationId,
        candidate: lk_agents_abi::agent::OutputCandidate,
        hosted: HostedTool,
        cx: telemetry::Cx,
    ) {
        let shared = Arc::clone(&self.shared);
        let cancelled = GenEvent::OutputValidated {
            speech,
            validation,
            verdict: lk_agents_core::agent_run::OutputVerdict::Failed(
                "output validation cancelled".into(),
            ),
            value: None,
        };
        let mut callback = PendingHostCallback::bind(&shared, hosted, cancelled);
        let id = HostCallbackId::OutputValidation(validation);
        self.speeches.callback(speech, id, "output validation", async move {
            let hosted = callback.hosted.as_ref().expect("callback is pending");
            // The voice run retains its live host value, not a persistence snapshot.
            let (verdict, value, _) = crate::output_validation::project(
                hosted.host.validate_output(&cx, &hosted.invocation, candidate).await,
            );
            callback.completed();
            shared.report(GenEvent::OutputValidated { speech, validation, verdict, value });
        });
    }

    /// A refusal's template runs with its speech; a deferred tool reply's outlives it.
    fn render_tool_handling_template(
        &mut self,
        rendering: Rendering,
        template: lk_agents_abi::agent::ToolTemplate,
        hosted: HostedTool,
        cx: telemetry::Cx,
    ) {
        let shared = Arc::clone(&self.shared);
        let cancelled = GenEvent::ToolHandlingTemplateRendered { rendering, text: Ok(None) };
        let mut callback = PendingHostCallback::bind(&shared, hosted, cancelled);
        let work = async move {
            let hosted = callback.hosted.as_ref().expect("callback is pending");
            let text = hosted
                .host
                .render_template(&cx, &hosted.invocation, template.key(), template)
                .await;
            callback.completed();
            shared.report(GenEvent::ToolHandlingTemplateRendered { rendering, text });
        };
        match rendering {
            Rendering::Refusal { speech, call } => {
                let id = HostCallbackId::ToolRefusal(call);
                self.speeches.callback(speech, id, "tool refusal template", work);
            }
            Rendering::ToolReply { .. } => {
                self.spawn("deferred tool reply template", work);
            }
        }
    }

    /// Opens `provider`'s stream off the loop and reports it, or that the provider is gone, as
    /// `VoiceEvent::StreamOpened`. A close abandons an open still in flight; whatever it opened
    /// is closed.
    fn open_stream(&mut self, open: u64, leg: StreamKind, provider: Option<PendingOpen>) {
        let Some(pending) = provider else {
            let error = ProviderError::Local(
                ApiError::new("provider resource is no longer available").with_retryable(false),
            );
            let _ = self.reports.send_opened(StreamOpened { open, leg, outcome: Err(error) });
            return;
        };
        let tx = self.reports.clone();
        let time = self.shared.time.clone();
        let resources = self.shared.resources.clone();
        let background = self.shared.background();
        let closing = self.shared.closing.clone();
        self.spawn("stream open", async move {
            // Abandoning an open is safe: a host open dropped unsettled is cancelled, and a stream
            // it returned unread is closed (`py_streams::StreamCall`); a native one owns its socket.
            let outcome = tokio::select! {
                biased;
                () = closing.cancelled() => return,
                outcome = pending.open(resources, time, background) => outcome,
            };
            // A refused send drops the opened stream, which closes it, joined by the session.
            let _ = tx.send_opened(StreamOpened { open, leg, outcome });
        });
    }
}

/// The speech takes the voice partition: the partition's history, which replaces the speech's `local` one,
/// is reported and returned, and the partition's loss is watched. An interrupted speech stops waiting
/// for a partition another holder has: an abandoned acquire releases whatever it may have taken.
async fn reload_history(
    handle: &super::SpeechHandle,
    storage: Arc<dyn lk_agents_abi::storage::voice::VoiceStorage>,
    local: Vec<lk_agents_abi::ChatItem>,
    queue: &crate::voice::session::Tx,
) -> Result<Option<Vec<lk_agents_abi::ChatItem>>, lk_agents_abi::storage::StorageError> {
    let speech = handle.id();
    let acquired = tokio::select! {
        biased;
        acquired = storage.acquire(speech.0, local.clone()) => acquired,
        () = handle.wait_for_interrupted() => return Ok(None),
    };
    match acquired {
        Ok(history) => {
            let _ =
                queue.send(VoiceEvent::StorageReloaded { read: local, history: history.clone() });
            let _ = storage.clone().dispatch(
                lk_agents_abi::storage::voice::VoiceStorageCommand::ObserveLoss {
                    speech: speech.0,
                },
                Arc::new(super::session::storage::StorageReporter(queue.clone())),
            );
            Ok(Some(history))
        }
        Err(error) => {
            let _ = queue.send(VoiceEvent::StorageFailed { speech, error: error.clone() });
            Err(error)
        }
    }
}

/// A provider whose stream is being opened.
enum PendingOpen {
    /// The STT, and the activity its stream runs under.
    Stt(Arc<dyn lk_agents_abi::Stt>, telemetry::Cx),
    /// The VAD, and the session its stream runs under.
    Vad(Arc<dyn lk_agents_abi::Vad>, telemetry::Cx),
}

impl PendingOpen {
    async fn open(
        self,
        resources: crate::voice::resources::Resources,
        time: crate::voice::session::Timebase,
        background: lk_agents_net::owned::Background,
    ) -> Result<Opened, ProviderError> {
        let (channel, sink, source) = match self {
            PendingOpen::Stt(stt, cx) => {
                let options = stt.conn_options().unwrap_or_default();
                let Some(stream) =
                    lk_agents_inference::stt::stream(stt, options, time.clock(), &cx, None, None)
                        .await?
                else {
                    return Err(ProviderError::Local(ApiError::new(
                        "the new agent's STT does not stream",
                    )));
                };
                let (sink, source) = stream.split();
                (AudioChannel::Stt, sink, PendingLeg::Stt(source))
            }
            PendingOpen::Vad(vad, cx) => {
                let (sink, source) = vad.stream(&cx).await?.split();
                (AudioChannel::Vad, sink, PendingLeg::Vad(source))
            }
        };
        Ok(Opened::new(channel, Arc::from(sink), source, resources, &background))
    }
}
