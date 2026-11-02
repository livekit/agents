//! Executes the decision core's effects after synchronous settlement.

use super::legs::PendingLeg;
use super::{HostCallbackId, VoiceRuntime};
use crate::voice::events::{Opened, StreamOpened};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_voice::agent_activity::realtime::RealtimeWake;
use lk_agents_voice::agent_session::effects::{StreamKind, VoiceEffect};
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

/// Dropping an awaited host operation must cancel its captured invocation and report its
/// terminal receipt. Otherwise the shared run can wait forever after the Rust future ends.
struct PendingHostCallback {
    shared: Arc<crate::voice::session::SessionShared>,
    hosted: Option<HostedTool>,
    on_cancel: Option<GenEvent>,
}

impl PendingHostCallback {
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
    /// Retire only callbacks belonging to this reply. Detached tool bodies and deferred
    /// tool-reply rendering have their own lifetimes and must not be cancelled here.
    fn cancel_speech_callbacks(
        &mut self,
        speech: lk_agents_voice::agent_activity::scheduling::SpeechId,
    ) {
        let tasks: Vec<_> = self
            .host_callbacks
            .iter()
            .filter_map(|((owner, _), task)| (*owner == speech).then_some(*task))
            .collect();
        self.host_callbacks.retain(|(owner, _), _| *owner != speech);
        for task in tasks {
            self.tasks.cancel(task);
        }
    }

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

    fn deliver_session_event(&mut self, event: SessionEvent) {
        self.peer.publish(SystemTap::Session(event.clone()));
        let _ = self.peer.deliver(VoiceHostEvent::Session(event));
    }

    fn perform_voice_effect(&mut self, effect: VoiceEffect) {
        use lk_agents_voice::agent_session::effects::{AudioOp, GateOp, VoiceEffect};
        // Journalled already: VoiceDriver recorded each effect when the domain produced it.
        match effect {
            VoiceEffect::RealtimePlayer { speech, event } => {
                let Some(player) = self.realtime_players.get(&speech) else {
                    return;
                };
                if let Err(error) = player.try_send(*event) {
                    if matches!(error, tokio::sync::mpsc::error::TrySendError::Full(_)) {
                        log::error!(target: "livekit.agents", speech = speech.0,
                            capacity = crate::voice::generation::realtime_generation::PLAYER_CAPACITY;
                            "a realtime speech fell too far behind its reply; it ends on what it holds");
                    }
                    self.realtime_players.remove(&speech);
                    let _ = self.reports.send(RealtimeWake::PlayerEnded(speech));
                }
            }
            VoiceEffect::CloseRealtimePlayer(speech) => {
                self.realtime_players.remove(&speech);
            }
            VoiceEffect::CloseRealtimePlayers => self.realtime_players.clear(),
            VoiceEffect::CancelSpeech(speech) => {
                self.cancel_speech_callbacks(speech);
                self.speeches.cancel(speech);
            }
            VoiceEffect::Speech(speech, update) => {
                if matches!(&update, lk_agents_voice::speech_handle::SpeechUpdate::Released) {
                    self.cancel_speech_callbacks(speech);
                }
                self.shared.speeches.apply(speech, update, &self.reports);
            }
            VoiceEffect::AttachRecorder { recorder, output: _ } => {
                self.recorder_observers.detach(&self.shared, None);
                let Some(recorder) = self.shared.resources.recorder(recorder.0) else { return };
                let tap =
                    crate::voice::recorder::RecorderTap::new(recorder, self.shared.time.clock());
                self.publications.recorder.send_replace(Some(tap));
                for output in self.driver.active_audio_outputs() {
                    self.recorder_observers.attach(&self.shared, output);
                }
            }
            VoiceEffect::DetachRecorder(_) => {
                self.recorder_observers.detach(&self.shared, None);
                self.publications.recorder.send_replace(None);
            }
            VoiceEffect::FlushRecorder(at) => {
                if let Some(tap) = self.shared.recorder_tap() {
                    let ok = tap.recorder.flush(at).is_ok();
                    self.shared
                        .journal
                        .record(lk_agents_voice::trace::PlayoutStep::RecorderFlushed { ok });
                }
            }
            VoiceEffect::PushKeyterms { stt, terms, warn_unsupported } => {
                if let Some(stt) = stt.and_then(|id| self.shared.resources.stt(id)) {
                    if warn_unsupported {
                        log::warn!(target: "livekit.agents", stt = stt.label().as_str();
                            "keyterms are not supported by this STT, ignoring keyterms update");
                    }
                    stt.update_keyterms(&terms);
                }
                self.publications.keyterms.send_replace(Some(terms));
            }
            VoiceEffect::RequestKeyterms { llm, req, request } => {
                let shared = Arc::clone(&self.shared);
                let Some(llm) = shared.resources.llm(llm) else {
                    shared.report(
                        lk_agents_voice::keyterm_detection::DetectionPass::opening_failed(req),
                    );
                    return;
                };
                let task = crate::voice::keyterm_detection::detection_pass(
                    Arc::clone(&shared),
                    llm,
                    *request,
                    req,
                );
                let handle = self.spawn("keyterm detection", task);
                self.keyterms = Some((req, handle));
            }
            VoiceEffect::CancelKeyterms(req) => {
                if self.keyterms.as_ref().is_some_and(|(pending, _)| *pending == req)
                    && let Some((_, task)) = self.keyterms.take()
                {
                    self.tasks.cancel(task);
                }
            }
            VoiceEffect::AdoptOpenedInput { channel, sink } => {
                if let Some(opened) = self.pending_legs.remove(&sink) {
                    let (_, sink, stream) = opened.adopt();
                    self.drain(channel, sink, stream);
                }
            }
            VoiceEffect::DiscardOpenedInput(sink) => {
                if let Some(opened) = self.pending_legs.remove(&sink) {
                    opened.retire(&self.tasks);
                } else {
                    self.shared.resources.release(sink);
                }
            }
            VoiceEffect::Journal(step) => self.shared.journal.record(*step),
            // A late or unknown call's answer is dropped.
            VoiceEffect::Answer(call, answer) => {
                if let Some(reply_to) = self.calls.take(call) {
                    reply_to.settle(answer);
                }
            }
            VoiceEffect::Host(event) => {
                if let Err(event) = self.peer.deliver(event)
                    && let VoiceHostEvent::Hook(call) = *event
                {
                    let _ = self.reports.send(lk_agents_voice::agent_session::Command::HookDone {
                        id: call.id(),
                        outcome: lk_agents_voice::agent_session::HookOutcome::Failed,
                    });
                }
            }
            VoiceEffect::StorageAbort(speech) => {
                self.storage_command(lk_agents_abi::storage::voice::VoiceStorageCommand::Abort {
                    speech: speech.0,
                })
            }
            VoiceEffect::StorageCommit { speech, history } => {
                self.storage_command(lk_agents_abi::storage::voice::VoiceStorageCommand::Settle {
                    speech: speech.0,
                    history,
                })
            }
            VoiceEffect::ChatContext(snapshot) => {
                self.publications.chat_ctx.send_replace(snapshot);
            }
            VoiceEffect::Outputs(outputs) => {
                self.publications.output.send_replace(outputs);
            }
            VoiceEffect::Tap(tap) => self.peer.publish(tap),
            VoiceEffect::SessionView(view) => {
                self.publications.view.send_replace(view);
            }
            VoiceEffect::Closing => {
                self.publications.closing.send_replace(true);
            }
            VoiceEffect::SttConversationItem { stt, role, text } => {
                if let Some(stt) = self.shared.resources.stt(stt)
                    && stt.capabilities().chat_context
                {
                    stt.push_conversation_item(&role, &text);
                }
            }
            VoiceEffect::StartSpeech { speech, context, llm, tts, stored } => {
                let Some(handle) = self.shared.speeches.get(speech) else { return };
                // A realtime speech plays what the socket sends it, through its own player.
                let player = matches!(context.source, ReplySource::Realtime).then(|| {
                    let capacity = crate::voice::generation::realtime_generation::PLAYER_CAPACITY;
                    let (tx, rx) = tokio::sync::mpsc::channel(capacity);
                    self.realtime_players.insert(handle.id(), tx);
                    rx
                });
                let mut context = match super::generation::ReplyExecution::bind(
                    *context,
                    llm,
                    tts,
                    &self.shared.resources,
                ) {
                    Ok(context) => context,
                    Err(error) => {
                        self.shared.report(GenEvent::Failed {
                            speech,
                            error: Box::new(lk_agents_voice::generation::fatal(error.to_string())),
                        });
                        return;
                    }
                };
                let shared = Arc::clone(&self.shared);
                let task = Arc::clone(&handle);
                let storage = self.storage.clone().filter(|_| stored);
                let queue = self.reports.clone();
                let speeches = self.speeches.clone();
                self.speeches.spawn(handle.id(), "speech", async move {
                    if let Some(storage) = storage {
                        match storage.acquire(task.id().0, context.plan.request.chat_ctx.items.clone()).await {
                            Ok(history) => {
                                let read = std::mem::replace(
                                    &mut context.plan.request.chat_ctx.items,
                                    history.clone(),
                                );
                                let _ = queue.send(VoiceEvent::StorageReloaded { read, history });
                                let _ = storage.clone().dispatch(
                                    lk_agents_abi::storage::voice::VoiceStorageCommand::ObserveLoss { speech: task.id().0 },
                                    Arc::new(super::storage::StorageReporter(queue.clone())),
                                );
                            }
                            Err(error) => {
                                log::warn!(target: "livekit.agents", "the voice lane could not be acquired: {error}");
                                if storage.lease_lost() {
                                    let _ = queue.send(VoiceEvent::StorageLost(task.id()));
                                }
                            }
                        }
                    }
                    crate::voice::generation::pipeline_reply::pipeline_reply_task(
                        shared, task, context, player, speeches,
                    )
                    .await;
                });
            }
            // Not traced here: VoiceDriver journalled it where it was decided.
            VoiceEffect::Emit(ev) => {
                self.deliver_session_event(ev);
            }
            VoiceEffect::FlushStream(sink) => {
                if let Some(sink) = self.shared.resources.audio_sink(sink) {
                    let _ = sink.flush();
                }
            }
            // A session without a turn detector asks for no flush.
            VoiceEffect::EotFlush(reason) => {
                if let Some(d) = self.eot.as_ref() {
                    d.flush(Some(reason.as_str().to_owned()));
                }
            }
            VoiceEffect::EotPredict { req, timeout } => {
                if self.eot.is_some() {
                    self.shared.journal.record(lk_agents_voice::trace::InputStep::LegRequested {
                        leg: lk_agents_voice::trace::DetectorName::Eot,
                        req: req.0,
                    });
                    self.predict(req, timeout);
                }
            }
            // `_generate_reply` is the one `timed_out=True` site. A session without a detector
            // asks for none.
            VoiceEffect::EotCancel { req: _, timed_out } => {
                if let Some(d) = self.eot.as_ref() {
                    d.cancel_inference(timed_out);
                }
            }
            // Not awaited, like the reference's `send_nowait` on `_interruption_ch`. A session
            // with no detector raises no signals.
            VoiceEffect::InterruptionSignal { overlap, signal } => {
                if let Some(d) = self.interruption.clone() {
                    match signal {
                        InterruptionSignal::OverlapStarted { .. } => {
                            let overlap =
                                overlap.unwrap_or(lk_agents_abi::turn::detector::ReqId(0));
                            d.signal_overlap(overlap, signal);
                        }
                        InterruptionSignal::AgentSpeechStarted
                        | InterruptionSignal::AgentSpeechEnded
                        | InterruptionSignal::OverlapEnded { .. } => d.signal(signal),
                    }
                }
            }
            // The open runs off the loop (the provider awaits a connection); the stream comes back
            // as `VoiceEvent::StreamOpened`.
            VoiceEffect::OpenSttStream { open, stt } => {
                let stt = self.shared.resources.stt(stt);
                self.open_stream(open, StreamKind::Stt, stt.map(PendingOpen::Stt));
            }
            VoiceEffect::OpenVadStream { open, vad } => {
                let vad = self.shared.resources.vad(vad);
                self.open_stream(open, StreamKind::Vad, vad.map(PendingOpen::Vad));
            }
            VoiceEffect::ProviderPrewarm(id) => self.shared.resources.prewarm(id),
            VoiceEffect::ProviderRelease(id) => self.shared.resources.release(id),
            VoiceEffect::CloseRealtimeSession(id) => {
                if self.realtime.is_none() {
                    let Some(session) = self.shared.resources.realtime(id) else { return };
                    self.realtime = Some(super::legs::RealtimeConnection::new(
                        session,
                        self.shared.time.clone(),
                        self.shared.spawner.clone(),
                    ));
                }
                if let Some(connection) = &mut self.realtime {
                    connection.close(&self.tasks, self.reports.clone());
                }
            }
            VoiceEffect::EotClose => {
                for task in self.predictions.drain(..) {
                    self.tasks.cancel(task);
                }
            }
            VoiceEffect::Audio(AudioOp::Gate { speech: _, op, cause: _, output }) => {
                let Some(sink) = self.shared.resources.audio_output(output) else {
                    return;
                };
                match op {
                    GateOp::Pause => sink.pause(),
                    GateOp::Resume => sink.resume(),
                    // A bare clear: the speech's own playout, cut by the same interruption, closes
                    // the segment it pushed (`clear_sink`). A flush here would end a synchronised
                    // segment none of whose audio reached the sink as if it had been said.
                    GateOp::ClearBuffer => sink.clear_buffer(),
                }
            }
            VoiceEffect::Audio(AudioOp::SetAudioOutput { previous, next, enabled }) => {
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
            VoiceEffect::Audio(AudioOp::SetTranscriptionOutput { previous, next, enabled }) => {
                let resources = &self.shared.resources;
                if let Some(old) = previous.and_then(|id| resources.text_output(id)) {
                    old.on_detached();
                }
                if let Some(new) = next.and_then(|id| resources.text_output(id)) {
                    if enabled { new.on_attached() } else { new.on_detached() }
                }
            }

            VoiceEffect::Audio(AudioOp::SetOutputEnabled { output, sink, enabled }) => {
                let resources = &self.shared.resources;
                match output {
                    lk_agents_voice::trace::OutputKind::Audio => {
                        if let Some(sink) = sink.and_then(|id| resources.audio_output(id)) {
                            if enabled { sink.on_attached() } else { sink.on_detached() }
                        }
                    }
                    lk_agents_voice::trace::OutputKind::Transcription => {
                        if let Some(sink) = sink.and_then(|id| resources.text_output(id)) {
                            if enabled { sink.on_attached() } else { sink.on_detached() }
                        }
                    }
                }
            }
            VoiceEffect::UpdateAudioInputRouting(command) => self.command(command),
            VoiceEffect::Audio(AudioOp::SetAudioInput(input)) => {
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
            VoiceEffect::Audio(AudioOp::SetVideoInput(input)) => self.set_video_input(input),
            VoiceEffect::Realtime { session, command } => {
                if let Some(session) = self.shared.resources.realtime(session) {
                    session.send(command);
                }
            }
            VoiceEffect::CancelTool { hosted, .. } => {
                let invocation = self.shared.speeches.invocation(&hosted);
                if self.shared.tool_bodies.claim_cancellation(&invocation) {
                    hosted.host.cancel(&invocation);
                }
            }
            VoiceEffect::AnswerExpert { hosted, answer } => {
                hosted.host.answer_expert(&self.shared.speeches.invocation(&hosted), answer)
            }
            VoiceEffect::ValidateOutput { speech, validation, candidate, hosted } => {
                let shared = Arc::clone(&self.shared);
                let invocation = shared.speeches.invocation(&hosted);
                let mut hosted = hosted;
                hosted.invocation = invocation;
                let mut callback = PendingHostCallback {
                    shared: Arc::clone(&shared),
                    hosted: Some(hosted),
                    on_cancel: Some(GenEvent::OutputValidated {
                        speech,
                        validation,
                        verdict: lk_agents_core::agent_run::OutputVerdict::Failed(
                            "output validation cancelled".into(),
                        ),
                        value: None,
                    }),
                };
                let task = self.speeches.spawn(speech, "output validation", async move {
                    let hosted = callback.hosted.as_ref().expect("callback is pending");
                    // The voice run retains its live host value, not a persistence snapshot.
                    let (verdict, value, _) = crate::output_validation::project(
                        hosted.host.validate_output(&hosted.invocation, candidate).await,
                    );
                    callback.completed();
                    shared.report(GenEvent::OutputValidated { speech, validation, verdict, value });
                });
                if let Some(previous) = self
                    .host_callbacks
                    .insert((speech, HostCallbackId::OutputValidation(validation)), task)
                {
                    self.tasks.cancel(previous);
                }
            }
            VoiceEffect::CancelOutputValidation { speech, validation } => {
                if let Some(task) = self
                    .host_callbacks
                    .remove(&(speech, HostCallbackId::OutputValidation(validation)))
                {
                    self.tasks.cancel(task);
                }
            }
            VoiceEffect::RenderToolHandlingTemplate { rendering, template, hosted } => {
                let shared = Arc::clone(&self.shared);
                let invocation = shared.speeches.invocation(&hosted);
                let mut hosted = hosted;
                hosted.invocation = invocation;
                let mut callback = PendingHostCallback {
                    shared: Arc::clone(&shared),
                    hosted: Some(hosted),
                    on_cancel: Some(GenEvent::ToolHandlingTemplateRendered {
                        rendering,
                        text: None,
                    }),
                };
                let work = async move {
                    let hosted = callback.hosted.as_ref().expect("callback is pending");
                    let rendered = hosted.host.render_template(&hosted.invocation, template).await;
                    let text = rendered.unwrap_or_else(|e| {
                        log::error!(target: "livekit.agents", "a tool_handling template failed: {e}");
                        None
                    });
                    callback.completed();
                    shared.report(GenEvent::ToolHandlingTemplateRendered { rendering, text });
                };
                match rendering {
                    Rendering::Refusal { speech, call } => {
                        let task = self.speeches.spawn(speech, "tool refusal template", work);
                        self.host_callbacks
                            .insert((speech, HostCallbackId::ToolRefusal(call)), task);
                    }
                    Rendering::ToolReply { .. } => {
                        self.spawn("deferred tool reply template", work);
                    }
                }
            }
        }
    }

    /// Opens `provider`'s stream off the loop and reports it, or that the provider is gone, as
    /// `VoiceEvent::StreamOpened`. A close abandons an open still in flight: nothing would take it.
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
        let spawner = self.shared.spawner.clone();
        let tasks = self.tasks.clone();
        let mut closing = self.shared.closing.clone();
        self.spawn("stream open", async move {
            let opening = async move {
                match pending {
                    PendingOpen::Stt(stt) => match stt.stream(None, None).await {
                        Ok(Some(stream)) => {
                            let (sink, source) = stream.split();
                            Ok(Opened::new(
                                AudioChannel::Stt,
                                Arc::from(sink),
                                PendingLeg::Stt(source),
                                resources,
                                time,
                                spawner,
                            ))
                        }
                        Err(e) => Err(e),
                        Ok(None) => Err(ProviderError::Local(ApiError::new(
                            "the new agent's STT does not stream",
                        ))),
                    },
                    PendingOpen::Vad(vad) => vad.stream().await.map(|stream| {
                        let (sink, source) = stream.split();
                        Opened::new(
                            AudioChannel::Vad,
                            Arc::from(sink),
                            PendingLeg::Vad(source),
                            resources,
                            time,
                            spawner,
                        )
                    }),
                }
            };
            let outcome = tokio::select! {
                biased;
                _ = closing.wait_for(|c| *c) => return,
                outcome = opening => outcome,
            };
            if let Err(StreamOpened { outcome: Ok(stream), .. }) =
                tx.send_opened(StreamOpened { open, leg, outcome })
            {
                stream.retire(&tasks);
            }
        });
    }
}

/// A provider whose stream is being opened.
enum PendingOpen {
    Stt(Arc<dyn lk_agents_abi::Stt>),
    Vad(Arc<dyn lk_agents_abi::Vad>),
}
