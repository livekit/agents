//! What the provider tells the driver of the server's happenings: the response it was told of
//! and only that one's messages, words and audio; its end and metrics; the items the server
//! added; the person's speech and transcript; and the errors.
use std::time::Duration;

use lk_agents_abi::{
    AudioFrame, ChatItem, Ts,
    errors::ApiError,
    llm::realtime::{
        AudioDelta, Event, FunctionCallReady, GenerateReplyFailed, GenerationCreated,
        GenerationDone, GenerationStatus, InputSpeechStarted, InputSpeechStopped,
        InputTranscription, MessageDone, MessageModalities, MessageStarted, Modalities,
        RealtimeError, RealtimeMetrics, RealtimeModelError, RemoteItemAdded, TextDelta,
    },
    metrics::base::{AgentMetrics, Metadata, RealtimeModelMetrics, SttMetrics},
    usage::TokenUsage,
};

use super::{
    DELTA_SAMPLES, Generation, LABEL, MODEL, MODEL_BUILT_BEFORE, PROVIDER, SAMPLE_RATE, Socket,
    State, TRANSCRIPTION_MODEL,
    server::{Happening, function_call, message},
};
use crate::fakes::Control;
use crate::record::{Detail, Kind, Tag};

impl Socket {
    // --- what the session is told ----------------------------------------------------------

    pub(super) fn told(&self, st: &mut State, now: Ts, happening: Happening) {
        let audio = !self.setup.text_only;
        let active = |st: &State, r: u32| st.generation.as_ref().is_some_and(|g| g.r == r);
        match happening {
            Happening::Created { r } => {
                let (id, request) = {
                    let resp = &st.server.responses[r as usize];
                    (resp.id.clone(), resp.request.clone())
                };
                if let Some(req_id) = request.clone().filter(|req| st.cancelled.remove(req)) {
                    // Cancelled before creation: cancelled by id, its end told to nobody.
                    st.server.cancel(now, Some(&id));
                    let error = RealtimeError::new("generate_reply cancelled");
                    st.events.push_back(Event::GenerateReplyFailed(GenerateReplyFailed {
                        req_id,
                        error,
                    }));
                    return;
                }
                let slot =
                    self.registry.register(Kind::Generation, Detail::None, Control::Generation);
                st.generations.insert(r, slot);
                st.server.registered(r, slot.nth);
                st.server.told_created(r, now);
                st.close_generation(audio);
                let req_id = request.filter(|req| st.asked.remove(req));
                st.generation = Some(Generation {
                    r,
                    id: id.clone(),
                    created_at: now,
                    first_token: None,
                    messages: Vec::new(),
                });
                st.events.push_back(Event::GenerationCreated(GenerationCreated {
                    response_id: id,
                    req_id,
                }));
            }
            Happening::MessageAdded { r, previous } => {
                let item = st.server.responses[r as usize].item.clone().unwrap_or_default();
                let added =
                    stamped(message(&item, lk_agents_abi::Role::Assistant), self.clock.wall());
                if let Some(g) = st.generation.as_mut().filter(|g| g.r == r) {
                    let started = Event::MessageStarted(MessageStarted {
                        message_id: item.clone(),
                        response_id: Some(g.id.clone()),
                    });
                    st.events.push_back(started);
                    let settled = !audio;
                    g.messages.push((item.clone(), true, settled));
                    if settled {
                        let g_id = g.id.clone();
                        st.events.push_back(modalities(&item, &g_id, false));
                    }
                }
                remote_added(st, added, previous);
            }
            Happening::Part { r } => {
                if let Some(g) = st.generation.as_mut().filter(|g| g.r == r) {
                    let mut settle = Vec::new();
                    for (id, _, settled) in g.messages.iter_mut().filter(|m| !m.2) {
                        *settled = true;
                        settle.push(id.clone());
                    }
                    let g_id = g.id.clone();
                    for id in settle {
                        st.events.push_back(modalities(&id, &g_id, audio));
                    }
                }
            }
            Happening::Text { r, delta } => {
                if let Some(g) = st.generation.as_mut().filter(|g| g.r == r)
                    && let Some((id, ..)) = g.messages.last()
                {
                    g.first_token.get_or_insert(now);
                    let event = TextDelta {
                        message_id: id.clone(),
                        response_id: Some(g.id.clone()),
                        delta,
                        start_time: None,
                    };
                    st.events.push_back(Event::TextDelta(event));
                }
            }
            Happening::Audio { r, chunk } => {
                let source = st.generations.get(&r).map(|s| s.nth);
                if let Some(source) = source {
                    self.registry.record_generated(source, chunk);
                }
                if let Some(g) = st.generation.as_mut().filter(|g| g.r == r)
                    && let Some((id, ..)) = g.messages.last()
                {
                    g.first_token.get_or_insert(now);
                    let pcm = Tag::generated(source.unwrap_or(0), chunk)
                        .to_le_bytes()
                        .repeat(DELTA_SAMPLES);
                    let frame = AudioFrame::new(pcm, SAMPLE_RATE, 1, DELTA_SAMPLES as u32);
                    let event = AudioDelta {
                        message_id: id.clone(),
                        response_id: Some(g.id.clone()),
                        frame,
                    };
                    st.events.push_back(Event::AudioDelta(event));
                }
            }
            Happening::CallAdded { r } => {
                let resp = &st.server.responses[r as usize];
                let (fc, name) = resp.function_call.clone().expect("a call");
                let previous = resp.item.clone();
                let call = stamped(function_call(&fc, &name), self.clock.wall());
                remote_added(st, call, previous);
            }
            Happening::MessageDone { r } => {
                if let Some(g) = st.generation.as_mut().filter(|g| g.r == r) {
                    let g_id = g.id.clone();
                    let open: Vec<String> = g
                        .messages
                        .iter_mut()
                        .filter(|m| m.1)
                        .map(|m| {
                            m.1 = false;
                            m.0.clone()
                        })
                        .collect();
                    for id in open {
                        st.events.push_back(message_done(&id, &g_id));
                    }
                }
            }
            Happening::CallDone { r } => {
                if active(st, r) {
                    let (fc, name) =
                        st.server.responses[r as usize].function_call.clone().expect("a call");
                    let ChatItem::FunctionCall(call) =
                        stamped(function_call(&fc, &name), self.clock.wall())
                    else {
                        unreachable!()
                    };
                    st.events
                        .push_back(Event::FunctionCall(FunctionCallReady { call: Box::new(call) }));
                }
            }
            Happening::Done { r, status } => {
                if let Some(slot) = st.generations.remove(&r) {
                    self.registry.mark_closed(slot);
                }
                if !active(st, r) {
                    return;
                }
                let g = st.generation.take().expect("active");
                for (id, open, settled) in &g.messages {
                    if !settled {
                        st.events.push_back(modalities(id, &g.id, audio));
                    }
                    if *open {
                        st.events.push_back(message_done(id, &g.id));
                    }
                }
                let (input, output) = st.server.usage(r);
                let usage = TokenUsage {
                    input_tokens: input,
                    output_tokens: output,
                    input_text_tokens: Some(input),
                    output_audio_tokens: Some(30),
                    output_text_tokens: Some(10),
                    ..TokenUsage::default()
                };
                let metrics = self.response_metrics(&g, now, status, &usage);
                st.events.push_back(Event::GenerationDone(GenerationDone {
                    response_id: g.id.clone(),
                    status,
                    usage: Box::new(usage),
                }));
                st.events.push_back(metrics);
                if status == GenerationStatus::Failed {
                    let message =
                        format!("{PROVIDER} response failed with error type: server_error");
                    st.events.push_back(Event::Error(self.error(
                        message,
                        Some("server_error"),
                        true,
                    )));
                }
            }
            Happening::ItemAdded { item, previous } => {
                remote_added(st, stamped(*item, self.clock.wall()), previous);
            }
            Happening::SpeechStarted { item } => {
                st.speech_started.insert(item, self.wall_secs());
                st.events.push_back(Event::InputSpeechStarted(InputSpeechStarted {}));
            }
            Happening::SpeechStopped => {
                st.events.push_back(Event::InputSpeechStopped(InputSpeechStopped {
                    user_transcription_enabled: true,
                }));
            }
            Happening::Transcript { item, text, last } => {
                let turn_started_at = if last { st.speech_started.remove(&item) } else { None };
                if last {
                    st.transcripts.remove(&item);
                } else {
                    st.transcripts.insert(item.clone(), text.clone());
                }
                let words = text.split(' ').count() as u32;
                st.events.push_back(Event::InputTranscription(InputTranscription {
                    item_id: item,
                    transcript: text,
                    is_final: last,
                    confidence: None,
                    turn_started_at,
                }));
                if last {
                    st.events.push_back(self.transcription_metrics(words));
                }
            }
            Happening::TranscriptFailed { item } => {
                // What had come of it is final; with nothing, nothing is told.
                let turn_started_at = st.speech_started.remove(&item);
                if let Some(transcript) = st.transcripts.remove(&item) {
                    st.events.push_back(Event::InputTranscription(InputTranscription {
                        item_id: item,
                        transcript,
                        is_final: true,
                        confidence: None,
                        turn_started_at,
                    }));
                }
            }
            Happening::Error { code, message, request } => {
                if let Some(req_id) = request.filter(|req| st.asked.remove(req)) {
                    let error = RealtimeError::with_code(message, code.map(str::to_owned));
                    st.events.push_back(Event::GenerateReplyFailed(GenerateReplyFailed {
                        req_id,
                        error,
                    }));
                }
                let message = format!("{PROVIDER} returned an error: {message}");
                st.events.push_back(Event::Error(self.error(message, code, true)));
            }
            Happening::Fatal => {
                let message = format!(
                    "{PROVIDER} returned an error: You exceeded your current quota, please check your plan and billing details."
                );
                // A fallback adapter takes it as recoverable while it has a model to swap to.
                let error = self.error(message, Some("insufficient_quota"), self.setup.fallback);
                st.events.push_back(Event::Error(error));
                self.fatal(st);
            }
        }
    }

    // --- what the session reports ----------------------------------------------------------

    fn error(&self, message: String, code: Option<&str>, recoverable: bool) -> RealtimeModelError {
        RealtimeModelError {
            timestamp: self.now().saturating_add(MODEL_BUILT_BEFORE),
            label: LABEL.into(),
            message,
            recoverable,
            req_id: None,
            code: code.map(str::to_owned),
            provider_error: None,
        }
    }

    pub(super) fn emit_error(&self, st: &mut State, error: ApiError, recoverable: bool) {
        let event = RealtimeModelError {
            provider_error: Some(error.clone()),
            ..self.error(error.message, None, recoverable)
        };
        st.events.push_back(Event::Error(event));
    }

    fn metadata() -> Metadata {
        Metadata { model_name: Some(MODEL.into()), model_provider: Some("livekit".into()) }
    }

    /// `_report_connection_acquired`: the connection's timing, with no usage.
    pub(super) fn connection_metrics(&self, acquire: Duration) -> Event {
        let metrics = RealtimeModelMetrics {
            label: String::new(),
            request_id: String::new(),
            timestamp: self.wall_secs(),
            duration: 0.0,
            connection_duration: None,
            ttft: None,
            cancelled: false,
            tokens: TokenUsage::default(),
            tokens_per_second: 0.0,
            acquire_time: Some(acquire.as_secs_f64()),
            connection_reused: false,
            metadata: Some(Self::metadata()),
        };
        Event::Metrics(RealtimeMetrics { metrics: Box::new(AgentMetrics::RealtimeModel(metrics)) })
    }

    /// The finished response's metrics, stamped when it was created.
    fn response_metrics(
        &self,
        g: &Generation,
        now: Ts,
        status: GenerationStatus,
        usage: &TokenUsage,
    ) -> Event {
        let secs = |t: Ts| t.as_nanos() as f64 / 1e9;
        let duration = secs(now) - secs(g.created_at);
        let metrics = RealtimeModelMetrics {
            label: LABEL.into(),
            request_id: g.id.clone(),
            timestamp: self.wall_secs() - duration,
            duration,
            connection_duration: None,
            ttft: g.first_token.map(|f| secs(f) - secs(g.created_at)),
            cancelled: status == GenerationStatus::Cancelled,
            tokens_per_second: if duration > 0.0 {
                usage.output_tokens as f64 / duration
            } else {
                0.0
            },
            tokens: usage.clone(),
            acquire_time: None,
            connection_reused: false,
            metadata: Some(Self::metadata()),
        };
        Event::Metrics(RealtimeMetrics { metrics: Box::new(AgentMetrics::RealtimeModel(metrics)) })
    }

    /// The input transcription's usage, as `STTMetrics` named by the transcription model.
    fn transcription_metrics(&self, words: u32) -> Event {
        let output = words;
        let metrics = SttMetrics {
            label: LABEL.into(),
            request_id: String::new(),
            timestamp: self.wall_secs(),
            duration: None,
            audio_duration: None,
            streamed: true,
            tokens: TokenUsage {
                input_tokens: 20,
                output_tokens: output.into(),
                input_audio_tokens: Some(20),
                ..TokenUsage::default()
            },
            acquire_time: None,
            connection_reused: false,
            metadata: Some(Metadata {
                model_name: Some(TRANSCRIPTION_MODEL.into()),
                model_provider: Some("livekit".into()),
            }),
        };
        Event::Metrics(RealtimeMetrics { metrics: Box::new(AgentMetrics::Stt(metrics)) })
    }
}

/// The server added an item: the driver is told, and the session's mirror holds it after
/// `previous` from now on.
fn remote_added(st: &mut State, item: ChatItem, previous: Option<String>) {
    if !st.desired.iter().any(|i| i.id() == item.id()) {
        let at = previous
            .as_deref()
            .and_then(|p| st.desired.iter().position(|i| i.id() == p))
            .map_or(st.desired.len(), |i| i + 1);
        st.desired.insert(at, item.clone());
    }
    let added = RemoteItemAdded { previous_item_id: previous, item: Box::new(item) };
    st.events.push_back(Event::RemoteItemAdded(added));
}

pub(super) fn modalities(message_id: &str, response_id: &str, audio: bool) -> Event {
    Event::MessageModalities(MessageModalities {
        message_id: message_id.into(),
        response_id: Some(response_id.into()),
        modalities: if audio { Modalities::AudioAndText } else { Modalities::TextOnly },
    })
}

pub(super) fn cancelled(response_id: String) -> Event {
    Event::GenerationDone(GenerationDone {
        response_id,
        status: GenerationStatus::Cancelled,
        usage: Box::default(),
    })
}

pub(super) fn message_done(message_id: &str, response_id: &str) -> Event {
    Event::MessageDone(MessageDone {
        message_id: message_id.into(),
        response_id: Some(response_id.into()),
    })
}

/// An item's `created_at` is the session's wall clock when it heard of it.
fn stamped(mut item: ChatItem, wall: std::time::SystemTime) -> ChatItem {
    match &mut item {
        ChatItem::Message(m) => m.created_at = wall,
        ChatItem::FunctionCall(f) => f.created_at = wall,
        ChatItem::FunctionCallOutput(f) => f.created_at = wall,
        ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => {}
    }
    item
}
