//! The server's side: every `Incoming` event, and what it does to the generation, the transcript
//! and the mirrored conversation.

use super::{Generation, MessageState, RealtimeMachine, Step};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use lk_agents_abi::ChatItem;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ApiError;
use serde_json::Value;

use crate::realtime::dialect::is_fatal_error;
use crate::realtime::wire::{self, Incoming, TranscriptionCompleted};
use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, GenerateReplyFailed, GenerationCreated, GenerationDone, GenerationStatus,
    InputSpeechStopped, InputTranscription, MessageDone, MessageModalities, MessageStarted,
    Modalities, RealtimeError, TextDelta,
};

impl RealtimeMachine {
    pub fn incoming(&mut self, now: Ts, msg: Incoming) -> Step {
        if self.closing {
            // "draining after aclose; the generation is already discarded"
            return Step::default();
        }
        match msg {
            Incoming::SpeechStarted { item_id } => {
                let mut step = Step::default();
                // xAI: a new turn began while another item's transcript was held; finalise it now
                // or it is lost.
                if self.dialect.defer_transcript_finalization
                    && self.held.as_ref().is_some_and(|h| Some(&h.item_id) != item_id.as_ref())
                {
                    step.merge(self.flush_held_transcript(now));
                }
                if let Some(id) = item_id {
                    // xAI re-announces onset for an item; keep the original start time.
                    self.speech_started_at.entry(id).or_insert(now);
                }
                step.events.push(Event::InputSpeechStarted(lk_agents_abi::llm::realtime::InputSpeechStarted {}));
                step
            }

            Incoming::SpeechStopped => Step::event(Event::InputSpeechStopped(InputSpeechStopped {
                user_transcription_enabled: self.opts.transcription_model.is_some(),
            })),

            Incoming::ResponseCreated { response_id, client_event_id } => {
                self.on_response_created(response_id, client_event_id)
            }

            Incoming::OutputItemAdded { item_id, item_type } => {
                if item_type != "message" {
                    return Step::default();
                }
                let Generation::Active { response_id, messages, order, .. } = &mut self.generation else {
                    return Step::default();
                };
                let response_id = Some(response_id.clone());
                messages.insert(item_id.clone(), MessageState { open: true, ..Default::default() });
                order.push(item_id.clone());
                let mut step = Step::event(Event::MessageStarted(MessageStarted {
                    message_id: item_id.clone(),
                    response_id: response_id.clone(),
                }));
                if !self.caps.audio_output {
                    // Text-only: settle modalities now, no audio can arrive.
                    if let Generation::Active { messages, .. } = &mut self.generation
                        && let Some(m) = messages.get_mut(&item_id)
                    {
                        m.modalities = Some(Modalities::TextOnly);
                    }
                    step.events.push(Event::MessageModalities (MessageModalities {
                        message_id: item_id,
                        response_id,
                        modalities: Modalities::TextOnly,
                    }));
                }
                step
            }

            Incoming::ContentPartAdded { item_id, part_type } => {
                let modalities = if part_type == "text" {
                    Modalities::TextOnly
                } else {
                    Modalities::AudioAndText
                };
                self.settle_modalities(&item_id, modalities)
            }

            Incoming::ItemAdded { previous_item_id, item } => {
                self.on_item_added(now, previous_item_id, item)
            }

            Incoming::ItemDeleted { mut item_id } => {
                // xAI acks a delete with `item_id: ""`: resolve it against the oldest in-flight
                // delete, or `update_chat_ctx` burns its 5 s timeout.
                if item_id.is_empty() && self.dialect.repair_empty_delete_item_id {
                    let oldest = self
                        .ctx_update
                        .as_ref()
                        .and_then(|c| c.awaiting_delete.iter().min().cloned());
                    match oldest {
                        Some(id) => item_id = id,
                        None => return Step::default(),
                    }
                }
                self.transcript_acc.remove(&item_id);
                self.speech_started_at.remove(&item_id);
                let _ = self.remote.delete(&item_id);
                self.settle_ctx_item(true, &item_id)
            }

            Incoming::TranscriptionDelta { item_id, content_index, delta } => {
                if delta.is_empty() {
                    return Step::default();
                }
                let acc = self.transcript_acc.entry(item_id.clone()).or_default();
                let text = acc.entry(content_index).or_default();
                text.push_str(&delta);
                let transcript = text.clone();
                Step::event(Event::InputTranscription(InputTranscription {
                    item_id,
                    transcript,
                    is_final: false,
                    confidence: None,
                    turn_started_at: None,
                }))
            }

            Incoming::TranscriptionUpdated { item_id, transcript } => {
                // xAI-only, and never final: it is the whole transcript so far, re-sent.
                if item_id.is_empty() || transcript.is_empty() {
                    return Step::default();
                }
                Step::event(Event::InputTranscription(InputTranscription {
                    item_id,
                    transcript,
                    is_final: false,
                    confidence: None,
                    turn_started_at: None,
                }))
            }

            Incoming::TranscriptionCompleted(t) => self.on_transcription_completed(now, t),

            Incoming::TranscriptionFailed { item_id, content_index } => {
                // Close any open partial stream so a consumer waiting for `is_final` does not
                // hang. With nothing accumulated nothing is emitted.
                let partial = self.take_accumulated(&item_id, content_index);
                let turn_started_at = self.speech_started_at.remove(&item_id);
                match partial {
                    None => Step::default(),
                    Some(transcript) => {
                        Step::event(Event::InputTranscription(InputTranscription {
                            item_id,
                            transcript,
                            is_final: true,
                            confidence: None,
                            turn_started_at,
                        }))
                    }
                }
            }

            Incoming::TextDelta { item_id, delta } => {
                let mut step = self.on_output(now, &item_id);
                let response_id = self.response_id();
                if let Generation::Active { messages, .. } = &mut self.generation {
                    let Some(m) = messages.get_mut(&item_id) else { return step };
                    m.transcript.push_str(&delta);
                }
                step.events.push(Event::TextDelta(TextDelta {
                    message_id: item_id,
                    response_id,
                    delta,
                    start_time: None,
                }));
                step
            }

            Incoming::AudioTranscriptDelta { item_id, delta, start_time } => {
                if matches!(self.generation, Generation::Discarded | Generation::None) {
                    return Step::default();
                }
                if let Generation::Active { messages, .. } = &mut self.generation
                    && let Some(m) = messages.get_mut(&item_id)
                {
                    m.transcript.push_str(&delta);
                }
                let response_id = self.response_id();
                Step::event(Event::TextDelta(TextDelta { message_id: item_id, response_id, delta, start_time }))
            }

            Incoming::AudioDelta { item_id, pcm } => {
                let mut step = self.on_output(now, &item_id);
                step.merge(self.settle_modalities(&item_id, Modalities::AudioAndText));
                if matches!(self.generation, Generation::Active { .. }) {
                    step.events.push(Event::AudioDelta (AudioDelta {
                        message_id: item_id,
                        response_id: self.response_id(),
                        frame: wire::audio_frame(pcm),
                    }));
                }
                step
            }

            Incoming::OutputItemDone { item } => self.on_output_item_done(item),

            Incoming::ResponseDone(done) => self.on_response_done(now, done),

            Incoming::Error(err) => self.on_error(now, err),

            // Log-only in the reference.
            Incoming::IdleTimeoutTriggered | Incoming::SessionCreated { .. } => Step::default(),
            Incoming::Unhandled { .. } => Step::default(),
        }
    }

    fn on_response_created(
        &mut self,
        response_id: String,
        client_event_id: Option<String>,
    ) -> Step {
        let mut step = Step::default();

        // xAI: `force_message`'s `response.created` has no `client_event_id`; the oldest say on
        // the wire claims it (FIFO, the server's answer order).
        let client_event_id = match client_event_id {
            Some(id) => Some(id),
            None if self.dialect.force_message_say => self.say_tags.pop_front(),
            None => None,
        };

        if self.dialect.discard_silent_responses {
            step.merge(self.discard_abandoned());
        }

        if let Some(id) = &client_event_id
            && self.discarded.remove(id)
        {
            // Abandoned before creation: cancel it *by id* (an untargeted cancel hits whatever is
            // in flight) and skip its trailing deltas.
            self.stale_say.remove(id);
            self.say_tags.retain(|t| t != id);
            step.frames.push(wire::response_cancel(Some(&response_id)));
            step.merge(self.close_generation());
            self.generation = Generation::Discarded;
            return step;
        }

        step.merge(self.close_generation());
        self.response_spoke = false;
        self.generation = Generation::Active {
            response_id: response_id.clone(),
            messages: HashMap::new(),
            order: Vec::new(),
        };

        // A still-pending `req_id` asked for this turn; a settled one timed out, and the turn is
        // surfaced but not as user-initiated.
        let req_id = client_event_id.filter(|id| self.pending_replies.remove(id).is_some());
        if let Some(id) = &req_id {
            self.stale_say.remove(id);
        }
        step.events.push(Event::GenerationCreated(GenerationCreated { response_id, req_id }));
        step
    }

    fn on_item_added(&mut self, now: Ts, previous_item_id: Option<String>, item: Value) -> Step {
        let mut previous_item_id = previous_item_id;
        // "the server can anchor to an item it just deleted; the item belongs at the tail"
        if let Some(prev) = &previous_item_id
            && self.remote.get(prev).is_none()
        {
            previous_item_id = self.remote.tail_id().map(str::to_owned);
        }
        // xAI: a null anchor means the tail, not the head.
        if previous_item_id.is_none() && self.dialect.anchor_root_at_tail {
            previous_item_id = self.remote.tail_id().map(str::to_owned);
        }

        let mut step = Step::default();
        let created_at = wall_from(now);
        match wire::from_wire_item(&item, created_at) {
            Ok(lk_item) => {
                let id = lk_item.id().to_owned();
                if self.remote.insert(previous_item_id.as_deref(), lk_item.clone()).is_ok() {
                    step.events
                        .push(Event::RemoteItemAdded(lk_agents_abi::llm::realtime::RemoteItemAdded { previous_item_id, item: Box::new(lk_item) }));
                }
                step.merge(self.settle_ctx_item(false, &id));
            }
            Err(_) => return step,
        }
        step
    }

    fn on_transcription_completed(&mut self, now: Ts, t: TranscriptionCompleted) -> Step {
        if self.dialect.defer_transcript_finalization {
            let mut step = Step::default();
            // Only hold a settled transcript; `in_progress` ones will be revised.
            if !t.in_progress {
                if self.held.as_ref().is_some_and(|h| h.item_id != t.item_id) {
                    step.merge(self.flush_held_transcript(now));
                }
                self.held = Some(t.clone());
            }
            // Always interim. The final one comes out of `flush_held_transcript`.
            step.events.push(Event::InputTranscription(InputTranscription {
                item_id: t.item_id,
                transcript: t.transcript,
                is_final: false,
                confidence: None,
                turn_started_at: None,
            }));
            return step;
        }
        self.finalize_transcript(t)
    }

    /// "The agent has started answering": xAI finalises the user's transcript here, the last
    /// moment it can be revised.
    fn on_output(&mut self, now: Ts, _item_id: &str) -> Step {
        if matches!(self.generation, Generation::Discarded | Generation::None) {
            return Step::default();
        }
        self.response_spoke = true;
        if self.dialect.defer_transcript_finalization {
            return self.flush_held_transcript(now);
        }
        Step::default()
    }

    fn on_output_item_done(&mut self, item: Value) -> Step {
        if !matches!(self.generation, Generation::Active { .. }) {
            return Step::default();
        }
        let ty = item.get("type").and_then(Value::as_str).unwrap_or("");
        if ty == "function_call" {
            let name = item.get("name").and_then(Value::as_str).unwrap_or("");
            // xAI emits `function_call` items for its own server-side tools; they have no local
            // handler.
            if self.dialect.ignore_unknown_function_calls
                && !self.tools.iter().any(|t| t.name == name)
            {
                return Step::default();
            }
            let s = |k: &str| item.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
            let call = Box::new(lk_agents_abi::FunctionCall {
                id: s("id"),
                call_id: s("call_id"),
                arguments: s("arguments"),
                name: s("name"),
                created_at: SystemTime::UNIX_EPOCH,
                extra: Default::default(),
                group_id: None,
            });
            return Step::event(Event::FunctionCall(lk_agents_abi::llm::realtime::FunctionCallReady { call }));
        }
        if ty != "message" {
            return Step::default();
        }
        let Some(item_id) = item.get("id").and_then(Value::as_str) else { return Step::default() };
        let item_id = item_id.to_owned();
        let mut step = Step::default();
        let default_modalities = self.default_modalities();
        let response_id = self.response_id();
        if let Generation::Active { messages, .. } = &mut self.generation
            && let Some(m) = messages.get_mut(&item_id)
        {
            m.open = false;
            if m.modalities.is_none() {
                m.modalities = Some(default_modalities);
                step.events.push(Event::MessageModalities (MessageModalities {
                    message_id: item_id.clone(),
                    response_id: response_id.clone(),
                    modalities: default_modalities,
                }));
            }
        }
        step.events.push(Event::MessageDone(MessageDone { message_id: item_id, response_id }));
        step
    }

    fn on_response_done(&mut self, now: Ts, done: crate::realtime::wire::ResponseDone) -> Step {
        if matches!(self.generation, Generation::Discarded) {
            self.generation = Generation::None;
            return Step::default();
        }
        // "OpenAI has a race condition where we could receive response.done without any previous
        // response.created (this happens generally during interruption)".
        let Generation::Active { response_id, messages, order } =
            std::mem::replace(&mut self.generation, Generation::None)
        else {
            return Step::default();
        };

        let mut step = Step::default();
        let default_modalities = self.default_modalities();
        for id in &order {
            let Some(m) = messages.get(id) else { continue };
            if m.modalities.is_none() {
                step.events.push(Event::MessageModalities (MessageModalities {
                    message_id: id.clone(),
                    response_id: Some(response_id.clone()),
                    modalities: default_modalities,
                }));
            }
            // The agent's answer joins the mirror, and from there the next turn's context.
            if let Some(ChatItem::Message(remote)) = self.remote.get_mut(id) {
                remote.content.push(lk_agents_abi::ChatContent::Text(m.transcript.clone()));
            }
            if m.open {
                step.events.push(Event::MessageDone(MessageDone { message_id: id.clone(), response_id: Some(response_id.clone()) }));
            }
        }

        step.events.push(Event::GenerationDone (GenerationDone {
            response_id: done.response_id.clone(),
            status: done.status,
            usage: done.usage,
        }));
        debug_assert!(response_id == done.response_id || done.response_id.is_empty());

        if done.status == GenerationStatus::Failed {
            // "failures are largely undocumented by openai, so we assume optimistically
            // recoverable unless the code is a known-fatal one".
            let fatal = is_fatal_error(done.error_code.as_deref(), done.error_type.as_deref());
            let message = format!(
                "{} response failed with error type: {}",
                self.dialect.provider_label,
                done.error_type.as_deref().unwrap_or("unknown")
            );
            if fatal {
                step.fatal = Some(ApiError::new(message.clone()).with_retryable(false));
            }
            step.events.push(Event::Error(self.error(now, message, !fatal)));
        }
        step
    }

    fn on_error(&mut self, now: Ts, err: crate::realtime::wire::ServerError) -> Step {
        let mut step = Step::default();
        let fatal = is_fatal_error(err.code.as_deref(), err.type_.as_deref());

        if let Some(event_id) = &err.event_id {
            // A rejected item event gets no `deleted`/`added` reply; settle it here or
            // `update_chat_ctx` stalls.
            if let Some(update) = &mut self.ctx_update
                && let Some((is_delete, item_id)) = update.by_event_id.remove(event_id)
            {
                // A duplicate id means the item is already there: success.
                if err.code.as_deref() != Some("item_create_duplicate_item_id") {
                    update.rejected.push(err.message.clone());
                }
                step.merge(self.settle_ctx_item(is_delete, &item_id));
                if !fatal {
                    return step;
                }
            } else if self.pending_replies.remove(event_id).is_some() {
                // A rejected `response.create` never gets `response.created`; fail it now rather
                // than at the 10 s timeout.
                self.say_tags.retain(|t| t != event_id);
                step.events.push(Event::GenerateReplyFailed (GenerateReplyFailed {
                    req_id: event_id.clone(),
                    error: RealtimeError::with_code(err.message.clone(), err.code.clone()),
                }));
            }
        }

        // "Cancellation failed": a cancel raced a finished response. Nothing went wrong.
        if err.message.starts_with("Cancellation failed") {
            return step;
        }
        // The server VAD commits each segment itself, so our commit lands on an emptied buffer.
        if err.code.as_deref() == Some("input_audio_buffer_commit_empty")
            && self.opts.turn_detection.is_some()
        {
            return step;
        }

        let message = format!("{} returned an error", self.dialect.provider_label);
        if fatal {
            step.fatal = Some(ApiError::new(message.clone()).with_retryable(false));
        }
        step.events.push(Event::Error(self.error(now, message, !fatal)));
        step
    }
}

/// `ChatItem.created_at` from the monotonic clock anchored at the epoch: the stamps are only
/// compared with each other. The driver overwrites them from `Clock::wall` if needed.
fn wall_from(now: Ts) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_nanos(now.as_nanos())
}
