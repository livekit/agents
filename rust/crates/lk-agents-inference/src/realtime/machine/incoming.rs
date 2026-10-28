//! The server's side: every `Incoming` event, and what it does to the generation, the transcript
//! and the mirrored conversation.

use super::{Generation, MessageState, Outgoing, RealtimeMachine, SPOKEN_REMEMBERED, Step};
use std::collections::BTreeMap;

use lk_agents_abi::ChatItem;
use lk_agents_abi::Ts;

use crate::realtime::protocol::{Incoming, TranscriptionCompleted};
use lk_agents_abi::llm::realtime::{
    AudioDelta, Event, GenerateReplyFailed, GenerationCreated, GenerationDone, GenerationStatus,
    InputSpeechStopped, InputTranscription, MessageDone, MessageModalities, MessageStarted,
    Modalities, RealtimeError, RealtimeModelError, TextDelta,
};

impl RealtimeMachine {
    pub(super) fn incoming(&mut self, now: Ts, msg: Incoming) -> Step {
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
                self.user_speaking = true;
                step.events.push(Event::InputSpeechStarted(
                    lk_agents_abi::llm::realtime::InputSpeechStarted {},
                ));
                step
            }

            Incoming::SpeechStopped => {
                self.user_speaking = false;
                Step::event(Event::InputSpeechStopped(InputSpeechStopped {
                    user_transcription_enabled: self.opts.transcription.is_some(),
                }))
            }

            Incoming::ResponseCreated { response_id, client_event_id } => {
                let step = self.on_response_created(response_id, client_event_id);
                if matches!(self.generation, Generation::Active { .. }) {
                    self.response_created_at = Some(now);
                    self.first_token_at = None;
                }
                step
            }

            Incoming::OutputItemAdded { item_id, item_type } => {
                if item_type != "message" {
                    return Step::default();
                }
                let Generation::Active { response_id, messages, order, .. } = &mut self.generation
                else {
                    return Step::default();
                };
                let response_id = Some(response_id.clone());
                messages.insert(item_id.clone(), MessageState { open: true, ..Default::default() });
                order.push(item_id.clone());
                if self.spoken.len() == SPOKEN_REMEMBERED {
                    self.spoken.pop_front();
                }
                self.spoken.push_back(item_id.clone());
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
                    step.events.push(Event::MessageModalities(MessageModalities {
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
                    let oldest =
                        self.ctx_update.as_ref().and_then(|c| c.awaiting_delete.first().cloned());
                    match oldest {
                        Some(id) => item_id = id,
                        None => return Step::default(),
                    }
                }
                self.transcript_acc.remove(&item_id);
                self.speech_started_at.remove(&item_id);
                let awaiting = self
                    .ctx_update
                    .as_ref()
                    .is_some_and(|update| update.awaiting_delete.contains(&item_id));
                if !awaiting && let Ok(item) = self.desired.delete(&item_id) {
                    self.discard_context_item(item);
                }
                if let Ok(item) = self.remote.delete(&item_id) {
                    self.discard_context_item(item);
                }
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
                let turn_started_at =
                    self.speech_started_at.remove(&item_id).map(|t| self.wall_secs(t));
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
                if self.stale(&item_id) {
                    return Step::default();
                }
                self.first_token(now);
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
                if matches!(self.generation, Generation::Discarded | Generation::None)
                    || self.stale(&item_id)
                {
                    return Step::default();
                }
                if let Generation::Active { messages, .. } = &mut self.generation
                    && let Some(m) = messages.get_mut(&item_id)
                {
                    m.transcript.push_str(&delta);
                }
                let response_id = self.response_id();
                Step::event(Event::TextDelta(TextDelta {
                    message_id: item_id,
                    response_id,
                    delta,
                    start_time,
                }))
            }

            Incoming::AudioDelta { item_id, frame } => {
                if self.stale(&item_id) {
                    return Step::default();
                }
                self.first_token(now);
                let mut step = self.on_output(now, &item_id);
                step.merge(self.settle_modalities(&item_id, Modalities::AudioAndText));
                if matches!(self.generation, Generation::Active { .. }) {
                    step.events.push(Event::AudioDelta(AudioDelta {
                        message_id: item_id,
                        response_id: self.response_id(),
                        frame,
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
            step.frames.push(Outgoing::ResponseCancel { response_id: Some(response_id.clone()) });
            if matches!(self.generation, Generation::Active { .. }) {
                // Created late, behind a reply the server opened meanwhile, which still generates:
                // that one stays, and the discarded one's end is skipped.
                self.discarded_responses.insert(response_id);
                if self.discarded_responses.len() >= super::DISCARDED_RESPONSES {
                    step.recycle = true;
                }
            } else {
                step.merge(self.close_generation());
                self.generation = Generation::Discarded;
            }
            // Created and cancelled: the server takes the next reply.
            if !self.settled_discards.remove(id) {
                let error = RealtimeError::new("generate_reply cancelled");
                step.events.push(Event::GenerateReplyFailed(GenerateReplyFailed {
                    req_id: id.clone(),
                    error,
                }));
            }
            return step;
        }

        if let Generation::Active { messages, .. } = &self.generation {
            self.closed_messages = messages.keys().cloned().collect();
        }
        step.merge(self.close_generation());
        self.response_spoke = false;
        self.generation = Generation::Active {
            response_id: response_id.clone(),
            messages: BTreeMap::new(),
            order: Vec::new(),
        };

        // A still-pending `req_id` asked for this turn; a settled one timed out, and the turn is
        // surfaced but not as user-initiated.
        let req_id = client_event_id.filter(|id| self.pending_replies.remove(id));
        if let Some(id) = &req_id {
            self.stale_say.remove(id);
            self.reply_deadlines.remove(id);
            self.reply_frames.remove(id);
            self.say_frames.remove(id);
            self.forget_deferred_reply(id);
        }
        step.events.push(Event::GenerationCreated(GenerationCreated { response_id, req_id }));
        step
    }

    fn on_item_added(
        &mut self,
        _now: Ts,
        mut previous_item_id: Option<String>,
        item: ChatItem,
    ) -> Step {
        if let Some(prev) = &previous_item_id
            && self.remote.get(prev).is_none()
        {
            previous_item_id = self.remote.tail_id().map(str::to_owned);
        }
        if previous_item_id.is_none() && self.dialect.anchor_root_at_tail {
            previous_item_id = self.remote.tail_id().map(str::to_owned);
        }
        let mut step = Step::default();
        let id = item.id().to_owned();
        let awaiting =
            self.ctx_update.as_ref().is_some_and(|update| update.awaiting_create.contains(&id));
        let added = self.remote.insert(previous_item_id.as_deref(), item.clone()).is_ok();
        if added {
            step.events.push(Event::RemoteItemAdded(
                lk_agents_abi::llm::realtime::RemoteItemAdded {
                    previous_item_id,
                    item: Box::new(item),
                },
            ));
        }
        if added && !awaiting && self.desired.get(&id).is_none() {
            step.merge(self.place_server_item(_now, &id));
        }
        step.merge(self.settle_ctx_item(false, &id));
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
        let metrics = self.transcription_metrics(now, &t);
        let mut step = self.finalize_transcript(t);
        step.events.extend(metrics);
        step
    }

    /// `_first_token_timestamp`: the response's first text or audio.
    fn first_token(&mut self, now: Ts) {
        if self.first_token_at.is_none() && matches!(self.generation, Generation::Active { .. }) {
            self.first_token_at = Some(now);
        }
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

    fn on_output_item_done(&mut self, item: ChatItem) -> Step {
        if let ChatItem::FunctionCall(call) = &item
            && let Some(ChatItem::FunctionCall(remote)) = self.remote.get_mut(&call.id)
        {
            remote.arguments = call.arguments.clone();
            if let Some(ChatItem::FunctionCall(desired)) = self.desired.get_mut(&call.id) {
                desired.arguments = call.arguments.clone();
            }
        }
        if !matches!(self.generation, Generation::Active { .. }) {
            return Step::default();
        }
        if let ChatItem::FunctionCall(call) = item {
            if self.dialect.ignore_unknown_function_calls
                && !self.tools.iter().any(|t| t.name == call.name)
            {
                return Step::default();
            }
            return Step::event(Event::FunctionCall(
                lk_agents_abi::llm::realtime::FunctionCallReady { call: Box::new(call) },
            ));
        }
        let ChatItem::Message(message) = item else {
            return Step::default();
        };
        let item_id = message.id;
        let mut step = Step::default();
        let default_modalities = self.default_modalities();
        let response_id = self.response_id();
        if let Generation::Active { messages, .. } = &mut self.generation
            && let Some(m) = messages.get_mut(&item_id)
        {
            m.open = false;
            if m.modalities.is_none() {
                m.modalities = Some(default_modalities);
                step.events.push(Event::MessageModalities(MessageModalities {
                    message_id: item_id.clone(),
                    response_id: response_id.clone(),
                    modalities: default_modalities,
                }));
            }
        }
        step.events.push(Event::MessageDone(MessageDone { message_id: item_id, response_id }));
        step
    }

    /// A delta of a message of the generation the open one replaced: the cancelled response's
    /// tail, which the server still streams after the next response was created (a cancel and a
    /// new reply in one breath, `corpus/probes/probe-cancel_then_create.json`).
    fn stale(&self, item_id: &str) -> bool {
        self.closed_messages.contains(item_id)
    }

    fn on_response_done(&mut self, now: Ts, done: crate::realtime::protocol::ResponseDone) -> Step {
        if self.discarded_responses.remove(&done.response_id) {
            return Step::default();
        }
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
        // A cancelled response's `response.done` can come after the next response's
        // `response.created` (a cancel and a new reply in one breath: the server frees the
        // conversation at the cancel). That response was closed when the next one opened; its end
        // must not close the one now generating (nor take its metrics' clock).
        if !done.response_id.is_empty() && done.response_id != response_id {
            self.generation = Generation::Active { response_id, messages, order };
            return Step::default();
        }

        let mut step = Step::default();
        let default_modalities = self.default_modalities();
        for id in &order {
            let Some(m) = messages.get(id) else { continue };
            if m.modalities.is_none() {
                step.events.push(Event::MessageModalities(MessageModalities {
                    message_id: id.clone(),
                    response_id: Some(response_id.clone()),
                    modalities: default_modalities,
                }));
            }
            // The agent's answer joins the mirror, and from there the next turn's context.
            if let Some(ChatItem::Message(remote)) = self.remote.get_mut(id) {
                remote.content.push(lk_agents_abi::ChatContent::Text(m.transcript.clone()));
                if let Some(ChatItem::Message(desired)) = self.desired.get_mut(id) {
                    desired.content.push(lk_agents_abi::ChatContent::Text(m.transcript.clone()));
                }
            }
            if m.open {
                step.events.push(Event::MessageDone(MessageDone {
                    message_id: id.clone(),
                    response_id: Some(response_id.clone()),
                }));
            }
        }

        step.events.push(Event::GenerationDone(GenerationDone {
            response_id: done.response_id.clone(),
            status: done.status,
            usage: done.usage,
        }));
        if let Some(metrics) =
            self.response_metrics(now, &done.response_id, done.status, &done.usage)
        {
            step.events.push(metrics);
        }

        if done.status == GenerationStatus::Failed {
            // "failures are largely undocumented by openai, so we assume optimistically
            // recoverable unless the code is a known-fatal one".
            let fatal =
                self.dialect.is_fatal(done.error_code.as_deref(), done.error_type.as_deref());
            let message = format!(
                "{} response failed with error type: {}",
                self.dialect.provider_label,
                done.error_type.as_deref().unwrap_or("unknown")
            );
            let error = RealtimeModelError {
                code: done.error_code.clone(),
                ..self.error(now, message, !fatal)
            };
            log::error!(target: "livekit.agents", code = error.code.as_deref().unwrap_or(""); "{}", error.message);
            // Raised in the reference when fatal: the session stops, and reports it once.
            if fatal { step.fatal = Some(error) } else { step.events.push(Event::Error(error)) }
        }
        step
    }

    fn on_error(&mut self, now: Ts, err: crate::realtime::protocol::ServerError) -> Step {
        let mut step = Step::default();
        let fatal = self.dialect.is_fatal(err.code.as_deref(), err.type_.as_deref());

        if let Some(event_id) = &err.event_id {
            // A rejected item event gets no `deleted`/`added` reply; settle it here or
            // `update_chat_ctx` stalls.
            if let Some(update) = &mut self.ctx_update
                && let Some((is_delete, item_id, sequence)) = update.by_event_id.remove(event_id)
            {
                if err.code.as_deref() != Some("item_create_duplicate_item_id") {
                    self.reject_ctx_item(is_delete, &item_id, sequence);
                }
                let confirmed = if !is_delete
                    && err.code.as_deref() == Some("item_create_duplicate_item_id")
                    && self.ctx_event_is_latest(&item_id, sequence)
                    && self.remote.get(&item_id).is_none()
                {
                    self.desired.get(&item_id).map(|item| {
                        (self.desired.previous_id(&item_id).map(str::to_owned), item.clone())
                    })
                } else {
                    None
                };
                if let Some((previous, item)) = confirmed {
                    // A duplicate rejection confirms existence even when the original added
                    // acknowledgement was lost. Install it before settling the desired state.
                    if self.remote.insert(previous.as_deref(), item.clone()).is_ok() {
                        step.events.push(Event::RemoteItemAdded(
                            lk_agents_abi::llm::realtime::RemoteItemAdded {
                                previous_item_id: previous,
                                item: Box::new(item),
                            },
                        ));
                    }
                }
                step.merge(self.settle_ctx_event(is_delete, &item_id, Some(sequence)));
                if !fatal {
                    return step;
                }
            } else if self.pending_replies.remove(event_id) {
                self.reply_deadlines.remove(event_id);
                self.reply_frames.remove(event_id);
                self.say_frames.remove(event_id);
                self.forget_deferred_reply(event_id);
                // A rejected `response.create` never gets `response.created`; fail it now rather
                // than at the 10 s timeout.
                self.say_tags.retain(|t| t != event_id);
                step.events.push(Event::GenerateReplyFailed(GenerateReplyFailed {
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

        // The server's own words and code: "returned an error" alone says nothing of a bad voice,
        // a tool schema or the quota.
        let message = format!("{} returned an error: {}", self.dialect.provider_label, err.message);
        let error =
            RealtimeModelError { code: err.code.clone(), ..self.error(now, message, !fatal) };
        log::error!(
            target: "livekit.agents",
            code = err.code.as_deref().unwrap_or(""),
            error_type = err.type_.as_deref().unwrap_or("");
            "{}", error.message
        );
        // Raised in the reference when fatal: the session stops, and reports it once.
        if fatal {
            step.fatal = Some(error)
        } else {
            step.events.push(Event::Error(error))
        }
        step
    }
}
