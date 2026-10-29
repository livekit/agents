//! The server's side of the client's frames: what each `Outgoing` does to the conversation,
//! the buffer and the responses, and what the real server refuses.
use lk_agents_abi::{ChatItem, Maybe, Ts};
use lk_agents_inference::realtime::{Incoming, Outgoing, SessionOptions, protocol::ServerError};

use super::{Breach, Hooks, Item, Response, SAMPLE_RATE, Server, later, unhandled};

/// The client event type a frame goes out as.
pub fn wire_name(frame: &Outgoing) -> &'static str {
    match frame {
        Outgoing::SessionUpdate { .. }
        | Outgoing::InstructionsUpdate { .. }
        | Outgoing::OptionsUpdate { .. }
        | Outgoing::ToolsUpdate { .. } => "session.update",
        Outgoing::ItemCreate { .. } | Outgoing::ForceMessage { .. } => "conversation.item.create",
        Outgoing::ItemDelete { .. } => "conversation.item.delete",
        Outgoing::ItemTruncate { .. } => "conversation.item.truncate",
        Outgoing::AudioAppend { .. } => "input_audio_buffer.append",
        Outgoing::AudioCommit => "input_audio_buffer.commit",
        Outgoing::AudioClear => "input_audio_buffer.clear",
        Outgoing::ResponseCreate { .. } => "response.create",
        Outgoing::ResponseCancel { .. } => "response.cancel",
    }
}

/// The `event_id` a frame goes out with: a response request's is its correlation id.
pub fn event_id_of(frame: &Outgoing) -> Option<&str> {
    match frame {
        Outgoing::SessionUpdate { event_id, .. }
        | Outgoing::InstructionsUpdate { event_id, .. }
        | Outgoing::OptionsUpdate { event_id, .. }
        | Outgoing::ToolsUpdate { event_id, .. }
        | Outgoing::ItemCreate { event_id, .. }
        | Outgoing::ItemDelete { event_id, .. } => Some(event_id),
        Outgoing::ResponseCreate { req_id, .. } | Outgoing::ForceMessage { req_id, .. } => {
            Some(req_id)
        }
        Outgoing::ItemTruncate { .. }
        | Outgoing::AudioAppend { .. }
        | Outgoing::AudioCommit
        | Outgoing::AudioClear
        | Outgoing::ResponseCancel { .. } => None,
    }
}

impl Server {
    /// One client frame, arrived at `now`; what it breached goes to the hooks.
    pub fn client(&mut self, now: Ts, frame: &Outgoing, hooks: &mut dyn Hooks) {
        self.breathe(now);
        self.receive(now, frame, hooks);
        self.report(hooks);
    }

    /// The client's breath before `now` is over: an output that arrived after a reply asked in
    /// it, and no reply was asked after it, was read by none.
    pub(super) fn breathe(&mut self, now: Ts) {
        self.unanswered.retain(|(.., at)| *at == now);
        let (over, open): (Vec<_>, Vec<_>) = self.early.drain(..).partition(|(.., at)| *at < now);
        self.early = open;
        for (response, call_id, _) in over {
            let request =
                self.responses.iter().find(|r| r.id == response).and_then(|r| r.request.clone());
            self.breaches.push(Breach::AskedBeforeOutput { response, request, call_id });
        }
    }

    pub(super) fn report(&mut self, hooks: &mut dyn Hooks) {
        for breach in self.breaches.drain(..) {
            hooks.breach(breach);
        }
    }

    fn receive(&mut self, now: Ts, frame: &Outgoing, hooks: &mut dyn Hooks) {
        match frame {
            Outgoing::SessionUpdate { opts, .. } => {
                if self.refuses_voice(Some(&opts.voice)) {
                    return self.refuse_voice(now, frame);
                }
                self.configure(opts);
                self.updated(now);
            }
            Outgoing::OptionsUpdate { patch, .. } => {
                let voice = match &patch.voice {
                    Maybe::Given(voice) => Some(voice),
                    Maybe::NotGiven => None,
                };
                if self.refuses_voice(voice) {
                    return self.refuse_voice(now, frame);
                }
                if let Maybe::Given(td) = &patch.turn_detection {
                    self.config.turn_detection(td.as_ref());
                }
                if let Maybe::Given(t) = &patch.transcription {
                    self.config.transcription = t.is_some();
                }
                if let Maybe::Given(choice) = &patch.tool_choice {
                    self.config.tool_choice = choice.clone();
                }
                if let Some(voice) = voice {
                    self.config.voice = Some(voice.clone());
                }
                self.updated(now);
            }
            Outgoing::InstructionsUpdate { instructions, .. } => {
                self.config.instructions = Some(instructions.clone());
                self.updated(now);
            }
            Outgoing::ToolsUpdate { tools, .. } => {
                self.config.tools = tools.iter().map(|t| t.name.clone()).collect();
                self.updated(now);
            }
            Outgoing::AudioAppend { frame } => {
                let ms = (frame.data.len() as u64 / 2) * 1000 / SAMPLE_RATE as u64;
                self.buffer_ms += ms;
                self.heard_ms += ms;
                hooks.heard(&frame.data);
            }
            Outgoing::AudioCommit => {
                // At least 100 ms, as the server says.
                if self.buffer_ms < 100 {
                    // The client could count what it appended unless the server emptied the
                    // buffer on its own, or an emptying is still on its way to it.
                    let knowing = self.own_empty && self.untold_empties == 0;
                    let why = "Error committing input audio buffer: buffer too small. Expected at least 100ms of audio.";
                    self.refuse(now, "input_audio_buffer_commit_empty", why, true, knowing, frame);
                } else {
                    let item = self.speaking.take().unwrap_or_else(|| self.mint_item("user"));
                    let lag = self.answer_lag(now);
                    self.own_empty = true;
                    self.commit(later(now, lag), item);
                }
            }
            Outgoing::AudioClear => {
                self.buffer_ms = 0;
                self.own_empty = true;
                self.untold_empties += 1;
                self.after(now, 90, unhandled("input_audio_buffer.cleared"));
            }
            Outgoing::ResponseCreate { req_id, instructions, tool_choice } => {
                if self.active.is_some() {
                    // Knowing only if the client had been told of the response refusing it: one
                    // it told the server to cancel no longer is (a cancel frees the conversation
                    // at once), and one the server opened in the same breath it could not know of.
                    let knowing = self
                        .active
                        .and_then(|r| self.responses[r as usize].told_created)
                        .is_some_and(|at| at < now);
                    let why = "Conversation already has an active response in progress. Wait until the response is finished before creating a new one.";
                    let code = "conversation_already_has_active_response";
                    return self.refuse(now, code, why, true, knowing, frame);
                }
                // Asked over a call whose output is not in yet: it reads the call without it.
                let tail = self.items.last().filter(|i| i.kind == "function_call");
                let unanswered = tail.and_then(|i| i.call_id.clone());
                // Answered in the order asked: a reply cancelled at once still opens before the
                // one asked after it.
                let delay = self.answer_after(now, "created_after_create");
                // A response's own `tool_choice` and `instructions` override the session's.
                let r = self.respond(
                    later(now, delay),
                    Some(req_id.clone()),
                    tool_choice.clone(),
                    instructions.clone(),
                );
                // A reply asked after the outputs reads them.
                self.early.retain(|(.., at)| *at != now);
                if let Some(call) = unanswered {
                    self.unanswered.push((self.responses[r as usize].id.clone(), call, now));
                }
            }
            Outgoing::ResponseCancel { response_id } => {
                self.cancel_frame(now, response_id.as_deref(), frame);
            }
            Outgoing::ItemCreate { item, previous_item_id, .. } => {
                self.create_item(now, item, previous_item_id.as_deref(), frame);
            }
            Outgoing::ItemDelete { item_id, .. } => match self.position(item_id) {
                Some(at) => {
                    self.items.remove(at);
                    let lag = self.answer_lag(now);
                    self.after(now, lag, Incoming::ItemDeleted { item_id: item_id.clone() });
                }
                None => {
                    let why = "Error deleting item: the item does not exist.";
                    self.refuse(now, "item_delete_invalid_item_id", why, true, true, frame);
                }
            },
            Outgoing::ItemTruncate { item_id, audio_end_ms } => {
                let end = *audio_end_ms;
                let Some(at) = self.position(item_id) else {
                    let why = "Item with item_id not found";
                    return self.refuse(
                        now,
                        "item_truncate_invalid_item_id",
                        why,
                        true,
                        true,
                        frame,
                    );
                };
                if !self.items[at].assistant() {
                    let why = "Only model output audio messages can be truncated";
                    return self.refuse(now, "unsupported_content_type", why, false, true, frame);
                }
                // Within the audio it holds, a millisecond of rounding allowed; what it holds is
                // cut there, and a later truncation is judged against the cut.
                let audio = self.items[at].audio_ms;
                if end > audio + 1 {
                    let why = "Audio content is already shorter than audio_end_ms";
                    return self.refuse(now, "invalid_value", why, false, true, frame);
                }
                self.items[at].audio_ms = end.min(audio);
                self.items[at].pcm_ms = end.min(self.items[at].pcm_ms);
                self.items[at].cut_ms = Some(end);
                let lag = self.answer_lag(now);
                self.put_about(later(now, lag), None, "conversation.item.truncated", item_id);
            }
            // xAI's `force_message`: an item type this server does not know.
            Outgoing::ForceMessage { .. } => {
                let why = "Invalid value: not a supported client event.";
                self.refuse(now, "invalid_value", why, true, true, frame);
            }
        }
    }

    /// The voice is fixed once assistant audio exists; an update naming another is refused whole.
    fn refuses_voice(&self, voice: Option<&String>) -> bool {
        self.voiced && voice.is_some_and(|v| Some(v) != self.config.voice.as_ref())
    }

    fn refuse_voice(&mut self, now: Ts, frame: &Outgoing) {
        let why = "Cannot update a conversation's voice if assistant audio is present.";
        self.refuse(now, "cannot_update_voice", why, false, true, frame);
    }

    fn updated(&mut self, now: Ts) {
        self.after(now, 5, unhandled("session.updated"));
    }

    fn cancel_frame(&mut self, now: Ts, named: Option<&str>, frame: &Outgoing) {
        if let Some(id) = named
            && !self.responses.iter().any(|r| r.id == id)
        {
            let why = "Response with id not found";
            return self.refuse(now, "response_cancel_not_found", why, false, true, frame);
        }
        let target =
            self.active.filter(|r| named.is_none_or(|id| self.responses[*r as usize].id == id));
        match target {
            Some(r) => {
                // The server's answer to the person's turn, nobody has heard a word of, no onset
                // told over it: the client cancels it on nothing the server saw.
                let resp = &self.responses[r as usize];
                if resp.turn && !resp.prompted && resp.audio_ms == 0 {
                    let (response, generation) = (resp.id.clone(), resp.source);
                    self.breaches.push(Breach::CancelledUnprompted { response, generation });
                }
                // What it had generated still streams out first: a reply asked in the same breath
                // is created before this one's `response.done` (probe-cancel_then_create).
                let lag = self.answer_after(now, "done_after_cancel");
                self.cancel(later(now, lag), r, "client_cancelled");
            }
            None => {
                // Knowing when the client had been told the end of the response it names (of
                // every one, unnamed); a response it asked for and has not heard of, or one whose
                // end is on its way, makes the cancel a race.
                let told_done = |r: &Response| r.finished.is_some_and(|at| at < now);
                let knowing = match named {
                    Some(id) => self.responses.iter().any(|r| r.id == id && told_done(r)),
                    None => self.responses.iter().all(told_done),
                };
                let why = "Cancellation failed: no active response found";
                self.refuse(now, "response_cancel_not_active", why, true, knowing, frame);
            }
        }
    }

    fn create_item(&mut self, now: Ts, item: &ChatItem, given: Option<&str>, frame: &Outgoing) {
        let id = item.id().to_owned();
        let previous = match given {
            Some("root") => None,
            Some(p) => Some(p.to_owned()),
            None => self.items.last().map(|i| i.id.clone()),
        };
        let output_of = match item {
            ChatItem::FunctionCallOutput(o) => Some(o.call_id.clone()),
            _ => None,
        };
        let called = |call: &String| {
            self.items.iter().any(|i| i.kind == "function_call" && i.call_id.as_ref() == Some(call))
        };
        let refusal = if self.position(&id).is_some() {
            let why = "Error adding item: an item with this id already exists.";
            Some(("item_create_duplicate_item_id", why, true))
        } else if previous.as_deref().is_some_and(|p| self.position(p).is_none()) {
            let why = "Error adding item: the previous item does not exist.";
            Some(("item_create_invalid_previous_item_id", why, true))
        } else if output_of.as_ref().is_some_and(|call| !called(call)) {
            Some(("invalid_tool_call_id", "Tool call ID not found in conversation.", false))
        } else {
            None
        };
        if let Some((code, why, cited)) = refusal {
            return self.refuse(now, code, why, cited, true, frame);
        }
        let at = match &previous {
            _ if given.is_none() => self.items.len(),
            Some(p) => self.position(p).map_or(self.items.len(), |i| i + 1),
            None => 0,
        };
        self.items.insert(at, Item::of(item));
        if let Some(call) = output_of {
            for (response, c, at) in &self.unanswered {
                if *c == call && *at == now {
                    self.early.push((response.clone(), call.clone(), now));
                }
            }
        }
        let lag = self.answer_lag(now);
        let added = Incoming::ItemAdded { previous_item_id: previous, item: item.clone() };
        self.after(now, lag, added);
        self.put_about(later(now, lag), None, "conversation.item.done", &id);
    }

    fn position(&self, id: &str) -> Option<usize> {
        self.items.iter().position(|i| i.id == id)
    }

    fn configure(&mut self, opts: &SessionOptions) {
        self.config.turn_detection(opts.turn_detection.as_ref());
        self.config.transcription = opts.transcription.is_some();
        self.config.audio = opts.audio_output;
        self.config.tool_choice = opts.tool_choice.clone();
        if let Some(instructions) = &opts.instructions {
            self.config.instructions = Some(instructions.clone());
        }
        self.config.voice = Some(opts.voice.clone());
    }

    /// The client's frame is refused with an `error` event, and the breach is noted.
    /// `cited`: the error names the client event it refuses (`error.event_id`), as the real
    /// server does for most refusals and not for some (`corpus/probes/probe-refusals.json`).
    fn refuse(
        &mut self,
        now: Ts,
        code: &'static str,
        why: &'static str,
        cited: bool,
        knowing: bool,
        frame: &Outgoing,
    ) {
        let event = wire_name(frame).to_owned();
        self.breaches.push(Breach::Refused { event, code, why, knowing });
        let lag = self.answer_lag(now);
        let error = ServerError {
            event_id: event_id_of(frame).filter(|_| cited).map(str::to_owned),
            code: Some(code.into()),
            type_: Some("invalid_request_error".into()),
            message: why.into(),
        };
        self.after(now, lag, Incoming::Error(error));
    }
}
