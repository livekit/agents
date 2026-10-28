//! Requests, cancellation and open-generation identity for the realtime session.

use super::*;

/// The realtime session's requests and its one open generation. The loop owns it; the `Driver`
/// methods below decide and act. State is updated before any action that can re-enter (a fatal
/// error closes the session, and the close interrupts this machine).
#[derive(Default)]
pub struct RealtimeSessionState {
    pub(super) caps: RealtimeCapabilities,
    next_key: u64,
    pub(super) pending: BTreeMap<String, Pending>,
    /// Requests ready to go while the server was busy, oldest first: sent one at a time once
    /// it is not (the server takes one reply at a time and refuses the rest).
    queue: std::collections::VecDeque<String>,
    /// Every generation still generating, oldest first: a server-initiated turn created while
    /// another plays is queued behind it (`_on_generation_created` schedules a new speech).
    open: Vec<Open>,
    /// The generations this session closed last; their late deltas are dropped silently.
    closed: std::collections::VecDeque<String>,
    /// A new session took over (`SessionSwapped`): it gets its frame once the speeches the
    /// swap cut are released, so the conversation it is given holds what the caller heard.
    pub(super) swap: Option<Swap>,
    /// Tools ran on a self-replying model: its next server-initiated generation is the
    /// follow-up, awaited under the timer keyed here.
    pub(super) auto_follow_up: Option<(Reuse, RequestKey)>,
    /// The last step's tools asked the server for a reply it generates itself: the agent stays
    /// `thinking` when that speech ends (the reference resets only when no reply is expected).
    /// Cleared when a generation opens.
    auto_reply_expected: bool,
    /// Requests cancelled (`CancelReply`) before the server answered: a generation that still
    /// answers one is interrupted (`_realtime_reply_task`: "the response already landed").
    /// Forgotten once asked.
    /// Each under its timer: a server that never answers frees the queue at its bound.
    pub(super) cancelled_requests: BTreeMap<String, RequestKey>,
    /// A generation answering a request already settled: its `GenerationDone` is dropped.
    pub(super) discarded: Option<String>,
    /// The generation whose foreign deltas were warned about; reset when a generation opens.
    foreign: Option<String>,
    /// Late words describe an older response and never acquire another speech.
    attributed: BTreeMap<String, AttributedTranscript>,
    attributed_bytes: usize,
    pub(super) closing: bool,
    ended: bool,
}

impl RealtimeSessionState {
    /// The model speaks a text it is given (`say()`).
    pub fn says(&self) -> bool {
        self.caps.supports_say
    }

    pub fn new(caps: RealtimeCapabilities) -> Self {
        RealtimeSessionState { caps, ..RealtimeSessionState::default() }
    }
}

/// A swap's frame, owed to the new session.
pub(super) struct Swap {
    /// The speeches the swap cut that are not released yet.
    pub(super) waiting: BTreeSet<SpeechId>,
    /// Ask again for the reply the swap cut.
    pub(super) regenerate: bool,
}

/// The timer set cannot key on a `String`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestKey(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Say {
        text: String,
        add_to_chat_ctx: bool,
    },
    GenerateReply {
        instructions: Option<String>,
        tool_choice: Option<ToolChoice>,
        /// A DTMF collection's read-back, whose guard the speech holds once it exists.
        read_back: bool,
    },
    /// No caller waits. `commit_audio`: commit the input buffer and ask, both or neither.
    UserTurn {
        commit_audio: bool,
    },
    ToolFollowUp {
        speech: SpeechId,
        step: u32,
        /// The step's verdict: `none` when the step budget is spent or the agent drains.
        tool_choice: Option<lk_agents_abi::ToolChoice>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Settled {
    Failed(RealtimeError),
    /// A late answer is discarded.
    TimedOut(RealtimeError),
    /// Cut by the interruption that also cut the server's turn, or by the close, for its cause.
    Interrupted(crate::trace::InterruptCause),
    /// The socket is gone: the reference's pending `generate_reply()` future fails with a
    /// `RealtimeError`, which `_realtime_reply_task` logs, as it does a timeout.
    Ended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Reuse {
    pub(super) speech: SpeechId,
    pub(super) step: u32,
}

pub(super) struct Pending {
    pub(super) key: RequestKey,
    pub(super) request: Request,
    pub(super) call: Option<SpeechCall>,
    pub(super) allow_interruptions: Option<bool>,
    /// The speech announced when it was asked for, which the server's answer plays into (a
    /// `Say`, a `GenerateReply` or a user turn: `_realtime_reply_task`'s handle).
    pub(super) speech: Option<SpeechId>,
    pub(super) phase: Phase,
}

/// Where a request is on its way to the server, with its command.
pub(super) enum Phase {
    /// Held until its speech is authorized, as the reference asks the server only then.
    Unsent(RealtimeCommand),
    /// Ready, and held while the server generates a reply or answers another request.
    Queued(RealtimeCommand),
    /// On the wire: asked again if the server refuses it for a reply it had started that the
    /// client had not heard of yet.
    Sent(RealtimeCommand),
}

struct Open {
    response_id: String,
    /// `None` until the generation's speech is started, in the same cycle.
    speech: Option<SpeechId>,
    /// `Interrupt` was sent for this response; a second stop sends nothing more.
    cancelled: bool,
}

// Share the closed-generation identity horizon and keep retained late text below one bounded
// duplex transcript backlog. Admission includes identifiers and spare String allocation.
const ATTRIBUTED_TRANSCRIPT_BYTES: usize = 256 * 1024;

struct AttributedTranscript {
    source_response_id: String,
    created_at: std::time::SystemTime,
    message: Option<ChatMessage>,
}

impl AttributedTranscript {
    fn retained_bytes(&self) -> usize {
        self.source_response_id.capacity()
            + self.message.as_ref().map_or(0, |message| {
                message.id.capacity()
                    + message
                        .content
                        .iter()
                        .map(|content| match content {
                            lk_agents_abi::ChatContent::Text(text) => text.capacity(),
                            _ => 0,
                        })
                        .sum::<usize>()
            })
    }
}

fn transcript_overload() -> ApiError {
    ApiError::new("realtime attributed transcript retention exceeded").with_retryable(false)
}

fn finish_transcript(mut transcript: AttributedTranscript) -> Option<ChatMessage> {
    transcript.message.take().map(|mut message| {
        message.metrics.provider_request_ids = Some(vec![transcript.source_response_id]);
        message
    })
}

pub(super) fn bare_reply(req_id: &str) -> RealtimeCommand {
    RealtimeCommand::GenerateReply(rt::GenerateReply {
        req_id: req_id.to_owned(),
        instructions: None,
        tool_choice: None,
    })
}

pub(crate) fn response_id_of(ev: &Event) -> Option<&str> {
    match ev {
        Event::MessageStarted(m) => m.response_id.as_deref(),
        Event::TextDelta(d) => d.response_id.as_deref(),
        Event::AudioDelta(d) => d.response_id.as_deref(),
        Event::MessageModalities(m) => m.response_id.as_deref(),
        Event::MessageDone(m) => m.response_id.as_deref(),
        Event::GenerationCreated(g) => Some(&g.response_id),
        Event::GenerationDone(g) => Some(&g.response_id),
        Event::TranscriptAttribution(a) => Some(&a.response_id),
        _ => None,
    }
}

/// Where a generation's event goes.
pub(super) enum Route {
    Dropped,
    /// A generation that is not the open one, warned about once: its id and the open one's.
    Foreign {
        id: String,
        open: String,
    },
    To(SpeechId),
}

impl RealtimeSessionState {
    pub(super) fn attribute_transcript(
        &mut self,
        attribution: rt::TranscriptAttribution,
        created_at: std::time::SystemTime,
    ) -> Result<(), ApiError> {
        if attribution.response_id == attribution.source_response_id
            || self.open.iter().any(|open| open.response_id == attribution.response_id)
        {
            return Err(ApiError::new("realtime transcript attribution reused a speech identity")
                .with_retryable(false));
        }
        if let Some(known) = self.attributed.get(&attribution.response_id) {
            return if known.source_response_id == attribution.source_response_id {
                Ok(())
            } else {
                Err(ApiError::new("realtime transcript attribution changed its source")
                    .with_retryable(false))
            };
        }
        let bytes = attribution.response_id.capacity() + attribution.source_response_id.capacity();
        if self.attributed.len() == CLOSED_REMEMBERED
            || bytes > ATTRIBUTED_TRANSCRIPT_BYTES.saturating_sub(self.attributed_bytes)
        {
            return Err(transcript_overload());
        }
        self.attributed_bytes += bytes;
        self.attributed.insert(
            attribution.response_id,
            AttributedTranscript {
                source_response_id: attribution.source_response_id,
                created_at,
                message: None,
            },
        );
        Ok(())
    }

    pub(super) fn attributed(&self, response_id: &str) -> bool {
        self.attributed.contains_key(response_id)
    }

    pub(super) fn transcript_event(
        &mut self,
        event: Event,
    ) -> Result<Option<ChatMessage>, ApiError> {
        let id = response_id_of(&event).expect("attributed event has response identity");
        if let Event::GenerationDone(done) = &event {
            let (id, transcript) =
                self.attributed.remove_entry(&done.response_id).expect("known attribution");
            self.attributed_bytes -= id.capacity() + transcript.retained_bytes();
            self.close(done.response_id.clone());
            return Ok(finish_transcript(transcript));
        }
        let transcript = self.attributed.get_mut(id).expect("known attribution");
        let before = transcript.retained_bytes();
        let completed = match event {
            Event::MessageStarted(message) => {
                if transcript.message.is_some() {
                    return Err(
                        ApiError::new("realtime transcript messages overlap").with_retryable(false)
                    );
                }
                if message.message_id.capacity()
                    > ATTRIBUTED_TRANSCRIPT_BYTES.saturating_sub(self.attributed_bytes)
                {
                    return Err(transcript_overload());
                }
                transcript.message = Some(ChatMessage::text(
                    message.message_id,
                    Role::Assistant,
                    "",
                    transcript.created_at,
                ));
                None
            }
            Event::TextDelta(delta) => {
                let Some(message) = &mut transcript.message else {
                    return Err(ApiError::new("realtime transcript delta has no message")
                        .with_retryable(false));
                };
                if message.id != delta.message_id {
                    return Err(ApiError::new("realtime transcript delta changed its message")
                        .with_retryable(false));
                }
                let lk_agents_abi::ChatContent::Text(text) = &mut message.content[0] else {
                    unreachable!("transcript has text only")
                };
                let available = ATTRIBUTED_TRANSCRIPT_BYTES.saturating_sub(self.attributed_bytes);
                if delta.delta.len() > available {
                    return Err(transcript_overload());
                }
                text.reserve_exact(delta.delta.len());
                text.push_str(&delta.delta);
                None
            }
            Event::MessageDone(done) => {
                if transcript.message.as_ref().is_some_and(|message| message.id != done.message_id)
                {
                    return Err(ApiError::new("realtime transcript ended another message")
                        .with_retryable(false));
                }
                transcript.message.take().map(|mut message| {
                    message.metrics.provider_request_ids =
                        Some(vec![transcript.source_response_id.clone()]);
                    message
                })
            }
            Event::AudioDelta(_) => {
                return Err(ApiError::new("an attributed transcript contained new audio")
                    .with_retryable(false));
            }
            _ => None,
        };
        self.attributed_bytes = self.attributed_bytes - before + transcript.retained_bytes();
        if self.attributed_bytes > ATTRIBUTED_TRANSCRIPT_BYTES {
            return Err(transcript_overload());
        }
        Ok(completed)
    }

    pub fn ended(&self) -> bool {
        self.ended
    }

    /// The model answers a tool result with a reply of its own.
    pub fn auto_tool_reply(&self) -> bool {
        self.caps.auto_tool_reply_generation
    }

    /// `true` when this starts the close (it was not closing).
    pub fn begin_closing(&mut self) -> bool {
        !std::mem::replace(&mut self.closing, true)
    }

    /// The socket ended: every open generation closes, nothing is followed up any more.
    pub(super) fn end(&mut self) {
        self.ended = true;
        for o in std::mem::take(&mut self.open) {
            self.close(o.response_id);
        }
        self.auto_follow_up = None;
        self.cancelled_requests.clear();
        self.attributed.clear();
        self.attributed_bytes = 0;
    }

    /// A request: pending under `req_id` until answered, its timer keyed by the returned key.
    /// `speech` is the speech announced for it.
    pub(super) fn register(
        &mut self,
        req_id: String,
        request: Request,
        call: Option<SpeechCall>,
        allow_interruptions: Option<bool>,
        speech: Option<SpeechId>,
        phase: Phase,
    ) -> RequestKey {
        let key = self.mint_key();
        let pending = Pending { key, request, call, allow_interruptions, speech, phase };
        self.pending.insert(req_id, pending);
        key
    }

    /// The request whose speech is `speech` and that waits for its authorization to be sent:
    /// its id, command and timer key.
    pub(super) fn unsent(&self, speech: SpeechId) -> Option<(String, RealtimeCommand, RequestKey)> {
        self.pending.iter().find_map(|(id, p)| match &p.phase {
            Phase::Unsent(cmd) if p.speech == Some(speech) => {
                Some((id.clone(), cmd.clone(), p.key))
            }
            _ => None,
        })
    }

    pub(super) fn take_by_key(&mut self, key: RequestKey) -> Option<(String, Pending)> {
        let req_id = self.pending.iter().find(|(_, p)| p.key == key).map(|(id, _)| id.clone())?;
        let pending = self.pending.remove(&req_id)?;
        Some((req_id, pending))
    }

    /// One key space for requests and the auto follow-up's wait: both time out on
    /// `Timer::RealtimeRequest`.
    fn mint_key(&mut self) -> RequestKey {
        self.next_key += 1;
        RequestKey(self.next_key - 1)
    }

    /// Tools ran on a self-replying model: its next server-initiated generation continues
    /// `speech`, awaited under the returned timer key.
    pub(super) fn follow_up_with(&mut self, speech: SpeechId, step: u32) -> RequestKey {
        let key = self.mint_key();
        self.auto_follow_up = Some((Reuse { speech, step }, key));
        self.auto_reply_expected = true;
        key
    }

    /// The follow-up's wait timed out: the speech it would have continued.
    pub(super) fn auto_follow_up_timed_out(&mut self, key: RequestKey) -> Option<SpeechId> {
        let (reuse, k) = self.auto_follow_up?;
        (k == key).then(|| {
            self.auto_follow_up = None;
            reuse.speech
        })
    }

    /// The server would refuse a reply now: it generates one not told to stop, or a request
    /// other than `req_id` went out and has not been answered (a cancelled one too: the
    /// server creates its response before it is cancelled).
    fn busy(&self, req_id: &str) -> bool {
        self.open.iter().any(|o| !o.cancelled)
            || !self.cancelled_requests.is_empty()
            || self.pending.iter().any(|(id, p)| id != req_id && matches!(p.phase, Phase::Sent(_)))
    }

    /// `req_id`'s command goes out now (`Some`, marked sent) or waits in the queue.
    pub(super) fn send_or_queue(&mut self, req_id: &str) -> Option<RealtimeCommand> {
        let busy = self.busy(req_id);
        let p = self.pending.get_mut(req_id)?;
        let cmd = match &p.phase {
            Phase::Unsent(cmd) | Phase::Queued(cmd) | Phase::Sent(cmd) => cmd.clone(),
        };
        if busy {
            p.phase = Phase::Queued(cmd);
            self.queue.push_back(req_id.to_owned());
            return None;
        }
        p.phase = Phase::Sent(cmd.clone());
        Some(cmd)
    }

    /// The server refused `req_id` for a reply of its own it had started as the request
    /// crossed it: back at the head of the queue. Its timer key, to cancel.
    pub(super) fn requeue(&mut self, req_id: &str) -> Option<RequestKey> {
        let p = self.pending.get_mut(req_id)?;
        let Phase::Sent(cmd) = &p.phase else { return None };
        p.phase = Phase::Queued(cmd.clone());
        self.queue.push_front(req_id.to_owned());
        Some(p.key)
    }

    /// The queue's next request, once the server is free: its command and timer key.
    pub(super) fn next_queued(&mut self) -> Option<(RealtimeCommand, RequestKey)> {
        while let Some(req_id) = self.queue.front().cloned() {
            if !self.pending.get(&req_id).is_some_and(|p| matches!(p.phase, Phase::Queued(_))) {
                self.queue.pop_front();
                continue;
            }
            if self.busy(&req_id) {
                return None;
            }
            self.queue.pop_front();
            let p = self.pending.get_mut(&req_id)?;
            let Phase::Queued(cmd) = &p.phase else { return None };
            let cmd = cmd.clone();
            p.phase = Phase::Sent(cmd.clone());
            return Some((cmd, p.key));
        }
        None
    }

    /// A generation is still generating and not yet told to stop.
    pub(crate) fn generating(&self) -> bool {
        self.open.iter().any(|o| !o.cancelled)
    }

    /// The speeches of the generations still generating.
    pub(super) fn open_speeches(&self) -> Vec<SpeechId> {
        self.open.iter().filter_map(|o| o.speech).collect()
    }

    pub(crate) fn auto_reply_expected(&self) -> bool {
        self.auto_reply_expected
    }

    pub(super) fn open_generation(&mut self, response_id: String, speech: Option<SpeechId>) {
        self.foreign = None;
        self.auto_reply_expected = false;
        self.open.push(Open { response_id, speech, cancelled: false });
    }

    pub(super) fn set_open_speech(&mut self, response_id: &str, speech: SpeechId) {
        if let Some(open) = self.open.iter_mut().find(|o| o.response_id == response_id) {
            open.speech = Some(speech);
        }
    }

    /// Where an event of `response_id` goes: its generation's speech, or nowhere. Without an
    /// id, the newest generation's.
    pub(super) fn route(&mut self, response_id: Option<&str>) -> Route {
        let Some(id) = response_id else {
            return match self.open.last().and_then(|o| o.speech) {
                Some(speech) => Route::To(speech),
                None => Route::Dropped,
            };
        };
        if let Some(open) = self.open.iter().find(|o| o.response_id == id) {
            return open.speech.map_or(Route::Dropped, Route::To);
        }
        let late = self.discarded.as_deref() == Some(id) || self.closed.iter().any(|c| c == id);
        match self.open.last() {
            Some(newest) if !late && self.foreign.as_deref() != Some(id) => {
                self.foreign = Some(id.to_owned());
                Route::Foreign { id: id.to_owned(), open: newest.response_id.clone() }
            }
            Some(_) | None => Route::Dropped,
        }
    }

    /// A late delta comes right after its generation's end: the last few are remembered.
    fn close(&mut self, response_id: String) {
        if self.closed.len() == CLOSED_REMEMBERED {
            self.closed.pop_front();
        }
        self.closed.push_back(response_id);
    }

    /// `GenerationDone` for `response_id`: `Some(speech)` when it closed an open generation; a
    /// discarded one ends like a closed one (its tail is late, not foreign).
    pub(super) fn generation_done(&mut self, response_id: &str) -> Option<Option<SpeechId>> {
        if self.discarded.as_deref() == Some(response_id) {
            self.discarded = None;
            self.close(response_id.to_owned());
            return None;
        }
        let at = self.open.iter().position(|o| o.response_id == response_id)?;
        let open = self.open.remove(at);
        self.close(open.response_id);
        Some(open.speech)
    }

    /// One `Interrupt` per generation still generating whose speech `stops` (or that has
    /// none), once each. A generation whose speech carries on is spared.
    pub(super) fn cancel_where(
        &mut self,
        stops: impl Fn(Option<SpeechId>) -> bool,
    ) -> Vec<RealtimeCommand> {
        if self.auto_follow_up.is_some_and(|(reuse, _)| stops(Some(reuse.speech))) {
            self.auto_follow_up = None;
        }
        if self.ended {
            return Vec::new();
        }
        let mut out = Vec::new();
        for open in self.open.iter_mut().filter(|o| !o.cancelled && stops(o.speech)) {
            open.cancelled = true;
            let response_id = Some(open.response_id.clone());
            out.push(RealtimeCommand::Interrupt(rt::Interrupt { response_id }));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributed_transcript_retention_is_refused_and_released_on_completion() {
        let mut state = RealtimeSessionState::default();
        let attribution = |id: String| rt::TranscriptAttribution {
            response_id: id,
            source_response_id: "spoken-response".into(),
        };
        for id in 0..CLOSED_REMEMBERED {
            state.attribute_transcript(attribution(id.to_string()), std::time::UNIX_EPOCH).unwrap();
        }
        assert!(
            state
                .attribute_transcript(attribution("overflow".into()), std::time::UNIX_EPOCH)
                .is_err()
        );
        state
            .transcript_event(Event::GenerationDone(rt::GenerationDone {
                response_id: "0".into(),
                status: rt::GenerationStatus::Completed,
                usage: Default::default(),
            }))
            .unwrap();
        state
            .attribute_transcript(attribution("replacement".into()), std::time::UNIX_EPOCH)
            .unwrap();
        state = RealtimeSessionState::default();

        let mut id = String::with_capacity(ATTRIBUTED_TRANSCRIPT_BYTES / 2);
        id.push('x');
        state.attribute_transcript(attribution(id), std::time::UNIX_EPOCH).unwrap();
        assert!(
            state
                .attribute_transcript(
                    attribution("y".repeat(ATTRIBUTED_TRANSCRIPT_BYTES / 2)),
                    std::time::UNIX_EPOCH
                )
                .is_err()
        );
        state
            .transcript_event(Event::GenerationDone(rt::GenerationDone {
                response_id: "x".into(),
                status: rt::GenerationStatus::Completed,
                usage: Default::default(),
            }))
            .unwrap();
        // Completion releases the stored key's capacity, not the shorter acknowledgement's.
        state
            .attribute_transcript(
                attribution("y".repeat(ATTRIBUTED_TRANSCRIPT_BYTES / 2)),
                std::time::UNIX_EPOCH,
            )
            .unwrap();
    }
}
