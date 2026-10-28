//! Everything that became true since the last input, written down.
//!
//! Answers, host events, what the fakes did, what the driver journalled, and the public state of
//! every speech handle. Nothing here awaits: what it finds is what was ready without giving the
//! driver another turn.
use super::apply::Settled;
use super::*;

impl Run {
    /// Everything that became true since the last input: answers, host events, what the fakes did,
    /// what the driver journalled, and the public state of every speech handle. Nothing here
    /// awaits, so what it finds is what was ready without giving the driver another turn.
    pub(super) fn collect(&mut self) {
        for nth in self.pending.keys().copied().collect::<Vec<_>>() {
            if let Some(settled) = self.pending.get_mut(&nth).and_then(Pending::try_recv) {
                self.pending.remove(&nth);
                self.record_answer(nth, settled);
            }
        }
        // A close in flight is the host awaiting `aclose()`: it moves when the run looks.
        if let Some(closing) = &mut self.closing
            && super::apply::ready_now(closing.as_mut()).is_some()
        {
            self.closing = None;
        }
        let from_host = self.record.host().len();
        let host = std::mem::take(&mut *self.host.lock().unwrap());
        for (at, seen) in host {
            // The handle is kept, not just recorded: `AwaitSpeech` needs the real object.
            if let Host::Session(SessionEvent::SpeechCreated { speech, .. }) = &seen {
                self.handles.entry(speech.id().0).or_insert_with(|| speech.clone());
            }
            // `_agent_state_changed`: the thinking sound plays while the agent thinks.
            if let (Some(player), Host::Session(SessionEvent::AgentStateChanged { new, .. })) =
                (&self.background, &seen)
            {
                let thinking = *new == lk_agents_abi::voice::events::AgentState::Thinking;
                let sound = fakes::FakeSoundStream::sound(5);
                let _ = player.on_agent_state_for_host(vec![sound], fakes::HALF, thinking, true);
            }
            self.record.record_host(at, seen);
        }
        self.sync_fakes();
        self.avatar_reports();
        let from_journal = self.record.journal().len();
        self.absorb_journal();
        self.attribute_chats(from_journal, from_host);

        // A snapshot takes a stamp of its own: what it observed came after everything stamped
        // before it and before everything stamped after, so a speech seen done here and a stream
        // closed by the next input never share an instant a timeline rule cannot order.
        let now = self.stamps.next();
        for (id, handle) in &self.handles {
            let done = handle.done();
            // A handle seen done keeps the error it ended with: no need to read it again.
            let ended = self.record.handles().get(id).is_some_and(|seen| seen.done);
            self.record.record_handle(
                *id,
                Handle {
                    kind: None,
                    done,
                    done_since: done.then_some(now),
                    interrupted: handle.interrupted_for_host(),
                    scheduled: handle.scheduled(),
                    allow_interruptions: handle.allow_interruptions(),
                    error: if ended {
                        None
                    } else {
                        handle.exception().ok().flatten().map(|e| format!("{e:?}"))
                    },
                },
            );
        }
        self.record.set_driver_finished(self.runtime.is_none());
        self.record.set_resources_left(self.api.resources().live());
        self.record.set_parks(tokio::runtime::Handle::current().metrics().worker_park_count(0));
        self.record.set_now(now);
    }

    pub(super) fn record_answer(&mut self, nth: u32, settled: Settled) {
        let at = self.stamps.next();
        if let Some(handle) = settled.handle {
            self.handles.entry(handle.id().0).or_insert(handle);
        }
        self.record.record_result(at, nth, settled.answer);
        self.record.close(at, Kind::Answer, nth);
    }

    /// The host dropped the await of answer `nth`: closed, with nothing answered.
    pub(super) fn record_abandoned(&mut self, nth: u32) {
        self.record_answer(nth, Settled::of(crate::record::Answer::Abandoned));
    }

    /// Take everything the fakes have recorded, and close whatever they no longer hold.
    ///
    /// This is where the harness takes ownership: a control arrives with the invocation that
    /// opened it and goes straight into the harness's own table, so nothing the fakes can reach
    /// is ever mutated from two sides. The per-kind counters behind the records and the
    /// recording's advance together, so the n-th chat the fakes registered is the n-th chat the
    /// recording opened and nothing has to be matched up.
    ///
    /// Closing is read off the controls rather than tracked per delivery, because "is there still
    /// a receiver for this" is one question about the table and not a fact each input can know.
    fn sync_fakes(&mut self) {
        while let Ok(record) = self.records.try_recv() {
            match record {
                Record::Opened { at, slot, detail, control } => {
                    if let Some(control) = control {
                        self.controls.insert(slot, control);
                    }
                    self.record.open(at, slot.kind, Id::Slot(slot.nth), detail);
                }
                Record::Closed { at, slot } => {
                    if self.record.live().has(slot.kind, slot.nth) {
                        self.record.close(at, slot.kind, slot.nth);
                    }
                    // Nothing more can be said on a closed recognition stream.
                    if slot.kind == Kind::Stt {
                        self.end_utterances_on(at, slot.nth);
                    }
                }
                Record::Produced { at, tts, chunk, text, stamps } => {
                    self.record.record_produced(at, tts, chunk, text, stamps);
                }
                Record::Delivered { at, slot } => {
                    self.record.record_delivered(at, slot.kind, slot.nth);
                }
                Record::Generated { at, generation, chunk } => {
                    self.record.record_generated(at, generation % 255, chunk);
                }
                Record::Breach { at, socket, breach } => {
                    self.record.record_breach(at, socket, breach);
                }
                Record::Heard { at, generation, ms, whole } => {
                    self.record.record_heard(at, generation % 255, ms, whole);
                }
                Record::Held { at, socket, items } => {
                    let items = items.into_iter().map(|(g, ms)| (g % 255, ms)).collect();
                    self.record.record_held(at, socket, items);
                }
                Record::Forwarded { at, slot, tags } => {
                    self.record.record_forwarded(at, slot.kind, slot.nth, tags);
                }
                Record::Sink { at, sink } => {
                    // The output reported every pending segment of its own the moment the product
                    // cleared it, so those controls are spent and the table has to say so before
                    // the next input can point at one.
                    if let Sink::Clear { output, .. } | Sink::Reset { output, .. } = &sink {
                        let segments = &self.segments;
                        self.controls.retain(|slot, _| {
                            slot.kind != Kind::Segment || segments.get(&slot.nth) != Some(output)
                        });
                    }
                    self.record.record_sink(at, sink);
                }
                Record::Node { at, agent, node } => {
                    self.record.record_node(at, agent, node);
                }
                Record::Segment { segment, output } => {
                    self.segments.insert(segment, output);
                }
            }
        }
        let at = self.stamps.now();
        for kind in [Kind::Chat, Kind::Tts, Kind::Stt, Kind::Tool, Kind::Segment, Kind::Prediction]
        {
            for nth in self.record.live().of(kind) {
                if !self.controls.get(&Slot { kind, nth }).is_some_and(Control::open) {
                    self.record.close(at, kind, nth);
                }
            }
        }
    }

    /// Absorb the journal, opening and closing the four resources the driver owns as it goes.
    ///
    /// A speech is the driver's to create and to finish, and so is a hold, a filler and a hook
    /// call. Reading them from the driver's own steps is not a shortcut: any second ledger beside
    /// the journal ends up disagreeing with it, and the journal is the account a rule is read
    /// against, so a case would then point at things the driver does not believe it has.
    fn absorb_journal(&mut self) {
        let steps = std::mem::take(&mut *self.steps.lock().unwrap());
        for (at, step) in steps {
            match &step {
                Step::Emitted(SessionEvent::SpeechCreated { speech, kind, user_initiated }) => {
                    let detail = Detail::Speech { kind: *kind, user_initiated: *user_initiated };
                    self.open_once(at, Kind::Speech, Id::Speech(speech.id()), detail);
                }
                Step::Emitted(SessionEvent::SpeechDone { speech, .. }) => {
                    self.close_by_id(at, Kind::Speech, Id::Speech(*speech));
                }
                Step::Speech(SpeechStep::IdleAnswered { hold: Some(hold), .. }) => {
                    self.open_once(at, Kind::Hold, Id::Hold(*hold), Detail::None);
                }
                Step::Speech(SpeechStep::IdleHoldReleased { hold }) => {
                    self.close_by_id(at, Kind::Hold, Id::Hold(*hold));
                }
                Step::Generation(GenerationStep::FillerStarted { filler, .. }) => {
                    self.open_once(at, Kind::Filler, Id::Filler(*filler), Detail::None);
                }
                Step::Generation(GenerationStep::FillerStopped { filler, .. }) => {
                    self.close_by_id(at, Kind::Filler, Id::Filler(*filler));
                }
                // The developer's `on_user_turn_exceeded` runs on its own and is never answered by
                // the case: it says nothing back the session waits on, and the callback still
                // running is what makes a second crossing in the same activity a no-op.
                Step::Handoff(HandoffStep::HookIssued {
                    kind: lk_agents_voice::agent_session::HookKind::OnUserTurnExceeded,
                    ..
                }) => {}
                Step::Handoff(HandoffStep::HookIssued { hook, kind, .. }) => {
                    self.open_once(at, Kind::Hook, Id::Hook(*hook), Detail::Hook { kind: *kind });
                }
                Step::Handoff(HandoffStep::HookFinished { hook, .. }) => {
                    self.close_by_id(at, Kind::Hook, Id::Hook(*hook));
                }
                // The output the case attached replaced the one before it: the old one is gone
                // from the chain, and a resource nothing can reach any more is closed.
                Step::Playout(PlayoutStep::OutputReplaced { audio, transcription }) => {
                    for is_audio in [(*audio, true), (*transcription, false)]
                        .into_iter()
                        .filter_map(|(replaced, is_audio)| replaced.then_some(is_audio))
                    {
                        let mut of_kind: Vec<u32> = self
                            .record
                            .live()
                            .of(Kind::Output)
                            .into_iter()
                            .filter(|nth| {
                                matches!(
                                    self.record.resource(Kind::Output, *nth).map(|r| r.detail()),
                                    Some(Detail::Output { audio: a }) if *a == is_audio
                                )
                            })
                            .collect();
                        of_kind.sort_unstable();
                        of_kind.pop();
                        for old in of_kind {
                            self.record.close(at, Kind::Output, old);
                        }
                    }
                }
                // A detached tool ran on past its answer; its final update is its end.
                Step::Tool(ToolStep::Updated { call_id, is_final: true, .. }) => {
                    if let Some(nth) = self.tool_by_call_id(call_id) {
                        self.controls.remove(&Slot { kind: Kind::Tool, nth });
                        self.record.close(at, Kind::Tool, nth);
                    }
                }
                _ => {}
            }
            self.record.record_journal(at, step);
        }
    }

    /// Open a driver-owned resource unless that identity is already known. The journal can name one
    /// twice -- a second `SpeechDone`, a hook re-announced -- and a run that opened it twice would
    /// hand the case two references to one thing.
    fn open_once(&mut self, at: Observed, kind: Kind, id: Id, detail: Detail) {
        if self.record.resources(kind).all(|opened| opened.id() != id) {
            self.record.open(at, kind, id, detail);
        }
    }

    fn close_by_id(&mut self, at: Observed, kind: Kind, id: Id) {
        let Some(nth) = self.record.resources(kind).find(|o| o.id() == id).map(|o| o.nth()) else {
            return;
        };
        if self.record.live().has(kind, nth) {
            self.record.close(at, kind, nth);
        }
    }

    /// Which speech each chat request was made for, recorded as a fact on the request.
    ///
    /// The driver itself supplies the association: it reports `LlmMetrics` for every request,
    /// carrying the response id this crate's own model fake minted and the speech the driver made
    /// the request for. Nothing is parsed out of message content, nothing is inferred from timing.
    ///
    /// Metrics arrive when a request finishes, so a request still open, or one cancelled or failed
    /// before it reported, has none. For those the journal's `ChatRequested` stands in: the driver
    /// journals it and opens the request in one poll, with no await between, so the requests a
    /// context's plans opened are those plans in order, and each takes the earliest request of its
    /// context nobody has taken.
    fn attribute_chats(&mut self, from_journal: usize, from_host: usize) {
        let plans: Vec<(u64, Vec<String>)> = self.record.journal()[from_journal..]
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Generation(GenerationStep::ChatRequested { speech, ctx_ids, .. }) => {
                    Some((speech.0, ctx_ids.clone()))
                }
                _ => None,
            })
            .collect();
        for (speech, ctx_ids) in plans {
            let chat = self
                .record
                .resources(Kind::Chat)
                .filter(|o| {
                    matches!(o.detail(), Detail::Chat { ctx_ids: seen, speech: None, .. } if *seen == ctx_ids)
                })
                .map(|o| o.nth())
                .next();
            if let Some(chat) = chat {
                self.record.attribute_chat(chat, speech);
            }
        }
        // A request's own metrics name its speech exactly, so they have the last word.
        for position in from_host..self.record.host().len() {
            let Host::Session(SessionEvent::MetricsCollected { metrics, speech: Some(speech) }) =
                &self.record.host()[position].1
            else {
                continue;
            };
            let AgentMetrics::Llm(metrics) = &**metrics else { continue };
            let Some(chat) = fakes::request_of(&metrics.request_id) else { continue };
            let speech = speech.0;
            self.record.attribute_chat(chat, speech);
        }
    }
}
