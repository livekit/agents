use super::apply::Settled;
use super::*;
use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;

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
            self.closed_probe();
        }
        self.absorb_probes();
        let from_host = self.record.host().len();
        let host = std::mem::take(&mut *self.host.lock().unwrap());
        for (at, seen, handle) in host {
            // The handle is kept, not just recorded: `AwaitSpeech` needs the real object.
            if let (Host::Session(SessionEvent::SpeechCreated { speech, .. }), Some(handle)) =
                (&seen, handle)
            {
                self.handles.entry(speech.0).or_insert(handle);
            }
            // `_agent_state_changed`: the thinking sound plays while the agent thinks.
            if let (Some(player), Host::Session(SessionEvent::AgentStateChanged { new, .. })) =
                (&self.background, &seen)
            {
                let thinking = *new == lk_agents_abi::voice::events::AgentState::Thinking;
                let sound = fakes::FakeSoundStream::sound(5);
                let error = player.on_agent_state(super::background::StateOptions {
                    sounds: lk_agents_abi::voice::background_audio::Sounds::Choice(vec![sound]),
                    draw: fakes::HALF,
                    thinking,
                });
                self.record.record_boundary(
                    self.stamps.next(),
                    crate::record::BoundaryOutcome {
                        operation: crate::record::BoundaryOperation::BackgroundState,
                        error,
                    },
                );
            }
            self.record.record_host(at, seen);
        }
        self.sync_fakes();
        self.avatar_reports();
        let from_journal = self.record.journal().len();
        self.absorb_journal();
        self.absorb_amd();
        self.absorb_amd_runs();
        self.attribute_chats(from_journal, from_host);

        // The snapshot takes a stamp of its own, so what it saw never shares an instant with what
        // the next input does.
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
                    interrupted: HostSpeech::interrupted(&**handle),
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
        self.record.set_driver_finished(self.session.is_none());
        self.record.set_resources_left(self.api.resources().live());
        let sources = self.room.local_participant().sources();
        self.record.set_room_speakers(sources.iter().map(|source| source.speaker()).collect());
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
                Record::Boundary { at, outcome } => self.record.record_boundary(at, outcome),
                Record::Storage { at, step } => self.record.record_storage(at, step),
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
                Record::OutputAttachment { at, output, attached } => {
                    self.record.record_output_attachment(at, output, attached);
                    let active = self
                        .record
                        .host_view()
                        .outputs()
                        .into_iter()
                        .find(|o| o.nth() == output)
                        .map_or(0, |o| {
                            o.attachments()
                                .fold(0i32, |n, (_, attached)| n + if attached { 1 } else { -1 })
                        });
                    if attached || active > 0 {
                        continue;
                    }
                    // Enable toggles can reattach the selected endpoint. Only a detached,
                    // replaced endpoint has relinquished its admitted binding.
                    let retired = self.record.host_view().outputs().into_iter().any(|o| {
                        if o.nth() != output {
                            return false;
                        }
                        o.replaced_at().is_some_and(|replacement| replacement <= at)
                            || self.record.applied().iter().any(|applied| {
                                applied.accepted
                                    && applied.at >= o.attached_at()
                                    && applied.at <= at
                                    && matches!(
                                        (&applied.input, o.is_audio()),
                                        (Input::SetAudioOutput { attach: false }, true)
                                            | (
                                                Input::SetTranscriptionOutput { attach: false },
                                                false
                                            )
                                    )
                            })
                            || self
                                .record
                                .inputs_view()
                                .close_asked_at()
                                .is_some_and(|close| close <= at)
                    });
                    if retired && self.record.live().has(Kind::Output, output) {
                        self.record.close(at, Kind::Output, output);
                    }
                }
                Record::Produced { at, tts, chunk, text, stamps } => {
                    self.record.record_produced(at, tts, chunk, text, stamps);
                }
                Record::Delivered { at, slot } => {
                    self.record.record_delivered(at, slot.kind, slot.nth);
                }
                Record::FirstAudioDue { at, slot, after_ms } => {
                    self.record.record_first_audio_due(slot.nth, at.at_ms + after_ms);
                }
                Record::Generated { at, generation, chunk } => {
                    self.record.record_generated(at, generation % 255, chunk);
                }
                Record::Fault { at, site, fault, on } => {
                    self.record.record_fault(at, site, fault);
                    if let Some(slot) = on.filter(|_| fault == crate::record::Fault::Hang)
                        && !site.is_close()
                    {
                        self.record.record_read_hung(at, slot.kind, slot.nth);
                    }
                    // A stream that panicked unwound away, and one whose close hangs was let go:
                    // either is gone, and nothing more reaches it.
                    let spent = fault == crate::record::Fault::Panic || site.is_close();
                    if let Some(slot) = on.filter(|_| spent) {
                        self.controls.remove(&slot);
                        if self.record.live().has(slot.kind, slot.nth) {
                            self.record.close(at, slot.kind, slot.nth);
                        }
                        if slot.kind == Kind::Stt {
                            self.end_utterances_on(at, slot.nth);
                        }
                    }
                }
                Record::Remote { at, step } => self.record.record_remote(at, step),
                Record::CancelRefused { at, call_id } => {
                    self.record.record_cancel_refused(at, call_id);
                }
                Record::ProviderFailed { at, kind, recoverable } => {
                    self.record.record_provider_failed(at, kind, recoverable);
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
                Record::Recognition { at, stt, report, emitted } => {
                    self.record.record_recognition(at, stt, report, emitted);
                }
                Record::InputEnded { at, slot } => {
                    self.record.record_input_ended(at, slot.kind, slot.nth);
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
                Record::ProcessorSpan { at, processor, label } => {
                    self.record.record_processor_span(at, processor, label);
                }
                Record::HostBinding { at, object, thread } => {
                    self.record.record_host_binding(at, object, thread);
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
                    self.open_once(at, Kind::Speech, Id::Speech(*speech), detail);
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
                    let nth = self.record.resources(Kind::Hook).find(|o| o.id() == Id::Hook(*hook));
                    if let Some(nth) = nth.map(|o| o.nth())
                        && self.registry.hang(crate::record::Site::Hook)
                    {
                        self.hung_hooks.insert(nth);
                    }
                }
                Step::Handoff(HandoffStep::HookFinished { hook, .. }) => {
                    self.close_by_id(at, Kind::Hook, Id::Hook(*hook));
                }
                // A detached tool ran on past its answer; its final update is its end.
                Step::Tool(ToolStep::Updated { call_id, is_final: true, .. }) => {
                    if let Some(nth) = self.tool_by_call_id(call_id) {
                        self.controls.remove(&Slot { kind: Kind::Tool, nth });
                        self.record.close(at, Kind::Tool, nth);
                    }
                }
                // A body that detached in the instant the driver ended its call on a report of its
                // own: nothing is left to end it (`tool_call_ended`).
                Step::Tool(ToolStep::CallEnded { call_id, end: Some(_), .. }) => {
                    if let Some(nth) = self.tool_by_call_id(call_id)
                        && self.tool_detached(nth)
                    {
                        self.controls.remove(&Slot { kind: Kind::Tool, nth });
                        self.record.close(at, Kind::Tool, nth);
                    }
                }
                // A body that detached after its call had already ended: the driver refuses its
                // report, as it will its final update, so nothing is left for it to end.
                Step::Tool(ToolStep::UpdateRefused { call_id }) => {
                    if let Some(nth) = self.tool_by_call_id(call_id)
                        && self.tool_detached(nth)
                    {
                        self.controls.remove(&Slot { kind: Kind::Tool, nth });
                        self.record.close(at, Kind::Tool, nth);
                    }
                }
                _ => {}
            }
            self.record.record_journal(at, step);
        }
    }

    /// Absorb what the runs entered through AMD's executor told the host: each entry, the events
    /// sent since, and `execute()` once it returned.
    fn absorb_amd_runs(&mut self) {
        let entered = self.entered.clone();
        let mut entered = entered.lock().unwrap();
        for e in entered.iter_mut() {
            if !std::mem::replace(&mut e.recorded, true) {
                let at = self.stamps.next();
                self.record.record_amd_entered(at, e.result.clone());
            }
            let Ok(run) = e.result else { continue };
            if let Some(events) = &mut e.events {
                while let Ok(event) = events.try_recv() {
                    let at = self.stamps.next();
                    self.record.record_amd_event(at, run, event);
                }
            }
            if let Some(execute) = e.execute.as_mut()
                && let Some(outcome) = super::apply::ready_now(execute.as_mut())
            {
                e.execute = None;
                let at = self.stamps.next();
                self.record.record_amd_outcome(at, run, outcome);
            }
            if let Some(close) = e.close.as_mut()
                && super::apply::ready_now(close.as_mut()).is_some()
            {
                e.close = None;
            }
        }
    }

    /// Absorb what the AMD gate observed: a committed turn opens an `AmdTurn` the answerer can
    /// name, and its run's detach closes every turn of that run.
    fn absorb_amd(&mut self) {
        use lk_agents_voice::amd::AmdObservation;
        let seen = std::mem::take(&mut *self.amd.lock().unwrap());
        for (at, run, observation) in seen {
            match &observation {
                AmdObservation::TurnCommitted { turn, .. } => {
                    let detail = Detail::AmdTurn { run: run.0, turn: turn.0 };
                    self.record.open(at, Kind::AmdTurn, Id::Pending, detail);
                }
                AmdObservation::Detached { run: detached } => {
                    let turns: Vec<u32> = self
                        .record
                        .resources(Kind::AmdTurn)
                        .filter(|t| matches!(t.detail(), Detail::AmdTurn { run, .. } if *run == detached.0))
                        .map(|t| t.nth())
                        .filter(|nth| self.record.live().has(Kind::AmdTurn, *nth))
                        .collect();
                    for nth in turns {
                        self.record.close(at, Kind::AmdTurn, nth);
                    }
                }
                _ => {}
            }
            self.record.record_amd(at, run, observation);
        }
    }

    /// Open a driver-owned resource unless that identity is already known. The journal can name one
    /// twice -- a second `SpeechDone`, a hook re-announced -- and a run that opened it twice would
    /// hand the case two references to one thing.
    pub(super) fn open_once(&mut self, at: Observed, kind: Kind, id: Id, detail: Detail) {
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
    /// context nobody has taken. A retry the request stream arms reopens the request after its
    /// backoff with no `ChatRequested` of its own, so its `RetryArmed` plans the speech's request
    /// again and takes the first of its context opened once the backoff is over, until the
    /// speech's request finishes.
    fn attribute_chats(&mut self, from_journal: usize, from_host: usize) {
        for position in from_journal..self.record.journal().len() {
            let speech = match &self.record.journal()[position].1 {
                Step::Generation(GenerationStep::ChatRequested { speech, ctx_ids, .. }) => {
                    let (speech, ctx_ids) = (speech.0, ctx_ids.clone());
                    // A retry armed earlier reopened its request first.
                    self.take_retried_chats();
                    self.take_chat(speech, &ctx_ids, 0);
                    continue;
                }
                Step::Generation(GenerationStep::RetryArmed {
                    speech,
                    leg: ProviderKind::Llm,
                    at,
                    ..
                }) => (speech, at.saturating_duration_since(Ts::ZERO).as_millis() as u64),
                Step::Generation(GenerationStep::LlmRequestFinished { speech, .. }) => {
                    let speech = speech.0;
                    self.retried_chats.retain(|(retried, ..)| *retried != speech);
                    continue;
                }
                _ => continue,
            };
            let (speech, due) = speech;
            let planned =
                self.record.journal()[..position].iter().rev().find_map(|(_, step)| match step {
                    Step::Generation(GenerationStep::ChatRequested {
                        speech: planned,
                        ctx_ids,
                        ..
                    }) if planned == speech => Some((speech.0, ctx_ids.clone(), due)),
                    _ => None,
                });
            self.retried_chats.extend(planned);
            self.take_retried_chats();
        }
        self.take_retried_chats();
        // A request's own metrics or error name its speech exactly, so they have the last word.
        // Context matching alone can confuse a hung open with a later identical request.
        for position in from_host..self.record.host().len() {
            let (request_id, speech) = match &self.record.host()[position].1 {
                Host::Session(SessionEvent::MetricsCollected {
                    metrics,
                    speech: Some(speech),
                    ..
                }) => {
                    let AgentMetrics::Llm(metrics) = &**metrics else { continue };
                    (metrics.request_id.as_str(), *speech)
                }
                Host::Session(SessionEvent::Error(error))
                    if error.source == lk_agents_voice::events::ErrorSource::Llm =>
                {
                    let lk_agents_abi::errors::ProviderError::Local(provider) = &error.error else {
                        continue;
                    };
                    let (Some(request_id), Some(speech)) = (&provider.request_id, error.speech)
                    else {
                        continue;
                    };
                    (request_id.as_str(), speech)
                }
                _ => continue,
            };
            let Some(chat) = self.registry.request_of(request_id) else { continue };
            self.record.attribute_chat(chat, speech.0);
        }
    }

    /// The earliest request of this context nobody has taken, opened at `from_ms` or later, is the
    /// speech's, if one opened.
    fn take_chat(&mut self, speech: u64, ctx_ids: &[String], from_ms: u64) -> bool {
        let chat = self
            .record
            .resources(Kind::Chat)
            .filter(|o| {
                o.opened_at().at_ms >= from_ms
                    && matches!(o.detail(), Detail::Chat { ctx_ids: seen, speech: None, .. } if seen == ctx_ids)
            })
            .map(|o| o.nth())
            .next();
        if let Some(chat) = chat {
            self.record.attribute_chat(chat, speech);
        }
        chat.is_some()
    }

    /// Each armed retry whose reopened request has been seen takes it, oldest first.
    fn take_retried_chats(&mut self) {
        let retried = std::mem::take(&mut self.retried_chats);
        for (speech, ctx_ids, due) in retried {
            if !self.take_chat(speech, &ctx_ids, due) {
                self.retried_chats.push((speech, ctx_ids, due));
            }
        }
    }
}
