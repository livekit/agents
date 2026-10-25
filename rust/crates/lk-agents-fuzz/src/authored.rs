//! Hand-written recordings, for the examples and counterexamples every contract must carry.
//!
//! No driver runs here. A real recording is what the harness saw; these are recordings the
//! driver never produced -- most importantly the ones it must never produce, so that a rule which
//! cannot fail is caught on the day it is written rather than the year nobody notices.
//!
//! The facts are written with the same production types a real recording holds, in the order they
//! happened, so a rule cannot tell an authored recording from a real one. Only the provenance
//! differs, and that is the point: a counterexample validates the checker, never the runtime.
//! Nothing outside `rules/` should use this.
use std::sync::Arc;

use lk_agents_abi::FunctionCall;
use lk_agents_voice::{
    agent_activity::scheduling::{SpeechId, SpeechKind},
    events::{SessionEvent, SpeechOutcome, ToolEndStatus, ToolLifecycle},
    speech_handle::{InputDetails, SpeechHandle},
    trace::Step,
};

use crate::{
    input::Input,
    record::{
        Answer, Detail, Handle, Host, Id, Kind, Observed, Phase, Recording, Sink, Tag, Task,
        TaskEnd,
    },
};

/// Builds a `Recording` fact by fact, as if a driver had produced it.
///
/// Each call takes the next stamp, so the order of the calls is the order of occurrence. That is
/// the whole interface: a rule that depends on ordering is tested by writing the facts in the
/// wrong order, which is what most counterexamples here are.
#[derive(Default)]
pub struct Authored {
    recording: Recording,
    seq: u64,
    later_ms: u64,
}

impl Authored {
    pub fn new() -> Self {
        Self::default()
    }

    fn next(&mut self) -> Observed {
        self.seq += 1;
        Observed::at(self.seq, self.seq + self.later_ms)
    }

    // --- the host's side -----------------------------------------------------------------------

    pub fn host(mut self, event: Host) -> Self {
        let at = self.next();
        self.recording.record_host(at, event);
        self
    }

    /// The host received this session event.
    pub fn session(self, event: SessionEvent) -> Self {
        self.host(Host::Session(event))
    }

    /// The host was told a speech exists. Builds a real, idle `SpeechHandle` for it, and opens the
    /// speech in the run's own history so a `Ref<Speech>` resolves to it.
    pub fn speech_created(mut self, speech: u64, kind: SpeechKind) -> Self {
        let at = self.next();
        let user_initiated = matches!(kind, SpeechKind::Say | SpeechKind::GenerateReply);
        self.recording.open(
            at,
            Kind::Speech,
            Id::Speech(SpeechId(speech)),
            Detail::Speech { kind, user_initiated },
        );
        let handle = SpeechHandle::new(
            SpeechId(speech),
            format!("speech_{speech}"),
            true,
            InputDetails::default(),
        );
        self.session(SessionEvent::SpeechCreated { speech: Arc::new(handle), kind, user_initiated })
    }

    pub fn speech_done(self, speech: u64, outcome: SpeechOutcome) -> Self {
        self.session(SessionEvent::SpeechDone { speech: SpeechId(speech), outcome })
    }

    pub fn tool_started(self, call_id: &str) -> Self {
        self.session(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
            call: FunctionCall {
                id: format!("item_{call_id}"),
                call_id: call_id.into(),
                arguments: "{}".into(),
                name: "lookup".into(),
                created_at: std::time::UNIX_EPOCH,
                extra: Default::default(),
                group_id: None,
            },
        }))
    }

    pub fn tool_ended(self, call_id: &str, status: ToolEndStatus) -> Self {
        self.session(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
            id: format!("item_{call_id}"),
            call_id: call_id.into(),
            message: None,
            status,
        }))
    }

    // --- the driver's side ---------------------------------------------------------------------

    pub fn journal(mut self, step: Step) -> Self {
        let at = self.next();
        self.recording.record_journal(at, step);
        self
    }

    /// A speech handle as the public API would report it. A handle that is already done is
    /// stamped `done_since` here, because a rule that measures how long it has been done needs
    /// one.
    pub fn speech(mut self, speech: u64, mut handle: Handle) -> Self {
        if handle.done && handle.done_since.is_none() {
            handle.done_since = Some(self.next());
        }
        self.recording.record_handle(speech, handle);
        self
    }

    // --- what the run opened -------------------------------------------------------------------

    /// Open a resource of this kind, returning the builder. Its index is the next of its kind,
    /// exactly as in a real run, so the `nth` a rule reads is predictable from the call order.
    pub fn opens(mut self, kind: Kind, detail: Detail) -> Self {
        let at = self.next();
        let nth = self.recording.live().opened(kind);
        self.recording.open(at, kind, Id::Slot(nth), detail);
        self
    }

    /// One model request the driver planned for `speech`.
    pub fn chat(self, speech: u64) -> Self {
        self.opens(
            Kind::Chat,
            Detail::Chat { ctx_ids: vec![], speech: Some(speech), tools: vec![] },
        )
    }

    /// One tool body the runner was handed, for `speech`.
    pub fn tool_call(self, call_id: &str, speech: u64) -> Self {
        self.opens(
            Kind::Tool,
            Detail::Tool { call_id: call_id.into(), name: "lookup".into(), speech },
        )
    }

    /// One synthesis stream.
    pub fn tts(self) -> Self {
        self.opens(Kind::Tts, Detail::None)
    }

    /// The case attached an output instance.
    pub fn output(self, audio: bool) -> Self {
        self.opens(Kind::Output, Detail::Output { audio })
    }

    /// The driver closed this resource.
    pub fn closed(mut self, kind: Kind, nth: u32) -> Self {
        let at = self.next();
        self.recording.close(at, kind, nth);
        self
    }

    // --- what left the session -------------------------------------------------------------------

    pub fn sink(mut self, sink: Sink) -> Self {
        let at = self.next();
        self.recording.record_sink(at, sink);
        self
    }

    /// One chunk of audio the n-th synthesis stream produced.
    pub fn produced(mut self, tts: u32, chunk: u16, text: &str) -> Self {
        let at = self.next();
        self.recording.record_produced(at, tts, chunk, text.into());
        self
    }

    /// One frame the driver pushed to the n-th stream of a provider kind, decoded to its sources.
    pub fn forwarded(mut self, kind: Kind, nth: u32, tags: Vec<Tag>) -> Self {
        let at = self.next();
        self.recording.record_forwarded(at, kind, nth, tags);
        self
    }

    // --- what the case did -----------------------------------------------------------------------

    /// The case sent this input, and it was taken.
    pub fn sent(self, input: Input) -> Self {
        self.attempted(input, true)
    }

    /// The case sent this input, it was taken, and it opened these resources.
    ///
    /// They share the input's stamp, as a real run records them. A rule pairs a resource with its
    /// caller on that equality, so stamping them apart would test a run that cannot happen.
    pub fn sent_opening(mut self, input: Input, opens: &[(Kind, Detail)]) -> Self {
        let at = self.next();
        self.recording.record_applied(at, input, true);
        for (kind, detail) in opens {
            let nth = self.recording.live().opened(*kind);
            self.recording.open(at, *kind, Id::Slot(nth), detail.clone());
        }
        self
    }

    /// The case sent this input and the receiver refused it.
    pub fn refused(self, input: Input) -> Self {
        self.attempted(input, false)
    }

    fn attempted(mut self, input: Input, accepted: bool) -> Self {
        let at = self.next();
        self.recording.record_applied(at, input, accepted);
        self
    }

    /// Developer code made this API call and is awaiting what it returns.
    pub fn api_call(self, input: Input) -> Self {
        self.sent_opening(input, &[(Kind::Answer, Detail::None)])
    }

    /// The most recent API call returned this.
    pub fn returns(self, value: Answer) -> Self {
        let call = self.recording.live().opened(Kind::Answer).saturating_sub(1);
        self.result(call, value)
    }

    /// The driver answered the n-th call it owed.
    pub fn result(mut self, answer: u32, result: Answer) -> Self {
        let at = self.next();
        self.recording.record_result(at, answer, result);
        self
    }

    /// The runner applied one more input after everything so far.
    ///
    /// An `Advance`, because a rule that gives the driver "one input of grace" counts only inputs
    /// that yield to the executor.
    pub fn then_act(self) -> Self {
        self.sent(Input::Advance { ms: crate::input::Millis(1) })
    }

    // --- where the run got to ---------------------------------------------------------------------

    /// The runner stopped adding work here.
    pub fn released(mut self) -> Self {
        let at = self.next();
        self.recording.enter_phase(Phase::Releasing);
        self.recording.mark_released(at);
        self
    }

    /// The test had answered everything the driver was waiting on.
    pub fn everything_answered(mut self) -> Self {
        let at = self.next();
        self.recording.enter_phase(Phase::Settling);
        self.recording.mark_settling(at);
        self
    }

    pub fn driver_finished(mut self) -> Self {
        self.recording.enter_phase(Phase::Closed);
        self.recording.set_driver_finished(true);
        self
    }

    pub fn driver_panicked(mut self, panic: &str) -> Self {
        self.recording.set_driver_panic(panic.to_owned());
        self.driver_finished()
    }

    /// The session spawned a task here, which the run's join found ended this way.
    #[track_caller]
    pub fn task(mut self, end: TaskEnd) -> Self {
        let spawned = self.next();
        let site = std::panic::Location::caller();
        self.recording.record_task(Task { spawned, site, end });
        self
    }

    /// Virtual time passes with nothing else observed.
    pub fn later(mut self, ms: u64) -> Self {
        self.later_ms += ms;
        self
    }

    pub fn done(mut self) -> Recording {
        self.recording.set_now(Observed::at(self.seq + 1, self.seq + 1 + self.later_ms));
        self.recording
    }
}
