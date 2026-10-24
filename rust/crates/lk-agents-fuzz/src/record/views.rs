//! What a rule reads: four views over one recording.
//!
//! A run is inputs in and outputs out. [`Recording::inputs_view`] is what the case sent. The other
//! three views are the driver's output, seen from three places: [`Recording::host_view`] is what
//! developer code saw, [`Recording::internal_view`] is what the driver wrote about itself, and
//! [`Recording::inference_view`] is what the models received. A rule reads the views its
//! guarantee names and nothing else in this crate; a rule that reads only the host view is a
//! public contract, one that reads two views is a correspondence claim between two accounts of
//! the same output.
//!
//! Every noun and question here is taken from a guarantee sentence. Stamps are ordered, so "the
//! speech was created before it was announced done" is `created < done` and nothing else.
use std::collections::BTreeSet;

use lk_agents_abi::voice::events::CloseReason;
use lk_agents_voice::{
    agent_activity::scheduling::SpeechKind,
    events::{SessionEvent, SpeechOutcome, ToolLifecycle},
    trace::{
        CloseStep, GenerationStep, InterruptCause, InterruptStep, PlayoutStep, SinkCall,
        SpeechStep, Step, ToolStep,
    },
};

use super::{
    Answer, Applied, Detail, Handle, Host, Id, Kind, Observed, Opened, Recording, Sink, Tag,
};
use crate::{check::Subject, input::Input};

impl Recording {
    /// What the case sent.
    pub fn inputs_view(&self) -> InputsView<'_> {
        InputsView(self)
    }

    /// What developer code saw: session events, the API calls it awaited, handles, the outputs it
    /// attached, the tool bodies its runner ran. One of the three places the driver's output can be seen.
    pub fn host_view(&self) -> HostView<'_> {
        HostView(self)
    }

    /// What the driver wrote about itself: its decisions, refusals and timers. The second place
    /// the driver's output can be seen, and the only one where its reasoning is.
    pub fn internal_view(&self) -> InternalView<'_> {
        InternalView(self)
    }

    /// What the models received: the frames, requests and text the driver sent them. The third
    /// place the driver's output can be seen.
    pub fn inference_view(&self) -> InferenceView<'_> {
        InferenceView::new(self)
    }
}

// ==============================================================================================
// Inputs
// ==============================================================================================

/// What the case sent: every input, taken or refused, in order.
#[derive(Clone, Copy)]
pub struct InputsView<'a>(&'a Recording);

impl<'a> InputsView<'a> {
    /// Every input sent, taken or refused, in order.
    pub fn sent(&self) -> &'a [Applied] {
        &self.0.applied
    }

    /// The inputs the product took, in order.
    pub fn accepted(&self) -> impl Iterator<Item = &'a Applied> {
        self.0.applied.iter().filter(|a| a.accepted)
    }

    /// The inputs the product took strictly between two stamps, in order.
    pub fn accepted_between(
        &self,
        from: Observed,
        to: Observed,
    ) -> impl Iterator<Item = &'a Applied> {
        self.accepted().filter(move |a| from < a.at && a.at < to)
    }

    /// When the host first asked the session to wind down: a drain, or the close itself. A
    /// `say` or `generate_reply` made after this reaches a session that has begun closing.
    pub fn closing_asked_at(&self) -> Option<Observed> {
        self.accepted()
            .find(|a| matches!(a.input, Input::Drain { .. } | Input::Close { .. }))
            .map(|a| a.at)
    }

    /// When the test had answered everything the driver was waiting on -- every request,
    /// synthesis, playout, hook and hold -- and found nothing new to answer twice over. From here
    /// the driver waits on nobody but itself, and the runner closes the session only after this.
    /// `None` when the run never got there.
    pub fn everything_answered_at(&self) -> Option<Observed> {
        self.0.settling
    }

    /// When the host asked to close the session.
    pub fn close_asked_at(&self) -> Option<Observed> {
        self.0.applied.iter().find(|a| matches!(a.input, Input::Close { .. })).map(|a| a.at)
    }

    /// Whether the driver has had a turn since `since`.
    ///
    /// Only `Advance` and the awaits yield to the executor: sending a command, feeding a stream
    /// and answering a fake all return without awaiting, by design, so that a case can build a
    /// burst. A rule that allows the driver "one input of grace" counts those and nothing else,
    /// or it fires on an effect the driver was never scheduled to produce.
    pub fn driver_ran_after(&self, since: Observed) -> bool {
        self.0.applied.iter().any(|a| a.at > since && yields(&a.input))
    }

    /// The driver's first turn after `at`: from the first yielding input after `at` to the input
    /// applied after it, which is where the effects of that turn have all landed. `None` until
    /// the driver has had a turn at all.
    pub fn driver_turn_after(&self, at: Observed) -> Option<Turn> {
        let applied = &self.0.applied;
        let began = applied.iter().position(|a| a.at > at && yields(&a.input))?;
        Some(Turn { began: applied[began].at, ended: applied.get(began + 1).map(|a| a.at) })
    }

    /// The audio input switch: the leg's enable, and attaching or detaching the microphone.
    pub fn audio_input(&self) -> Switch {
        self.switch(|input| match input {
            Input::SetIoEnabled {
                leg: lk_agents_voice::agent_session::IoKind::AudioInput,
                enabled,
            } => Some(*enabled),
            Input::SetAudioInput { attach } => Some(*attach),
            _ => None,
        })
    }

    /// The transcription path: the leg's enable, and attaching or detaching the text output.
    pub fn transcription_output(&self) -> Switch {
        self.switch(|input| match input {
            Input::SetIoEnabled {
                leg: lk_agents_voice::agent_session::IoKind::Transcription,
                enabled,
            } => Some(*enabled),
            Input::SetTranscriptionOutput { attach } => Some(*attach),
            _ => None,
        })
    }

    /// The audio output: attached or detached.
    pub fn audio_output(&self) -> Switch {
        self.switch(|input| match input {
            Input::SetAudioOutput { attach } => Some(*attach),
            _ => None,
        })
    }

    fn switch(&self, toggle: impl Fn(&Input) -> Option<bool>) -> Switch {
        Switch {
            changes: self
                .accepted()
                .filter_map(|a| toggle(&a.input).map(|on| (a.at, on)))
                .collect(),
        }
    }
}

/// Whether an input gives the driver a turn.
fn yields(input: &Input) -> bool {
    matches!(
        input,
        Input::Advance { .. }
            | Input::AwaitAnswer { .. }
            | Input::AwaitSpeech { .. }
            | Input::AwaitExit { .. }
    )
}

/// One turn of the driver: when it began, and when the next input landed after it.
#[derive(Clone, Copy, Debug)]
pub struct Turn {
    pub began: Observed,
    /// `None` while that turn is the latest thing recorded.
    pub ended: Option<Observed>,
}

/// Something the case switches off and on. On from the start: every leg is attached before the
/// first input.
#[derive(Clone, Debug)]
pub struct Switch {
    /// The accepted changes, in order.
    changes: Vec<(Observed, bool)>,
}

impl Switch {
    /// Whether it was on at `at`, as of the last change at or before it.
    pub fn on_at(&self, at: Observed) -> bool {
        self.changes.iter().rev().find(|(when, _)| *when <= at).is_none_or(|(_, on)| *on)
    }

    /// Whether it was switched off strictly between two stamps.
    pub fn off_within(&self, from: Observed, to: Observed) -> bool {
        self.changes.iter().any(|(when, on)| !on && from < *when && *when < to)
    }

    /// Whether it was on at `from` and never switched off before `to`.
    pub fn on_throughout(&self, from: Observed, to: Observed) -> bool {
        self.on_at(from) && !self.off_within(from, to)
    }

    /// Whether it was ever switched off.
    pub fn ever_off(&self) -> bool {
        self.changes.iter().any(|(_, on)| !on)
    }
}

/// A stretch of the run, open-ended when it had not closed by the end.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub from: Observed,
    pub to: Option<Observed>,
}

impl Window {
    /// Whether the window and the stretch `from..=to` share any instant.
    pub fn overlaps(&self, from: Observed, to: Observed) -> bool {
        self.from <= to && self.to.is_none_or(|end| end >= from)
    }
}

// ==============================================================================================
// The host's view
// ==============================================================================================

/// What developer code saw.
#[derive(Clone, Copy)]
pub struct HostView<'a>(&'a Recording);

impl<'a> HostView<'a> {
    /// The session events the host received, in order.
    pub fn events(&self) -> Vec<(Observed, &'a SessionEvent)> {
        self.0
            .host
            .iter()
            .filter_map(|(at, host)| match host {
                Host::Session(event) => Some((*at, event)),
                _ => None,
            })
            .collect()
    }

    /// Every speech the host knows of: one it was told about, or one it holds a handle for.
    pub fn speeches(&self) -> Vec<SpeechView<'a>> {
        let mut ids: BTreeSet<u64> = self.0.handles.keys().copied().collect();
        ids.extend(self.0.index.host_by_speech.keys().copied());
        ids.extend(self.0.opened.iter().filter_map(|o| match o.id {
            Id::Speech(id) if o.kind == Kind::Speech => Some(id.0),
            _ => None,
        }));
        ids.into_iter().map(|id| self.speech(id)).collect()
    }

    /// One speech, by the id the driver minted for it. An id nothing happened to reads as empty.
    pub fn speech(&self, id: u64) -> SpeechView<'a> {
        SpeechView { r: self.0, id }
    }

    /// Every API call developer code made that returns a value -- `say`, `interrupt`,
    /// `wait_for_idle`, `wait_for_idle_and_hold`, `commit_user_turn`, `handoff`, `drain`, a DTMF start --
    /// in the order they were made. Each is a future the developer awaits.
    pub fn api_calls(&self) -> Vec<ApiCallView<'a>> {
        self.0
            .opened
            .iter()
            .filter(|o| o.kind == Kind::Answer)
            .map(|opened| ApiCallView { r: self.0, opened })
            .collect()
    }

    /// Every tool body the host's runner ran.
    pub fn tools_run(&self) -> Vec<ToolRunView<'a>> {
        self.0
            .opened
            .iter()
            .filter(|o| o.kind == Kind::Tool)
            .map(|opened| ToolRunView { r: self.0, opened })
            .collect()
    }

    /// Whether the runner ran a body for this call.
    pub fn tool_ran(&self, call_id: &str) -> bool {
        self.tools_run().iter().any(|tool| tool.call_id() == call_id)
    }

    /// The output instances the host attached, in order.
    pub fn outputs(&self) -> Vec<OutputView<'a>> {
        self.0
            .opened
            .iter()
            .filter(|o| o.kind == Kind::Output)
            .map(|opened| OutputView { r: self.0, opened })
            .collect()
    }

    /// Every push of audio into one of the host's outputs, decoded to the sources it came from.
    pub fn audio_pushed(&self) -> Vec<AudioPushed<'a>> {
        self.0
            .sink
            .iter()
            .filter_map(|(at, sink)| match sink {
                Sink::Audio { output, tags } => {
                    Some(AudioPushed { at: *at, output: *output, tags })
                }
                _ => None,
            })
            .collect()
    }

    /// Every floor hold a caller took.
    pub fn holds(&self) -> Vec<HoldView<'a>> {
        self.0
            .opened
            .iter()
            .filter(|o| o.kind == Kind::Hold)
            .map(|opened| HoldView { r: self.0, opened })
            .collect()
    }

    /// When the case ended the stream a `say` opened for it to feed, by the stream's number.
    /// `None` while the case still holds it open.
    pub fn stream_ended_at(&self, stream: u32) -> Option<Observed> {
        self.0
            .opened
            .iter()
            .find(|o| o.kind == Kind::Stream && o.nth == stream)
            .and_then(|opened| closed_at(self.0, opened))
    }
}

/// One speech as the host knows it: what it was told, and what the handle reports.
#[derive(Clone, Copy)]
pub struct SpeechView<'a> {
    r: &'a Recording,
    pub id: u64,
}

impl<'a> SpeechView<'a> {
    /// What kind of speech the driver created, once it said.
    pub fn kind(&self) -> Option<SpeechKind> {
        self.r.opened.iter().find_map(|o| match (&o.id, &o.detail) {
            (Id::Speech(id), Detail::Speech { kind, .. }) if id.0 == self.id => Some(*kind),
            _ => None,
        })
    }

    /// Started by the driver on a preflight transcript, before the turn ended.
    pub fn is_speculative(&self) -> bool {
        self.kind() == Some(SpeechKind::Preemptive)
    }

    /// When the host was told this speech exists.
    pub fn created_at(&self) -> Option<Observed> {
        self.events()
            .find(|(_, e)| matches!(e, SessionEvent::SpeechCreated { .. }))
            .map(|(at, _)| at)
    }

    /// Every time the host was told this speech finished, with the outcome it was told.
    pub fn done_announcements(&self) -> Vec<(Observed, SpeechOutcome)> {
        self.events()
            .filter_map(|(at, e)| match e {
                SessionEvent::SpeechDone { outcome, .. } => Some((at, *outcome)),
                _ => None,
            })
            .collect()
    }

    /// Whether the host holds a handle for it at all.
    pub fn is_known(&self) -> bool {
        self.handle().is_some()
    }

    /// The handle says the speech is done.
    pub fn is_done(&self) -> bool {
        self.handle().is_some_and(|h| h.done)
    }

    /// The first instant the handle was seen done.
    pub fn done_at(&self) -> Option<Observed> {
        self.handle().and_then(|h| h.done_since)
    }

    pub fn is_interrupted(&self) -> bool {
        self.handle().is_some_and(|h| h.interrupted)
    }

    pub fn is_scheduled(&self) -> bool {
        self.handle().is_some_and(|h| h.scheduled)
    }

    /// The error the handle reports, if the speech failed.
    pub fn error(&self) -> Option<&'a str> {
        self.handle().and_then(|h| h.error.as_deref())
    }

    /// Done, not interrupted, no error: the speech ran to its natural end.
    pub fn completed_normally(&self) -> bool {
        self.handle().is_some_and(|h| h.done && !h.interrupted && h.error.is_none())
    }

    /// How the speech ended, as the handle reports it. `None` while it is not done.
    pub fn outcome(&self) -> Option<SpeechOutcome> {
        let handle = self.handle().filter(|h| h.done)?;
        Some(if handle.error.is_some() {
            SpeechOutcome::Failed
        } else if handle.interrupted {
            SpeechOutcome::Interrupted
        } else {
            SpeechOutcome::Completed
        })
    }

    fn handle(&self) -> Option<&'a Handle> {
        self.r.handles.get(&self.id)
    }

    fn events(&self) -> impl Iterator<Item = (Observed, &'a SessionEvent)> {
        let r = self.r;
        let positions = r.index.host_by_speech.get(&self.id).map_or(&[][..], Vec::as_slice);
        positions.iter().filter_map(move |p| match &r.host[*p] {
            (at, Host::Session(event)) => Some((*at, event)),
            _ => None,
        })
    }
}

impl std::fmt::Debug for SpeechView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.handle() {
            Some(h) => write!(
                f,
                "speech {} (done: {}, interrupted: {}, scheduled: {}, error: {:?})",
                self.id, h.done, h.interrupted, h.scheduled, h.error
            ),
            None => write!(f, "speech {} (no handle)", self.id),
        }
    }
}

/// One API call developer code made and awaited: which call, when, and what it returned.
#[derive(Clone, Copy)]
pub struct ApiCallView<'a> {
    r: &'a Recording,
    opened: &'a Opened,
}

impl<'a> ApiCallView<'a> {
    /// Which call, counting from zero: what a `Ref<Answer>` in a case names.
    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    /// The call itself: the input whose variant is the API method and whose fields are its
    /// arguments. Paired by the one instant the call and what it opened share.
    pub fn input(&self) -> &'a Input {
        self.r
            .applied
            .iter()
            .find(|a| a.at == self.opened.at)
            .map(|a| &a.input)
            .expect("an API call is one of the test's inputs")
    }

    pub fn called_at(&self) -> Observed {
        self.opened.at
    }

    /// What the call returned, and when. `None` while the developer is still awaiting it.
    pub fn returned(&self) -> Option<(Observed, &'a Answer)> {
        self.r.results.iter().find(|(_, nth, _)| *nth == self.opened.nth).map(|(at, _, a)| (*at, a))
    }

    /// The future was dropped without ever returning: the driver let go of the channel.
    pub fn never_returned(&self) -> bool {
        matches!(self.returned(), Some((_, Answer::Lost)))
    }

    /// The stream the same `say` opened for the test to feed, if it opened one.
    pub fn stream(&self) -> Option<u32> {
        self.r
            .opened
            .iter()
            .find(|o| o.kind == Kind::Stream && o.at == self.opened.at)
            .map(|o| o.nth)
    }
}

/// One tool body the host's runner ran, and what the host was told about that call.
#[derive(Clone, Copy)]
pub struct ToolRunView<'a> {
    r: &'a Recording,
    opened: &'a Opened,
}

impl<'a> ToolRunView<'a> {
    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    /// The id the model gave the call.
    pub fn call_id(&self) -> &'a str {
        match &self.opened.detail {
            Detail::Tool { call_id, .. } => call_id,
            _ => "",
        }
    }

    /// The speech the call was made for.
    pub fn speech(&self) -> u64 {
        match &self.opened.detail {
            Detail::Tool { speech, .. } => *speech,
            _ => 0,
        }
    }

    /// Every time the host was told this call started.
    pub fn started_announcements(&self) -> Vec<Observed> {
        self.told(|update| matches!(update, ToolLifecycle::Started { .. }))
    }

    /// Every time the host was told this call ended.
    pub fn ended_announcements(&self) -> Vec<Observed> {
        self.told(|update| matches!(update, ToolLifecycle::Ended { .. }))
    }

    fn told(&self, pick: impl Fn(&ToolLifecycle) -> bool) -> Vec<Observed> {
        let positions =
            self.r.index.host_by_tool.get(self.call_id()).map_or(&[][..], Vec::as_slice);
        positions
            .iter()
            .filter_map(|p| match &self.r.host[*p] {
                (at, Host::Session(SessionEvent::ToolExecutionUpdated(update))) if pick(update) => {
                    Some(*at)
                }
                _ => None,
            })
            .collect()
    }
}

/// One output instance the host attached.
#[derive(Clone, Copy)]
pub struct OutputView<'a> {
    r: &'a Recording,
    opened: &'a Opened,
}

impl<'a> OutputView<'a> {
    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    /// An audio output, as opposed to a transcription one.
    pub fn is_audio(&self) -> bool {
        matches!(self.opened.detail, Detail::Output { audio: true })
    }

    pub fn attached_at(&self) -> Observed {
        self.opened.at
    }

    /// When the next output of the same kind was attached in its place. `None` while it is
    /// still the current one.
    pub fn replaced_at(&self) -> Option<Observed> {
        let audio = matches!(self.opened.detail, Detail::Output { audio: true });
        self.r
            .opened
            .iter()
            .find(|o| {
                o.kind == Kind::Output
                    && matches!(o.detail, Detail::Output { audio: a } if a == audio)
                    && o.at > self.opened.at
            })
            .map(|o| o.at)
    }

    /// Every call that landed on this instance, in order.
    pub fn received(&self) -> Vec<(Observed, &'a Sink)> {
        self.r
            .sink
            .iter()
            .filter(|(_, sink)| sink_output(sink) == Some(self.opened.nth))
            .map(|(at, sink)| (*at, sink))
            .collect()
    }
}

/// The instance a sink call landed on, for the calls that name one.
fn sink_output(sink: &Sink) -> Option<u32> {
    match sink {
        Sink::Audio { output, .. }
        | Sink::Flush { output, .. }
        | Sink::Clear { output, .. }
        | Sink::Text { output, .. }
        | Sink::TextFlush { output }
        | Sink::Paused { output }
        | Sink::Resumed { output }
        | Sink::Reset { output, .. } => Some(*output),
        Sink::Played { .. }
        | Sink::Started { .. }
        | Sink::Progressed { .. }
        | Sink::Recorder(_) => None,
    }
}

/// One push of audio into an output, decoded to where each sample came from.
#[derive(Clone, Copy, Debug)]
pub struct AudioPushed<'a> {
    pub at: Observed,
    pub output: u32,
    pub tags: &'a [Tag],
}

/// One floor hold a caller took.
#[derive(Clone, Copy)]
pub struct HoldView<'a> {
    r: &'a Recording,
    opened: &'a Opened,
}

impl HoldView<'_> {
    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    pub fn taken_at(&self) -> Observed {
        self.opened.at
    }

    /// When the driver released it. `None` while it is still held.
    pub fn released_at(&self) -> Option<Observed> {
        closed_at(self.r, self.opened)
    }
}

fn closed_at(r: &Recording, opened: &Opened) -> Option<Observed> {
    r.closed
        .iter()
        .find(|(_, kind, nth)| *kind == opened.kind && *nth == opened.nth)
        .map(|(at, ..)| *at)
}

// ==============================================================================================
// The driver's own view
// ==============================================================================================

/// What the driver wrote about itself: its journal, verbatim.
#[derive(Clone, Copy)]
pub struct InternalView<'a>(&'a Recording);

impl<'a> InternalView<'a> {
    /// Every step the driver journalled, in order.
    pub fn steps(&self) -> &'a [(Observed, Step)] {
        &self.0.journal
    }

    /// What the driver journalled about one speech.
    pub fn speech(&self, id: u64) -> SpeechTrace<'a> {
        SpeechTrace { r: self.0, id }
    }

    /// Every interruption the driver journalled, with the speech and the cause it named.
    pub fn interruptions(&self) -> Vec<Interruption> {
        self.0
            .journal
            .iter()
            .filter_map(|(at, step)| match step {
                Step::Interrupt(InterruptStep::Interrupted { speech, cause, .. }) => {
                    Some(Interruption { at: *at, speech: speech.0, cause: *cause })
                }
                _ => None,
            })
            .collect()
    }

    /// When the driver began closing, and why: the host asked, or its own error policy did.
    pub fn close_requested(&self) -> Option<(Observed, CloseReason)> {
        self.0.journal.iter().find_map(|(at, step)| match step {
            Step::Close(CloseStep::Requested { reason, .. }) => Some((*at, *reason)),
            _ => None,
        })
    }

    /// Whether the driver's own error policy asked to close after `at`: any close request with
    /// `CloseReason::Error`, not only the first close of the run, which may be the host's.
    pub fn error_close_requested_after(&self, at: Observed) -> bool {
        self.0.journal.iter().any(|(when, step)| {
            *when > at
                && matches!(
                    step,
                    Step::Close(CloseStep::Requested { reason: CloseReason::Error, .. })
                )
        })
    }

    /// Whether a newer commit superseded the one waiting at `at`: the driver drops the older
    /// commit's future on purpose, and the API reports it as superseded.
    pub fn commit_superseded_after(&self, at: Observed) -> bool {
        self.0.journal.iter().any(|(when, step)| {
            *when > at
                && matches!(step, Step::Turn(lk_agents_voice::trace::TurnStep::CommitSuperseded))
        })
    }

    /// Every tool call the driver started, and whether it handed it to the runner (`hosted`) or
    /// answered the model in place.
    pub fn tool_calls_started(&self) -> Vec<(Observed, &'a str, bool)> {
        self.0
            .journal
            .iter()
            .filter_map(|(at, step)| match step {
                Step::Tool(ToolStep::CallStarted { call_id, hosted, .. }) => {
                    Some((*at, call_id.as_str(), *hosted))
                }
                _ => None,
            })
            .collect()
    }

    /// The stretches in which the gate fed the STT leg silence instead of the room.
    pub fn stt_gate_closed(&self) -> Vec<Window> {
        let mut windows = Vec::new();
        let mut closed_from = None;
        for (at, step) in &self.0.journal {
            let Step::Playout(PlayoutStep::GatePublished { stt_real, .. }) = step else { continue };
            match (closed_from, stt_real) {
                (None, false) => closed_from = Some(*at),
                (Some(from), true) => {
                    windows.push(Window { from, to: Some(*at) });
                    closed_from = None;
                }
                _ => {}
            }
        }
        if let Some(from) = closed_from {
            windows.push(Window { from, to: None });
        }
        windows
    }
}

/// One interruption the driver journalled.
#[derive(Clone, Copy, Debug)]
pub struct Interruption {
    pub at: Observed,
    pub speech: u64,
    pub cause: InterruptCause,
}

/// What the driver journalled about one speech, in order.
#[derive(Clone, Copy)]
pub struct SpeechTrace<'a> {
    r: &'a Recording,
    id: u64,
}

impl<'a> SpeechTrace<'a> {
    /// Every step naming this speech, in order.
    pub fn steps(&self) -> impl Iterator<Item = (Observed, &'a Step)> {
        let r = self.r;
        let positions = r.index.journal_by_speech.get(&self.id).map_or(&[][..], Vec::as_slice);
        positions.iter().map(move |p| (r.journal[*p].0, &r.journal[*p].1))
    }

    /// When the driver journalled interrupting it.
    pub fn interrupted_at(&self) -> Option<Observed> {
        self.first(|s| matches!(s, Step::Interrupt(InterruptStep::Interrupted { .. })))
    }

    /// When the turn authorized it to play.
    pub fn authorized_at(&self) -> Option<Observed> {
        self.first(|s| matches!(s, Step::Speech(SpeechStep::Authorized { .. })))
    }

    /// When its first audio was pushed to the sink.
    pub fn first_sink_push_at(&self) -> Option<Observed> {
        self.first(|s| {
            matches!(s, Step::Playout(PlayoutStep::SinkCalled { call: SinkCall::Push { .. }, .. }))
        })
    }

    /// When the driver armed a retry for one of its requests.
    pub fn retry_armed_at(&self) -> Option<Observed> {
        self.first(|s| matches!(s, Step::Generation(GenerationStep::RetryArmed { .. })))
    }

    /// When the driver first asked the voice to synthesize real words for it.
    pub fn synthesis_requested_at(&self) -> Option<Observed> {
        self.first(|s| {
            matches!(s, Step::Generation(GenerationStep::TtsRequested { text, .. }) if !text.trim().is_empty())
        })
    }

    /// Whether any of its audio was reported played.
    pub fn audio_played(&self) -> bool {
        self.first(
            |s| matches!(s, Step::Playout(PlayoutStep::AudioPlayed { frames, .. }) if *frames > 0),
        )
        .is_some()
    }

    /// The transcript the driver forwarded for it, piece by piece, in order.
    pub fn transcript_forwarded(&self) -> Vec<&'a str> {
        self.steps()
            .filter_map(|(_, step)| match step {
                Step::Playout(PlayoutStep::TranscriptForwarded { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn first(&self, pick: impl Fn(&Step) -> bool) -> Option<Observed> {
        self.steps().find(|(_, step)| pick(step)).map(|(at, _)| at)
    }
}

// ==============================================================================================
// The models' view
// ==============================================================================================

/// What the models received: every stream the driver opened to one, and what went down it.
#[derive(Clone, Copy)]
pub struct InferenceView<'a> {
    r: &'a Recording,
}

impl<'a> InferenceView<'a> {
    fn new(r: &'a Recording) -> Self {
        InferenceView { r }
    }

    /// Every stream of one kind the driver opened, live or not, in order.
    pub fn streams(&self, kind: Kind) -> Vec<StreamView<'_, 'a>> {
        self.r
            .opened
            .iter()
            .filter(|o| o.kind == kind)
            .map(|opened| StreamView { view: self, opened })
            .collect()
    }

    /// Every request the driver made of the model.
    pub fn requests(&self) -> Vec<RequestView<'a>> {
        self.r
            .opened
            .iter()
            .filter(|o| o.kind == Kind::Chat)
            .map(|opened| RequestView { opened })
            .collect()
    }

    /// The requests the driver made for one speech, as its own metrics attribute them.
    pub fn requests_for(&self, speech: u64) -> Vec<u32> {
        self.requests()
            .into_iter()
            .filter(|q| q.speech() == Some(speech))
            .map(|q| q.nth())
            .collect()
    }

    /// The speech a request was made for, once the driver has said.
    pub fn request_for(&self, request: u32) -> Option<u64> {
        self.requests().into_iter().find(|q| q.nth() == request).and_then(|q| q.speech())
    }

    /// Whether the voice produced this chunk of this synthesis stream.
    pub fn synthesized(&self, tts: u32, chunk: u16) -> bool {
        self.r.index.produced_chunks.contains(&(tts, chunk))
    }

    /// Every microphone frame a listening stream heard, in order.
    pub fn frames_heard(&self) -> Vec<Heard<'a>> {
        self.r
            .forwarded
            .iter()
            .map(|(at, kind, nth, tags)| Heard { at: *at, kind: *kind, stream: *nth, tags })
            .collect()
    }
}

/// One stream the driver opened to a model.
#[derive(Clone, Copy)]
pub struct StreamView<'v, 'a> {
    view: &'v InferenceView<'a>,
    opened: &'a Opened,
}

impl StreamView<'_, '_> {
    pub fn kind(&self) -> Kind {
        self.opened.kind
    }

    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    pub fn opened_at(&self) -> Observed {
        self.opened.at
    }

    /// When the driver closed it. `None` while it is still open, or if it was only dropped.
    pub fn closed_at(&self) -> Option<Observed> {
        closed_at(self.view.r, self.opened)
    }

    /// Open at `from` and still open at `to`.
    pub fn open_throughout(&self, from: Observed, to: Observed) -> bool {
        self.opened.at <= from && self.closed_at().is_none_or(|closed| closed > to)
    }

    /// Whether this stream heard the microphone frame after it was pushed.
    pub fn heard_after(&self, mic: u32, chunk: u16, pushed: Observed) -> bool {
        self.view
            .r
            .index
            .heard
            .get(&(self.opened.kind, self.opened.nth, mic, chunk))
            .is_some_and(|ats| ats.iter().any(|at| *at > pushed))
    }
}

/// One request the driver made of the model.
#[derive(Clone, Copy)]
pub struct RequestView<'a> {
    opened: &'a Opened,
}

impl RequestView<'_> {
    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    /// The speech the driver made it for, once its metrics have said.
    pub fn speech(&self) -> Option<u64> {
        match &self.opened.detail {
            Detail::Chat { speech, .. } => *speech,
            _ => None,
        }
    }
}

/// One microphone frame batch a listening stream heard.
#[derive(Clone, Copy, Debug)]
pub struct Heard<'a> {
    pub at: Observed,
    pub kind: Kind,
    pub stream: u32,
    pub tags: &'a [Tag],
}

// ==============================================================================================
// What a finding is about
// ==============================================================================================

impl From<&SpeechView<'_>> for Subject {
    fn from(speech: &SpeechView<'_>) -> Self {
        Subject::Speech(speech.id)
    }
}

impl From<SpeechView<'_>> for Subject {
    fn from(speech: SpeechView<'_>) -> Self {
        Subject::Speech(speech.id)
    }
}

impl From<&ApiCallView<'_>> for Subject {
    fn from(call: &ApiCallView<'_>) -> Self {
        Subject::Resource(Kind::Answer, call.nth())
    }
}

impl From<&ToolRunView<'_>> for Subject {
    fn from(tool: &ToolRunView<'_>) -> Self {
        Subject::Resource(Kind::Tool, tool.nth())
    }
}

impl From<&OutputView<'_>> for Subject {
    fn from(output: &OutputView<'_>) -> Self {
        Subject::Resource(Kind::Output, output.nth())
    }
}

impl From<&StreamView<'_, '_>> for Subject {
    fn from(stream: &StreamView<'_, '_>) -> Self {
        Subject::Resource(stream.kind(), stream.nth())
    }
}

impl From<&HoldView<'_>> for Subject {
    fn from(hold: &HoldView<'_>) -> Self {
        Subject::Resource(Kind::Hold, hold.nth())
    }
}
