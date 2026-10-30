//! Each failing span of an input processor is one recoverable `AudioProcessing` error naming that
//! processor: never lost, never repeated, never invented.
//!
//! The fake stream knows the truth: it records each span as it hands its reader the failure
//! that opens it. A span is owed its error once the driver had a whole turn to take it before
//! the session finished closing, as a room event is; a close that cut it off may leave it
//! unsaid. A failure never costs its frame, which reaches the recognizer as any other:
//! `forwarded_while_enabled` holds every pushed frame to that.
use std::collections::BTreeMap;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::voice::rtc::ProcessorFailure;
use lk_agents_voice::events::{ErrorEvent, ErrorSource, SessionEvent};
use lk_agents_voice::trace::{CloseStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Faults, NoopRefs, PreConnect, ProcessorFaults, RoomLegs, RoomSetup, Setup},
    input::{Count, Input, Millis, Ref, RoomIn},
    record::{Host, Recording},
};

pub struct ProcessorSpanIsOneError;

impl Rule for ProcessorSpanIsOneError {
    const CONTRACT: Contract = Contract {
        id: "processor_span_is_one_error",
        guarantee: "Each input processor's failing span is one recoverable audio processing error naming it.",
        examples: &[one_span_one_error, a_span_cut_off_by_the_close],
        counterexamples: &[reported_twice, never_reported, reported_unrecoverable],
        probe: Some(failing_processors),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let spans = r.processor_spans();
        let inputs = r.inputs_view();
        let internal = r.internal_view();
        let closing = inputs
            .close_asked_at()
            .into_iter()
            .chain(internal.close_requested().map(|(at, _)| at))
            .min();
        let closed = r
            .journal_where(|step| match step {
                Step::Emitted(SessionEvent::Close { .. }) => Some(()),
                _ => None,
            })
            .map(|(at, ())| at)
            .next();
        // Per processor: its label, the spans it opened, and how many of them are owed an error.
        let mut opened: BTreeMap<u64, (&str, usize, usize)> = BTreeMap::new();
        for (at, processor, label) in spans {
            let owed = closing.is_none_or(|c| *at < c)
                && closed.is_none_or(|d| {
                    inputs.driver_turn_after(*at).is_some_and(|t| t.ended.is_some_and(|e| e < d))
                });
            let entry = opened.entry(*processor).or_insert((label, 0, 0));
            entry.1 += 1;
            entry.2 += usize::from(owed);
        }
        let mut told: BTreeMap<u64, usize> = BTreeMap::new();
        for (at, event) in r.host() {
            let Host::Session(SessionEvent::Error(error)) = event else { continue };
            if error.source != ErrorSource::AudioProcessing {
                continue;
            }
            let failure = error.processor_failure.as_ref();
            check.expect(
                Subject::Session,
                error.recoverable && failure.is_some(),
                format!(
                    "an audio processing error at seq {} is unrecoverable or names no processor",
                    at.seq
                ),
            );
            let Some(failure) = failure else { continue };
            let processor = failure.processor.id();
            let label = opened.get(&processor).map(|(label, ..)| *label);
            check.expect(
                Subject::Session,
                label == Some(failure.label.as_str()),
                format!(
                    "{} {processor} reported failing at seq {} but opened no such span",
                    failure.label, at.seq
                ),
            );
            *told.entry(processor).or_default() += 1;
        }
        for (processor, (label, spans, owed)) in opened {
            let told = told.get(&processor).copied().unwrap_or(0);
            check.expect(
                Subject::Session,
                owed <= told && told <= spans,
                format!(
                    "{label} {processor} opened {spans} failing spans ({owed} before the close) and \
                     was reported {told} times"
                ),
            );
        }
    }
}

const DENOISER: &str = "fake.Denoiser";

fn error(recoverable: bool) -> SessionEvent {
    let refs: std::sync::Arc<dyn lk_agents_abi::errors::RefRegistry> =
        std::sync::Arc::new(NoopRefs);
    let processor = lk_agents_abi::HostRef::new(1, &refs);
    SessionEvent::Error(Box::new(ErrorEvent {
        source: ErrorSource::AudioProcessing,
        recoverable,
        error: ProviderError::Local(ApiError::new("fake.Denoiser rejected the frame")),
        speech: None,
        processor_failure: Some(ProcessorFailure {
            label: DENOISER.into(),
            processor: processor.clone(),
            error: processor,
            message: "fake.Denoiser rejected the frame".into(),
        }),
    }))
}

fn advance() -> Input {
    Input::Advance { ms: Millis(20) }
}

/// The denoiser's span, reported on the driver's next turn.
fn one_span_one_error() -> Recording {
    Authored::new()
        .processor_span(1, DENOISER)
        .sent(advance())
        .session(error(true))
        .sent(advance())
        .done()
}

/// The session closed in the driver's turn that would have reported the span.
fn a_span_cut_off_by_the_close() -> Recording {
    let reason = lk_agents_abi::voice::events::CloseReason::Error;
    Authored::new()
        .processor_span(1, DENOISER)
        .sent(advance())
        .journal(Step::Close(CloseStep::Requested { reason, drain: false }))
        .journal(Step::Emitted(SessionEvent::Close { reason, error: None }))
        .sent(advance())
        .done()
}

/// One span, two errors.
fn reported_twice() -> Recording {
    Authored::new()
        .processor_span(1, DENOISER)
        .sent(advance())
        .session(error(true))
        .session(error(true))
        .sent(advance())
        .done()
}

/// The driver had its turns, and the span never reached the session.
fn never_reported() -> Recording {
    Authored::new().processor_span(1, DENOISER).sent(advance()).sent(advance()).done()
}

/// A processor failure is always recoverable.
fn reported_unrecoverable() -> Recording {
    Authored::new()
        .processor_span(1, DENOISER)
        .sent(advance())
        .session(error(false))
        .sent(advance())
        .done()
}

/// Prerequisite: the person's microphone, its pre-connect head replayed first, goes through a
/// denoiser broken for a few frames and a gain control that glitches every fourth frame.
fn failing_processors() -> Case {
    Case {
        setup: Setup {
            room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
            room_legs: Some(RoomLegs { pre_connect: Some(PreConnect::Sent) }),
            failing_processors: Some(ProcessorFaults {
                denoiser: Faults::Span { from: 2, frames: 30 },
                gain_control: Some(Faults::Flapping { period: 4, fails: 1 }),
            }),
            ..Setup::default()
        },
        inputs: vec![
            Input::Room { event: RoomIn::Linked },
            advance(),
            Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
            Input::Advance { ms: Millis(400) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
            Input::Advance { ms: Millis(400) },
        ],
        ..Case::default()
    }
}
