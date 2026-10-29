//! A speech fails only from a failure the case injected into one of its own requests, or from
//! closing before it was scheduled, from a synthesis the case failed while it was live, from the
//! realtime socket the case dropped while it was live, from an error the realtime server sent
//! while it was live (a refusal, a timeout, a failed connection: the server's, reported by the
//! session), from a server that created no reply within the client's bound, or from a refusal
//! the driver journals itself.
use lk_agents_voice::{
    events::{ErrorSource, SessionEvent},
    trace::{PlayoutStep, SpeechStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{Input, Ref, RtEvent},
    record::{Applied, Handle, Recording, Sink},
};

pub struct SpeechErrorIsInjected;

impl Rule for SpeechErrorIsInjected {
    const CONTRACT: Contract = Contract {
        id: "speech_error_is_injected",
        guarantee: "A failed speech had a provider failure delivered to one of its own requests, lost the realtime socket it waited on or heard its server err, waited out the reply bound on a stalled server, was refused by Close before scheduling, was a say the driver refused on the journal, or lost its own keyed sink completion after its declared deadline with no playback receipt in between.",
        examples: &[
            failed_by_injection,
            failed_from_its_own_missing_playback,
            output_refusal_with_its_own_diagnostic,
        ],
        counterexamples: &[
            failed_without_injection,
            playback_timeout_without_its_own_deadline,
            playback_timeout_after_a_console_receipt,
            output_refusal_without_its_own_diagnostic,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let (host, internal, inference) = (r.host_view(), r.internal_view(), r.inference_view());
        for speech in host.speeches() {
            let Some(error) = speech.error() else { continue };

            // The only provider that can fail is our fake, and it fails only when told to. So a
            // failed speech must have had a failure sent to one of the chat requests the run
            // attributes to it, unless Close refused it before it was ever scheduled.
            let requests = inference.requests_for(speech.id);
            // A refused failure never reached a provider, so it cannot be what broke the speech.
            let injected = inputs.accepted().any(
                |sent| matches!(&sent.input, Input::LlmFail { chat } if requests.contains(&chat.nth)),
            );
            // A synthesis failure is attributed by time rather than by request: the fake voice fails
            // only when told to, and a speech that ended after one was told to has its cause.
            let before_done = |sent: &&Applied| speech.done_at().is_none_or(|done| sent.at < done);
            let voice_failed = inputs
                .accepted()
                .filter(before_done)
                .any(|sent| matches!(sent.input, Input::TtsFail { .. }));
            // The voice was down when the speech ended: the outage as the case last set it before.
            let voice_down = inputs
                .accepted()
                .filter(before_done)
                .filter_map(|sent| match sent.input {
                    Input::TtsOutage { down } => Some(down),
                    _ => None,
                })
                .last()
                .is_some_and(|down| down);
            // A realtime speech waits on the socket for its reply; dropped, the request fails.
            let socket_dropped = inputs.accepted().filter(before_done).any(|sent| {
                matches!(
                    sent.input,
                    Input::RealtimeEvent { event: RtEvent::Disconnected | RtEvent::FatalError, .. }
                )
            });
            // The realtime server is a simulation of the real one, and refuses and times out as
            // that does: a `response.create` over its own response, a connection slower than the
            // client's bound. What it said reached the host as the session's realtime error.
            // The same instant counts: a server that never connects fails the reply and reports
            // the connection timed out off one stall, the session's two bounds being equal, in
            // either order.
            let server_erred = host.events().iter().any(|(at, ev)| {
                matches!(ev, SessionEvent::Error(e) if e.source == ErrorSource::Realtime)
                    && speech.done_at().is_none_or(|done| at.at_ms <= done.at_ms)
            }) || crate::fakes::realtime::server::REFUSALS
                .iter()
                .any(|code| error.contains(code));
            // The server is slow as real ones are, now and then past the client's bound.
            let server_stalled = error.contains("generate_reply timed out");
            let refused_by_close = inputs.close_asked_at().is_some() && !speech.is_scheduled();
            // A `say` the driver refused on its own terms -- words with no voice to speak them --
            // fails without ever making a request; the refusal is on the journal.
            let refused_by_driver = requests.is_empty()
                && internal
                    .steps()
                    .iter()
                    .any(|(_, step)| matches!(step, Step::Speech(SpeechStep::SayRefused { .. })));
            // Missing external completion is a sink failure, not a provider failure. Accept only
            // the keyed diagnostic for this speech after its own declared completion deadline,
            // and only when no receipt came in between: the console's CLI loses none.
            let missing_playback = internal.steps().iter().any(|(lost_at, step)| {
                let Step::Playout(PlayoutStep::FinishLost { speech: lost_speech, segment, .. }) =
                    step
                else {
                    return false;
                };
                lost_speech.0 == speech.id
                    && error.contains(&format!(
                        "playback report timed out for attachment {} segment {}",
                        segment.attachment, segment.segment,
                    )) && internal.steps().iter().any(|(due_at, step)| {
                    matches!(step, Step::Playout(PlayoutStep::FinishDue { speech: due_speech, at })
                            if due_speech == lost_speech
                                && due_at <= lost_at
                                && at.saturating_duration_since(lk_agents_abi::Ts::ZERO).as_millis()
                                    <= u128::from(lost_at.at_ms)
                                && !r.sink().iter().any(|(receipt_at, sink)| {
                                    matches!(sink, Sink::ConsoleReceipt)
                                        && due_at < receipt_at
                                        && receipt_at <= lost_at
                                }))
                })
            });
            let output_refused = internal.steps().iter().any(|(_, step)| {
                matches!(step, Step::Playout(PlayoutStep::OutputRefused { speech: refused, reason })
                    if refused.0 == speech.id && error.contains(&format!("output admission refused: {reason}")))
            });
            let accounted = injected
                || voice_failed
                || voice_down
                || socket_dropped
                || server_erred
                || server_stalled
                || refused_by_close
                || refused_by_driver
                || missing_playback
                || output_refused;
            check.expect(
                &speech,
                accounted,
                format!(
                    "speech failed with {error} but no failure was sent to its requests {requests:?}"
                ),
            );
        }
    }
}

/// The case failed the speech's own chat request, and the speech failed.
fn failed_by_injection() -> Recording {
    Authored::new()
        .chat(4)
        .sent(Input::LlmFail { chat: Ref::new(0) })
        .speech(4, Handle::finished().failed("injected provider failure"))
        .done()
}

/// A speech failed; nothing was ever injected and nobody closed.
fn failed_without_injection() -> Recording {
    Authored::new().speech(4, Handle::finished().failed("boom")).done()
}

fn playback_timeout_record(deadline: bool, receipt: bool) -> Recording {
    use lk_agents_abi::{Ts, voice::io::PlaybackSegment};
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    let segment = PlaybackSegment { attachment: 8, segment: 3 };
    let mut authored = Authored::new();
    if deadline {
        authored = authored
            .journal(Step::Playout(PlayoutStep::FinishDue { speech: SpeechId(4), at: Ts::ZERO }));
    }
    if receipt {
        authored = authored.sink(Sink::ConsoleReceipt);
    }
    authored
        .journal(Step::Playout(PlayoutStep::FinishLost {
            speech: SpeechId(4),
            segment,
            position: std::time::Duration::ZERO,
        }))
        .speech(
            4,
            Handle::finished().failed("playback report timed out for attachment 8 segment 3"),
        )
        .done()
}

fn failed_from_its_own_missing_playback() -> Recording {
    playback_timeout_record(true, false)
}

fn playback_timeout_without_its_own_deadline() -> Recording {
    playback_timeout_record(false, false)
}

/// The console's CLI acknowledged the flush before the deadline expired: no receipt was lost.
fn playback_timeout_after_a_console_receipt() -> Recording {
    playback_timeout_record(true, true)
}

fn output_refusal_record(traced: bool) -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    use lk_agents_voice::agent_session::output::OutputBindingError;
    let reason = OutputBindingError::Capacity;
    let mut authored = Authored::new();
    if traced {
        authored = authored
            .journal(Step::Playout(PlayoutStep::OutputRefused { speech: SpeechId(4), reason }));
    }
    authored
        .speech(4, Handle::finished().failed(&format!("output admission refused: {reason}")))
        .done()
}

fn output_refusal_with_its_own_diagnostic() -> Recording {
    output_refusal_record(true)
}

fn output_refusal_without_its_own_diagnostic() -> Recording {
    output_refusal_record(false)
}
