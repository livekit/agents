//! A failed model or voice request is retried as the session's `conn_options` say, and no further.
//!
//! The budgets are the session's (`SessionConnectOptions`): an `llm_conn_options` or
//! `tts_conn_options` the driver ignored retries on its own defaults. Each failed attempt is the
//! session's `error` event, `recoverable` exactly when a retry follows it. A voice that does not
//! stream is spoken sentence by sentence, and a sentence that fails is synthesized again, on its
//! own budget, even after the reply's earlier sentences were heard: the reference's adapter retries
//! each sentence's synthesis, and the reply's voice does not stop at the first transient failure.
use std::collections::BTreeMap;

use lk_agents_voice::{
    events::{ErrorSource, SessionEvent},
    trace::{GenerationStep, ProviderKind, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::ConnSetup,
    input::{Input, Ref},
    record::{Kind, Recording},
};

pub struct RetriesFollowConnOptions;

impl Rule for RetriesFollowConnOptions {
    const CONTRACT: Contract = Contract {
        id: "retries_follow_conn_options",
        guarantee: "Model and voice requests are retried within the session's conn_options, each failure reported recoverable exactly when retried, and a failed sentence of a voice that does not stream is synthesized again.",
        examples: &[retried_once_and_reported, sentence_synthesized_again],
        counterexamples: &[retried_past_budget, recoverable_but_not_retried, sentence_dropped],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let Some((conn, chunked)) = r.conn() else { return };
        let budget = |leg: ProviderKind| match leg {
            ProviderKind::Llm => conn.llm,
            ProviderKind::Tts => conn.tts,
        };
        let internal = r.internal_view();
        // (speech, leg) -> (retries armed, failures reported recoverable)
        let mut tally: BTreeMap<(u64, ProviderKind), (u32, u32)> = BTreeMap::new();
        for (_, step) in internal.steps() {
            match step {
                Step::Generation(GenerationStep::RetryArmed { speech, leg, attempt, .. }) => {
                    check.expect(
                        Subject::Speech(speech.0),
                        *attempt <= budget(*leg),
                        format!(
                            "{leg:?} attempt {attempt} of speech {} was retried past its budget \
                             of {} retries",
                            speech.0,
                            budget(*leg)
                        ),
                    );
                    tally.entry((speech.0, *leg)).or_default().0 += 1;
                }
                Step::Emitted(SessionEvent::Error(ev)) if ev.recoverable => {
                    let leg = match ev.source {
                        ErrorSource::Llm => ProviderKind::Llm,
                        ErrorSource::Tts => ProviderKind::Tts,
                        _ => continue,
                    };
                    let Some(speech) = ev.speech else { continue };
                    tally.entry((speech.0, leg)).or_default().1 += 1;
                }
                _ => {}
            }
        }
        for ((speech, leg), (armed, recoverable)) in tally {
            check.expect(
                Subject::Speech(speech),
                armed == recoverable,
                format!(
                    "speech {speech}'s {leg:?} reported {recoverable} recoverable failures for \
                     {armed} retries"
                ),
            );
        }
        if chunked {
            sentences_retried(r, conn, check);
        }
    }
}

/// Each sentence's synthesis that failed after its audio began is followed by another of the same
/// text while the budget lasts, unless the reply was cut or the session closed meanwhile.
fn sentences_retried(r: &Recording, conn: ConnSetup, check: &mut Check) {
    let text_of = |nth: u32| -> String {
        r.produced().iter().filter(|(_, tts, _, _)| *tts == nth).map(|(_, _, _, t)| t.as_str()).collect()
    };
    let internal = r.internal_view();
    let inputs = r.inputs_view();
    for sent in inputs.accepted() {
        let Input::TtsFail { tts } = &sent.input else { continue };
        let text = text_of(tts.nth);
        if text.is_empty() {
            continue;
        }
        let tries = (0..=tts.nth).filter(|&n| text_of(n) == text).count() as u32;
        if tries > conn.tts {
            continue;
        }
        // A reply cut, or a session closing, after the failure has no sentence left to say.
        let cut = internal.interruptions().iter().any(|i| i.at >= sent.at)
            || internal.close_requested().is_some_and(|(at, _)| at >= sent.at)
            || r.close_begun().is_some_and(|at| at >= sent.at);
        if cut {
            continue;
        }
        let again = r.resources(Kind::Tts).any(|s| s.nth() > tts.nth && text_of(s.nth()) == text);
        check.expect(
            Subject::Resource(Kind::Tts, tts.nth),
            again,
            format!("the sentence {text:?} failed on synthesis {} and was not synthesized again", tts.nth),
        );
    }
}

fn budgets() -> ConnSetup {
    ConnSetup { stt: 1, llm: 1, tts: 1, interval_ms: 100 }
}

fn armed(attempt: u32) -> Step {
    use lk_agents_abi::Ts;
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Step::Generation(GenerationStep::RetryArmed {
        speech: SpeechId(4),
        leg: ProviderKind::Llm,
        attempt,
        at: Ts::ZERO,
    })
}

fn error(recoverable: bool) -> Step {
    use lk_agents_abi::errors::{ApiError, ProviderError};
    use lk_agents_voice::{agent_activity::scheduling::SpeechId, events::ErrorEvent};
    Step::Emitted(SessionEvent::Error(Box::new(ErrorEvent {
        source: ErrorSource::Llm,
        recoverable,
        error: ProviderError::Local(ApiError::new("injected")),
        speech: Some(SpeechId(4)),
    })))
}

/// One failure, reported recoverable, then the retry its budget allows.
fn retried_once_and_reported() -> Recording {
    Authored::new().conn(budgets(), false).journal(error(true)).journal(armed(1)).done()
}

/// A second retry on a budget of one.
fn retried_past_budget() -> Recording {
    Authored::new()
        .conn(budgets(), false)
        .journal(error(true))
        .journal(armed(1))
        .journal(error(true))
        .journal(armed(2))
        .done()
}

/// A failure reported recoverable that nothing retried.
fn recoverable_but_not_retried() -> Recording {
    Authored::new().conn(budgets(), false).journal(error(true)).done()
}

fn failed_sentence() -> Authored {
    Authored::new()
        .conn(budgets(), true)
        .tts()
        .produced(0, 0, "Hello.")
        .tts()
        .produced(1, 0, "How are you?")
        .sent(Input::TtsFail { tts: Ref::new(1) })
}

/// The failed sentence is synthesized again, and its audio follows.
fn sentence_synthesized_again() -> Recording {
    failed_sentence().tts().produced(2, 0, "How are you?").done()
}

/// The reply's voice stopped at the failed sentence.
fn sentence_dropped() -> Recording {
    failed_sentence().done()
}
