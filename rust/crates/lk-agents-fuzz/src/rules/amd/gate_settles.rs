use lk_agents_voice::amd::{AmdObservation, TurnEnd};
use lk_agents_voice::audio_recognition::TurnId;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{
        Kind, Observed, Recording,
        amd::{GatedTurn, Request},
    },
};

// Every turn committed under the gate reaches a decision, a supersession or an exit within a
// bound, while AMD makes progress. The scripted answerer is that progress: it can answer only
// what the session tells it, so the session must report each commit at once; once AMD answered
// (or a newer turn superseded it), the turn must end within the bound. Its outcome follows the
// decision applied: an allow is heard, unless something independent of the session's own report
// explains why not; a deny keeps the person's words and answers nothing. A session that does
// nothing fails here.

pub struct GateSettles;

/// How long an answered turn may take to end, past its customer hook.
const BOUND_MS: u64 = 2_000;

/// How long an allowed reply may take to be heard once its decision applied.
const HEARD_MS: u64 = 15_000;

impl Rule for GateSettles {
    const CONTRACT: Contract = Contract {
        id: "gate_settles",
        guarantee: "Every turn committed under the AMD gate is reported to AMD at its commit, \
                    ends within a bound once AMD answered it or a newer turn superseded it, and \
                    follows the decision applied: an allowed reply is heard unless a fault, a \
                    refusal, newer speech or the close explains why not, and a denied turn keeps \
                    its words with no reply.",
        examples: &[answered_and_ended, superseded_and_ended, allowed_and_heard, denied_and_kept],
        counterexamples: &[
            never_reported,
            answered_never_ended,
            allowed_with_no_reply,
            allowed_reply_never_heard,
            denied_but_answered,
        ],
        probe: Some(a_denied_turn),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        for turn in amd.gated_turns() {
            let observed = amd.observed(turn.run);
            let reported = observed.iter().any(|(at, o)| {
                matches!(o, AmdObservation::TurnCommitted { turn: t, .. } if *t == TurnId(turn.turn))
                    && *at <= next_input(r, turn.committed)
            });
            check.expect_with(Subject::Session, reported, || {
                format!(
                    "turn {} ({:?}) committed under run {} at {} ms and AMD was never told",
                    turn.turn, turn.words, turn.run, turn.committed.at_ms
                )
            });
            let answer = amd.answers_for(&turn).into_iter().next();
            let from = match (answer, turn.superseded) {
                (Some(a), Some(s)) => a.at().min(s),
                (Some(a), None) => a.at(),
                (None, Some(s)) => s,
                (None, None) => continue,
            };
            // Past the turn's own customer hook, which is issued only once every older turn's
            // hook returned: no bound on the session covers customer code.
            let Some(hook) = amd.hook_returned(&turn) else { continue };
            let from = from.max(hook);
            let due = from.at_ms + BOUND_MS;
            if r.now().at_ms < due || r.close_begun().is_some_and(|c| c.at_ms <= due) {
                continue;
            }
            let ended = observed.iter().find_map(|(at, o)| match o {
                AmdObservation::TurnEnded { turn: t, how } if *t == TurnId(turn.turn) => {
                    Some((*at, *how))
                }
                _ => None,
            });
            // `Detached` is the run's last observation: a turn its detach settled ends after it.
            let detached = (amd.attachments().into_iter())
                .find(|a| a.run == turn.run)
                .and_then(|a| a.detached)
                .is_some_and(|d| d.at_ms <= due && ended.is_none_or(|(at, _)| at > d));
            let ended_in_time = ended.is_some_and(|(at, _)| at.at_ms <= due);
            check.expect_with(Subject::Session, ended_in_time || detached, || {
                format!(
                    "turn {} was answered or superseded at {} ms and had not ended by {due} ms \
                     ({ended:?})",
                    turn.turn, from.at_ms
                )
            });
            outcome(r, &turn, check);
        }
    }
}

/// The turn's outcome, by the decision applied to it.
fn outcome(r: &Recording, turn: &GatedTurn, check: &mut Check) {
    let amd = r.amd_view();
    let requests = amd.requests_answering(turn);
    let made: Vec<_> = requests.iter().filter(|q| !amd.preemptive(q)).collect();
    let heard = requests.iter().filter_map(|q| amd.first_audio(q.speech?)).min();
    // A turn with no words has none to keep; `StopResponse` ignores the turn, as Python does
    // (`agent_activity.py`); a turn dropped behind a speech it could not interrupt keeps none.
    let kept = turn.item_id.is_some()
        || turn.words.trim().is_empty()
        || amd.stopped(turn)
        || amd.dropped_unanswered(turn);
    let Some((answer, applied)) = amd.applied(turn) else {
        // Superseded before a decision applied: the words stay, nothing answers it.
        if turn.superseded.is_some() {
            check.expect_with(Subject::Session, kept && heard.is_none(), || {
                format!(
                    "turn {} was superseded and was heard at {heard:?}, or lost its words",
                    turn.turn
                )
            });
        }
        return;
    };
    // A turn that asked for no reply gets none, whatever AMD decided.
    if !answer.allows() || amd.skips_reply(turn.turn) {
        check.expect_with(Subject::Session, kept && made.is_empty() && heard.is_none(), || {
            format!(
                "turn {} was denied ({answer:?}) and kept={kept}, was asked {} times, heard at \
                 {heard:?}",
                turn.turn,
                made.len()
            )
        });
        return;
    }
    let due = applied.at_ms + HEARD_MS;
    if r.now().at_ms < due || heard.is_some_and(|h| h.at_ms <= due) {
        return;
    }
    check.expect_with(Subject::Session, unheard_explained(r, turn, &requests, due), || {
        format!(
            "turn {} was allowed ({answer:?}, applied at {} ms), asked {} times, and not heard \
             by {due} ms with nothing to explain it",
            turn.turn,
            applied.at_ms,
            made.len()
        )
    });
}

/// Evidence, independent of the session's own report, that an allowed reply could go unheard
/// by `due` ms: the hook stopped the turn, a refusal the session traced, a fault the search
/// injected, the person or the host acting after the commit, the close, or a model or voice the
/// case had not finished.
fn unheard_explained(r: &Recording, turn: &GatedTurn, requests: &[Request<'_>], due: u64) -> bool {
    use lk_agents_voice::trace::{SpeechStep, Step};
    let amd = r.amd_view();
    let after = |at: Observed| at > turn.committed && at.at_ms <= due;
    if amd.stopped(turn) || amd.dropped_unanswered(turn) {
        return true;
    }
    let refused = r.journal().iter().any(|(at, step)| {
        after(*at) && matches!(step, Step::Speech(SpeechStep::ScheduleRejected { .. }))
    });
    let faulted = r.faults().iter().any(|(at, _, _)| after(*at));
    let acted = r.applied().iter().filter(|a| a.accepted && after(a.at)).any(|a| {
        matches!(
            a.input,
            Input::LlmFail { .. }
                | Input::TtsFail { .. }
                | Input::TtsHold { .. }
                | Input::TtsOutage { down: true }
                | Input::PlayoutFinishLost { .. }
                | Input::AvatarError { .. }
                | Input::VadStart { .. }
                | Input::VadInference { speaking: true, .. }
                | Input::SttStart { .. }
                | Input::CommitTurn { .. }
                | Input::CommitUserTurn { .. }
                | Input::StartUserTurn { .. }
                | Input::EndUserTurn { .. }
                | Input::Interrupt { .. }
                | Input::InterruptSpeech { .. }
                | Input::SetAllowInterruptions { .. }
                | Input::SetAudioOutput { attach: false }
                | Input::SetIoEnabled { leg: lk_agents_voice::agent_session::IoKind::AudioOutput, enabled: false }
                | Input::Close { .. }
                | Input::Drain { .. }
                | Input::Handoff { .. }
                | Input::UpdateModels { .. }
                | Input::AmdDetach { forbid: true, .. }
        )
    }) || amd.answers().iter().any(|a| {
        // A real run's completion that forbids replies.
        matches!(a, crate::record::amd::Answer::Detach { at, forbid: true, .. } if after(*at))
    });
    // The model never asked, still answering, or answering with no words or a tool call.
    let spoke = |chat: u32| {
        r.applied().iter().any(|a| {
            a.accepted
                && matches!(&a.input, Input::LlmText { chat: c, text } if c.nth == chat && !text.0.trim().is_empty())
        })
    };
    let model_owes = requests.iter().any(|q| r.live().has(Kind::Chat, q.chat) || !spoke(q.chat));
    // A voice or a playout the case still owes.
    let voice_owes = [Kind::Tts, Kind::Stream, Kind::Segment].into_iter().any(|kind| {
        r.opened().iter().any(|o| {
            o.kind == kind
                && after(o.at)
                && !r.closed().iter().any(|(at, k, n)| *k == kind && *n == o.nth && at.at_ms <= due)
        })
    });
    refused || faulted || acted || model_owes || voice_owes
}

/// The first input the run applied after `at`: a report made in the commit's own transition
/// is recorded before it.
fn next_input(r: &Recording, at: Observed) -> Observed {
    r.applied().iter().map(|a| a.at).find(|seen| *seen > at).unwrap_or(Observed::END)
}

fn turn() -> Authored {
    Authored::new().sent(Input::AmdAttach {}).user_turn(Some(1), 1, "hello")
}

fn ended(how: TurnEnd) -> AmdObservation {
    AmdObservation::TurnEnded { turn: TurnId(1), how }
}

fn answered_and_ended() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .request(10, 1, "hello", &[], &[])
        .amd(1, ended(TurnEnd::Replied))
        .later(3_000)
        .done()
}

fn superseded_and_ended() -> Recording {
    turn()
        .user_turn(Some(1), 2, "hello again")
        .amd(1, ended(TurnEnd::Superseded))
        .later(3_000)
        .done()
}

fn never_reported() -> Recording {
    Authored::new()
        .sent(Input::AmdAttach {})
        .user_turn(None, 1, "hello")
        .sent(Input::Advance { ms: crate::input::Millis(1) })
        .later(3_000)
        .done()
}

fn answered_never_ended() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Deny })
        .later(3_000)
        .done()
}

fn allowed_and_heard() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .request(10, 1, "hello", &[], &[])
        .amd(1, ended(TurnEnd::Replied))
        .pushed(10)
        .later(20_000)
        .done()
}

fn denied_and_kept() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Deny })
        .amd(1, ended(TurnEnd::Denied))
        .later(3_000)
        .done()
}

/// Allowed, asked, the model answered with words, and nothing was heard or explains why.
fn allowed_reply_never_heard() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .request(10, 1, "hello", &[], &[])
        .sent(Input::LlmText {
            chat: Ref::new(0),
            text: crate::input::AgentText("Hi, who is this?".into()),
        })
        .sent(Input::LlmEnd { chat: Ref::new(0) })
        .closed(Kind::Chat, 0)
        .amd(1, ended(TurnEnd::Replied))
        .later(20_000)
        .done()
}

/// Denied, and answered anyway.
fn denied_but_answered() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Deny })
        .request(10, 1, "hello", &[], &[])
        .amd(1, ended(TurnEnd::Denied))
        .later(3_000)
        .done()
}

/// Allowed, reported as replied, and never asked: the session's own report is no evidence.
fn allowed_with_no_reply() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .amd(1, ended(TurnEnd::Replied))
        .later(20_000)
        .done()
}

/// A turn AMD denies is reported and ends with its words saved.
fn a_denied_turn() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("please hold", AmdVerdict::Wait)
}
