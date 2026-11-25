use lk_agents_voice::events::SessionEvent;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref, RtEvent},
    record::{Recording, RtSent},
};

// Proposal, Realtime: with client-side turn detection, the session commits the realtime server's
// audio buffer at a gated turn's commit, before its decision, as Python does
// (`agent_activity.py` `_user_turn_completed_impl` commits before `should_reply`), so a denied
// `wait` turn leaves no audio for the next allowed reply. Judged at the fake server: every gated
// turn adds one buffer commit, and the commits of every turn up to a reply's own precede that
// reply's response request. Commits are counted, not timed: the socket's queue keeps their
// order but may lag the gate's commit. Between a turn's own commit and its reply's request, only
// another turn or a host commit adds one: a second commit there commits audio the turn did not
// hold. A turn the close or a lost connection overtook is not judged.

pub struct RealtimeTurnAudioCommitted;

impl Rule for RealtimeTurnAudioCommitted {
    const CONTRACT: Contract = Contract {
        id: "realtime_turn_audio_committed",
        guarantee: "On a realtime model, every turn committed under the AMD gate commits the \
                    server's audio buffer, denied or allowed, before the reply to it is asked.",
        examples: &[each_turn_committed],
        counterexamples: &[
            denied_turn_left_in_the_buffer,
            committed_with_the_reply,
            committed_at_the_commit_and_again_with_the_reply,
        ],
        probe: Some(a_wait_then_a_person),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        if !amd.realtime() || connection_lost(r) {
            return;
        }
        let close = r.inputs_view().close_asked_at();
        let commits = amd.rt_commits();
        let turns = amd.gated_turns();
        for (i, turn) in turns.iter().enumerate() {
            if close.is_some_and(|c| c <= turn.committed) {
                continue;
            }
            let Some(attached) = amd.attachments().into_iter().find(|a| a.run == turn.run) else {
                continue;
            };
            // This run's turns up to this one, each owed its own commit.
            let owed =
                turns[..=i].iter().filter(|t| t.run == turn.run && t.committed > attached.attached);
            let owed = owed.count();
            let made = |before: crate::record::Observed| {
                commits.iter().filter(|c| **c > attached.attached && **c < before).count()
            };
            // The reply's response: the first dispatched for a speech the session reported for
            // the turn, else the first asked in the turn's window.
            let response = (amd.rt_responses_for(turn).into_iter().next())
                .or_else(|| amd.rt_responses_answering(turn).into_iter().next());
            if let Some(response) = response {
                check.expect_with(Subject::Session, made(response.at) >= owed, || {
                    format!(
                        "turn {}'s realtime reply was asked at {} ms after {} buffer commits; \
                         the run's {owed} turns so far each owe one before it",
                        turn.turn,
                        response.at.at_ms,
                        made(response.at)
                    )
                });
                // Its own commit is the only one before the response, but for each turn or host
                // commit in between: a second one commits audio the turn did not hold.
                let between = |at: &crate::record::Observed| *at > turn.committed && *at < response.at;
                let own = commits.iter().filter(|c| **c >= turn.committed && between(c)).count();
                let others = (r.host_view().events().iter())
                    .filter(|(at, e)| between(at) && matches!(e, SessionEvent::UserTurnCompleted(_)))
                    .count()
                    + (r.applied().iter())
                        .filter(|a| a.accepted && between(&a.at))
                        .filter(|a| {
                            matches!(a.input, Input::CommitTurn { .. } | Input::CommitUserTurn { .. })
                        })
                        .count();
                check.expect_with(Subject::Session, own <= 1 + others, || {
                    format!(
                        "turn {} committed the server's buffer {own} times before its reply was \
                         asked at {} ms, with {others} other commits between",
                        turn.turn, response.at.at_ms
                    )
                });
            } else if !amd.answers_for(turn).is_empty() {
                let until = close.unwrap_or(crate::record::Observed::END);
                check.expect_with(Subject::Session, made(until) >= owed, || {
                    format!(
                        "turn {} was decided with no reply and the server's buffer holds its \
                         audio: {} commits for the run's {owed} turns so far",
                        turn.turn,
                        made(until)
                    )
                });
            }
        }
    }
}

/// A fatal error or a dropped connection may lose what the socket queued.
fn connection_lost(r: &Recording) -> bool {
    r.applied().iter().any(|a| {
        a.accepted
            && matches!(
                a.input,
                Input::RealtimeEvent { event: RtEvent::Disconnected | RtEvent::FatalError, .. }
            )
    })
}

/// A realtime session, its tools sent at the open.
fn wait_then_person() -> Authored {
    Authored::new()
        .rt_sent(RtSent::Tools(Vec::new()))
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "please hold")
}

fn each_turn_committed() -> Recording {
    wait_then_person()
        .rt_sent(RtSent::CommitAudio)
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Wait })
        .later(1_000)
        .user_turn(Some(1), 2, "hello who is this")
        .rt_sent(RtSent::CommitAudio)
        .sent(Input::AmdDecide { turn: Ref::new(1), verdict: AmdVerdict::Allow })
        .rt_response("rt_1", None)
        .done()
}

/// The denied turn's audio stays in the buffer and the next reply answers it too.
fn denied_turn_left_in_the_buffer() -> Recording {
    wait_then_person()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Wait })
        .later(1_000)
        .user_turn(Some(1), 2, "hello who is this")
        .sent(Input::AmdDecide { turn: Ref::new(1), verdict: AmdVerdict::Allow })
        .rt_sent(RtSent::CommitAudio)
        .rt_response("rt_1", None)
        .done()
}

/// The ungated path's order: the buffer is committed only with the response request, so the
/// denied turn's audio rides the person's reply.
fn committed_with_the_reply() -> Recording {
    wait_then_person()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Wait })
        .later(1_000)
        .user_turn(Some(1), 2, "hello who is this")
        .sent(Input::AmdDecide { turn: Ref::new(1), verdict: AmdVerdict::Allow })
        .reply_created(1, 2, 7)
        .rt_sent(RtSent::CommitAudio)
        .rt_request("rt_1", 7, None)
        .done()
}

/// The gate commits at the commit, and the reply commits again with its request, as a request
/// built with `commit_audio` would.
fn committed_at_the_commit_and_again_with_the_reply() -> Recording {
    Authored::new()
        .rt_sent(RtSent::Tools(Vec::new()))
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "hello who is this")
        .rt_sent(RtSent::CommitAudio)
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .reply_created(1, 1, 7)
        .rt_sent(RtSent::CommitAudio)
        .rt_request("rt_1", 7, None)
        .done()
}

/// A `wait` turn, then a person: each turn's audio is committed, and the person's reply follows.
/// Python: `test_suppressed_reply_keeps_only_the_realtime_transcript` (`rt.committed`).
fn a_wait_then_a_person() -> crate::drive::Case {
    use crate::fakes::amd::probe::{advance, attached, decide, says};
    let mut inputs = attached(None);
    inputs.extend(says(0, "please hold"));
    inputs.push(decide(0, AmdVerdict::Wait));
    inputs.push(advance(1_000));
    inputs.extend(says(1, "hello who is this"));
    inputs.push(decide(1, AmdVerdict::Allow));
    inputs.push(advance(4_000));
    crate::fakes::amd::probe::realtime_call(inputs)
}
