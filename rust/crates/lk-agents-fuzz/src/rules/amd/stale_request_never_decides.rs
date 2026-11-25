use crate::{
    authored::{Authored, amd_bound, amd_predicted, amd_prediction},
    check::{Check, Contract, Rule, Subject},
    input::AmdSays,
    record::{
        Recording,
        amd_run::{AskKind, says_category},
    },
};
use lk_agents_voice::amd::effects::ReqId;
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Category, Stage};
use lk_agents_voice::amd::AmdRunId;
use lk_agents_voice::audio_recognition::TurnId;
use lk_agents_voice::trace::AmdStep;

// Spec §14 invariant 2: a stale request never decides. A prediction's source request was issued
// before it and never cancelled, for this turn, or for an older turn whose result this empty
// turn reuses (spec §7.3). Only a timeout, or a reuse with no request pending, has no source.
// The outcome side: a model prediction's raw category is what the classifier answered a request
// opened after that request was issued, and the host heard the same category for the turn.

pub struct StaleRequestNeverDecides;

impl Rule for StaleRequestNeverDecides {
    const CONTRACT: Contract = Contract {
        id: "stale_request_never_decides",
        guarantee: "A turn is decided only by a request issued for it, or by the pending request \
                    an empty turn reuses, and never by a request AMD abandoned.",
        examples: &[decided_by_its_own, reused_by_an_empty_turn],
        counterexamples: &[decided_by_a_cancelled, decided_by_another_turns, decided_by_no_answer],
        probe: Some(a_newer_turn_cancels),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        let steps = view.steps();
        let entered: Vec<AmdRunId> = view.entered().into_iter().map(|(_, run)| run).collect();
        let asks = view.asks();
        for p in view.predicted().into_iter().filter(|p| entered.contains(&p.run)) {
            let before = |at: crate::record::Observed| at < p.at;
            let Some(req) = p.source_req else {
                check.expect_with(
                    Subject::Session,
                    matches!(p.reason, AmdReason::InferenceTimeout | AmdReason::Reused),
                    || format!("AMD turn {}'s {:?} prediction names no request", p.turn_id, p.reason),
                );
                continue;
            };
            let issued = steps.iter().find_map(|(at, s)| match s {
                AmdStep::RequestIssued { run, req: r, turn } if *run == p.run && *r == req => {
                    Some((*at, *turn))
                }
                _ => None,
            });
            let cancelled = steps.iter().any(|(at, s)| {
                matches!(s, AmdStep::RequestCancelled { run, req: r } if *run == p.run && *r == req)
                    && before(*at)
            });
            let issued_for = |turn: TurnId| issued.is_some_and(|(at, t)| before(at) && t == turn);
            let own = issued_for(p.turn);
            let reused = p.reason == AmdReason::Reused
                && issued.is_some_and(|(at, t)| before(at) && t != p.turn && empty(&steps, p.run, p.turn));
            check.expect_with(Subject::Session, (own || reused) && !cancelled, || {
                format!(
                    "AMD turn {}'s {:?} prediction came from request {req:?}, issued as {issued:?}, \
                     cancelled: {cancelled}",
                    p.turn_id, p.reason
                )
            });
            if p.reason != AmdReason::Prediction {
                continue;
            }
            let issued_at = issued.map(|(at, _)| at);
            let answered = view.replies().iter().any(|reply| {
                reply.accepted
                    && before(reply.at)
                    && says_category(reply.says) == p.raw
                    && asks.iter().any(|(nth, opened, asked)| {
                        *nth == reply.ask
                            && matches!(asked.kind, AskKind::Classify | AskKind::Choice)
                            && issued_at.is_some_and(|issued| issued <= *opened)
                    })
            });
            let told = view.predictions(p.run).iter().any(|(_, e)| {
                e.turn_id == p.turn_id && e.category == p.category && e.reason == p.reason
            });
            check.expect_with(Subject::Session, answered && told, || {
                format!(
                    "AMD turn {}'s prediction {:?} (raw {:?}) has no answer to its request \
                     ({answered}) or no matching event ({told})",
                    p.turn_id, p.category, p.raw
                )
            });
        }
    }
}

/// The turn committed with no text: AMD bound it without text.
fn empty(steps: &[(crate::record::Observed, AmdStep)], run: AmdRunId, turn: TurnId) -> bool {
    steps.iter().any(|(_, s)| {
        matches!(s, AmdStep::TurnBound { run: r, turn: t, has_text: false, .. } if *r == run && *t == turn)
    })
}

fn issued(req: u64, turn: u64) -> AmdStep {
    AmdStep::RequestIssued { run: AmdRunId(1), req: ReqId(req), turn: TurnId(turn) }
}

fn cancelled(req: u64) -> AmdStep {
    AmdStep::RequestCancelled { run: AmdRunId(1), req: ReqId(req) }
}

fn human(turn: u64, turn_id: u64, req: u64) -> AmdStep {
    let human = Some(Category::Human);
    amd_predicted(1, turn, turn_id, Some(req), human, Category::Human, Stage::Human, AmdReason::Prediction, false)
}

fn told(turn_id: u64) -> AmdEvent {
    AmdEvent::Prediction(amd_prediction(turn_id, Category::Human, Stage::Human, AmdReason::Prediction))
}

const STATE: &str = r#"{"stage": "uncertain"}"#;

fn first_turn() -> Authored {
    Authored::new()
        .amd_entered(1)
        .amd_step(amd_bound(1, 4, 1))
        .amd_step(issued(1, 4))
        .amd_asked(AskKind::Classify, STATE, "hello")
}

fn decided_by_its_own() -> Recording {
    first_turn().amd_said(0, AmdSays::Human, true).amd_step(human(4, 1, 1)).amd_event(1, told(1)).done()
}

fn reused_by_an_empty_turn() -> Recording {
    let empty = lk_agents_voice::trace::AmdStep::TurnBound {
        run: AmdRunId(1),
        turn: TurnId(5),
        turn_id: 2,
        has_text: false,
        late_wait: false,
    };
    let raw = Some(Category::Human);
    let reused =
        amd_predicted(1, 5, 2, Some(1), raw, Category::Human, Stage::Human, AmdReason::Reused, false);
    first_turn()
        .amd_step(empty)
        .amd_said(0, AmdSays::Human, true)
        .amd_step(human(4, 1, 1))
        .amd_event(1, told(1))
        .amd_step(reused)
        .done()
}

fn decided_by_a_cancelled() -> Recording {
    first_turn()
        .amd_step(amd_bound(1, 5, 2))
        .amd_step(cancelled(1))
        .amd_step(issued(2, 5))
        .amd_said(0, AmdSays::Human, true)
        .amd_step(human(5, 2, 1))
        .amd_event(1, told(2))
        .done()
}

fn decided_by_another_turns() -> Recording {
    first_turn()
        .amd_step(amd_bound(1, 5, 2))
        .amd_said(0, AmdSays::Human, true)
        .amd_step(human(5, 2, 1))
        .amd_event(1, told(2))
        .done()
}

fn decided_by_no_answer() -> Recording {
    first_turn().amd_said(0, AmdSays::Voicemail, true).amd_step(human(4, 1, 1)).amd_event(1, told(1)).done()
}

/// A second turn with text commits while the first turn's request is in flight; the first
/// request's late answer arrives after.
fn a_newer_turn_cancels() -> crate::drive::Case {
    use crate::fakes::amd::{executor_probe::call, probe};
    use crate::input::{Input, Ref};
    let mut inputs = probe::says(0, "hello");
    inputs.extend(probe::says(1, "please leave a message after the tone"));
    inputs.push(Input::AmdReply { ask: Ref::new(0), says: AmdSays::Human });
    inputs.push(Input::AmdReply { ask: Ref::new(1), says: AmdSays::Voicemail });
    inputs.extend(probe::answered(1));
    inputs.push(probe::advance(8_000));
    call(Default::default(), inputs)
}
