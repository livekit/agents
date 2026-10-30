//! A lost voice-lane lease never ends the call: what the agent says after it is heard.
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::search;
use lk_agents_voice::trace::{SpeechStep, Step};

// Catches: a lost lease cancelled the speech under it and every later one at its creation, each
// announced `Completed` with nothing said, and a speech whose commit was answered after the loss
// never finished; a workflow asking again at once looped at one instant (`speech_is_heard`,
// `no_premature_completion`, `virtual_time_advances`, survey seeds 12, 341, 615; seed 11 draws the scenario since the room inputs).
// voice-session.md: storage failures are logged and never end the call.
#[test]
fn the_agent_is_heard_after_its_lane_is_lost() {
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(11));
    assert!(setup.storage_lost_at.is_some(), "the scenario: the call's lane is stored");
    let case = search::explore(setup, 11, 3_000).case;
    let (outcome, journal) = simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        let journal: Vec<Step> = run.record.journal().iter().map(|(_, s)| s.clone()).collect();
        (finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await, journal)
    });
    let lost = journal
        .iter()
        .position(|s| matches!(s, Step::Speech(SpeechStep::LaneLost { .. })))
        .expect("the scenario: the lane's lease is lost mid-call");
    assert!(
        journal[lost..]
            .iter()
            .any(|s| matches!(s, Step::Speech(SpeechStep::AgentStartedSpeaking { .. }))),
        "the agent speaks after the loss"
    );
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
