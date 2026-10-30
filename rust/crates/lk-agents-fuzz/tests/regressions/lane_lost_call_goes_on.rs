//! A lost voice-lane lease never ends the call: what the agent says after it is heard.
use lk_agents_fuzz::drive::{Outcome, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, UserText};
use lk_agents_voice::trace::{SpeechStep, Step};

// Catches: a lost lease cancelled the speech under it and every later one at its creation, each
// announced `Completed` with nothing said, and a speech whose commit was answered after the loss
// never finished (`speech_is_heard`, `no_premature_completion`, survey seeds 12, 341, 615;
// 62855980b). voice-session.md: storage failures are logged and never end the call.
#[test]
fn the_agent_is_heard_after_its_lane_is_lost() {
    // The store loses the lease with its answer to the first turn.
    let setup = Setup { storage_lost_at: Some(1), ..Setup::default() };
    let reply = |chat: u32, said: &str| {
        [
            Input::GenerateReply {
                user_input: Some(UserText("hello?".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(chat), text: AgentText(said.into()) },
            Input::LlmEnd { chat: Ref::new(chat) },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutStarted { segment: Ref::new(chat) },
            Input::PlayoutProgressed { segment: Ref::new(chat), ms: Millis(5_000) },
            Input::PlayoutFinished { segment: Ref::new(chat) },
            Input::Advance { ms: Millis(1_000) },
        ]
    };
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in reply(0, "Hi, how can I help?").into_iter().chain(reply(1, "Sure.")) {
            run.apply(&input).await;
        }
        let journal: Vec<&Step> = run.record.journal().iter().map(|(_, s)| s).collect();
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
        let outcome = finish(run, Outcome::default()).await;
        assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
    });
}
