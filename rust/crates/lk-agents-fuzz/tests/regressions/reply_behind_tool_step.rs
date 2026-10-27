//! A reply made while a speech with tools plays asks its model once that speech's step is on the
//! record, not once the whole speech is over.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs};
use lk_agents_voice::trace::{GenerationStep, Step};

// Catches: a reply created while a speech played waited for that speech to finish before planning
// its request. A speech whose tools answered as its step ended goes on to a follow-up step, which
// needs the floor; the new reply had taken the floor and waited for the speech's end, so both
// waited until the replan's bound (the interruption timeout and a second): six seconds of dead air
// (`speech_settles`, survey seed 14338). The reply is planned once the step is recorded.
#[test]
fn a_reply_behind_a_tool_step_asks_its_model_at_once() {
    let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("Sure, one moment.".into()) },
        // A tool the agent does not have: its error output asks the model for a follow-up step.
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "missing".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        // Asked while the first speech plays and its tool runs.
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(200) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(7_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let ended = run
        .record
        .applied()
        .iter()
        .find(|a| matches!(a.input, Input::PlayoutFinished { .. }))
        .map(|a| a.at.at_ms)
        .expect("the step ended");
    // The requests made after the step ended: the follow-up's and the second reply's.
    let asked: Vec<(u64, u64)> = run
        .record
        .journal()
        .iter()
        .filter_map(|(at, s)| match s {
            Step::Generation(GenerationStep::ChatRequested { speech, .. }) if at.at_ms >= ended => {
                Some((speech.0, at.at_ms - ended))
            }
            _ => None,
        })
        .collect();
    assert!(!asked.is_empty(), "nothing was asked after the step ended");
    assert!(asked.iter().all(|(_, after)| *after < 1_000), "asked late: {asked:?}");
}
