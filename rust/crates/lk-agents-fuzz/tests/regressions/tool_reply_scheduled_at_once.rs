//! A background tool's next deferred reply is scheduled in the cycle that makes it.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
use lk_agents_voice::trace::{SpeechStep, Step, ToolStep};

// Catches: a detached tool's second update, landing while the reply to its first was playing,
// delivered by the cycle's waiter pass once that reply ended, after the cycle's scheduling pass
// had run: the new reply sat queued with the floor free until some unrelated wake scheduled it
// (the user-away timer, 15 s later in survey seed 70), the agent silent with the tool's news in
// hand (`speech_is_heard`). What the waiters start is now scheduled in the same cycle.
#[test]
fn a_second_tool_update_reply_is_authorized_at_once() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        tools: vec!["lookup".into()],
        max_tool_steps: Some(1),
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        // The tool answers with a progress update and keeps running in the background.
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Detached,
            text: AgentText("Checking.".into()),
        },
        // The model speaks the progress and calls the tool again, which returns nothing.
        Input::Advance { ms: Millis(3_000) },
        Input::ToolAnswered {
            tool: Ref::new(1),
            answer: ToolAnswer::Return,
            text: AgentText(String::new()),
        },
        Input::PlayoutStarted { segment: Ref::new(1) },
        Input::PlayoutFinished { segment: Ref::new(1) },
        Input::Advance { ms: Millis(1_000) },
        // The first news, with the agent idle: its reply plays.
        Input::ToolUpdate {
            tool: Ref::new(0),
            progress: Some(AgentText("Found it.".into())),
            end: ToolAnswer::Return,
        },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(2) },
        // The second news lands while that reply plays, and waits for it to end.
        Input::ToolUpdate {
            tool: Ref::new(0),
            progress: Some(AgentText("It shipped.".into())),
            end: ToolAnswer::Return,
        },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutFinished { segment: Ref::new(2) },
        Input::Advance { ms: Millis(20_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let replies: Vec<_> = journal
        .iter()
        .filter_map(|(at, s)| match s {
            Step::Tool(ToolStep::ReplyScheduled { speech, .. }) => Some((*at, *speech)),
            _ => None,
        })
        .collect();
    let [_, (at, speech)] = replies[..] else { panic!("two deferred replies: {replies:?}") };
    let authorized = journal.iter().find_map(|(a, s)| match s {
        Step::Speech(SpeechStep::Authorized { speech: sp }) if *sp == speech => Some(*a),
        _ => None,
    });
    assert!(
        authorized.is_some_and(|a| a.at_ms == at.at_ms),
        "made at {} ms, authorized at {:?}",
        at.at_ms,
        authorized.map(|a| a.at_ms)
    );
}
