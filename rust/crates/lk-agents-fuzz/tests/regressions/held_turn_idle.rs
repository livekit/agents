//! A turn held for the agent a handoff brings in does not hold idle off.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{
    AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
};

// Catches: a user turn decided while a handoff drained was held for the arriving agent and held
// idle off, while the drain waited on a speech whose tool waited on idle: nothing moved until the
// close (`speech_settles`, survey seeds 1923, 3526, 4514). The reference has no held turn
// (`on_end_of_turn` skips it while scheduling is paused), so it never blocks `wait_for_idle`.
#[test]
fn a_tool_waiting_for_idle_lets_a_drain_holding_a_turn_finish() {
    let setup = Setup {
        tools: vec!["lookup".into()],
        stt: true,
        turn: TurnMode::Manual,
        ..Setup::default()
    };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::Handoff {
                agent: "next".into(),
                instructions: AgentText("You are next.".into()),
                tools: vec![],
                pause_previous: true,
                resume: false,
            },
            Input::Advance { ms: Millis(20) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("yeah hi".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::CommitUserTurn {
                transcript_timeout: Millis(100),
                stt_flush: Millis(0),
                skip_reply: false,
            },
            // Past the endpointing delay: the turn is decided, and held for the arriving agent.
            Input::Advance { ms: Millis(1_000) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::AfterIdle,
                text: AgentText(String::new()),
            },
            Input::Advance { ms: Millis(200) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let speeches = run.record.host_view().speeches();
    let first = speeches.first().expect("the reply");
    assert!(first.is_done(), "the speech waits on a tool that waits on a held turn");
    assert_eq!(speeches.len(), 2, "the held turn reached the arriving agent and was answered");
}
