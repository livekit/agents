//! A step whose model failed gives the floor back before waiting for its tools.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

// Catches: a step whose model request failed after dispatching a tool kept its speech current
// while it waited for the tool, so a tool asking `ctx.wait_for_idle()` waited on its own speech
// until the close (`speech_settles`, survey seed 4514). The reference marks the generation done
// before awaiting the tools (`_pipeline_reply_task_impl`, `_mark_generation_done`).
#[test]
fn a_tool_waiting_for_idle_outlives_its_failed_step() {
    let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
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
            Input::LlmFail { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::AfterIdle,
                text: AgentText(String::new()),
            },
            Input::Advance { ms: Millis(100) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        assert!(run.record.applied().iter().all(|a| a.accepted), "every input landed");
        run
    });
    let speeches = run.record.host_view().speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done(), "the speech waits on a tool that waits on it");
}
