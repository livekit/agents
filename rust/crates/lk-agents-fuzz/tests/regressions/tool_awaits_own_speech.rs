//! A tool awaiting its own speech is refused (`RuntimeError`) instead of waiting forever.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

// Catches: `await ctx.speech_handle` inside a blocking tool waited for a speech that waited for
// that tool: the call hung silently. The reference raises (`SpeechHandle.wait_for_playout`'s
// circular-wait guard); the port's comments claimed the shim had it, and nothing did.
#[test]
fn a_tool_awaiting_its_own_speech_is_refused() {
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
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::AwaitOwnSpeech,
                text: AgentText(String::new()),
            },
            Input::Advance { ms: Millis(200) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let ended = run.record.host_view().tools_run().iter().all(|t| !t.ended_announcements().is_empty());
    assert!(ended, "the tool's wait on its own speech was refused and the tool returned");
    let speeches = run.record.host_view().speeches();
    assert!(speeches[0].is_done(), "the speech no longer waits on a tool waiting on it");
}
