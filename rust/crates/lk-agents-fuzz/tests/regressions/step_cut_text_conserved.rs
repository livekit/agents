//! A host's unforced `interrupt()` during playout ends the whole speech: the cut step loses its
//! words, the tool's answer is committed, and no follow-up is asked for.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};

// Catches: `text_conservation` holding a speech that completed to every word its steps were
// given, though the host's unforced `interrupt()` stops only the playing step (its words are
// never forwarded) and the tool's follow-up then completes the speech (survey seed 2750). The
// completion clause now skips a speech the driver journalled interrupting. The step-scoped cut is
// gone since: as `_interrupt_fut` is one-shot, the speech ends interrupted and nothing follows.
#[test]
fn a_cut_step_does_not_owe_its_words() {
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
            Input::Advance { ms: Millis(20) },
            Input::InterruptSpeech { speech: Ref::new(0), force: false },
            // Words of the cut step: never forwarded.
            Input::LlmText {
                chat: Ref::new(0),
                text: AgentText("Your balance is due on the fourteenth.".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Value,
                text: AgentText("done".into()),
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(100) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let speeches = run.record.host_view().speeches();
    assert!(speeches[0].is_interrupted(), "the host's interrupt ends the speech");
    assert!(
        run.record.internal_view().speech(speeches[0].id).interrupted_at().is_some(),
        "its first step was cut"
    );
    let requests = run.record.inference_view().requests_for(speeches[0].id);
    assert_eq!(requests.len(), 1, "no follow-up after the interrupt: {requests:?}");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"text_conservation"), "{failures:?}");
}
