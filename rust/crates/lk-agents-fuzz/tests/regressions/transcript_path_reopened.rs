//! A transcript path reopened mid-speech forwards the later segments only, in order.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
use lk_agents_voice::trace::{PlayoutStep, Step};

// Catches: `text_conservation` requiring the forwarded words to be a prefix of the speech's text
// even when the transcription output was detached for its first segments. The sink is re-read per
// segment, so a reattach part-way shows what comes after it and skips what came before (survey
// seed 5066: step 1 spoken unshown, the tool reply's step shown).
#[test]
fn a_reattached_path_shows_what_follows() {
    let inputs = [
        Input::SetTranscriptionOutput { attach: false },
        Input::SetAudioOutput { attach: false },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("First part.".into()) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::SetTranscriptionOutput { attach: true },
        Input::Advance { ms: Millis(20) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("ok".into()),
        },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(1), text: AgentText("Second part.".into()) },
        Input::LlmEnd { chat: Ref::new(1) },
        Input::Advance { ms: Millis(100) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let forwarded: String = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Playout(PlayoutStep::TranscriptForwarded { text, .. }) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(forwarded, "Second part.", "only the step after the reattach was shown");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"text_conservation"), "{failed:?}");
}
