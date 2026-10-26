//! The session's `tts_text_transforms` reach the voice: by default markdown and emoji never do.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

// Catches: the resolved `tts_text_transforms` option (default `filter_markdown` + `filter_emoji`)
// never reaching the synthesis pipeline, which read only a constructor argument no host passes, so
// every reply's markdown and emoji were spoken; and the markdown stage starting mid-line, so a
// reply's leading `# ` heading was spoken too.
#[test]
fn markdown_and_emoji_never_reach_the_voice() {
    let say = Input::Say {
        text: AgentText("# Today\n**Great news!** Your order shipped \u{1F389}".into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in [say, Input::Advance { ms: Millis(100) }] {
            run.apply(&input).await;
        }
        run
    });
    let spoken: String =
        run.record.produced().iter().map(|(_, _, _, text)| text.as_str()).collect();
    assert!(spoken.contains("Great news!"), "the words reached the voice: {spoken:?}");
    assert!(!spoken.contains(['#', '*', '\u{1F389}']), "markup reached the voice: {spoken:?}");
}
