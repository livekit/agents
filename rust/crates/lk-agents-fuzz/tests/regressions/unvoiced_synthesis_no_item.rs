//! A speech whose voice failed before any of its audio played commits no assistant item.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_core::tts::markup::ChatMessageExt;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
use lk_agents_voice::{events::SessionEvent, trace::Step};

// Catches: the failed synthesis's words committed as an interrupted assistant message the caller
// never heard. The reference's `forward_generation` leaves a segment whose TTS failed with none of
// its audio captured "skipped", so `forwarded_text` is empty and no message is added; found by
// `scripts/parity.py tts_failure`.
#[test]
fn an_unvoiced_speech_commits_nothing() {
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let inputs = [
            Input::TtsOutage { down: true },
            Input::Say {
                text: AgentText("This will not be heard.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(30_000) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let replies: Vec<_> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, step)| match step {
            Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                ChatItem::Message(m) if m.role == Role::Assistant => m.text_content(),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(replies.is_empty(), "{replies:?}");
}
