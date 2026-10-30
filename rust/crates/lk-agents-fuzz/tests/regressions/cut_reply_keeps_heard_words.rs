//! A reply cut mid-playout is committed as the words heard when the output reports the cut late.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_core::tts::markup::ChatMessageExt;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};
use lk_agents_voice::events::SessionEvent;

const TEXT: &str = "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve.";

// Catches: a cut speech reading the synchroniser's words only if the output's report of the cut
// landed within a few scheduler hops. A host audio source reports from the host's loop, later
// than that, so the reply was committed whole: on real calls every interrupted reply's history
// held words the caller never heard (the room's transcript, paced, stopped where the audio did).
// The reference awaits the output's playout event for the synchronised transcript.
#[test]
fn a_cut_reply_is_committed_as_heard() {
    let setup = Setup { sync_transcription: true, lagging_output: true, ..Setup::default() };
    let inputs = [
        Input::Say {
            text: AgentText(TEXT.into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(1_000) },
        Input::Interrupt { force: true, wait: false },
        Input::Advance { ms: Millis(2_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let said: Vec<String> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::Assistant => m.text_content(),
                _ => None,
            },
            _ => None,
        })
        .collect();
    // The fake's audio is one short chunk and no progress was reported: nothing was heard.
    assert!(
        said.iter().all(|t| TEXT.starts_with(t.trim_end()) && t.len() < TEXT.len()),
        "{said:?}"
    );
}
