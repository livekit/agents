//! A duplex model's words that arrive after the sound they describe still enter the conversation.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref};
use lk_agents_voice::events::SessionEvent;

// Catches: the last words of a duplex model's utterance, which its text stream delivers after the
// utterance's audio ended, being lost or failing the session: the segmenter attributes them to
// the completed speech (`TranscriptAttribution`) and the driver commits them as an item of their
// own, with no error.
#[test]
fn a_late_transcript_is_committed_after_its_speech() {
    let mut attributed = 0;
    for seed in 0..16 {
        let setup = Setup {
            realtime: Some(RealtimeSetup { duplex: true, seed, ..RealtimeSetup::default() }),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(1_000) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::Advance { ms: Millis(10_000) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let events = run.record.host_view().events();
        let errors: Vec<_> =
            events.iter().filter(|(_, ev)| matches!(ev, SessionEvent::Error(_))).collect();
        assert!(errors.is_empty(), "seed {seed}: {errors:?}");
        let said = events
            .iter()
            .filter(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => {
                    matches!(item.as_ref(), ChatItem::Message(m) if m.role == Role::Assistant)
                }
                _ => false,
            })
            .count();
        // The utterance, and its late words as an item of their own.
        attributed += usize::from(said >= 2);
    }
    assert!(attributed > 0, "no seed's model sent its last words late");
}
