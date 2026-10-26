//! A spoken user turn is committed with its end-of-turn metrics.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode, UserText};
use lk_agents_voice::events::SessionEvent;

// Catches: the user message built with an empty `MetricsReport`, though the turn's anchors and
// delays were computed for the EOU metrics. The reference's `_init_metrics_from_end_of_turn`
// stamps them on the message, and the reply's `e2e_latency` is measured from its
// `stopped_speaking_at`: with them gone no reply on a call had an end-to-end latency, and the
// session report showed no turn timings.
#[test]
fn a_spoken_turn_carries_its_metrics() {
    let setup = Setup { vad: true, stt: true, turn: TurnMode::Stt, ..Setup::default() };
    let inputs = [
        Input::Advance { ms: Millis(20) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("hello there".into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(2000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let users: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::User => Some(m.metrics.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(users.len(), 1, "one user turn");
    let m = &users[0];
    assert!(m.started_speaking_at.is_some() && m.stopped_speaking_at.is_some(), "{m:?}");
    assert!(m.end_of_turn_delay.is_some() && m.transcription_delay.is_some(), "{m:?}");
}
