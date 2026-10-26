//! An adopted preemptive reply measures its end-to-end latency from the turn it answers.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, TurnMode, UserText};
use lk_agents_voice::events::SessionEvent;

// Catches: a preemptive reply started before its user turn was committed, so it never learned
// the turn's metrics and its message had no `e2e_latency`; with preemptive generation on (the
// default) no reply on a call had one. The reference writes the turn's metrics onto the
// preemptive generation's user message when it adopts it.
#[test]
fn an_adopted_preemptive_reply_has_its_latency() {
    let setup = Setup {
        stt: true,
        vad: true,
        preemptive: true,
        turn: TurnMode::Stt,
        ..Setup::default()
    };
    let words = || UserText("what's my balance".into());
    let inputs = [
        Input::Advance { ms: Millis(20) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(600) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
        Input::SttPreflight { utterance: Ref::new(0), text: words(), speaker: None },
        Input::Advance { ms: Millis(50) },
        Input::SttFinal { utterance: Ref::new(0), text: words(), speaker: None },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(1000) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("It's forty dollars.".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::TtsRelease { tts: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(500) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let replies: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::Assistant => Some(m.metrics.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(replies.len(), 1, "one reply");
    assert!(replies[0].e2e_latency.is_some(), "{:?}", replies[0]);
}
