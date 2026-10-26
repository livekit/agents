//! A turn's metrics name the models that produced it.
use lk_agents_abi::{ChatItem, MetricsReport, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, TurnMode, UserText};
use lk_agents_voice::events::SessionEvent;

// Catches: the user and assistant messages committed without `stt_metadata`, `llm_metadata` and
// `tts_metadata`. The reference stamps each with the provider's `metrics_metadata`
// (`_init_metrics_from_end_of_turn`, `_pipeline_reply_task_impl`); on a real call every chat item,
// and the turn metrics exported from them, had no model name.
#[test]
fn a_spoken_turn_and_its_reply_name_their_models() {
    let setup = Setup { vad: true, stt: true, turn: TurnMode::Stt, ..Setup::default() };
    let inputs = [
        Input::Advance { ms: Millis(20) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("how much is it".into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(2_000) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("It's forty dollars.".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
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
    let of = |role: Role| -> Vec<MetricsReport> {
        run.record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::Message(m) if m.role == role => Some(m.metrics.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    };
    let (users, replies) = (of(Role::User), of(Role::Assistant));
    let ([user], [reply]) = (users.as_slice(), replies.as_slice()) else {
        panic!("one turn and one reply: {users:?} {replies:?}");
    };
    assert!(user.stt_metadata.is_some(), "{user:?}");
    assert!(reply.llm_metadata.is_some() && reply.tts_metadata.is_some(), "{reply:?}");
}
