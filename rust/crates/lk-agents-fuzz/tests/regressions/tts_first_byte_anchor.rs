//! A reply's TTS timings count from when the provider was sent text, when it says so.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::events::SessionEvent;

// Catches: `tts_node_ttfb` measured from the first text the TTS node was given, though the
// gateway client stamps when its first sentence went out; with sentence buffering that read
// every reply's TTFB a few hundred milliseconds slow. `llm_node_ttfs` was the same stamp, so it
// equalled the LLM's time to first token. The reference anchors both on the stream's
// `USERDATA_TTS_STARTED_TIME`, and reports no time to first sentence for a TTS without one, as
// this fake.
#[test]
fn a_tts_without_a_start_stamp_reports_no_time_to_first_sentence() {
    let setup = Setup::default();
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(300) },
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
    let [reply] = replies.as_slice() else { panic!("one reply: {replies:?}") };
    assert!(reply.tts_node_ttfb.is_some(), "{reply:?}");
    assert!(reply.llm_node_ttfs.is_none(), "{reply:?}");
}
