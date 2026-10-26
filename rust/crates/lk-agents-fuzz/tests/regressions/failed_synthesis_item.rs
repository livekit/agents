//! A reply whose voice failed mid-step commits what was heard of it, marked cut short.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::{events::SessionEvent, trace::Step};

// Catches: a TTS failing after its first chunk played committed the reply as complete. The
// reference commits the synchroniser's words up to the played position and marks the message
// interrupted, so the LLM knows it was cut short (`_pipeline_reply_task_impl`,
// `interrupted=speech_handle.interrupted or any(out.tts_failed ...)`, agent_activity.py:4176).
#[test]
fn a_failed_synthesis_commits_what_was_heard_as_interrupted() {
    let text = |t: &str| AgentText(t.into());
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(20) },
            Input::LlmText { chat: Ref::new(0), text: text("Hello there.") },
            Input::Advance { ms: Millis(20) },
            Input::TtsHold { tts: Ref::new(0) },
            Input::LlmText { chat: Ref::new(0), text: text(" How are you?") },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::TtsFail { tts: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let messages: Vec<_> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, step)| match step {
            Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                ChatItem::Message(msg) if msg.role == Role::Assistant => Some(msg.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let [msg] = &messages[..] else { panic!("one reply, got {messages:?}") };
    assert!(msg.interrupted, "the reply is marked cut short");
    // The synchroniser heard the played segment through: its words are all of it.
    assert_eq!(msg.text_content().as_deref(), Some("Hello there. How are you?"));
}
