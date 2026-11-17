//! A bound voice conversation saves pending recognition independently of agent speech.

use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, StorageSetup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode, UserText};
use lk_agents_fuzz::record::{Kind, StorageStep};

/// Interim updates are durable before any reply starts; accepting the turn replaces its draft
/// under the same item identity, even when no reply is requested.
#[test]
fn recognition_is_checkpointed_before_a_reply_and_a_silent_turn_is_saved() {
    let run = simulate(async {
        let setup = Setup {
            stt: true,
            turn: TurnMode::Manual,
            storage: Some(StorageSetup::default()),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("book".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("book a table".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
            Input::CommitUserTurn {
                transcript_timeout: Millis(0),
                stt_flush: Millis(0),
                skip_reply: true,
            },
            Input::Advance { ms: Millis(2_000) },
            Input::Close { drain: false },
            Input::AwaitExit {},
        ] {
            run.apply(&input).await;
        }
        run
    });
    assert_eq!(run.record.resources(Kind::Chat).count(), 0, "the user turn requested no reply");
    let storage = run.record.host_view().storage();
    let drafts: Vec<_> = storage
        .iter()
        .filter_map(|(_, step)| match step {
            StorageStep::ConversationCheckpoint { pending_user_turns, .. } => {
                pending_user_turns.first()
            }
            _ => None,
        })
        .collect();
    assert!(
        drafts.iter().any(|message| message.raw_text_content().as_deref() == Some("book")),
        "the first interim was durable"
    );
    let draft = drafts
        .iter()
        .find(|message| message.raw_text_content().as_deref() == Some("book a table"))
        .expect("the later interim replaced it durably");
    assert!(
        drafts.iter().all(|message| message.id == draft.id),
        "interim revisions retained one identity"
    );
    let saved = storage
        .iter()
        .rev()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { items } => Some(items),
            _ => None,
        })
        .expect("the closed store was read back");
    let users: Vec<_> = saved
        .iter()
        .filter_map(ChatItem::as_message)
        .filter(|message| message.role == Role::User)
        .collect();
    assert_eq!(users.len(), 1, "the draft was not duplicated: {saved:?}");
    assert_eq!(users[0].id, draft.id);
    assert_eq!(users[0].raw_text_content().as_deref(), Some("book a table"));
}

/// A reply finishes while the person is speaking, releasing the storage run. Their final
/// transcript then opens a new run; its accepted timestamp must replace the older draft.
#[test]
fn a_final_user_turn_replaces_its_draft_after_a_reply_releases_storage() {
    use lk_agents_fuzz::input::AgentText;
    use lk_agents_voice::events::SessionEvent;

    let run = simulate(async {
        let setup = Setup {
            stt: true,
            tts: false,
            turn: TurnMode::Manual,
            storage: Some(StorageSetup::default()),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("book a table".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(1_000) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Take your time.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
            Input::CommitUserTurn {
                transcript_timeout: Millis(0),
                stt_flush: Millis(0),
                skip_reply: true,
            },
            Input::Advance { ms: Millis(2_000) },
            Input::Close { drain: false },
            Input::AwaitExit {},
        ] {
            run.apply(&input).await;
        }
        run
    });
    let host = run.record.host_view();
    let accepted = host
        .events()
        .into_iter()
        .find_map(|(_, event)| match event {
            SessionEvent::ConversationItemAdded { item, .. } => {
                item.as_message().filter(|message| message.role == Role::User)
            }
            _ => None,
        })
        .expect("the completed user turn was accepted");
    let storage = host.storage();
    let saved = storage
        .iter()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { items } => items
                .iter()
                .filter_map(ChatItem::as_message)
                .find(|message| message.id == accepted.id),
            _ => None,
        })
        .expect("the accepted user turn was saved");
    assert_eq!(saved.created_at, accepted.created_at, "a stored draft replaced the final turn");
    assert_eq!(saved.raw_text_content(), accepted.raw_text_content());
    let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The ledger places a tool result with its call. Later snapshots must preserve that accepted
/// timestamp instead of projecting the raw execution result over it again.
#[test]
fn a_committed_tool_result_keeps_its_timestamp_in_later_checkpoints() {
    use lk_agents_fuzz::input::{AgentText, ToolArgs};
    use lk_agents_voice::events::SessionEvent;

    let run = simulate(async {
        let setup =
            Setup { tts: false, storage: Some(StorageSetup::default()), ..Setup::default() };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SetAudioOutput { attach: false },
            Input::GenerateReply {
                user_input: Some(UserText("Can you look that up?".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(1_000) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "missing".into(),
                args: ToolArgs("{}".into()),
            },
            Input::Advance { ms: Millis(200) },
            Input::UpdateChatCtx {
                items: vec![lk_agents_fuzz::input::Role::Assistant],
                text: AgentText("A private model reminder.".into()),
            },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "also_missing".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmText { chat: Ref::new(0), text: AgentText("I cannot use that tool.".into()) },
            Input::LlmFail { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
            Input::Close { drain: false },
            Input::AwaitExit {},
        ] {
            run.apply(&input).await;
        }
        run
    });
    let accepted = run
        .record
        .host_view()
        .events()
        .into_iter()
        .find_map(|(_, event)| match event {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::FunctionCallOutput(output) => Some(output),
                _ => None,
            },
            _ => None,
        })
        .expect("the tool failure was recorded for the conversation");
    let storage = run.record.host_view().storage();
    let saved = storage
        .iter()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { items } => items.iter().find_map(|item| match item {
                ChatItem::FunctionCallOutput(output) if output.id == accepted.id => Some(output),
                _ => None,
            }),
            _ => None,
        })
        .expect("the accepted result was saved");
    assert_eq!(saved, accepted, "the raw execution result replaced the committed result");
    let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Replacing an agent's private prompt cannot turn the reminder for an unfinished tool
/// into a durable call revision or a fabricated result.
#[test]
fn a_running_tool_reminder_does_not_replace_the_saved_call() {
    use lk_agents_fuzz::input::{AgentText, ToolAnswer, ToolArgs};

    let run = simulate(async {
        let setup = Setup {
            tts: false,
            tools: vec!["lookup".into()],
            storage: Some(StorageSetup::default()),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(20) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(1_000) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Detached,
                text: AgentText("Checking.".into()),
            },
            Input::Advance { ms: Millis(1_000) },
            Input::Interrupt { force: true, wait: true },
            Input::Advance { ms: Millis(6_000) },
            // The next prompt lacks the call, so generation adds its private reminder.
            Input::UpdateChatCtx { items: Vec::new(), text: AgentText(String::new()) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        let chats = run.record.resources(Kind::Chat).count();
        assert!(chats >= 2, "the new reply started while lookup was still running");
        let chat = Ref::new((chats - 1) as u32);
        for input in [
            Input::LlmText { chat, text: AgentText("Anything else?".into()) },
            Input::LlmEnd { chat },
            Input::Advance { ms: Millis(1_000) },
            Input::Close { drain: false },
            Input::AwaitExit {},
        ] {
            run.apply(&input).await;
        }
        run
    });
    let storage = run.record.host_view().storage();
    let saved = storage
        .iter()
        .rev()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { items } => Some(items),
            _ => None,
        })
        .expect("the closed store was read back");
    let call = saved
        .iter()
        .find_map(|item| match item {
            ChatItem::FunctionCall(call) => Some(call),
            _ => None,
        })
        .expect("the original accepted tool call remains saved");
    let prompt = run.record.resources(Kind::Chat).last().expect("the later model request ran");
    let lk_agents_fuzz::record::Detail::Chat { ctx_ids, .. } = prompt.detail() else {
        panic!("the model request exposes its prompt identities")
    };
    assert!(ctx_ids.contains(&call.id), "the new prompt carried the running tool's reminder");
    assert!(call.extra.is_empty(), "the model's private reminder replaced the call: {call:?}");
    let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}
