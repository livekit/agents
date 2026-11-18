use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::{
    drive::{RealtimeSetup, Run, Setup, StorageSetup, simulate},
    fakes::RealtimeDelays,
    input::{Count, Input, Millis, Ref, RtEvent},
    record::StorageStep,
};
use lk_agents_voice::events::SessionEvent;

fn setup(delays: RealtimeDelays, storage: StorageSetup) -> Setup {
    Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(0),
            delays,
            ..Default::default()
        }),
        storage: Some(storage),
        ..Default::default()
    }
}

async fn advance(run: &mut Run, millis: u64) {
    crate::playout::apply_with_playout(run, &Input::Advance { ms: Millis(millis) }).await;
}

async fn turn(run: &mut Run) {
    run.apply(&Input::MicFrames { mic: Ref::new(0), chunks: Count(30) }).await;
    advance(run, 300).await;
    run.apply(&Input::CommitTurn { skip_reply: false }).await;
}

async fn close(run: &mut Run) {
    run.apply(&Input::Close { drain: true }).await;
    run.apply(&Input::AwaitExit {}).await;
}

fn saved(run: &Run) -> Vec<ChatItem> {
    run.record
        .host_view()
        .storage()
        .iter()
        .rev()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { items } => Some(items.clone()),
            _ => None,
        })
        .expect("the closed SQLite partition was read back")
}

fn messages(items: &[ChatItem]) -> Vec<(Role, String, String)> {
    items
        .iter()
        .filter_map(ChatItem::as_message)
        .map(|m| (m.role, m.id.clone(), m.raw_text_content().unwrap_or_default()))
        .collect()
}

/// Reproduces real provider ordering: input ASR completes after the assistant has already
/// answered and its speech has been durably saved. A final must fill the original user slot.
#[test]
fn a_late_transcript_stays_before_its_already_saved_answer() {
    let run = simulate(async {
        let setup = setup(
            RealtimeDelays { transcription: Some(20_000), ..Default::default() },
            StorageSetup::default(),
        );
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 10_000).await;
        let events = run.record.host_view().events();
        assert!(events.iter().any(|(_, event)| matches!(event, SessionEvent::ConversationItemAdded { item, .. } if matches!(item.as_ref(), ChatItem::Message(m) if m.role == Role::Assistant))));
        assert!(!events.iter().any(|(_, event)| matches!(event, SessionEvent::ConversationItemAdded { item, .. } if matches!(item.as_ref(), ChatItem::Message(m) if m.role == Role::User))));
        advance(&mut run, 12_000).await;
        close(&mut run).await;
        run
    });
    let saved = messages(&saved(&run));
    assert_eq!(
        saved.iter().map(|m| m.0).collect::<Vec<_>>(),
        [Role::User, Role::Assistant],
        "{saved:#?}"
    );
    assert!(!saved[0].2.is_empty());
}

/// Two ASR jobs finish in reverse order, while speech commits and conversation checkpoints
/// pass through the real SQLite store. Arrival order must not become durable turn order.
#[test]
fn out_of_order_transcript_finals_preserve_both_turns_in_storage() {
    let run = simulate(async {
        let setup = setup(
            RealtimeDelays {
                first_transcription: Some(25_000),
                transcription: Some(100),
                ..Default::default()
            },
            StorageSetup::default(),
        );
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 10_000).await;
        turn(&mut run).await;
        advance(&mut run, 10_000).await;
        advance(&mut run, 10_000).await;
        close(&mut run).await;
        run
    });
    let saved = messages(&saved(&run));
    assert_eq!(
        saved.iter().map(|m| m.0).collect::<Vec<_>>(),
        [Role::User, Role::Assistant, Role::User, Role::Assistant],
        "{saved:#?}"
    );
    let users: Vec<_> = saved.iter().filter(|m| m.0 == Role::User).collect();
    assert_eq!(users.len(), 2);
    assert_ne!(users[0].1, users[1].1);
}

/// A failed ASR job settles its partial under its own item ID. A later successful turn must
/// neither inherit that prefix nor cause a second copy of the failed turn to be stored.
#[test]
fn a_failed_transcription_keeps_its_partial_and_the_next_turn_is_independent() {
    let run = simulate(async {
        let setup = setup(
            RealtimeDelays {
                first_transcription: Some(20_000),
                transcription: Some(100),
                transcription_partial: Some(100),
                ..Default::default()
            },
            StorageSetup::default(),
        );
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 500).await;
        run.apply(&Input::RealtimeEvent {
            socket: Ref::new(0),
            event: RtEvent::TranscriptionFailed,
        })
        .await;
        advance(&mut run, 10_000).await;
        turn(&mut run).await;
        advance(&mut run, 12_000).await;
        close(&mut run).await;
        run
    });
    let saved = messages(&saved(&run));
    assert_eq!(
        saved.iter().map(|m| m.0).collect::<Vec<_>>(),
        [Role::User, Role::Assistant, Role::User, Role::Assistant],
        "{saved:#?}"
    );
    let users: Vec<_> = saved.iter().filter(|m| m.0 == Role::User).collect();
    assert!(!users[0].2.is_empty(), "the received partial was retained");
    assert_ne!(users[0].1, users[1].1);
    assert!(
        users[0].2.split_whitespace().count() < users[1].2.split_whitespace().count(),
        "the failed partial was not filled with the next turn's words: {users:?}"
    );
}

/// A transcription may fail before producing any words while response generation also fails.
/// The next turn remains usable and storage must not invent a user transcript for the failure.
#[test]
fn transcription_and_response_failures_do_not_invent_words_or_poison_the_next_turn() {
    let run = simulate(async {
        let setup = setup(
            RealtimeDelays {
                first_transcription: Some(20_000),
                transcription: Some(100),
                transcription_partial: Some(2_000),
                ..Default::default()
            },
            StorageSetup::default(),
        );
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 500).await;
        for event in [RtEvent::TranscriptionFailed, RtEvent::ResponseFailed, RtEvent::ServerError] {
            run.apply(&Input::RealtimeEvent { socket: Ref::new(0), event }).await;
        }
        advance(&mut run, 3_000).await;
        turn(&mut run).await;
        advance(&mut run, 12_000).await;
        close(&mut run).await;
        run
    });
    let saved = messages(&saved(&run));
    let users: Vec<_> = saved.iter().filter(|m| m.0 == Role::User).collect();
    assert_eq!(users.len(), 2, "one server item per committed audio turn: {saved:#?}");
    assert!(users[0].2.is_empty(), "the failed ASR job must not invent words");
    assert!(!users[1].2.is_empty());
    let told_users: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, event)| match event {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(m) if m.role == Role::User => Some(m.id.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(
        told_users,
        [users[1].1.clone()],
        "the empty model placeholder is not a human transcript"
    );
    assert!(
        run.record
            .host_view()
            .events()
            .iter()
            .any(|(_, event)| matches!(event, SessionEvent::Error(_))),
        "provider failures reach the host"
    );
}

/// A write can apply and lose its acknowledgement after speech was saved. The late final is
/// recovered by the production store's idempotent resend, without a duplicate user item.
#[test]
fn a_late_transcript_is_saved_once_when_a_write_acknowledgement_is_lost() {
    let run = simulate(async {
        let storage = StorageSetup { lose_commit_answer_at: Some(20_250), ..Default::default() };
        let setup =
            setup(RealtimeDelays { transcription: Some(20_000), ..Default::default() }, storage);
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 30_000).await;
        close(&mut run).await;
        run
    });
    let saved = messages(&saved(&run));
    assert_eq!(
        saved.iter().map(|m| m.0).collect::<Vec<_>>(),
        [Role::User, Role::Assistant],
        "{saved:#?}"
    );
    assert!(!saved[0].2.is_empty(), "{saved:#?}");
    let checkpoints = run.record.host_view().storage();
    assert!(
        checkpoints.iter().any(|(_, step)| matches!(step, StorageStep::CommitAnswerLost)),
        "the history write must apply and lose its first answer"
    );
    assert!(checkpoints.iter().any(|(_, step)| matches!(step, StorageStep::ConversationCheckpoint { items, .. } if items.iter().any(|item| item.id() == saved[0].1 && item.as_message().is_some_and(|m| !m.raw_text_content().unwrap_or_default().is_empty())))), "the final reached an acknowledged checkpoint");
}

/// A failed conversation checkpoint is terminal: the host learns that the late transcript
/// was not durably accepted, rather than receiving a successful close over an empty saved row.
#[test]
fn a_failed_late_transcript_checkpoint_closes_with_an_explicit_storage_error() {
    let run = simulate(async {
        let storage = StorageSetup { outage: Some((20_250, 500)), ..Default::default() };
        let setup =
            setup(RealtimeDelays { transcription: Some(20_000), ..Default::default() }, storage);
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 30_000).await;
        close(&mut run).await;
        run
    });
    assert!(
        run.record.host_view().storage().iter().any(|(_, step)| matches!(
            step,
            StorageStep::ConversationAnswered { result: Err(_), .. }
        )),
        "the outage must refuse a conversation checkpoint"
    );
    assert!(run.record.host_view().events().iter().any(|(_, event)| matches!(event, SessionEvent::Close { reason: lk_agents_abi::voice::events::CloseReason::Error, error: Some(error) } if error.source == lk_agents_abi::voice::events::ErrorSource::Storage)), "the failed durable save is an explicit storage failure");
}

/// A disconnect ends only that connection's ASR work. Reconnection settles its received
/// partial and a later input gets its own item instead of inheriting the interrupted transcript.
#[test]
fn reconnect_settles_a_partial_transcript_before_accepting_the_next_turn() {
    let run = simulate(async {
        let setup = setup(
            RealtimeDelays {
                first_transcription: Some(20_000),
                transcription: Some(100),
                transcription_partial: Some(100),
                ..Default::default()
            },
            StorageSetup::default(),
        );
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 500).await;
        run.apply(&Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::Disconnected })
            .await;
        advance(&mut run, 100).await;
        run.apply(&Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::Reconnected }).await;
        advance(&mut run, 1_000).await;
        turn(&mut run).await;
        advance(&mut run, 25_000).await;
        close(&mut run).await;
        run
    });
    let saved = messages(&saved(&run));
    let users: Vec<_> = saved.iter().filter(|m| m.0 == Role::User).collect();
    assert_eq!(users.len(), 2, "{saved:#?}");
    assert!(!users[0].2.is_empty());
    assert!(!users[1].2.is_empty());
    assert_ne!(users[0].1, users[1].1);
    assert!(
        users[0].2.split_whitespace().count() < users[1].2.split_whitespace().count(),
        "the interrupted prefix belongs only to its original item: {users:?}"
    );
}

/// The failed job need not be the most recent input. Failing an older turn after a newer final
/// must settle that older prefix under its own ID and retain both replies in conversation order.
#[test]
fn an_older_transcription_can_fail_after_a_newer_turn_was_answered() {
    let run = simulate(async {
        let setup = setup(
            RealtimeDelays {
                first_transcription: Some(30_000),
                transcription: Some(100),
                transcription_partial: Some(100),
                ..Default::default()
            },
            StorageSetup::default(),
        );
        let mut run = Run::start(&setup, 0, Default::default()).await;
        turn(&mut run).await;
        advance(&mut run, 10_000).await;
        turn(&mut run).await;
        advance(&mut run, 10_000).await;
        run.apply(&Input::RealtimeEvent {
            socket: Ref::new(0),
            event: RtEvent::OldestTranscriptionFailed,
        })
        .await;
        advance(&mut run, 1_000).await;
        close(&mut run).await;
        run
    });
    let saved = messages(&saved(&run));
    assert_eq!(
        saved.iter().map(|m| m.0).collect::<Vec<_>>(),
        [Role::User, Role::Assistant, Role::User, Role::Assistant],
        "{saved:#?}"
    );
    let users: Vec<_> = saved.iter().filter(|m| m.0 == Role::User).collect();
    let finals: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, event)| match event {
            SessionEvent::UserInputTranscribed(t) if t.is_final => t.item_id.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(
        finals,
        [users[1].1.clone(), users[0].1.clone()],
        "the failing ASR job was the older one"
    );
    assert!(
        users[0].2.split_whitespace().count() < 4,
        "the first job failed before its complete final"
    );
}
