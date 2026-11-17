use crate::StorageError;
use crate::context::store::ContextStore;
use crate::testing::Backend;
use crate::voice::VoiceStore;
use lk_agents_abi::storage::context::{ContextActivation as _, ContextField};
use lk_agents_abi::storage::voice::{
    VoiceHistory, VoiceStorage as _, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as _,
};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use serde_json::json;
use std::time::Duration;

#[tokio::test]
async fn a_delayed_voice_checkpoint_receipt_preserves_later_live_input() {
    use crate::Storage as _;
    use crate::testing::{Faults, TempDir, Wrapped, tasks};
    use std::sync::Arc;

    let dir = TempDir::new("voice_checkpoint_receipt_");
    let storage = crate::backends::sqlite::SqliteStorage::open(
        &dir.0,
        crate::thread::LeaseTiming::default(),
        tasks(),
    )
    .await
    .unwrap();
    let faults = Arc::<Faults>::default();
    let shared = faults.clone();
    let wrap = move |conn: crate::backends::sqlite::FileConnection| {
        Arc::new(shared.over(Box::new(conn))) as Arc<dyn crate::ThreadDbConnection>
    };
    let storage = Arc::new(Wrapped { storage: Arc::new(storage), wrap });
    let thread_id = storage.create_thread(Default::default()).await.unwrap().thread_id;
    let store = Arc::new(
        VoiceStore::bound(storage.clone(), thread_id, "desk".into(), None, Cx::NONE).await.unwrap(),
    );
    let initial = store.acquire(1, VoiceHistory::default()).await.unwrap();
    store.checkpoint(1, 1, "desk".into(), initial).await.unwrap();
    let activation = store.activation();
    let _attachment = activation.attach_session_sink(store.session_sink()).unwrap();
    let before = activation.session_state().unwrap();
    let (committed, release) = faults.hold_next_receipt("UPDATE partitions SET state =");
    let (done, receipt) = lk_agents_net::signal::once();
    let writing = activation.clone();
    let background = storage.background();
    let _flush = background.tasks.scoped(&*background.spawner, "voice flush", async move {
        done.resolve(writing.flush_session(vec![]).await);
    });
    committed.wait().await.unwrap();
    let later = ChatItem::Message(ChatMessage::text(
        "later-input",
        Role::User,
        "Actually, Saturday.",
        std::time::UNIX_EPOCH,
    ));
    let live =
        crate::session_state::voice_value("desk".into(), vec![later], Some(&before)).unwrap();
    activation.stage_session(live.clone()).unwrap();
    release.resolve(());
    receipt.wait().await.unwrap().unwrap();
    assert_eq!(activation.session_state(), Some(live), "a receipt cannot roll back later input");
    let saved = activation.load_session().await.unwrap().unwrap();
    assert_eq!(saved["state"]["chat_history"], json!([]), "later input was not in this write");
    store.aclose().await.unwrap();
    assert_eq!(activation.session_state(), Some(saved), "only the checkpoint was durable");
}

#[tokio::test]
async fn a_queued_voice_flush_keeps_remote_state_admitted_before_it_gets_the_write_owner() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let initial = store.acquire(1, VoiceHistory::default()).await.unwrap();
    store.checkpoint(1, 1, "desk".into(), initial).await.unwrap();
    let activation = store.activation();
    let _attachment = activation.attach_session_sink(store.session_sink()).unwrap();
    let owner = store.state.lock().await;
    let mut flush = std::pin::pin!(activation.flush_session(vec![]));
    assert!(
        std::future::Future::poll(
            flush.as_mut(),
            &mut std::task::Context::from_waker(std::task::Waker::noop()),
        )
        .is_pending(),
        "the flush waits for the existing write owner",
    );
    let mut admitted = activation.session_state().unwrap();
    let mut remote =
        lk_agents_abi::delegation::remote::RemoteAgentState::new("https://booking.test".into());
    remote.default_context_id = Some("new-trip".into());
    admitted["state"]["remote_agents"]["booking"] = serde_json::to_value(remote).unwrap();
    activation.stage_session(admitted).unwrap();
    drop(owner);
    flush.await.unwrap();
    let saved = activation.load_session().await.unwrap().unwrap();
    assert_eq!(
        saved["state"]["remote_agents"]["booking"]["default_context_id"],
        json!("new-trip"),
        "waiting for an earlier write must not restore the caller's older metadata",
    );
    store.aclose().await.unwrap();
    assert_eq!(activation.session_state(), Some(saved));
    fixture.cleanup(&[]).await;
}

#[tokio::test]
async fn a_voice_turn_reloads_remote_state_written_by_a_text_turn_between_speeches() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    let initial = store.acquire(1, VoiceHistory::default()).await.unwrap();
    let initial = store.checkpoint(1, 1, "desk".into(), initial).await.unwrap();
    let local = store.finish(Some(1), 1, initial).await.unwrap();

    let thread = fixture.thread().await;
    let holder = crate::Holder::new("text");
    let mut text =
        thread.partition("desk").open(crate::RunOptions::new("text", ""), &holder).await.unwrap();
    let mut updated = text.state().cloned().unwrap();
    let mut remote =
        lk_agents_abi::delegation::remote::RemoteAgentState::new("https://booking.test".into());
    remote.default_context_id = Some("continued-trip".into());
    updated["state"]["remote_agents"]["booking"] = serde_json::to_value(remote).unwrap();
    text.set_state(updated);
    text.commit(crate::CommitOptions::default()).await.unwrap();

    store.acquire(2, local).await.unwrap();
    let resumed = store.activation().session_state().unwrap();
    assert_eq!(
        resumed["state"]["remote_agents"]["booking"]["default_context_id"],
        json!("continued-trip"),
        "the prior voice activation must not overwrite the text turn's remote state",
    );
    store.aclose().await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn a_model_view_omitting_a_handoff_keeps_its_order_across_timestamp_ties() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    let handoff = ChatItem::AgentHandoff(lk_agents_abi::AgentHandoffItem {
        id: "handoff".into(),
        old_agent_id: None,
        new_agent_id: "desk".into(),
        created_at: std::time::UNIX_EPOCH,
    });
    let user =
        ChatItem::Message(ChatMessage::text("user", Role::User, "hello", std::time::UNIX_EPOCH));
    store.acquire(1, VoiceHistory::new(vec![handoff.clone(), user.clone()])).await.unwrap();
    store.finish(Some(1), 1, VoiceHistory::new(vec![handoff, user.clone()])).await.unwrap();
    let merged = store.acquire(2, VoiceHistory::new(vec![user])).await.unwrap();
    assert_eq!(merged.items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["handoff", "user"]);
    store.aclose().await.unwrap();
    fixture.cleanup(&[]).await;
}

/// A canceled acquisition can leave no active run when shutdown submits the final history.
/// Its successful receipt must still mean that history exists in the database.
#[tokio::test]
async fn a_final_save_without_a_completed_acquire_still_persists_its_items() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let item = ChatItem::Message(ChatMessage::text(
        "user",
        Role::User,
        "trailing words",
        std::time::UNIX_EPOCH,
    ));
    let (sink, mut events) = Events::new();
    store
        .clone()
        .dispatch(
            VoiceStorageCommand::Settle {
                speech: 1,
                sequence: 1,
                history: VoiceHistory::new(vec![item.clone()]),
            },
            sink,
        )
        .unwrap();
    assert!(matches!(events.recv().await, Some(VoiceStorageEvent::Settled { result: Ok(_), .. })));
    let thread = fixture.thread().await;
    let rows = thread.partition("desk").history().await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        lk_agents_abi::llm::chat_context::protocol::decode_chat_item(rows[0].payload.clone())
            .unwrap(),
        item
    );
    store.aclose().await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn a_late_save_reopens_ownership_instead_of_acknowledging_an_absent_run() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    store.acquire(1, VoiceHistory::default()).await.unwrap();
    let item = ChatItem::Message(ChatMessage::text(
        "user",
        Role::User,
        "trailing words",
        std::time::UNIX_EPOCH,
    ));
    let pending = store.begin(None, VoiceHistory::new(vec![item.clone()])).await.unwrap();
    // Another speech releases the shared run between the conversation's acquire and save.
    store.finish(Some(1), 2, VoiceHistory::default()).await.unwrap();
    store.finish(None, 3, pending).await.unwrap();

    let thread = fixture.thread().await;
    let rows = thread.partition("desk").history().await.unwrap();
    assert_eq!(rows.len(), 1, "a successful save receipt protects the pending item");
    assert_eq!(
        lk_agents_abi::llm::chat_context::protocol::decode_chat_item(rows[0].payload.clone())
            .unwrap(),
        item
    );
    store.aclose().await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

/// A supplied model history is not evidence that unseen database rows were deleted. The first
/// acquire reads under ownership and reconciles them, including writes before store binding.
#[tokio::test]
async fn a_first_acquire_preserves_rows_absent_from_supplied_history() {
    use crate::thread::run::ItemRow;
    use crate::{CommitOptions, Holder, RunOptions};
    use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    let foreign = ChatItem::Message(ChatMessage::text(
        "foreign",
        Role::User,
        "text message",
        std::time::UNIX_EPOCH,
    ));
    let local = ChatItem::Message(ChatMessage::text(
        "local",
        Role::User,
        "voice message",
        std::time::UNIX_EPOCH,
    ));
    let mut writer = thread
        .partition("desk")
        .open(RunOptions::new("text", ""), &Holder::new("text"))
        .await
        .unwrap();
    writer.history_mut().push(ItemRow {
        id: "foreign".into(),
        message_id: None,
        payload: encode_chat_item(&foreign),
    });
    writer.commit(CommitOptions::default()).await.unwrap();
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            Some(vec![local.clone()]),
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let merged = store.acquire(1, VoiceHistory::new(vec![local.clone()])).await.unwrap();
    assert_eq!(merged.items.iter().map(ChatItem::id).collect::<Vec<_>>(), ["foreign", "local"]);
    // A second speech may start before the first acquire's history reaches the core.
    // Its local snapshot and a progress write still cannot erase that unread foreign row.
    let overlapping = store.acquire(2, VoiceHistory::new(vec![local.clone()])).await.unwrap();
    assert_eq!(
        overlapping.items.iter().map(ChatItem::id).collect::<Vec<_>>(),
        ["foreign", "local"]
    );
    store.checkpoint(2, 1, "desk".into(), VoiceHistory::new(vec![local])).await.unwrap();
    let (sink, mut events) = Events::new();
    store
        .clone()
        .dispatch(VoiceStorageCommand::Settle { speech: 1, sequence: 3, history: merged }, sink)
        .unwrap();
    assert!(matches!(events.recv().await, Some(VoiceStorageEvent::Settled { result: Ok(_), .. })));
    let saved = thread
        .partition("desk")
        .history()
        .await
        .unwrap()
        .into_iter()
        .map(|row| decode_chat_item(row.payload).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(saved.iter().map(ChatItem::id).collect::<Vec<_>>(), ["foreign", "local"]);
    assert_eq!(saved[0], foreign);
    store.aclose().await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

/// A successful settle receipt is a durability barrier even when another speech shares the
/// active database run. A later abort must not lose the first speech's cut prefix or flag.
#[tokio::test]
async fn settling_one_overlapping_speech_persists_its_interrupted_item() {
    use lk_agents_abi::llm::chat_context::protocol::decode_chat_item;
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    store.acquire(1, VoiceHistory::default()).await.unwrap();
    store.acquire(2, VoiceHistory::default()).await.unwrap();
    let mut heard =
        ChatMessage::text("reply", Role::Assistant, "heard prefix", std::time::UNIX_EPOCH);
    heard.interrupted = true;
    let heard = ChatItem::Message(heard);
    let (sink, mut events) = Events::new();
    store
        .clone()
        .dispatch(
            VoiceStorageCommand::Settle {
                speech: 1,
                sequence: 3,
                history: VoiceHistory::new(vec![heard.clone()]),
            },
            sink,
        )
        .unwrap();
    assert!(matches!(
        events.recv().await,
        Some(VoiceStorageEvent::Settled { speech: 1, result: Ok(_), .. })
    ));
    let saved = thread
        .partition("desk")
        .history()
        .await
        .unwrap()
        .into_iter()
        .map(|row| decode_chat_item(row.payload).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(saved, [heard]);
    store.aclose().await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

/// Live recognition is a checkpointed draft, not an accepted message. Losing the activation
/// keeps its latest words, and the next owner recovers them once under the original item ID.
#[tokio::test]
async fn recognition_progress_is_durable_before_a_speech_and_recovered_once() {
    use lk_agents_abi::storage::voice::{VoiceConversationBoundary, VoiceConversationProgress};
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    let input = ChatMessage::text("input", Role::User, "book a table", std::time::UNIX_EPOCH);
    store
        .conversation(VoiceConversationProgress {
            sequence: 1,
            agent: "desk".into(),
            history: VoiceHistory::default(),
            pending_user_turns: vec![input.clone()],
            boundary: VoiceConversationBoundary::Progress,
        })
        .await
        .unwrap();
    assert!(
        thread.partition("desk").history().await.unwrap().is_empty(),
        "a pending input is not accepted history"
    );
    let loaded = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    let state = loaded.session_state().unwrap();
    assert_eq!(
        crate::session_state::pending_voice_inputs(&state).unwrap().as_slice(),
        std::slice::from_ref(&input)
    );
    store
        .activation()
        .stage_session(
            crate::session_state::voice_value(
                "desk".into(),
                vec![],
                store.activation().session_state().as_ref(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        crate::session_state::pending_voice_inputs(&store.activation().session_state().unwrap())
            .unwrap()
            .as_slice(),
        std::slice::from_ref(&input),
        "a host view update did not erase the draft"
    );
    store.aclose().await.unwrap();
    let resumed = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let recovered = resumed.acquire(2, VoiceHistory::default()).await.unwrap();
    assert_eq!(recovered.items, [ChatItem::Message(input)]);
    let (sink, mut events) = Events::new();
    resumed
        .clone()
        .dispatch(VoiceStorageCommand::Settle { speech: 2, sequence: 2, history: recovered }, sink)
        .unwrap();
    assert!(matches!(events.recv().await, Some(VoiceStorageEvent::Settled { result: Ok(_), .. })));
    resumed.aclose().await.unwrap();
    let loaded = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    let state = loaded.session_state().unwrap();
    assert!(crate::session_state::pending_voice_inputs(&state).unwrap().is_empty());
    assert_eq!(thread.partition("desk").history().await.unwrap().len(), 1);
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn restarting_voice_refuses_to_replay_a_tool_with_an_unknown_outcome() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    let mut run = thread
        .partition("desk")
        .open(crate::RunOptions::new("crashed", ""), &crate::Holder::new("old"))
        .await
        .unwrap();
    let call = ChatItem::FunctionCall(lk_agents_abi::FunctionCall {
        id: "tool".into(),
        call_id: "external-effect".into(),
        name: "charge".into(),
        arguments: "{}".into(),
        created_at: std::time::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    });
    run.checkpoint_session(
        crate::session_state::voice_value("desk".into(), vec![call.clone()], None).unwrap(),
        vec![lk_agents_abi::llm::chat_context::protocol::encode_chat_item(&call)],
    )
    .await
    .unwrap();
    run.abandon(crate::RunOutcome::Failed, "worker-lost").await.unwrap();
    let restored = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await;
    assert!(
        matches!(restored, Err(StorageError::Invalid(message)) if message.contains("reconcile"))
    );
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn a_host_flush_preserves_the_confirmed_partial_prefix() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    store.acquire(1, VoiceHistory::default()).await.unwrap();
    let partial = ChatItem::Message(ChatMessage::text(
        "reply",
        Role::Assistant,
        "heard prefix",
        std::time::UNIX_EPOCH,
    ));
    store.checkpoint(1, 1, "desk".into(), VoiceHistory::new(vec![partial])).await.unwrap();
    store
        .session_sink()
        .save_session(store.activation().session_state().unwrap(), vec![])
        .await
        .unwrap();
    store.aclose().await.unwrap();
    let thread = fixture.thread().await;
    let rows = thread.partition("desk").history().await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, "reply");
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn voice_checkpoints_and_terminal_saves_never_regress_out_of_order() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    store.acquire(1, VoiceHistory::default()).await.unwrap();
    let history = |text| {
        VoiceHistory::new(vec![ChatItem::Message(ChatMessage::text(
            "reply",
            Role::Assistant,
            text,
            std::time::UNIX_EPOCH,
        ))])
    };
    store.checkpoint(1, 2, "desk".into(), history("latest")).await.unwrap();
    store.checkpoint(1, 1, "desk".into(), history("stale")).await.unwrap();
    store.finish(Some(1), 1, history("stale terminal snapshot")).await.unwrap();
    store.aclose().await.unwrap();
    let thread = fixture.thread().await;
    let rows = thread.partition("desk").history().await.unwrap();
    let item =
        lk_agents_abi::llm::chat_context::protocol::decode_chat_item(rows[0].payload.clone())
            .unwrap();
    assert_eq!(item.as_message().unwrap().raw_text_content().as_deref(), Some("latest"));
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn observed_voice_lease_loss_refuses_new_work_and_discards_staged_saves() {
    let fixture =
        Backend::Sqlite.fixture(Some((Duration::from_secs(5), Duration::from_millis(10)))).await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    store
        .acquire(
            1,
            VoiceHistory::new(vec![ChatItem::Message(ChatMessage::text(
                "u",
                Role::User,
                "save",
                std::time::UNIX_EPOCH,
            ))]),
        )
        .await
        .unwrap();
    let activation = store.activation();
    let field = |value| {
        vec![ContextField {
            name: "count".into(),
            schema: json!({"type":"integer"}),
            value: json!(value),
        }]
    };
    let loaded = activation.load("root:desk".into(), "Draft".into(), field(0)).await.unwrap();
    activation
        .save("root:desk".into(), "Draft".into(), loaded.revision, field(1), None)
        .await
        .unwrap();
    fixture.orphan_lease("desk", "foreign").await;
    let (sink, mut events) = Events::new();
    std::sync::Arc::clone(&store)
        .dispatch(VoiceStorageCommand::ObserveLoss { speech: 1 }, sink.clone())
        .unwrap();
    let lost = tokio::time::timeout(Duration::from_secs(1), events.recv()).await.unwrap();
    assert!(matches!(lost, Some(VoiceStorageEvent::LeaseLost { speech: 1 })));
    assert!(store.lease_lost());
    assert!(matches!(
        store.acquire(2, VoiceHistory::default()).await,
        Err(StorageError::LeaseLost(_))
    ));
    std::sync::Arc::clone(&store).dispatch(VoiceStorageCommand::Abort { speech: 1 }, sink).unwrap();
    let abandoned = events.recv().await;
    assert!(matches!(abandoned, Some(VoiceStorageEvent::Abandoned { speech: 1, result: Ok(_) })));
    let thread = fixture.thread().await;
    assert!(thread.partition("desk").history().await.unwrap().is_empty());
    let restored = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    assert!(
        restored
            .load("root:desk".into(), "Draft".into(), field(0))
            .await
            .unwrap()
            .fields
            .is_empty()
    );
    store.aclose().await.unwrap();
    restored.close();
    fixture.cleanup(&[&thread]).await;
}

/// A lease the store's own canceled turn left behind (its cleanup lost) is re-entered by the
/// store's next turn at once, not after the lease TTL.
#[tokio::test]
async fn a_turn_reenters_the_lease_a_canceled_turn_left_behind() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    store.acquire(1, VoiceHistory::default()).await.unwrap();
    let (sink, mut events) = Events::new();
    std::sync::Arc::clone(&store).dispatch(VoiceStorageCommand::Abort { speech: 1 }, sink).unwrap();
    assert!(matches!(events.recv().await, Some(VoiceStorageEvent::Abandoned { .. })));
    fixture.orphan_lease("desk", store.holder.as_str()).await;
    let next =
        tokio::time::timeout(Duration::from_secs(2), store.acquire(2, VoiceHistory::default()))
            .await;
    assert!(matches!(next, Ok(Ok(_))), "the next turn waited for its own orphaned lease");
    store.aclose().await.unwrap();
    fixture.cleanup(&[]).await;
}

/// The voice store's events, as the voice executor receives them.
struct Events(tokio::sync::mpsc::UnboundedSender<VoiceStorageEvent>);

impl Events {
    fn new() -> (std::sync::Arc<Self>, tokio::sync::mpsc::UnboundedReceiver<VoiceStorageEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (std::sync::Arc::new(Events(tx)), rx)
    }
}

impl lk_agents_abi::StreamInput<VoiceStorageEvent> for Events {
    fn push(&self, item: VoiceStorageEvent) -> Result<(), lk_agents_abi::errors::ProviderError> {
        let _ = self.0.send(item);
        Ok(())
    }
    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

/// A call's partition metrics are the job's: Cloud receives only measurements keyed to a job.
#[tokio::test]
async fn a_turns_partition_metrics_record_under_the_calls_job() {
    use lk_agents_abi::telemetry::record::{JobKey, Record, RecordSink};
    #[derive(Default)]
    struct Jobs(parking_lot::Mutex<Vec<(&'static str, Option<JobKey>)>>);
    impl RecordSink for Jobs {
        fn record(&self, record: Record) {
            if let Record::Measurement(m) = record {
                self.0.lock().push((m.instrument, m.job));
            }
        }
    }
    let sink = std::sync::Arc::new(Jobs::default());
    let clock = std::sync::Arc::new(lk_agents_abi::VirtualClock::new());
    let tracer = lk_agents_telemetry::Tracer::new(sink.clone(), clock, 7);
    let job = JobKey("AJ_call".into());
    let cx = tracer.root().with_job(Some(job.clone()));
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        cx,
    )
    .await
    .unwrap();
    store.acquire(1, VoiceHistory::default()).await.unwrap();
    let measured = sink.0.lock().clone();
    assert!(measured.contains(&("lk.agents.partition.acquire_wait", Some(job))), "{measured:?}");
}

/// A turn's own edit to a committed row (a question answered since the last commit) survives
/// another writer's commit in between that left the row alone; the other writer's own rows and
/// edits are kept.
#[tokio::test]
async fn a_turns_edit_survives_another_writers_commit() {
    use crate::thread::run::ItemRow;
    use crate::{CommitOptions, Holder, RunOptions};
    use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;
    let said = |id: &str, role, text: &str| {
        ChatItem::Message(ChatMessage::text(id, role, text, std::time::UNIX_EPOCH))
    };
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let asked = vec![said("q", Role::Assistant, "which day?"), said("a", Role::User, "hello")];
    store.acquire(1, VoiceHistory::new(asked.clone())).await.unwrap();
    let (sink, mut events) = Events::new();
    std::sync::Arc::clone(&store)
        .dispatch(
            VoiceStorageCommand::Settle {
                speech: 1,
                sequence: 3,
                history: VoiceHistory::new(asked),
            },
            sink,
        )
        .unwrap();
    let Some(VoiceStorageEvent::Settled { speech: 1, result: Ok(mut local), .. }) =
        events.recv().await
    else {
        panic!("the first turn committed and returned its observed baseline")
    };
    // A text turn appends an exchange and rewrites `a`.
    let thread = fixture.thread().await;
    let mut run = thread
        .partition("desk")
        .open(RunOptions::new("text_1", ""), &Holder::new("text"))
        .await
        .unwrap();
    let rows = [
        said("q", Role::Assistant, "which day?"),
        said("a", Role::User, "hello, edited"),
        said("t", Role::User, "text me"),
    ];
    *run.history_mut() = rows
        .iter()
        .map(|item| ItemRow {
            id: item.id().into(),
            message_id: None,
            payload: encode_chat_item(item),
        })
        .collect();
    run.commit(CommitOptions::default()).await.unwrap();
    local.items = vec![said("q", Role::Assistant, "Monday"), said("a", Role::User, "hello")];
    let merged = store.acquire(2, local).await.unwrap();
    let text = |id: &str| {
        merged
            .items
            .iter()
            .find(|item| item.id() == id)
            .and_then(|item| item.as_message())
            .map(|m| m.content.iter().map(|c| format!("{c:?}")).collect::<String>())
    };
    assert!(text("q").is_some_and(|t| t.contains("Monday")), "{merged:?}");
    assert!(text("a").is_some_and(|t| t.contains("edited")), "{merged:?}");
    assert!(text("t").is_some(), "{merged:?}");
    store.aclose().await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn a_late_snapshot_keeps_its_own_baseline_after_another_turn_reads_a_foreign_edit() {
    use crate::thread::run::ItemRow;
    use crate::{CommitOptions, Holder, RunOptions};
    use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;

    let said = |text| {
        ChatItem::Message(ChatMessage::text("reply", Role::Assistant, text, std::time::UNIX_EPOCH))
    };
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    let first = store.acquire(1, VoiceHistory::new(vec![said("Original.")])).await.unwrap();
    let stale = store.finish(Some(1), 1, first).await.unwrap();

    let thread = fixture.thread().await;
    let mut text = thread
        .partition("desk")
        .open(RunOptions::new("text", ""), &Holder::new("text"))
        .await
        .unwrap();
    *text.history_mut() = vec![ItemRow {
        id: "reply".into(),
        message_id: None,
        payload: encode_chat_item(&said("Edited by text.")),
    }];
    text.commit(CommitOptions::default()).await.unwrap();

    let mut current = store.acquire(2, stale.clone()).await.unwrap();
    let late = store.acquire(3, stale).await.unwrap();
    assert_eq!(
        late.items,
        [said("Edited by text.")],
        "a newer read cannot relabel an old snapshot as a local edit"
    );

    current.items = vec![said("A new voice answer.")];
    let saved = store.checkpoint(2, 2, "desk".into(), current).await.unwrap();
    assert_eq!(
        saved.items,
        [said("A new voice answer.")],
        "an edit made against the current baseline still applies"
    );
    store.aclose().await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn a_newer_snapshot_can_replace_our_own_checkpoint_while_its_receipt_is_in_flight() {
    let item = |text| {
        ChatItem::Message(ChatMessage::text("reply", Role::Assistant, text, std::time::UNIX_EPOCH))
    };
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    let first = store.acquire(1, VoiceHistory::new(vec![item("Hello")])).await.unwrap();
    // Both snapshots were captured before either checkpoint's receipt reached the core.
    let mut final_words = first.clone();
    final_words.items = vec![item("Hello, how can I help?")];
    store.checkpoint(1, 1, "desk".into(), first).await.unwrap();
    let final_saved = store.checkpoint(1, 2, "desk".into(), final_words).await.unwrap();
    assert_eq!(final_saved.items, [item("Hello, how can I help?")]);
    store.aclose().await.unwrap();
    fixture.cleanup(&[]).await;
}

#[tokio::test]
async fn an_in_flight_receipt_does_not_turn_our_previous_write_into_a_foreign_edit() {
    let item = |text| {
        ChatItem::Message(ChatMessage::text("input", Role::User, text, std::time::UNIX_EPOCH))
    };
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = VoiceStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        None,
        Cx::NONE,
    )
    .await
    .unwrap();
    let before = store.acquire(1, VoiceHistory::new(Vec::new())).await.unwrap();
    let mut draft = before.clone();
    draft.items = vec![item("")];
    // Both snapshots are prepared before the first write's receipt reaches the session.
    let mut final_input = before.clone();
    final_input.items = vec![item("Please check my order.")];
    store.finish(Some(1), 1, draft).await.unwrap();
    let saved = store.finish(Some(2), 2, final_input).await.unwrap();
    assert_eq!(saved.items, [item("Please check my order.")]);
    let mut stale = before;
    stale.items = vec![item("")];
    let saved = store.finish(Some(3), 1, stale).await.unwrap();
    assert_eq!(saved.items, [item("Please check my order.")], "older submissions stay stale");
    store.aclose().await.unwrap();
    fixture.cleanup(&[]).await;
}

/// agent-db evicting the database idle refuses the turn's commit in flight unapplied (`retry`)
/// and closes the socket: the commit is sent again on the redialed socket and the turn settles,
/// so the call goes on. Before, the refusal ended the call as a storage failure.
#[tokio::test(start_paused = true)]
async fn a_commit_refused_by_an_idle_eviction_is_sent_again_and_the_call_goes_on() {
    use crate::Storage as _;
    use crate::testing::{Faults, TempDir, Wrapped, WriteFault, tasks};
    use std::sync::Arc;
    let dir = TempDir::new("evicted_commit_");
    let storage = crate::backends::sqlite::SqliteStorage::open(
        &dir.0,
        crate::thread::LeaseTiming::default(),
        tasks(),
    );
    let faults = Arc::<Faults>::default();
    let shared = faults.clone();
    let wrap = move |conn: crate::backends::sqlite::FileConnection| {
        Arc::new(shared.over(Box::new(conn))) as Arc<dyn crate::ThreadDbConnection>
    };
    let storage = Arc::new(Wrapped { storage: Arc::new(storage.await.unwrap()), wrap });
    let thread_id = storage.create_thread(Default::default()).await.unwrap().thread_id;
    let store = Arc::new(
        VoiceStore::bound(storage.clone(), thread_id.clone(), "desk".into(), None, Cx::NONE)
            .await
            .unwrap(),
    );
    let said = vec![ChatItem::Message(ChatMessage::text(
        "hi",
        Role::User,
        "hello",
        std::time::UNIX_EPOCH,
    ))];
    store.acquire(1, VoiceHistory::new(said.clone())).await.unwrap();
    faults.fail_next_commit(WriteFault::EvictIdle);
    let (sink, mut events) = Events::new();
    Arc::clone(&store)
        .dispatch(
            VoiceStorageCommand::Settle {
                speech: 1,
                sequence: 3,
                history: VoiceHistory::new(said),
            },
            sink,
        )
        .unwrap();
    match events.recv().await {
        Some(VoiceStorageEvent::Settled { speech: 1, result, .. }) => {
            assert!(result.is_ok(), "the call ended on a refusal that applied nothing: {result:?}")
        }
        _ => panic!("the turn did not settle"),
    }
    store.aclose().await.unwrap();
    let reopened = VoiceStore::bound(storage, thread_id, "desk".into(), None, Cx::NONE).await;
    let history = reopened.unwrap().initial_history().unwrap_or_default();
    assert_eq!(history.len(), 1, "the turn committed once: {history:?}");
}
