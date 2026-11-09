use crate::StorageError;
use crate::context::store::ContextStore;
use crate::testing::Backend;
use crate::voice::VoiceStore;
use lk_agents_abi::storage::context::{ContextActivation as _, ContextField};
use lk_agents_abi::storage::voice::{
    VoiceStorage as _, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as _,
};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use serde_json::json;
use std::time::Duration;

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
            vec![ChatItem::Message(ChatMessage::text(
                "u",
                Role::User,
                "save",
                std::time::UNIX_EPOCH,
            ))],
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
    activation.save("root:desk".into(), "Draft".into(), loaded.revision, field(1)).await.unwrap();
    fixture.orphan_lease("desk", "foreign").await;
    let (sink, mut events) = Events::new();
    std::sync::Arc::clone(&store)
        .dispatch(VoiceStorageCommand::ObserveLoss { speech: 1 }, sink.clone())
        .unwrap();
    let lost = tokio::time::timeout(Duration::from_secs(1), events.recv()).await.unwrap();
    assert!(matches!(lost, Some(VoiceStorageEvent::LeaseLost { speech: 1 })));
    assert!(store.lease_lost());
    assert!(matches!(store.acquire(2, Vec::new()).await, Err(StorageError::LeaseLost(_))));
    std::sync::Arc::clone(&store).dispatch(VoiceStorageCommand::Abort { speech: 1 }, sink).unwrap();
    let abandoned = events.recv().await;
    assert!(matches!(abandoned, Some(VoiceStorageEvent::Abandoned { speech: 1, result: Ok(()) })));
    let thread = fixture.thread().await;
    assert!(thread.lane("desk").history().await.unwrap().is_empty());
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
    store.acquire(1, Vec::new()).await.unwrap();
    let (sink, mut events) = Events::new();
    std::sync::Arc::clone(&store).dispatch(VoiceStorageCommand::Abort { speech: 1 }, sink).unwrap();
    assert!(matches!(events.recv().await, Some(VoiceStorageEvent::Abandoned { .. })));
    fixture.orphan_lease("desk", store.holder.as_str()).await;
    let next = tokio::time::timeout(Duration::from_secs(2), store.acquire(2, Vec::new())).await;
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

/// A call's lane metrics are the job's: Cloud receives only measurements keyed to a job.
#[tokio::test]
async fn a_turns_lane_metrics_record_under_the_calls_job() {
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
    store.acquire(1, Vec::new()).await.unwrap();
    let measured = sink.0.lock().clone();
    assert!(measured.contains(&("lk.agents.lane.acquire_wait", Some(job))), "{measured:?}");
}
