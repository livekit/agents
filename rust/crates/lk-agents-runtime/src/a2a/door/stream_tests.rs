use super::*;
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_net::outbox::outbox;

fn task() -> TaskRow {
    TaskRow {
        task_id: "task".into(),
        partition_key: "desk".into(),
        pos: 1,
        name: None,
        message_id: Some("input".into()),
        status: TaskState::Working,
        updated_at_ms: 1000,
        data: Default::default(),
        internal: None,
    }
}

fn exact() -> ChatMessage {
    ChatMessage::text(
        "source-owned-exact",
        Role::Assistant,
        "Reference ABC.",
        std::time::UNIX_EPOCH,
    )
}

#[tokio::test]
async fn exact_handoff_keeps_the_source_identity_and_working_state() {
    let (sender, mut receiver) = outbox(8);
    let publisher = Publisher::new(JsonRpcId::Number(1), "trip".into(), None, Some(sender));
    publisher.accepted(&task()).await;
    receiver.recv().await.unwrap().unwrap();
    publisher.exact("task", &exact()).await.unwrap();
    let frame = receiver.recv().await.unwrap().unwrap();
    let envelope: serde_json::Value = serde_json::from_str(&frame.json).unwrap();
    let a2a::StreamResponse::StatusUpdate(update) =
        serde_json::from_value(envelope["result"].clone()).unwrap()
    else {
        panic!("exact output must remain an SDK Working status update")
    };
    assert_eq!(update.status.state, a2a::TaskState::Working);
    let message = update.status.message.unwrap();
    assert_eq!(message.message_id, exact().id);
    assert_eq!(
        message.metadata.unwrap()[lk_agents_abi::a2a::metadata::VERBATIM_KEY],
        "Reference ABC."
    );
    assert_eq!(message.parts, vec![a2a::Part::text("Reference ABC.")]);
}

#[tokio::test]
async fn a_closed_or_slow_observer_cannot_produce_a_successful_exact_handoff() {
    for slow in [false, true] {
        let (sender, mut receiver) = outbox(2);
        let publisher = Publisher::new(JsonRpcId::Number(1), "trip".into(), None, Some(sender));
        publisher.accepted(&task()).await;
        if !slow {
            receiver.close();
        }
        let error = publisher.exact("task", &exact()).await.unwrap_err();
        assert_ne!(error.kind, lk_agents_abi::errors::HostErrorKind::Cancelled);
        assert!(publisher.state.lock().await.1.is_none(), "the failed observer remained attached");
        drop(publisher);
        receiver.recv().await.unwrap().unwrap();
        if slow {
            let detached = receiver.recv().await.unwrap().unwrap();
            assert!(
                detached.is_error,
                "slow observation ended without its existing protocol error"
            );
        }
        assert!(receiver.recv().await.is_none(), "a refused exact frame was still handed off");
    }
}
