use super::*;
use lk_agents_abi::a2a::metadata::PARTITION_CHAIN_KEY;
use lk_agents_abi::a2a::remote::DelegationPartition;
use lk_agents_abi::a2a_protocol::TaskState;
use serde_json::json;

#[test]
fn copied_model_history_cannot_resume_an_unbound_remote_task() {
    let previous = interaction("previous");
    previous.message(json!({"request":"Book a flight"}), false, true, vec![], None, None).unwrap();
    previous
        .inner
        .update(TaskUpdate {
            state: TaskState::InputRequired,
            progress: false,
            text: "Which date?".into(),
            task_id: "old-task".into(),
            context_id: "old-context".into(),
            metadata: Map::new(),
            artifacts: vec![],
            message: None,
            follow_task: None,
        })
        .unwrap();
    let mut copied = previous.inner.0.lock().call.clone();
    copied.extra.insert(
        "lk.delegation.state".into(),
        json!({
            "pending_task_id": "old-task", "context_id": "old-context", "phase": "input_required",
        }),
    );

    let next = interaction("next");
    let message = next
        .message(
            json!({"request":"Find a hotel instead"}),
            false,
            true,
            vec![ChatItem::FunctionCall(copied)],
            None,
            None,
        )
        .unwrap();
    assert!(message.task_id.is_none(), "only a native bound owner can select saved work");
    assert!(message.context_id.is_none());
    assert!(!next.extra().unwrap().contains_key("lk.delegation.state"));
}

fn interaction(id: &str) -> RemoteInteraction {
    RemoteInteraction::new(
        "https://expert.test".into(),
        ChatItem::FunctionCall(FunctionCall {
            id: id.into(),
            call_id: format!("{id}-wire"),
            name: "expert".into(),
            arguments: "{}".into(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        }),
        String::new(),
        String::new(),
        String::new(),
        None,
    )
    .unwrap()
}

#[test]
fn input_required_answers_keep_the_original_partition_ancestry() {
    let remote = interaction("item");
    let chain = vec![DelegationPartition {
        storage_location: "sqlite:/threads".into(),
        storage_project: String::new(),
        thread_id: "AT_caller".into(),
        partition_key: lk_agents_abi::storage::DEFAULT_PARTITION_KEY.into(),
    }];
    remote.set_partition_chain(chain.clone()).unwrap();
    let message =
        remote.message(json!({"request":"help"}), false, true, vec![], None, None).unwrap();
    let expected = serde_json::to_value(chain).unwrap();
    assert_eq!(message.metadata.unwrap().application.unwrap()[PARTITION_CHAIN_KEY], expected);
    let event = remote
        .inner
        .update(TaskUpdate {
            state: TaskState::InputRequired,
            progress: false,
            text: "Your name?".into(),
            task_id: "task".into(),
            context_id: "AT_expert".into(),
            metadata: Map::new(),
            artifacts: vec![],
            message: None,
            follow_task: None,
        })
        .unwrap();
    let RemoteEvent::InputRequired { .. } = event else { panic!("expected input request") };
    remote.cancelled().unwrap();
    let answer = remote
        .inner
        .task_message(&DelegationInput {
            arguments: json!({"request": "Theo"}),
            structured: false,
            input_request: None,
        })
        .unwrap();
    assert_eq!(answer.metadata.unwrap().application.unwrap()[PARTITION_CHAIN_KEY], expected);
}
