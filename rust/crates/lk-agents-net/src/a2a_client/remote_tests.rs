use super::*;
use lk_agents_abi::a2a::metadata::PARTITION_CHAIN_KEY;
use lk_agents_abi::a2a::remote::DelegationPartition;
use lk_agents_abi::storage::TaskState;
use serde_json::json;

#[test]
fn input_required_answers_keep_the_original_partition_ancestry() {
    let remote = RemoteInteraction::new(
        "https://expert.test".into(),
        "expert".into(),
        ChatItem::FunctionCall(FunctionCall {
            id: "item".into(),
            call_id: "call".into(),
            name: "expert".into(),
            arguments: "{}".into(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        }),
        vec![],
        String::new(),
        String::new(),
        String::new(),
    )
    .unwrap();
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
            text: "Your name?".into(),
            task_id: "task".into(),
            context_id: "AT_expert".into(),
            metadata: Map::new(),
            artifacts: vec![],
        })
        .unwrap();
    let RemoteEvent::InputRequired { request_id, .. } = event else {
        panic!("expected input request")
    };
    let answer = remote.inner.answer(&request_id, "Theo".into()).unwrap();
    assert_eq!(answer.metadata.unwrap().application.unwrap()[PARTITION_CHAIN_KEY], expected);
}
