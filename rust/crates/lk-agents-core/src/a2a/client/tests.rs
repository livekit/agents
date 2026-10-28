use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use serde_json::json;

fn card() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "name": "billing", "version": "1", "description": "Billing expert",
        "supportedInterfaces": [{"url": "https://example.test/agent/v1", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": true, "extensions": [{"uri": EXTENSION_URI}]}
    })).unwrap()
}

fn connected() -> Client {
    let mut client =
        Client::new("https://example.test/agent".into(), Some("credential".into())).unwrap();
    client.accept_card(&card()).unwrap();
    client
}

fn message(id: &str) -> Message {
    Message {
        message_id: id.into(),
        context_id: None,
        task_id: None,
        role: a2a::Role::User,
        parts: vec![Part::text("question")],
        metadata: None,
        extensions: None,
        reference_task_ids: None,
    }
}

fn task(id: &str, context: &str, state: &str) -> StreamResponse {
    serde_json::from_value(json!({"task": {"id": id, "contextId": context,
        "status": {"state": state}}}))
    .unwrap()
}

fn event(body: Value) -> StreamResponse {
    serde_json::from_value(body).unwrap()
}

fn status(state: &str, text: Option<&str>) -> StreamResponse {
    let mut s = json!({"state": state});
    if let Some(text) = text {
        s["message"] = serde_json::to_value(message("status-message")).unwrap();
        s["message"]["role"] = json!("ROLE_AGENT");
        s["message"]["parts"] = json!([{"text": text}]);
    }
    event(json!({"statusUpdate": {"taskId": "task", "contextId": "context", "status": s}}))
}

fn chunk(id: &str, parts: Value, append: bool) -> StreamResponse {
    event(json!({"artifactUpdate": {"taskId": "task", "contextId": "context",
        "artifact": {"artifactId": id, "name": "answer", "parts": parts}, "append": append}}))
}

#[test]
fn card_and_every_rpc_include_version_and_cancel_always_addresses_its_context() {
    let mut client = Client::new("https://example.test/agent/".into(), None).unwrap();
    let get = client.card_request().unwrap();
    assert_eq!(get.method, HttpMethod::Get);
    assert_eq!(get.url, "https://example.test/agent/.well-known/agent-card.json");
    assert!(get.headers.contains(&("A2A-Version".into(), "1.0".into())));
    client.accept_card(&card()).unwrap();
    let task = TaskRef { id: "task".into(), context_id: "context".into() };
    let requests = [
        client.send(message("stable-message-id"), None, None, "send".into()).unwrap(),
        client.cancel(&task, Some("caller-canceled"), "cancel".into()).unwrap(),
        client.subscribe(&task, Some("opaque-cursor"), "subscribe".into()).unwrap(),
    ];
    for request in &requests {
        assert!(request.headers.contains(&("A2A-Version".into(), "1.0".into())));
        assert_eq!(request.method, HttpMethod::Post);
        assert_eq!(request.url, "https://example.test/agent/v1");
    }
    let canceled = crate::a2a::rpc::decode(&requests[1].body).unwrap();
    assert!(!requests[0].task_response);
    assert!(requests[1].task_response);
    assert!(!requests[2].task_response);
    assert_eq!(
        canceled.call,
        crate::a2a::rpc::Call::Cancel {
            task: task.clone(),
            reason: Some("caller-canceled".into())
        }
    );
    let subscribed = crate::a2a::rpc::decode(&requests[2].body).unwrap();
    assert_eq!(
        subscribed.call,
        crate::a2a::rpc::Call::Subscribe { task, offset: Some("opaque-cursor".into()) }
    );
    let retry = requests[0].clone();
    assert_eq!(retry.body, requests[0].body);
    let sent = crate::a2a::rpc::decode(&retry.body).unwrap();
    let crate::a2a::rpc::Call::Send { message, .. } = sent.call else { panic!() };
    assert_eq!(message.message_id, "stable-message-id");
    assert_eq!(message.context_id, None);
}

#[test]
fn last_thread_routing_does_not_imply_a_task_and_explicit_continuation_wins() {
    let mut client = connected();
    let mut reply = ReplyState::default();
    let ack = reply
        .accept(task("original-task", "original-context", "TASK_STATE_SUBMITTED"))
        .unwrap()
        .unwrap();
    client.acknowledge(&ack).unwrap();
    let request = client.send(message("followup"), None, None, 2.into()).unwrap();
    let crate::a2a::rpc::Call::Send { message: sent, .. } =
        crate::a2a::rpc::decode(&request.body).unwrap().call
    else {
        panic!()
    };
    assert_eq!(sent.context_id.as_deref(), Some("original-context"));
    assert_eq!(sent.task_id, None);
    let continued = client
        .send(message("answer"), Some("other-context"), Some("pending-task"), 3.into())
        .unwrap();
    let crate::a2a::rpc::Call::Send { message: sent, .. } =
        crate::a2a::rpc::decode(&continued.body).unwrap().call
    else {
        panic!()
    };
    assert_eq!(sent.context_id.as_deref(), Some("other-context"));
    assert_eq!(sent.task_id.as_deref(), Some("pending-task"));
    assert_eq!(
        continued.task,
        Some(TaskRef { id: "pending-task".into(), context_id: "other-context".into() })
    );
    let mut embedded = message("embedded-task");
    embedded.task_id = Some("embedded-pending".into());
    let embedded = client.send(embedded, None, None, 4.into()).unwrap();
    assert_eq!(
        embedded.task,
        Some(TaskRef { id: "embedded-pending".into(), context_id: "original-context".into() })
    );
    let independent = connected().send(message("other-person"), None, None, 1.into()).unwrap();
    let crate::a2a::rpc::Call::Send { message: sent, .. } =
        crate::a2a::rpc::decode(&independent.body).unwrap().call
    else {
        panic!()
    };
    assert_eq!(sent.context_id, None);
}

#[test]
fn final_answer_assembles_text_and_data_artifacts_and_namespaced_metadata() {
    let mut reply = ReplyState::default();
    reply.accept(task("task", "context", "TASK_STATE_SUBMITTED")).unwrap();
    let progress = reply.accept(status("TASK_STATE_WORKING", Some("Checking"))).unwrap().unwrap();
    assert_eq!(progress.text, "Checking");
    reply.accept(chunk("answer", json!([{"text": "Booked "}]), false)).unwrap();
    reply
        .accept(chunk("answer", json!([{"text": "Tuesday"}, {"data": {"confirmation": 42}}]), true))
        .unwrap();
    let done = reply.accept(event(json!({"statusUpdate": {"taskId": "task", "contextId": "context",
        "status": {"state": "TASK_STATE_COMPLETED"}, "metadata": {
        "https://livekit.io/a2a/ext/agents/v1/verbatim": "Booked Tuesday", "livekit/verbatim": "spoofed", "app": 7
    }}}))).unwrap().unwrap();
    assert_eq!(done.state, TaskState::Completed);
    assert_eq!(done.text, "Booked Tuesday{\"confirmation\":42}");
    assert_eq!(done.metadata["livekit/verbatim"], "Booked Tuesday");
    assert_eq!(done.metadata["app"], 7);
    assert!(matches!(done.artifacts[0].parts[2].content, PartContent::Data(_)));
    assert!(reply.finish().is_ok());
}

#[test]
fn progress_and_partial_artifacts_cannot_become_a_success_at_eof_or_remote_failure() {
    for outcome in ["TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"] {
        let mut reply = ReplyState::default();
        reply.accept(task("task", "context", "TASK_STATE_SUBMITTED")).unwrap();
        reply.accept(status("TASK_STATE_WORKING", Some("Working"))).unwrap();
        reply.accept(chunk("partial", json!([{"text": "partial result"}]), false)).unwrap();
        assert_eq!(reply.finish(), Err(ClientError::Incomplete));
        let update = reply.accept(status(outcome, Some("stopped"))).unwrap().unwrap();
        assert_ne!(update.state, TaskState::Completed);
        assert_eq!(update.text, "stopped");
        assert!(update.is_settled());
        assert!(reply.finish().is_ok());
    }
}

#[test]
fn input_required_settles_the_stream_with_the_exact_pending_task() {
    let mut reply =
        ReplyState::new(Some(TaskRef { id: "task".into(), context_id: "context".into() }));
    let question = reply
        .accept(status("TASK_STATE_INPUT_REQUIRED", Some("Which pharmacy?")))
        .unwrap()
        .unwrap();
    assert_eq!(question.state, TaskState::InputRequired);
    assert_eq!(question.text, "Which pharmacy?");
    assert_eq!(reply.task_ref(), Some(TaskRef { id: "task".into(), context_id: "context".into() }));
    assert!(reply.finish().is_ok());
}

#[test]
fn foreign_rpc_and_task_responses_cannot_resolve_the_original_call() {
    let mut reply = ReplyState::default();
    let envelope = a2a::JsonRpcResponse::success(
        "wrong-request".into(),
        serde_json::to_value(task("task", "context", "TASK_STATE_COMPLETED")).unwrap(),
    );
    assert!(reply.decode(&serde_json::to_vec(&envelope).unwrap(), &"expected".into()).is_err());
    assert!(!reply.settled());
    reply.accept(task("task", "context", "TASK_STATE_SUBMITTED")).unwrap();
    for foreign in [
        task("foreign-task", "context", "TASK_STATE_COMPLETED"),
        task("task", "foreign-context", "TASK_STATE_COMPLETED"),
    ] {
        assert!(reply.accept(foreign).is_err());
        assert!(!reply.settled());
    }
    reply.accept(status("TASK_STATE_COMPLETED", Some("answer"))).unwrap();
    assert!(reply.accept(status("TASK_STATE_COMPLETED", Some("duplicate"))).is_err());
}

#[test]
fn a_merged_task_explicitly_names_the_answer_owner_instead_of_returning_partial_output() {
    let mut reply = ReplyState::default();
    reply.accept(task("task", "context", "TASK_STATE_SUBMITTED")).unwrap();
    reply.accept(chunk("partial", json!([{"text": "unaccepted partial answer"}]), false)).unwrap();
    let merged = reply.accept(event(json!({"statusUpdate": {"taskId": "task", "contextId": "context",
        "status": {"state": "TASK_STATE_COMPLETED"}, "metadata": { MERGED_INTO_KEY: "answer-owner" }}}))).unwrap().unwrap();
    assert_eq!(
        merged.follow_task,
        Some(TaskRef { id: "answer-owner".into(), context_id: "context".into() })
    );
    assert!(merged.text.is_empty());
}

#[test]
fn arbitrary_peer_message_answers_preserve_data_and_files_without_inventing_a_task() {
    let mut reply = ReplyState::default();
    let update = reply.accept(event(json!({"message": {"messageId": "reply", "role": "ROLE_AGENT", "parts": [
        {"text": "receipt "}, {"data": {"id": 42}}, {"url": "https://example.test/receipt", "mediaType": "application/pdf"}
    ]}}))).unwrap().unwrap();
    assert_eq!(update.state, TaskState::Completed);
    assert_eq!(update.text, "receipt {\"id\":42}");
    assert!(update.task_id.is_empty());
    assert!(update.context_id.is_empty());
    assert!(matches!(update.artifacts[0].parts[2].content, PartContent::Url(_)));
    assert_eq!(reply.task_ref(), None);
}

#[test]
fn unary_cancellation_decodes_the_observed_task_without_a_stream_wrapper() {
    let mut reply =
        ReplyState::new(Some(TaskRef { id: "task".into(), context_id: "context".into() }));
    let envelope = JsonRpcResponse::success(
        7.into(),
        json!({
            "id": "task", "contextId": "context", "status": {"state": "TASK_STATE_CANCELED"}
        }),
    );
    let update = reply.decode_task(&serde_json::to_vec(&envelope).unwrap(), &7.into()).unwrap();
    assert_eq!(update.state, TaskState::Canceled);
    assert!(reply.finish().is_ok());
}

#[test]
fn malformed_messages_are_refused_before_discovery_with_the_same_routing_as_send() {
    let client = Client::new("https://example.test/agent".into(), None).unwrap();
    client.validate_message(&message("valid"), None, None).unwrap();
    let mut missing_id = message("");
    let mut missing_parts = message("no-parts");
    missing_parts.parts.clear();
    let mut missing_role = message("no-role");
    missing_role.role = a2a::Role::Unspecified;
    let mut missing_context = message("pending-without-context");
    missing_context.task_id = Some("pending".into());
    for message in [&missing_id, &missing_parts, &missing_role, &missing_context] {
        assert!(matches!(
            client.validate_message(message, None, None),
            Err(ClientError::InvalidInput(_))
        ));
    }
    missing_id.message_id = "now-valid".into();
    client.validate_message(&missing_id, Some("context"), Some("pending")).unwrap();
    assert!(client.validate_message(&message("bad-routing"), Some(""), None).is_err());
    assert!(client.validate_message(&message("bad-routing"), None, Some("pending")).is_err());
    client.validate_message(&missing_context, Some("context"), None).unwrap();
    assert!(client.card().is_none());
}

#[test]
fn opening_a_new_task_cannot_change_the_explicitly_requested_thread() {
    let request =
        connected().send(message("new-task"), Some("authorized-context"), None, 1.into()).unwrap();
    assert_eq!(request.task, None);
    assert_eq!(request.context_id.as_deref(), Some("authorized-context"));
    let mut reply = ReplyState::for_request(&request);
    assert!(reply.accept(task("peer-task", "foreign-context", "TASK_STATE_SUBMITTED")).is_err());
    assert_eq!(reply.task_ref(), None);
    assert!(!reply.settled());
    let ack = reply
        .accept(task("peer-task", "authorized-context", "TASK_STATE_SUBMITTED"))
        .unwrap()
        .unwrap();
    assert_eq!(ack.context_id, "authorized-context");
    assert_eq!(ack.task_id, "peer-task");
    let mut message_only = ReplyState::for_request(&request);
    let mut foreign = message("foreign-message");
    foreign.context_id = Some("foreign-context".into());
    assert!(message_only.accept(StreamResponse::Message(foreign)).is_err());
    assert!(!message_only.settled());
}
