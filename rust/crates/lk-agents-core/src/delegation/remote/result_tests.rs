use super::*;
use crate::a2a::client::ReplyState;
use crate::delegation::remote::{DelegationUpdate, InteractionId, RemoteInteraction};
use lk_agents_abi::a2a_protocol::{Role, StreamResponse, Task, TaskState, TaskStatus};
use serde_json::json;

fn result(response: StreamResponse) -> ToolOutcome {
    let update = ReplyState::default().accept(response).unwrap().unwrap();
    let mut interaction = RemoteInteraction::new(InteractionId {
        remote_id: "travel".into(),
        parent_call_id: "original-call".into(),
        endpoint: "https://example.test/travel".into(),
    })
    .unwrap();
    let DelegationUpdate::Completed(result) = interaction.update(&update).unwrap() else {
        panic!("an accepted completed reply resolves its originating tool")
    };
    result
}

fn message(parts: Vec<Part>) -> Message {
    Message {
        message_id: "direct-reply".into(),
        context_id: None,
        task_id: None,
        role: Role::Agent,
        parts,
        metadata: None,
        extensions: None,
        reference_task_ids: None,
    }
}

fn value(outcome: ToolOutcome) -> String {
    let ToolOutcome::Value { text, verbatim: false, .. } = outcome else {
        panic!("ordinary peer content remains model-facing tool output")
    };
    text
}

#[test]
fn a_direct_json_value_is_not_joined_with_prose_or_confused_with_no_result() {
    for data in [json!({"receipt":42}), json!([1, 2]), json!(null), json!(false)] {
        let observed = result(StreamResponse::Message(message(vec![Part::data(data.clone())])));
        assert_eq!(serde_json::from_str::<Value>(&value(observed)).unwrap(), data);
    }
    let text = r#"{"receipt":42}"#;
    assert_eq!(
        value(result(StreamResponse::Message(message(vec![Part::text(text)])))),
        text,
        "JSON-looking text stays text"
    );
}

#[test]
fn direct_message_files_keep_their_identity_and_do_not_invent_a_task() {
    let mut reply = message(vec![
        Part::text("Receipt attached."),
        Part::url("https://files.example.test/receipt.pdf")
            .with_filename("receipt.pdf")
            .with_media_type("application/pdf"),
    ]);
    reply.extensions = Some(vec!["https://vendor.example.test/receipt".into()]);
    let model: Value =
        serde_json::from_str(&value(result(StreamResponse::Message(reply.clone())))).unwrap();
    assert_eq!(model, serde_json::to_value(reply).unwrap());
    assert!(model.get("taskId").is_none());
    assert_eq!(model["messageId"], "direct-reply");
}

#[test]
fn a_named_answer_does_not_discard_other_result_artifacts_or_read_inline_files() {
    let artifact = |id: &str, name: &str, parts: Vec<Part>| Artifact {
        artifact_id: id.into(),
        name: Some(name.into()),
        description: None,
        parts,
        metadata: None,
        extensions: Some(vec!["https://vendor.example.test/result".into()]),
    };
    let task = Task {
        id: "booking".into(),
        context_id: "trip".into(),
        status: TaskStatus { state: TaskState::Completed, message: None, timestamp: None },
        artifacts: Some(vec![
            artifact("answer", "answer", vec![Part::text("Booked.")]),
            artifact("data", "receipt", vec![Part::data(json!({"receipt":42}))]),
            artifact(
                "pdf",
                "attachment",
                vec![
                    Part::raw(vec![0x7f; MODEL_RESULT_BYTES * 2])
                        .with_filename("receipt.pdf")
                        .with_media_type("application/pdf"),
                ],
            ),
        ]),
        history: None,
        metadata: None,
    };
    let before = task.clone();
    let model: Value =
        serde_json::from_str(&value(result(StreamResponse::Task(task.clone())))).unwrap();
    assert_eq!(model[0]["parts"][0]["text"], "Booked.");
    assert_eq!(model[1]["parts"][0]["data"], json!({"receipt":42}));
    assert_eq!(model[2]["parts"][0]["filename"], "receipt.pdf");
    assert_eq!(model[2]["parts"][0]["mediaType"], "application/pdf");
    assert!(model[2]["parts"][0].get("raw").is_none());
    let description = model[2]["parts"][0]["text"].as_str().unwrap();
    assert!(description.contains("65536 bytes"));
    assert!(description.contains("artifact \"pdf\", part 0"));
    assert!(description.contains("have not been read"));
    assert_eq!(task, before, "model rendering never edits the retained SDK payload");
}

#[test]
fn exact_final_words_and_structured_material_have_separate_delivery_contracts() {
    let mut reply = message(vec![Part::text("Booked."), Part::data(json!({"receipt":42}))]);
    reply.metadata =
        Some([(lk_agents_abi::a2a::metadata::VERBATIM_KEY.into(), json!("Booked."))].into());
    let ToolOutcome::Value { text, verbatim: true, reply_required: true, more: Some(more) } =
        result(StreamResponse::Message(reply))
    else {
        panic!("exact words are delivered once while data remains available to the frontend")
    };
    assert_eq!(text, "Booked.");
    let material: Value = serde_json::from_str(&more).unwrap();
    assert_eq!(material["parts"][1]["data"], json!({"receipt":42}));
}

#[test]
fn oversized_completed_content_is_retained_without_becoming_retryable_business_work() {
    let reply = message(vec![Part::data(json!({"large":"x".repeat(MODEL_RESULT_BYTES)}))]);
    assert_eq!(value(result(StreamResponse::Message(reply))), RETAINED_RESULT);
    assert_eq!(value(result(StreamResponse::Message(message(Vec::new())))), NO_RESULT);
}
