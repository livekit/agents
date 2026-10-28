use super::*;
use serde_json::{Map, json};

fn identity(conversation: &str) -> InteractionId {
    InteractionId {
        delegation_id: format!("delegation-{conversation}"),
        tool_id: "reusable-expert".into(),
        parent_call_id: format!("call-{conversation}"),
        endpoint: "https://example.test/expert".into(),
    }
}

fn interaction(conversation: &str) -> RemoteInteraction {
    RemoteInteraction::new(identity(conversation)).unwrap()
}

fn update(state: TaskState, context: &str, task: &str, text: &str) -> TaskUpdate {
    TaskUpdate {
        state,
        text: text.into(),
        task_id: task.into(),
        context_id: context.into(),
        metadata: Map::new(),
        artifacts: vec![],
        follow_task: None,
    }
}

fn begin(interaction: &mut RemoteInteraction, extension: bool) -> Message {
    interaction
        .message(
            "help".into(),
            "initial-message".into(),
            Some("caller-thread"),
            "caller",
            None,
            extension,
            None,
        )
        .unwrap()
}

fn history() -> CallerHistory {
    CallerHistory::decode(json!({"items": [{"type": "message", "id": "public-message",
        "created_at": 1.0, "role": "user", "content": ["explicitly supplied context"]}]}))
    .unwrap()
}

#[test]
fn delegation_authority_matrix_never_treats_a_capability_as_thread_permission() {
    let history = history();
    let mut capable = interaction("capability-only");
    let outgoing = capable
        .message(
            "help".into(),
            "m1".into(),
            Some("caller-thread"),
            "caller",
            None,
            true,
            Some(&history),
        )
        .unwrap();
    assert_eq!(outgoing.context_id, None);
    assert_eq!(outgoing.parts.len(), 2);
    assert_eq!(outgoing.metadata.as_ref().unwrap()[KIND_KEY], "delegation");
    assert_eq!(CallerHistory::from_message(&outgoing).unwrap(), Some(history.clone()));

    let proof = VerifiedAuthority::from_local_verifier(
        "https://example.test/expert".into(),
        "caller-thread".into(),
        "caller".into(),
    )
    .unwrap();
    let mut trusted = interaction("trusted");
    let outgoing = trusted
        .message(
            "help".into(),
            "m2".into(),
            Some("caller-thread"),
            "caller",
            Some(&proof),
            true,
            Some(&history),
        )
        .unwrap();
    assert_eq!(outgoing.context_id.as_deref(), Some("caller-thread"));
    assert_eq!(outgoing.parts.len(), 1);
    assert_eq!(CallerHistory::from_message(&outgoing).unwrap(), None);

    for (endpoint, thread, caller) in [
        ("https://foreign.test/expert", "caller-thread", "caller"),
        ("https://example.test/expert", "different-thread", "caller"),
        ("https://example.test/expert", "caller-thread", "different-caller"),
    ] {
        let proof =
            VerifiedAuthority::from_local_verifier(endpoint.into(), thread.into(), caller.into())
                .unwrap();
        let mut refused = interaction("mismatched-proof");
        assert_eq!(
            refused.message(
                "help".into(),
                "m3".into(),
                Some("caller-thread"),
                "caller",
                Some(&proof),
                true,
                Some(&history)
            ),
            Err(DelegationError::InvalidAuthority)
        );
        assert_eq!(refused.snapshot().phase, InteractionPhase::Ready);
    }
}

#[test]
fn arbitrary_peer_receives_text_only_and_keeps_its_own_remote_task_context() {
    let mut remote = interaction("arbitrary");
    let outgoing = remote
        .message(
            "help".into(),
            "m1".into(),
            Some("private-thread"),
            "caller",
            None,
            false,
            Some(&history()),
        )
        .unwrap();
    assert_eq!(outgoing.parts, vec![Part::text("help")]);
    assert_eq!(outgoing.context_id, None);
    assert_eq!(outgoing.task_id, None);
    assert_eq!(outgoing.metadata, None);
    remote
        .update(&update(TaskState::InputRequired, "peer-context", "peer-task", "Which account?"))
        .unwrap();
    let answer = remote
        .message(
            "account 42".into(),
            "m2".into(),
            Some("private-thread"),
            "caller",
            None,
            false,
            Some(&history()),
        )
        .unwrap();
    assert_eq!(answer.context_id.as_deref(), Some("peer-context"));
    assert_eq!(answer.task_id.as_deref(), Some("peer-task"));
    assert_eq!(answer.metadata, None);
    assert_eq!(answer.parts, vec![Part::text("account 42")]);
}

#[test]
fn remote_interaction_continuation_is_isolated_and_restores_the_exact_pending_task() {
    let mut first = interaction("first");
    let mut second = interaction("second");
    assert_eq!(first.snapshot().id.tool_id, second.snapshot().id.tool_id);
    begin(&mut first, true);
    begin(&mut second, true);
    let first_question =
        first.update(&update(TaskState::InputRequired, "context-a", "task-a", "Account?")).unwrap();
    let second_question =
        second.update(&update(TaskState::InputRequired, "context-b", "task-b", "Date?")).unwrap();
    assert!(matches!(first_question, DelegationUpdate::InputRequired { .. }));
    assert!(matches!(second_question, DelegationUpdate::InputRequired { .. }));
    for (original, context, task) in
        [(&first, "context-a", "task-a"), (&second, "context-b", "task-b")]
    {
        let state =
            serde_json::from_value(serde_json::to_value(original.snapshot()).unwrap()).unwrap();
        let mut restored = RemoteInteraction::restore(state, &original.snapshot().id).unwrap();
        let request = restored.snapshot().request_id.unwrap();
        restored.accepts_reply(&request).unwrap();
        let answer = restored
            .message(
                "answer".into(),
                format!("message-{task}"),
                Some("caller-thread"),
                "caller",
                None,
                true,
                None,
            )
            .unwrap();
        assert_eq!(answer.context_id.as_deref(), Some(context));
        assert_eq!(answer.task_id.as_deref(), Some(task));
        assert_eq!(answer.metadata.as_ref().unwrap()[KIND_KEY], "delegation");
    }
    assert!(RemoteInteraction::restore(first.snapshot(), &second.snapshot().id).is_err());
}

#[test]
fn a_new_question_on_the_same_task_rejects_a_late_reply_to_the_previous_question() {
    let mut remote = interaction("questions");
    begin(&mut remote, true);
    let DelegationUpdate::InputRequired { request_id: first, .. } =
        remote.update(&update(TaskState::InputRequired, "context", "task", "Account?")).unwrap()
    else {
        panic!()
    };
    remote.accepts_reply(&first).unwrap();
    remote.message("42".into(), "answer-1".into(), None, "caller", None, true, None).unwrap();
    let DelegationUpdate::InputRequired { request_id: second, .. } =
        remote.update(&update(TaskState::InputRequired, "context", "task", "Date?")).unwrap()
    else {
        panic!()
    };
    assert_ne!(first, second);
    assert_eq!(remote.accepts_reply(&first), Err(DelegationError::NotWaiting));
    remote.accepts_reply(&second).unwrap();
}

#[test]
fn remote_cancellation_and_failure_settle_truthfully_and_verbatim_is_split_once() {
    for state in [TaskState::Canceled, TaskState::Failed, TaskState::Rejected] {
        let mut remote = interaction("failure");
        begin(&mut remote, true);
        let outcome =
            remote.update(&update(state, "context", "task", "remote explanation")).unwrap();
        match state {
            TaskState::Canceled => assert_eq!(outcome, DelegationUpdate::Cancelled),
            _ => assert_eq!(
                outcome,
                DelegationUpdate::Failed(ToolOutcome::ToolError("remote explanation".into()))
            ),
        }
        assert!(remote.ended().is_ok());
        assert_eq!(
            remote.update(&update(TaskState::Completed, "context", "task", "late success")),
            Err(DelegationError::Settled)
        );
    }
    let mut remote = interaction("complete");
    begin(&mut remote, true);
    let mut done = update(
        TaskState::Completed,
        "context",
        "task",
        "Your refund is confirmed. It arrives tomorrow.",
    );
    done.metadata.insert("livekit/verbatim".into(), "Your refund is confirmed.".into());
    let result = remote.update(&done).unwrap();
    assert_eq!(
        result,
        DelegationUpdate::Completed(ToolOutcome::Value {
            text: "Your refund is confirmed.".into(),
            verbatim: true,
            reply_required: true,
            more: Some("It arrives tomorrow.".into()),
        })
    );
    assert_eq!(remote.update(&done), Err(DelegationError::Settled));
}

#[test]
fn progress_eof_never_becomes_an_answer_and_local_cancel_keeps_uncertain_effects() {
    let mut remote = interaction("cancel");
    begin(&mut remote, true);
    assert_eq!(
        remote.update(&update(TaskState::Working, "context", "task", "Checking")).unwrap(),
        DelegationUpdate::Progress("Checking".into())
    );
    assert_eq!(remote.ended(), Err(DelegationError::MissingOutcome));
    let effect = remote.cancel().unwrap();
    assert_eq!(effect.task, TaskRef { id: "task".into(), context_id: "context".into() });
    assert_eq!(effect.reason, "caller-canceled");
    assert_eq!(remote.cancel(), None);
    let late =
        remote.update(&update(TaskState::Completed, "context", "task", "late answer")).unwrap();
    assert_eq!(late, DelegationUpdate::Cancelled);
    remote.cancellation_settled();
    assert_eq!(remote.snapshot().phase, InteractionPhase::Uncertain);
    assert_eq!(remote.ended(), Err(DelegationError::MissingOutcome));
}

#[test]
fn cancel_before_ack_still_emits_cancellation_once_the_remote_task_is_known() {
    let mut remote = interaction("cancel-before-ack");
    begin(&mut remote, true);
    assert_eq!(remote.cancel(), None);
    assert_eq!(
        remote.update(&update(TaskState::Submitted, "late-context", "late-task", "")).unwrap(),
        DelegationUpdate::Cancelled
    );
    let effect =
        remote.cancel().expect("the accepted remote task still owes cancellation delivery");
    assert_eq!(effect.task, TaskRef { id: "late-task".into(), context_id: "late-context".into() });
    assert_eq!(effect.reason, "caller-canceled");
    assert_eq!(remote.cancel(), None);
}

#[test]
fn a_foreign_update_cannot_resolve_or_retarget_a_pending_remote_call() {
    let mut remote = interaction("foreign-task");
    begin(&mut remote, true);
    remote.update(&update(TaskState::InputRequired, "context", "task", "Account?")).unwrap();
    let original = remote.snapshot();
    for (context, task) in [("context", "foreign-task"), ("foreign-context", "task")] {
        assert_eq!(
            remote.update(&update(TaskState::Completed, context, task, "foreign answer")),
            Err(DelegationError::WrongTask)
        );
        assert_eq!(remote.snapshot(), original);
    }
}

#[test]
fn input_required_retains_an_exact_prefix_while_the_rest_can_be_rephrased() {
    let mut remote = interaction("verbatim-question");
    begin(&mut remote, true);
    let mut question = update(
        TaskState::InputRequired,
        "context",
        "task",
        "Read this exactly. Which account should I use?",
    );
    question.metadata.insert("livekit/verbatim".into(), "Read this exactly.".into());
    let DelegationUpdate::InputRequired { question: text, verbatim, .. } =
        remote.update(&question).unwrap()
    else {
        panic!()
    };
    assert_eq!(text, "Read this exactly. Which account should I use?");
    assert_eq!(verbatim.as_deref(), Some("Read this exactly."));
}

#[test]
fn restoring_a_pending_question_preserves_the_sequence_and_rejects_previous_round_replies() {
    let mut remote = interaction("restored-question-sequence");
    begin(&mut remote, true);
    let DelegationUpdate::InputRequired { request_id: first, .. } =
        remote.update(&update(TaskState::InputRequired, "context", "task", "Account?")).unwrap()
    else {
        panic!()
    };
    let saved = serde_json::to_value(remote.snapshot()).unwrap();
    let saved = serde_json::from_value(saved).unwrap();
    let mut restored = RemoteInteraction::restore(saved, &remote.snapshot().id).unwrap();
    restored.accepts_reply(&first).unwrap();
    restored.message("42".into(), "answer-first".into(), None, "caller", None, true, None).unwrap();
    let DelegationUpdate::InputRequired { request_id: second, .. } =
        restored.update(&update(TaskState::InputRequired, "context", "task", "Date?")).unwrap()
    else {
        panic!()
    };
    assert_ne!(first, second);
    assert_eq!(restored.accepts_reply(&first), Err(DelegationError::NotWaiting));
    restored.accepts_reply(&second).unwrap();
    assert_eq!(restored.snapshot().question_sequence, remote.snapshot().question_sequence + 1);
}
