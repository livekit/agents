use super::*;
use serde_json::{Map, json};

fn identity(conversation: &str) -> InteractionId {
    InteractionId {
        remote_id: "remote".into(),
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
        progress: false,
        task_id: task.into(),
        context_id: context.into(),
        metadata: Map::new(),
        artifacts: vec![],
        message: None,
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
            None,
        )
        .unwrap()
}

#[test]
fn an_answer_retains_its_checkpointed_occurrence_until_the_next_peer_observation() {
    use lk_agents_abi::delegation::remote::{InputRequestRef, RemoteTaskRef};
    let mut remote = interaction("input-receipt");
    begin(&mut remote, false);
    remote.update(&update(TaskState::InputRequired, "trip", "booking", "Which day?")).unwrap();
    let reference = InputRequestRef::RemoteTask {
        remote_id: "remote".into(),
        task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
        request_id: "booking/input_3".into(),
    };
    remote.retain_input_request(Some(reference.clone())).unwrap();
    let mut foreign = reference.clone();
    let InputRequestRef::RemoteTask { task, .. } = &mut foreign else { unreachable!() };
    task.task_id = "other-booking".into();
    assert!(remote.retain_input_request(Some(foreign)).is_err());
    remote.message("Friday".into(), "response".into(), None, "", None, false, None, None).unwrap();
    let saved = remote.snapshot();
    assert_eq!(saved.input_request, Some(reference.clone()));
    remote.update(&update(TaskState::Working, "trip", "booking", "Checking.")).unwrap();
    assert_eq!(
        remote.snapshot().input_request,
        None,
        "the new observation cannot reuse an earlier answer selection"
    );
}

#[test]
fn a_task_clarification_does_not_claim_the_waiters_input_occurrence() {
    use lk_agents_abi::delegation::remote::{InputRequestRef, RemoteTaskRef};
    let mut remote = interaction("clarification");
    begin(&mut remote, false);
    remote.update(&update(TaskState::InputRequired, "trip", "booking", "Which day?")).unwrap();
    let reference = InputRequestRef::RemoteTask {
        remote_id: "remote".into(),
        task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
        request_id: "booking/input_3".into(),
    };
    remote.retain_input_request(Some(reference.clone())).unwrap();
    let message = remote
        .message(
            "Why do you need the date?".into(),
            "clarification".into(),
            None,
            "",
            None,
            false,
            None,
            None,
        )
        .unwrap();
    assert_eq!(message.task_id.as_deref(), Some("booking"));
    assert_eq!(message.context_id.as_deref(), Some("trip"));
    assert_eq!(remote.snapshot().input_request, Some(reference));
}

#[test]
fn a_working_task_accepts_a_correction_without_replacing_its_result_owner() {
    let mut remote = interaction("working-correction");
    begin(&mut remote, false);
    remote.update(&update(TaskState::Working, "trip", "booking", "Searching Friday")).unwrap();
    let original = remote.snapshot();
    let message = remote
        .task_message("Saturday, not Friday".into(), "correction".into(), None, false)
        .unwrap();
    assert_eq!(message.task_id.as_deref(), Some("booking"));
    assert_eq!(message.context_id.as_deref(), Some("trip"));
    assert_eq!(message.text(), Some("Saturday, not Friday"));
    assert_eq!(remote.snapshot(), original, "the existing observer still owns the original call");
    remote.update(&update(TaskState::Completed, "trip", "booking", "Saturday options")).unwrap();
    assert!(remote.task_message("again".into(), "late".into(), None, false).is_err());
}

#[test]
fn task_messages_revalidate_captured_occurrences_without_replacing_a_pending_question() {
    use lk_agents_abi::delegation::remote::{InputRequestRef, RemoteTaskRef};
    let mut remote = interaction("captured-correction");
    begin(&mut remote, false);
    remote.update(&update(TaskState::InputRequired, "trip", "booking", "Which day?")).unwrap();
    let reference = InputRequestRef::RemoteTask {
        remote_id: "remote".into(),
        task: RemoteTaskRef { context_id: "trip".into(), task_id: "booking".into() },
        request_id: "booking/input_3".into(),
    };
    remote.retain_input_request(Some(reference.clone())).unwrap();
    let original = remote.snapshot();
    remote
        .task_message("Why do you need the date?".into(), "clarification".into(), None, false)
        .unwrap();
    assert_eq!(remote.snapshot(), original);
    remote.task_message("Friday".into(), "answer".into(), Some(reference.clone()), false).unwrap();
    assert_eq!(remote.snapshot(), original);
    remote.update(&update(TaskState::InputRequired, "trip", "booking", "What budget?")).unwrap();
    assert!(remote.task_message("Friday".into(), "stale".into(), Some(reference), false).is_err());
    assert_eq!(remote.snapshot().id.parent_call_id, original.id.parent_call_id);
}

#[test]
fn new_work_reuses_the_saved_remote_context_without_inheriting_a_task_or_thread_authority() {
    let mut remote = interaction("new-work");
    let message = remote
        .message(
            "Another request".into(),
            "new-message".into(),
            Some("private-caller-thread"),
            "caller",
            None,
            false,
            None,
            Some("saved-remote-context"),
        )
        .unwrap();
    assert_eq!(message.context_id.as_deref(), Some("saved-remote-context"));
    assert_eq!(message.task_id, None, "context reuse does not select an existing task");
    assert_eq!(message.metadata, None, "a saved remote context is not caller sharing authority");
    assert_eq!(remote.snapshot().context_id, message.context_id);
    assert_eq!(begin(&mut interaction("different-conversation"), false).context_id, None);
}

#[test]
fn only_explicitly_flagged_working_updates_request_an_announcement() {
    for state in [TaskState::Unspecified, TaskState::Submitted, TaskState::Working] {
        for flag in [false, true] {
            let mut remote = interaction("progress");
            begin(&mut remote, false);
            let mut status = update(state.clone(), "context", "task", "Checking.");
            status.progress = flag;
            let outcome = remote.update(&status).unwrap();
            if flag && state == TaskState::Working {
                assert_eq!(outcome, DelegationUpdate::Progress("Checking.".into()));
            } else {
                assert_eq!(outcome, DelegationUpdate::Pending);
            }
            assert_eq!(remote.ended(), Err(DelegationError::MissingOutcome));
        }
    }
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
            None,
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
            Some("another-saved-context"),
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
                Some(&history),
                None,
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
            None,
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
            Some("different-default-context"),
        )
        .unwrap();
    assert_eq!(answer.context_id.as_deref(), Some("peer-context"));
    assert_eq!(answer.task_id.as_deref(), Some("peer-task"));
    assert_eq!(answer.metadata, None);
    assert_eq!(answer.parts, vec![Part::text("account 42")]);
}

#[test]
fn live_remote_interactions_keep_their_exact_pending_tasks_isolated() {
    let mut first = interaction("first");
    let mut second = interaction("second");
    assert_eq!(first.snapshot().id.remote_id, second.snapshot().id.remote_id);
    begin(&mut first, true);
    begin(&mut second, true);
    let first_question =
        first.update(&update(TaskState::InputRequired, "context-a", "task-a", "Account?")).unwrap();
    let second_question =
        second.update(&update(TaskState::InputRequired, "context-b", "task-b", "Date?")).unwrap();
    assert!(matches!(first_question, DelegationUpdate::InputRequired { .. }));
    assert!(matches!(second_question, DelegationUpdate::InputRequired { .. }));
    for (original, context, task) in
        [(&mut first, "context-a", "task-a"), (&mut second, "context-b", "task-b")]
    {
        let answer = original
            .message(
                "answer".into(),
                format!("message-{task}"),
                Some("caller-thread"),
                "caller",
                None,
                true,
                None,
                None,
            )
            .unwrap();
        assert_eq!(answer.context_id.as_deref(), Some(context));
        assert_eq!(answer.task_id.as_deref(), Some(task));
        assert_eq!(answer.metadata.as_ref().unwrap()[KIND_KEY], "delegation");
    }
}

#[test]
fn remote_cancellation_and_failure_settle_truthfully_and_verbatim_is_split_once() {
    for state in [TaskState::Canceled, TaskState::Failed, TaskState::Rejected] {
        let mut remote = interaction("failure");
        begin(&mut remote, true);
        let outcome =
            remote.update(&update(state.clone(), "context", "task", "remote explanation")).unwrap();
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
fn an_ended_observation_does_not_cancel_work_or_turn_progress_into_an_answer() {
    let mut remote = interaction("observation-ended");
    begin(&mut remote, true);
    let mut progress = update(TaskState::Working, "context", "task", "Checking");
    progress.progress = true;
    assert_eq!(remote.update(&progress).unwrap(), DelegationUpdate::Progress("Checking".into()));
    assert_eq!(remote.ended(), Err(DelegationError::MissingOutcome));
    assert_eq!(remote.snapshot().phase, InteractionPhase::Running);
    assert!(matches!(
        remote.update(&update(TaskState::Completed, "context", "task", "late answer")).unwrap(),
        DelegationUpdate::Completed(ToolOutcome::Value { text, .. }) if text == "late answer"
    ));
    assert!(remote.ended().is_ok());
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
fn delegation_rejects_an_owned_partition_including_default_keys_and_ancestors() {
    use lk_agents_abi::a2a::metadata::PARTITION_CHAIN_KEY;
    use lk_agents_abi::a2a::remote::DelegationPartition;
    let caller = DelegationPartition {
        storage_location: "https://project.example".into(),
        storage_project: String::new(),
        thread_id: "AT_shared".into(),
        partition_key: lk_agents_abi::storage::DEFAULT_PARTITION_KEY.into(),
    };
    let ancestor = DelegationPartition { partition_key: "front".into(), ..caller.clone() };
    let metadata = Map::from_iter([(PARTITION_CHAIN_KEY.into(), json!([ancestor, caller]))]);
    assert!(validate_partition_delegation(&metadata, &caller).is_err());
    assert!(
        validate_partition_delegation(
            &metadata,
            &DelegationPartition { partition_key: "front".into(), ..caller.clone() }
        )
        .is_err()
    );
    for independent in [
        DelegationPartition { partition_key: "backend".into(), ..caller.clone() },
        DelegationPartition { thread_id: "AT_other".into(), ..caller.clone() },
        DelegationPartition {
            storage_location: "https://another-project.example".into(),
            ..caller.clone()
        },
    ] {
        assert!(validate_partition_delegation(&metadata, &independent).is_ok());
    }
    assert!(
        validate_partition_delegation(&Map::new(), &caller).is_ok(),
        "ordinary independent requests are legal"
    );
}

#[test]
fn exact_working_output_survives_two_native_hops_without_settling_the_business_task() {
    let text = "Your booking reference is ABC-123.";
    let mut exact = text.to_owned();
    for hop in 0..2 {
        let mut remote = interaction(&format!("exact-hop-{hop}"));
        begin(&mut remote, false);
        let wire = crate::a2a::wire::exact_progress(
            "booking",
            "trip",
            exact,
            format!("exact-{hop}"),
            1000,
        );
        let encoded = serde_json::to_vec(&wire).unwrap();
        let decoded = serde_json::from_slice(&encoded).unwrap();
        let mut reader = crate::a2a::client::ReplyState::default();
        let observed = reader.accept(decoded).unwrap().unwrap();
        assert_eq!(observed.state, TaskState::Working);
        assert!(observed.progress);
        assert_eq!(observed.text, text);
        assert_eq!(observed.metadata["livekit/verbatim"], text);
        let DelegationUpdate::Exact(words) = remote.update(&observed).unwrap() else {
            panic!("an exact Working message was rephrased or interpreted as a final result")
        };
        exact = words;
        assert_eq!(remote.snapshot().phase, InteractionPhase::Running);
        assert!(remote.snapshot().input_request.is_none(), "exact progress cannot ask for input");

        let ordinary = crate::a2a::wire::progress(
            "booking",
            "trip",
            "Still checking.".into(),
            format!("progress-{hop}"),
            1001,
        );
        let ordinary = reader.accept(ordinary).unwrap().unwrap();
        assert_eq!(
            remote.update(&ordinary).unwrap(),
            DelegationUpdate::Progress("Still checking.".into())
        );
        let final_update =
            update(TaskState::Completed, "trip", "booking", "Reservation confirmed.");
        assert!(
            matches!(
                remote.update(&final_update).unwrap(),
                DelegationUpdate::Completed(ToolOutcome::Value { text, verbatim: false, .. })
                    if text == "Reservation confirmed."
            ),
            "progress exactness leaked into another event or settled the business task early"
        );
    }
    assert_eq!(exact, text);
}

#[test]
fn matching_task_names_cannot_move_an_input_occurrence_between_remote_ids() {
    let mut remote = interaction("scoped");
    begin(&mut remote, false);
    remote.update(&update(TaskState::InputRequired, "trip", "booking", "Which day?")).unwrap();
    let foreign = lk_agents_abi::session::InputRequestRef::RemoteTask {
        remote_id: "different-configured-id".into(),
        task: lk_agents_abi::delegation::remote::RemoteTaskRef {
            context_id: "trip".into(),
            task_id: "booking".into(),
        },
        request_id: "booking/input_3".into(),
    };
    assert!(remote.retain_input_request(Some(foreign.clone())).is_err());
    let mut replacement = interaction("replacement");
    assert!(replacement.respond_to(foreign).is_err());
}

#[test]
fn a_lost_observation_reads_a_known_task_again_and_never_guesses_without_one() {
    let mut fresh = interaction("lost");
    assert_eq!(fresh.observation_lost(true, 3), ObservationLoss::Unknown, "no task identity");
    let mut running = interaction("lost");
    running.update(&update(TaskState::Working, "trip", "booking", "")).unwrap();
    let task = TaskRef { id: "booking".into(), context_id: "trip".into() };
    assert_eq!(running.observation_lost(true, 3), ObservationLoss::Resubscribe(task.clone()));
    assert_eq!(running.observation_lost(false, 3), ObservationLoss::Snapshot(task.clone()));
    // Only the opening snapshot arrived again: no progress, so the bound keeps counting.
    running.update(&update(TaskState::Working, "trip", "booking", "")).unwrap();
    assert_eq!(running.observation_lost(true, 3), ObservationLoss::Resubscribe(task.clone()));
    assert_eq!(running.observation_lost(true, 3), ObservationLoss::Unknown, "bounded");
    let refused =
        crate::a2a::client::ClientError::Rpc { code: -32004, message: "no".into(), data: None };
    assert_eq!(
        ObservationLoss::resubscribe_failed(task.clone(), &refused),
        ObservationLoss::Snapshot(task.clone())
    );
    let lost = crate::a2a::client::ClientError::incomplete();
    assert_eq!(ObservationLoss::resubscribe_failed(task, &lost), ObservationLoss::Unknown);
}

#[test]
fn progress_after_a_resubscription_resets_the_recovery_bound() {
    let mut running = interaction("progress");
    running.update(&update(TaskState::Working, "trip", "booking", "")).unwrap();
    for _ in 0..5 {
        assert!(matches!(running.observation_lost(true, 1), ObservationLoss::Resubscribe(_)));
        running.update(&update(TaskState::Working, "trip", "booking", "")).unwrap();
        running.update(&update(TaskState::Working, "trip", "booking", "")).unwrap();
    }
    running.update(&update(TaskState::Completed, "trip", "booking", "done")).unwrap();
    assert_eq!(running.observation_lost(true, 1), ObservationLoss::Unknown, "settled");
}

#[test]
fn a_snapshot_of_a_still_running_lost_task_stays_unknown() {
    let mut running = interaction("snapshot");
    running.update(&update(TaskState::Working, "trip", "booking", "")).unwrap();
    let still: a2a::Task = serde_json::from_value(
        json!({"id":"booking","contextId":"trip","status":{"state":"TASK_STATE_WORKING"}}),
    )
    .unwrap();
    assert_eq!(running.reconcile_lost(a2a::StreamResponse::Task(still)).unwrap(), None);
    let done: a2a::Task = serde_json::from_value(
        json!({"id":"booking","contextId":"trip","status":{"state":"TASK_STATE_COMPLETED"},
            "artifacts":[{"artifactId":"r","parts":[{"text":"Booked"}]}]}),
    )
    .unwrap();
    assert!(matches!(
        running.reconcile_lost(a2a::StreamResponse::Task(done)).unwrap(),
        Some(DelegationUpdate::Completed(_))
    ));
}
