use super::*;

#[test]
fn silent_progress_remains_sdk_working_status_without_a_conversational_wakeup() {
    for reply_required in [false, true] {
        let event = progress_with_policy(
            "task",
            "conversation",
            "Looking up rooms".into(),
            "status".into(),
            0,
            reply_required,
        );
        let StreamResponse::StatusUpdate(update) = event else { panic!("a Working status") };
        assert_eq!(update.status.state, TaskState::Working);
        let message = update.status.message.unwrap();
        assert_eq!(message.message_id, "status");
        assert_eq!(message.task_id.as_deref(), Some("task"));
        assert_eq!(message.context_id.as_deref(), Some("conversation"));
        assert_eq!(message.parts, vec![Part::text("Looking up rooms")]);
        let flagged = message
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get(lk_agents_abi::a2a::metadata::PROGRESS_KEY))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        assert_eq!(flagged, reply_required);
    }
}
