use super::*;
use std::time::UNIX_EPOCH;

#[test]
fn text_to_voice_projection_retains_identity_and_private_remote_state() {
    let first =
        ChatItem::Message(ChatMessage::text("human", Role::User, "Remember Alice.", UNIX_EPOCH));
    let mut initial = initial_voice("desk".into(), vec![first.clone()]);
    let identity = initial.session_id.clone();
    initial.remote_agents.insert(
        "booking".into(),
        lk_agents_abi::delegation::remote::RemoteAgentState::new("https://booking.test".into()),
    );
    let current = snapshot_value(initial).unwrap();
    let second = ChatItem::Message(ChatMessage::text(
        "assistant",
        Role::Assistant,
        "Hello Alice.",
        UNIX_EPOCH,
    ));
    let incoming = voice_value("desk".into(), vec![first, second], Some(&current)).unwrap();
    let restored = SessionState::from_value(incoming).unwrap();
    assert_eq!(restored.session_id().as_deref(), Some(identity.as_str()));
    assert_eq!(restored.chat_history().len(), 2);
    assert_eq!(
        restored._snapshot_data().unwrap().unwrap().remote_agents["booking"].endpoint,
        "https://booking.test"
    );
}

#[test]
fn remote_views_group_observations_and_keep_their_bound_or_detached_owner() {
    use lk_agents_abi::a2a_protocol as a2a;
    use lk_agents_abi::delegation::remote::{RemoteAgentState, RemoteTaskState};
    use lk_agents_abi::session::remote::{RemoteAgentView, RemoteContextView, RemoteTaskView};

    let mut remote = RemoteAgentState::new("https://booking.test".into());
    let message = |id, context: Option<&str>| a2a::Message {
        message_id: id,
        context_id: context.map(str::to_owned),
        role: a2a::Role::Agent,
        parts: Vec::new(),
        task_id: None,
        metadata: None,
        extensions: None,
        reference_task_ids: None,
    };
    remote.messages.insert("direct".into(), message("direct".into(), None));
    remote.messages.insert("receipt".into(), message("receipt".into(), Some("messages")));
    remote.tasks.insert(
        "booking".into(),
        RemoteTaskState {
            task: a2a::Task {
                id: "booking".into(),
                context_id: "trip".into(),
                status: a2a::TaskStatus {
                    state: a2a::TaskState::InputRequired,
                    message: None,
                    timestamp: None,
                },
                history: None,
                artifacts: None,
                metadata: None,
            },
            input_requests: BTreeMap::new(),
            cancel_requested: false,
            next_input: 0,
        },
    );
    let mut initial = initial_voice("desk".into(), Vec::new());
    initial.remote_agents.insert("booking".into(), remote);
    let state = SessionState::from_value(snapshot_value(initial.clone()).unwrap()).unwrap();
    let detached = SessionState::new(Some(state.snapshot())).unwrap();
    let bound = state.remote_agents().remove("booking").unwrap();
    let fixed = detached.remote_agents().remove("booking").unwrap();
    assert_eq!(bound.messages().unwrap().len(), 2);
    let contexts = bound.contexts().unwrap();
    assert_eq!(contexts.keys().map(String::as_str).collect::<Vec<_>>(), ["messages", "trip"]);
    assert_eq!(contexts["messages"].messages().unwrap().len(), 1);
    assert!(contexts["messages"].tasks().unwrap().is_empty());
    assert!(contexts["trip"].messages().unwrap().is_empty());
    let task = contexts["trip"].tasks().unwrap().remove("booking").unwrap();
    assert!(task.task().unwrap().history.is_none(), "inspection never fabricates optional history");

    initial
        .remote_agents
        .get_mut("booking")
        .unwrap()
        .tasks
        .get_mut("booking")
        .unwrap()
        .task
        .status
        .state = a2a::TaskState::Completed;
    state.assign(&SessionState::from_value(snapshot_value(initial).unwrap()).unwrap());
    assert_eq!(task.task().unwrap().status.state, a2a::TaskState::Completed);
    assert_eq!(
        fixed.tasks().unwrap()["booking"].task().unwrap().status.state,
        a2a::TaskState::InputRequired
    );
    let mut value = task.task().unwrap();
    value.id = "changed by reader".into();
    assert_eq!(task.task().unwrap().id, "booking", "observed protocol values are detached copies");
}
