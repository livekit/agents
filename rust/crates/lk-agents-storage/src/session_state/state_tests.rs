use super::*;
use std::time::UNIX_EPOCH;

#[test]
fn dropping_an_unclosed_binding_releases_its_native_state_and_control_store() {
    use lk_agents_abi::storage::context::ContextActivation as _;
    use lk_agents_abi::storage::session::SessionBinding as _;

    let view = SessionState::new(None).unwrap();
    let weak = Arc::downgrade(&view.0);
    let store = crate::context::store::ContextStore::new(Vec::new(), None, false).unwrap();
    let binding = crate::session_binding::SessionBinding::new(store, view).unwrap();
    assert!(weak.upgrade().is_some());
    drop(binding);
    assert!(weak.upgrade().is_none(), "abandoned startup must not retain the view/store cycle");
}

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
            terminal_event: None,
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

#[test]
fn joined_control_readback_never_replaces_newer_local_progress_or_a_rebound_view() {
    use lk_agents_abi::storage::context::ContextActivation as _;
    let original = snapshot_value(initial_voice("desk".into(), vec![])).unwrap();
    let store =
        crate::context::store::ContextStore::new(vec![], Some(original.clone()), false).unwrap();
    let view = SessionState::from_value(original.clone()).unwrap();
    view.bind_control(store.clone()).unwrap();
    let expected = view.0.lock().clone();
    let local = SessionState::from_value(
        voice_value(
            "desk".into(),
            vec![
                ChatMessage::text("local-edit", Role::User, "Keep this newer draft", UNIX_EPOCH)
                    .into(),
            ],
            Some(&original),
        )
        .unwrap(),
    )
    .unwrap();
    view.assign(&local);
    let mut saved = initial_voice("desk".into(), vec![]);
    saved.remote_agents.insert(
        "remote".into(),
        lk_agents_abi::delegation::remote::RemoteAgentState::new("https://remote.test".into()),
    );
    let saved = SessionState::from_value(snapshot_value(saved).unwrap()).unwrap();
    view.publish_control_if(&store, &expected, &saved);
    assert_eq!(view._to_json(), local._to_json(), "a newer explicit edit wins over readback");
    let unchanged = SessionState::from_value(original).unwrap();
    unchanged.bind_control(store.clone()).unwrap();
    let expected = unchanged.0.lock().clone();
    unchanged.publish_control_if(&store, &expected, &saved);
    assert_eq!(unchanged._to_json(), saved._to_json(), "an unchanged joined view may advance");
    let expected = unchanged.0.lock().clone();
    store.close();
    let next = crate::context::store::ContextStore::new(vec![], None, false).unwrap();
    unchanged.bind_control(next).unwrap();
    unchanged.publish_control_if(&store, &expected, &local);
    assert_eq!(
        unchanged._to_json(),
        saved._to_json(),
        "old-store readback cannot publish after rebind"
    );
}
