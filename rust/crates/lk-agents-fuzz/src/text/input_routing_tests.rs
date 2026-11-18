//! A question, a side question and a response across fresh owners and real SQLite checkpoints.

use super::*;
use lk_agents_abi::ChoiceDelta;
use lk_agents_abi::agent::run::{RunOutcome, StreamEvent};
use lk_agents_abi::delegation::DelegationOptions;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::session::InputRequestState;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::session::{PersistedSessionState as _, SessionBinding as _};
use lk_agents_abi::text_session::SessionSnapshot;
use lk_agents_runtime::storage::context::store::ContextStore;
use lk_agents_runtime::storage::session_binding::SessionBinding;
use lk_agents_runtime::storage::session_state::SessionState;
use serde_json::Value;

fn text(value: &str) -> ChoiceDelta {
    ChoiceDelta { role: Some(Role::Assistant), content: Some(value.into()), ..Default::default() }
}

fn call(name: &str, arguments: Value) -> ChoiceDelta {
    ChoiceDelta {
        role: Some(Role::Assistant),
        tool_calls: vec![lk_agents_abi::FunctionToolCall {
            name: name.into(),
            arguments: arguments.to_string(),
            call_id: format!("call_{name}"),
            extra: None,
        }],
        ..Default::default()
    }
}

/// An ordinary request: restore, run the production executor, consume text, then end its lease.
/// Only model/tool boundaries are scripted. No routing or storage decisions are faked.
async fn turn(
    env: &Arc<Env>,
    database: &Store,
    human: &str,
    replies: Vec<ChoiceDelta>,
    tools: Vec<ToolOutcome>,
    verdict: Option<ChoiceDelta>,
    lose_checkpoint_ack: bool,
) -> (RunOutcome, SessionSnapshot) {
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    store.begin_session().await.unwrap();
    let view = SessionState::new(Some(store.load_session_snapshot().await.unwrap())).unwrap();
    let binding = Arc::new(SessionBinding::new(store.clone(), view.clone()).unwrap());
    let id = env.add_run(RunKind::Session);
    let frontend = RunFakes::new(env.clone(), id, None);
    frontend.script(replies, tools);
    let delegation = verdict.map(|verdict| {
        let router = RunFakes::new(env.clone(), env.add_run(RunKind::Direct), None);
        router.script(vec![verdict], vec![]);
        DelegationOptions { routing_model: Some(Arc::new(FakeLlm(router))) }
    });
    let agent = TextSessionAgent {
        definition: Env::agent(),
        llm: Arc::new(FakeLlm(frontend.clone())),
        host: Arc::new(FakeHost(frontend)),
        output_type_explicit: false,
        agent_tools: BTreeMap::new(),
        host_context: HostRef::new(0, &env.refs),
        tools_host: None,
    };
    let runner = lk_agents_runtime::text_session::Runner::new(
        &env.process,
        "desk".into(),
        vec![agent],
        4,
        delegation,
    )
    .unwrap();
    if lose_checkpoint_ack {
        // The activation already owns its lease. This fault hits a conversation checkpoint.
        database.lose_next_write_acknowledgement();
    }
    let mut updates = runner
        .run(TextSessionRequest {
            session_sink: Some(binding.clone()),
            session_id: "conversation".into(),
            id_seed: Some(23),
            execution_id: format!("exec_{id}"),
            task_id: None,
            message: ChatMessage::text(format!("message_{id}"), Role::User, human, env.wall())
                .into(),
            delegated: false,
            conn_options: Default::default(),
        })
        .unwrap();
    let mut delivered = BTreeMap::<String, String>::new();
    let mut outcome = None;
    while let Some(update) = updates.next().await.unwrap() {
        match update {
            TextSessionUpdate::Stream {
                event: StreamEvent::TextDelta { message_id, delta },
                ..
            } => {
                delivered.entry(message_id).or_default().push_str(&delta);
            }
            TextSessionUpdate::InvocationSettled(invocation_id)
            | TextSessionUpdate::Parked(invocation_id) => {
                runner
                    .release(TextSessionDelivery {
                        invocation_id,
                        messages: std::mem::take(&mut delivered),
                    })
                    .unwrap();
            }
            TextSessionUpdate::Finished(finished) => outcome = Some(finished.turn.result.outcome),
            TextSessionUpdate::Failed { error, .. } => panic!("the story's turn failed: {error}"),
            _ => {}
        }
    }
    let saved = view._snapshot_data().unwrap().unwrap();
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await, "every reconstructed turn releases its lease");
    (outcome.expect("the executor settles every turn"), saved)
}

#[tokio::test(start_paused = true)]
async fn a_presented_question_survives_a_side_question_restart_and_lost_checkpoint_ack() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone(), vec![]);
    let database = Store::open(23, false, clock).await;

    let (outcome, asked) = turn(
        &env,
        &database,
        "Book a room",
        vec![call("lookup", json!({"q":"room"}))],
        vec![ToolOutcome::InputRequired {
            question: "Which day?".into(),
            verbatim: true,
            tool: None,
        }],
        None,
        false,
    )
    .await;
    assert!(matches!(outcome, RunOutcome::InputRequired(_)));
    let (request_id, request) =
        asked.frames[&asked.root_frame_id].input_requests.first_key_value().unwrap();
    assert_eq!(request.state, InputRequestState::Open);
    assert!(request.presented_message_id.is_some(), "the person consumed the question's text");

    let (_, explained) = turn(
        &env,
        &database,
        "Why do you need the date?",
        vec![text("To check availability.")],
        vec![],
        Some(call("route_input_response", json!({"candidates":[],"unclear":false}))),
        false,
    )
    .await;
    assert_eq!(
        explained.frames[&explained.root_frame_id].input_requests[request_id].state,
        InputRequestState::Open
    );

    let (_, answered) = turn(
        &env,
        &database,
        "Friday",
        vec![text("Friday is selected.")],
        vec![],
        Some(call("route_input_response", json!({"candidates":[0],"unclear":false}))),
        true,
    )
    .await;
    let response = &answered.frames[&answered.root_frame_id].input_requests[request_id];
    assert_eq!(response.state, InputRequestState::Responded);
    assert_eq!(
        response.response_message.as_ref().unwrap().raw_text_content().as_deref(),
        Some("Friday")
    );
    assert!(database.faulted(), "SQLite committed a checkpoint before losing its acknowledgement");
    let dialogue: Vec<_> =
        answered.chat_history.iter().map(|message| message.raw_text_content().unwrap()).collect();
    assert_eq!(
        dialogue,
        [
            "Book a room",
            "Which day?",
            "Why do you need the date?",
            "To check availability.",
            "Friday",
            "Friday is selected."
        ]
    );
    let (_, restored) =
        turn(&env, &database, "Thanks", vec![text("You're welcome.")], vec![], None, false).await;
    assert_eq!(
        &restored.frames[&restored.root_frame_id].input_requests[request_id], response,
        "a fresh owner restores the admitted response without offering that question again"
    );
    database.close().await;
}
