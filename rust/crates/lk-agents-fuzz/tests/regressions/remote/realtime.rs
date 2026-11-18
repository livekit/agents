use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use lk_agents_abi::delegation::remote::{InputRequest, RemoteAgentState, RemoteTaskState};
use lk_agents_abi::session::InputRequestState;
use lk_agents_abi::text_session::{SessionSnapshot, SessionStateSink as _};
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_fuzz::drive::{OPENING_AGENT, RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::fakes::a2a_peer::{ENDPOINT, PeerScript};
use lk_agents_fuzz::fakes::remote_tool::REMOTE_TOOL;
use lk_agents_fuzz::input::{Input, Millis, UserText};
use lk_agents_fuzz::record::{Detail, Kind, RemoteStep};
use lk_agents_runtime::storage::session_binding::SessionBinding;
use lk_agents_voice::trace::{RealtimeStep, Step};
use serde_json::json;

/// A conversation whose desk task waits on "Which day?", already presented to the person.
fn waiting_for_a_day() -> Arc<SessionBinding> {
    let mut state = SessionSnapshot::new("call".into(), "start".into(), OPENING_AGENT.into());
    let question = ChatMessage::text("presented", Role::Assistant, "Which day?", UNIX_EPOCH);
    state.chat_history.push(question.clone());
    state.frames.get_mut(&state.root_frame_id).unwrap().history.push(question.into());
    let task = serde_json::from_value(json!({
        "id":"task-1", "contextId":"ctx-1", "status":{"state":"TASK_STATE_INPUT_REQUIRED"}
    }))
    .unwrap();
    let source = serde_json::from_value(json!({
        "messageId":"task-1-q1", "role":"ROLE_AGENT", "contextId":"ctx-1", "taskId":"task-1",
        "parts":[{"text":"Which day?"}]
    }))
    .unwrap();
    let mut remote = RemoteAgentState::new(ENDPOINT.into());
    remote.messages.insert("task-1-q1".into(), source);
    let request = InputRequest {
        state: InputRequestState::Open,
        source_message_id: Some("task-1-q1".into()),
        response_message_id: None,
        presented_message_id: Some("presented".into()),
        human_response_id: None,
    };
    remote.tasks.insert(
        "task-1".into(),
        RemoteTaskState {
            task,
            input_requests: BTreeMap::from([("day".into(), request)]),
            cancel_requested: false,
            next_input: 1,
            terminal_event: None,
            artifact_sources: BTreeMap::new(),
        },
    );
    state.remote_agents.insert(ENDPOINT.into(), remote);
    let binding = Arc::new(SessionBinding::memory());
    binding.stage_voice(state).unwrap();
    binding
}

/// The response recogniser's request, while it is still open.
fn routing_chat(run: &Run) -> Option<u32> {
    run.record.live().of(Kind::Chat).into_iter().find(|chat| {
        matches!(run.record.resource(Kind::Chat, *chat).map(|r| r.detail()),
            Some(Detail::Chat { tools, .. })
                if tools.iter().any(|t| t == lk_agents_core::session::input_routing::TOOL))
    })
}

// Catches: a realtime reply cut (an interruption, or a close that does not drain) while the call
// its model proposed still waits for the person's words to be routed kept that never-admitted
// call, and with it the speech and the routing model's request, alive until the close timeouts
// (about ten seconds). The cut must drop the call and cancel the routing work at once.
#[test]
fn a_cut_realtime_reply_drops_its_call_waiting_on_response_routing() {
    for cut in [Input::Interrupt { force: true, wait: false }, Input::Close { drain: false }] {
        let binding = waiting_for_a_day();
        simulate(async {
            let setup = Setup {
                realtime: Some(RealtimeSetup {
                    turn_detection: false,
                    tool_calls: Some(100),
                    ..RealtimeSetup::default()
                }),
                tts: false,
                tools: vec![REMOTE_TOOL.into()],
                input_routing: true,
                remote: Some(PeerScript::default()),
                ..Setup::default()
            };
            let mut run = Run::start_with(&setup, 0, Default::default(), |args, _| {
                args.session_sink = Some(binding.clone());
            })
            .await;
            run.apply(&Input::GenerateReply {
                user_input: Some(UserText("Friday".into())),
                instructions: None,
                allow: None,
            })
            .await;
            // The model's generation ends on its desk call while the recogniser has not
            // answered: the call waits for the person's words to be admitted.
            let mut generated = false;
            for _ in 0..80 {
                run.apply(&Input::Advance { ms: Millis(50) }).await;
                generated = run.record.journal().iter().any(|(_, step)| {
                    matches!(step, Step::Realtime(RealtimeStep::GenerationStatus { .. }))
                });
                if generated {
                    break;
                }
            }
            assert!(generated, "the realtime model never generated its reply");
            let routing = routing_chat(&run).expect("the recogniser's request is open");
            run.apply(&cut).await;
            run.apply(&Input::Advance { ms: Millis(1_000) }).await;
            assert!(
                !run.record.live().has(Kind::Chat, routing),
                "{cut:?}: the routing request outlived the cut reply"
            );
            assert!(
                !run.record
                    .remote()
                    .iter()
                    .any(|(_, s)| matches!(s, RemoteStep::BodyStarted { .. })),
                "{cut:?}: the never-admitted call ran"
            );
            let speeches_open = run.record.live().of(Kind::Speech);
            assert!(
                speeches_open.is_empty(),
                "{cut:?}: speeches {speeches_open:?} outlived the cut"
            );
        });
    }
}
