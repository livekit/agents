//! Shared response admission stays visible after a later voice reply is planned.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use lk_agents_abi::delegation::DelegationOptions;
use lk_agents_abi::delegation::remote::{InputRequest, RemoteAgentState, RemoteTaskState};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::session::InputRequestState;
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::text_session::{SessionSnapshot, SessionStateSink as _};
use lk_agents_abi::{BoxFuture, ChatMessage, ChatRequest, Llm, LlmStream, Role};
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{OPENING_AGENT, Run, Setup, simulate};
use lk_agents_fuzz::fakes::{FakeLlm, Registry};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs, TurnMode, UserText};
use lk_agents_runtime::storage::session_binding::SessionBinding;
use parking_lot::Mutex;
use serde_json::json;

const REMOTE: &str = "https://travel.test";

struct ObservedLlm {
    inner: Arc<FakeLlm>,
    requests: Arc<Mutex<Vec<ChatRequest>>>,
}

impl Llm for ObservedLlm {
    fn label(&self) -> String {
        "observed frontend".into()
    }

    fn chat<'a>(
        &'a self,
        cx: &Cx,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        self.requests.lock().push(request.clone());
        self.inner.chat(cx, request)
    }
}

fn waiting_for_date() -> Arc<SessionBinding> {
    let mut state = SessionSnapshot::new("call".into(), "start".into(), OPENING_AGENT.into());
    let question = ChatMessage::text("presented", Role::Assistant, "Which day?", UNIX_EPOCH);
    state.chat_history.push(question.clone());
    state.frames.get_mut(&state.root_frame_id).unwrap().history.push(question.into());
    let task = serde_json::from_value(json!({
        "id":"booking", "contextId":"trip", "status":{"state":"TASK_STATE_INPUT_REQUIRED"}
    }))
    .unwrap();
    let source = serde_json::from_value(json!({
        "messageId":"date", "role":"ROLE_AGENT", "contextId":"trip", "taskId":"booking",
        "parts":[{"text":"Which day?"}]
    }))
    .unwrap();
    let mut remote = RemoteAgentState::new(REMOTE.into());
    remote.messages.insert("date".into(), source);
    remote.tasks.insert(
        "booking".into(),
        RemoteTaskState {
            task,
            input_requests: BTreeMap::from([(
                "date-request".into(),
                InputRequest {
                    state: InputRequestState::Open,
                    source_message_id: Some("date".into()),
                    response_message_id: None,
                    presented_message_id: Some("presented".into()),
                    human_response_id: None,
                },
            )]),
            cancel_requested: false,
            next_input: 1,
        },
    );
    state.remote_agents.insert(REMOTE.into(), remote);
    let binding = Arc::new(SessionBinding::memory());
    binding.stage_voice(state).unwrap();
    binding
}

fn frontend(registry: &Arc<Registry>, requests: Arc<Mutex<Vec<ChatRequest>>>) -> Arc<dyn Llm> {
    Arc::new(ObservedLlm { inner: FakeLlm::new(registry.clone()), requests })
}

#[test]
fn admitted_remote_responses_remain_visible_to_later_frontend_replies() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let binding = waiting_for_date();
    simulate(async {
        // The current agent lacks travel: recognize and retain the response, then plan a
        // later reply as happens when background work produces another conversation turn.
        let mut run = Run::start_with(
            &Setup { tts: false, ..Default::default() },
            0,
            Default::default(),
            |args, registry| {
                args.session_sink = Some(binding.clone());
                args.delegation =
                    Some(DelegationOptions { routing_model: FakeLlm::new(registry.clone()) });
                args.llm = Some(frontend(registry, requests.clone()));
            },
        )
        .await;
        for input in [
            Input::GenerateReply {
                user_input: Some(UserText("Friday".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "route_input_response".into(),
                args: ToolArgs(r#"{"candidates":[0],"unclear":false}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(1), text: AgentText("I have your date.".into()) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(500) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        let held = binding.restore().unwrap().unwrap();
        assert_eq!(
            held.remote_agents[REMOTE].tasks["booking"].input_requests["date-request"].state,
            InputRequestState::Responding,
            "the response is held until an eligible agent returns"
        );
        let captured = requests.lock().clone();
        assert_eq!(captured.len(), 2, "admission and the later reply each reach the frontend");
        for request in captured.iter() {
            assert!(
                request.chat_ctx.items.iter().any(|item| item.as_message().is_some_and(
                    |message| message.role == Role::System
                        && message.raw_text_content().is_some_and(
                            |text| text.contains(REMOTE) && text.contains("date-request")
                        )
                )),
                "a later frontend request lost the recognized response's original destination"
            );
        }
        assert!(
            held.chat_history.iter().all(|message| message.role != Role::System),
            "private routing observations must not become perceived human dialogue"
        );
        run.apply(&Input::Close { drain: false }).await;
        run.apply(&Input::Advance { ms: Millis(5_000) }).await;
        run.apply(&Input::AwaitExit {}).await;
        let (failures, _) = check_all(&run.record);
        assert!(failures.is_empty(), "{failures:?}");
    });
}

#[test]
fn a_corrected_turn_is_classified_again_before_the_frontend_answers_it() {
    let routing = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let binding = waiting_for_date();
    simulate(async {
        let setup = Setup { stt: true, tts: false, turn: TurnMode::Manual, ..Default::default() };
        let mut run = Run::start_with(&setup, 0, Default::default(), |args, registry| {
            args.session_sink = Some(binding.clone());
            args.delegation =
                Some(DelegationOptions { routing_model: frontend(registry, routing.clone()) });
            args.llm = Some(frontend(registry, requests.clone()));
        })
        .await;
        for input in [
            Input::Advance { ms: Millis(100) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: UserText("Why do you need the date?".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(100) },
            Input::CommitTurn { skip_reply: false },
            Input::Advance { ms: Millis(1_000) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "route_input_response".into(),
                args: ToolArgs(r#"{"candidates":[],"unclear":false}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        assert_eq!(routing.lock().len(), 1, "the interim was recognized before its first reply");
        assert_eq!(requests.lock().len(), 1, "the frontend began answering the side question");

        // The final adds an answer while the original frontend is still generating.
        for input in [
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("Why do you need the date? Friday.".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        let captured = routing.lock().clone();
        assert_eq!(
            captured.len(),
            2,
            "the corrected words bypassed response recognition; frontend requests: {:?}",
            requests
                .lock()
                .iter()
                .map(|request| request
                    .chat_ctx
                    .items
                    .iter()
                    .filter_map(|item| item.as_message())
                    .filter(|message| message.role == Role::User)
                    .filter_map(ChatMessage::raw_text_content)
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        let prompt = captured[1]
            .chat_ctx
            .items
            .last()
            .unwrap()
            .as_message()
            .unwrap()
            .raw_text_content()
            .unwrap();
        let prompt: serde_json::Value = serde_json::from_str(&prompt).unwrap();
        assert_eq!(prompt["latest_message"], "Why do you need the date? Friday.");
        assert_eq!(requests.lock().len(), 1, "the revised frontend must wait for its verdict");
        for input in [
            Input::LlmToolCall {
                chat: Ref::new(2),
                name: "route_input_response".into(),
                args: ToolArgs(r#"{"candidates":[0],"unclear":false}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(2) },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        let held = binding.restore().unwrap().unwrap();
        let input = &held.remote_agents[REMOTE].tasks["booking"].input_requests["date-request"];
        assert_eq!(input.state, InputRequestState::Responding);
        let human = held
            .chat_history
            .iter()
            .find(|message| Some(&message.id) == input.human_response_id.as_ref())
            .unwrap();
        assert_eq!(human.raw_text_content().unwrap(), "Why do you need the date? Friday.");
        assert_eq!(
            held.chat_history.iter().filter(|message| message.role == Role::User).count(),
            1,
            "a final transcript revises one human turn instead of inventing another"
        );
        let latest = requests.lock().last().unwrap().clone();
        assert_eq!(
            latest
                .chat_ctx
                .items
                .iter()
                .filter_map(|item| item.as_message())
                .filter(|message| message.id.starts_with("input_route/"))
                .count(),
            1,
            "the revised verdict replaces its old private observation"
        );
        run.apply(&Input::Close { drain: false }).await;
        run.apply(&Input::Advance { ms: Millis(5_000) }).await;
        run.apply(&Input::AwaitExit {}).await;
        let (failures, _) = check_all(&run.record);
        assert!(failures.is_empty(), "{failures:?}");
    });
}
