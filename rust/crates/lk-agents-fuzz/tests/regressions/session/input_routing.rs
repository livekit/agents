//! Shared response admission stays visible after a later voice reply is planned.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use lk_agents_abi::a2a::client::AgentClientError;
use lk_agents_abi::a2a::remote::{DelegationInput, RemoteResponseClaim, RemoteTaskDriver};
use lk_agents_abi::agent::VoiceAgent as _;
use lk_agents_abi::delegation::DelegationOptions;
use lk_agents_abi::delegation::remote::{InputRequest, RemoteAgentState, RemoteTaskState};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::session::InputRequestState;
use lk_agents_abi::session::{InputRequestRef, SessionCommands as _};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::text_session::{SessionSnapshot, SessionStateSink as _};
use lk_agents_abi::voice::session::VoiceSession as _;
use lk_agents_abi::{BoxFuture, ChatMessage, ChatRequest, Llm, LlmStream, Role};
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{OPENING_AGENT, Run, Setup, simulate};
use lk_agents_fuzz::fakes::{FakeLlm, Registry};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs, TurnMode, UserText};
use lk_agents_runtime::storage::session_binding::SessionBinding;
use lk_agents_voice::tool_executor::{ToolOutcome, ToolReport, VoiceAgentFrame};
use parking_lot::Mutex;
use serde_json::json;

const REMOTE: &str = "https://travel.test";

/// These tests drive the native journal boundary directly; the admitted callback identity
/// remains real, while this live resource never opens a second transport of its own.
struct AdmissionCandidate;

impl RemoteTaskDriver for AdmissionCandidate {
    fn same_configuration(&self, other: &dyn RemoteTaskDriver) -> bool {
        (other as &dyn std::any::Any).is::<Self>()
    }

    fn cancellation(
        &self,
        _task: lk_agents_abi::delegation::remote::RemoteTaskRef,
        _retry_after: Option<String>,
    ) -> Arc<dyn lk_agents_abi::session::control::RemoteCancellation> {
        panic!("this admission fixture never dispatches a task control")
    }

    fn is_live(&self) -> bool {
        true
    }

    fn input<'a>(
        &'a self,
        _input: DelegationInput,
        _response_claim: Option<Arc<RemoteResponseClaim>>,
        _origin: Option<lk_agents_abi::session::ToolCallRef>,
    ) -> BoxFuture<'a, Result<(), AgentClientError>> {
        Box::pin(async { panic!("the native receipt test drives its original journal directly") })
    }

    fn resume<'a>(
        &'a self,
        _request_id: String,
        _request: Box<lk_agents_abi::a2a::protocol::SendMessageRequest>,
        _streaming: bool,
        _input_request: Option<InputRequestRef>,
        _origin: Option<lk_agents_abi::session::ToolCallRef>,
    ) -> BoxFuture<'a, Result<(), AgentClientError>> {
        Box::pin(async { panic!("the native receipt test only inspects saved recovery") })
    }

    fn reconcile(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

/// The session loop belongs to the deterministic harness. Poll a native command to enqueue
/// it, then pump all immediately runnable owner work before inspecting its actual receipt.
async fn native_receipt<T>(run: &mut Run, mut receipt: BoxFuture<'_, T>, what: &str) -> T {
    use futures_util::FutureExt as _;
    if let Some(value) = receipt.as_mut().now_or_never() {
        return value;
    }
    run.apply(&Input::Advance { ms: Millis(0) }).await;
    receipt
        .now_or_never()
        .unwrap_or_else(|| panic!("{what} did not settle after pumping its owner"))
}

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
            terminal_event: None,
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
                    Some(DelegationOptions { routing_model: Some(FakeLlm::new(registry.clone())) });
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
            args.delegation = Some(DelegationOptions {
                routing_model: Some(frontend(registry, routing.clone())),
            });
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

#[test]
fn recognition_retains_the_response_until_the_remote_tool_claims_it() {
    response_handoff(false);
}

#[test]
fn a_reused_provider_call_id_does_not_steal_the_original_remote_result() {
    response_handoff(true);
}

fn response_handoff(reused_call_ids: bool) {
    let host = Arc::new(Mutex::new(None));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let binding = waiting_for_date();
    simulate(async {
        let mut run = Run::start_with(
            &Setup { tools: vec!["travel".into()], tts: false, ..Default::default() },
            0,
            Default::default(),
            |args, registry| {
                args.session_sink = Some(binding.clone());
                args.delegation =
                    Some(DelegationOptions { routing_model: Some(FakeLlm::new(registry.clone())) });
                args.llm = Some(Arc::new(ObservedLlm {
                    inner: if reused_call_ids {
                        FakeLlm::reusing_call_ids(registry.clone())
                    } else {
                        FakeLlm::new(registry.clone())
                    },
                    requests: requests.clone(),
                }));
                let frame = args.voice_agent.as_ref().unwrap();
                let mut definition = frame.definition().clone();
                definition.tools[0].remote = Some(REMOTE.into());
                let callbacks = lk_agents_fuzz::fakes::FakeTools::new(registry.clone());
                args.tools = Some(definition.tools.clone());
                args.voice_agent = Some(Arc::new(VoiceAgentFrame::new(
                    definition,
                    callbacks.clone(),
                    frame.host_context().clone(),
                    frame.toolset_scope(),
                )));
                *host.lock() = Some(callbacks);
            },
        )
        .await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "travel".into(),
                args: ToolArgs(r#"{"label":"book"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::ToolUpdate {
                tool: Ref::new(0),
                progress: Some(AgentText("The request is with travel.".into())),
                end: lk_agents_fuzz::input::ToolAnswer::Detached,
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(1), text: AgentText("I am checking.".into()) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(500) },
        ] {
            run.apply(&input).await;
        }
        let lk_agents_fuzz::record::Detail::Tool { invocation_id, operation, call_id, .. } =
            run.record.resource(lk_agents_fuzz::record::Kind::Tool, 0).unwrap().detail()
        else {
            panic!("travel did not run");
        };
        run.registry
            .session()
            .unwrap()
            .tool_update(
                invocation_id.clone(),
                *operation,
                call_id.clone(),
                ToolReport {
                    outcome: ToolOutcome::InputRequired {
                        question: "Which day?".into(),
                        verbatim: true,
                    },
                    input_request: Some(Box::new(InputRequestRef::RemoteTask {
                        remote_id: REMOTE.into(),
                        task: lk_agents_abi::delegation::remote::RemoteTaskRef {
                            context_id: "trip".into(),
                            task_id: "booking".into(),
                        },
                        request_id: "date-request".into(),
                    })),
                    transition: None,
                },
            )
            .unwrap();
        for input in [
            Input::Advance { ms: Millis(500) },
            Input::GenerateReply {
                user_input: Some(UserText("Friday".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(100) },
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
        let callbacks = host.lock().clone().unwrap();
        assert!(
            callbacks.expert_answers.lock().unwrap().is_empty(),
            "recognition dispatched before the tool invocation, allowing a second send"
        );
        let held = binding.restore().unwrap().unwrap();
        let original_history = &held.frames[&held.root_frame_id].history;
        let original_call = original_history
            .iter()
            .find_map(|item| match item {
                lk_agents_abi::ChatItem::FunctionCall(call) => Some(call.clone()),
                _ => None,
            })
            .expect("the originating remote call was not recorded");
        let original_output_id = original_history
            .iter()
            .find_map(|item| match item {
                lk_agents_abi::ChatItem::FunctionCallOutput(output)
                    if output.call_id == original_call.call_id =>
                {
                    Some(output.id.clone())
                }
                _ => None,
            })
            .expect("the originating call has no pending output");
        assert_eq!(
            held.remote_agents[REMOTE].tasks["booking"].input_requests["date-request"].state,
            InputRequestState::Responding
        );
        // The frontend now calls the same public remote tool. Its rewritten arguments must not
        // replace the original human message, and a repeated call must not send another answer.
        for input in [
            Input::LlmToolCall {
                chat: Ref::new(3),
                name: "travel".into(),
                args: ToolArgs(r#"{"label":"the model rewrote Friday"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(3) },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        let lk_agents_fuzz::record::Detail::Tool {
            item_id: continuation_item,
            call_id: continuation,
            ..
        } = run.record.resource(lk_agents_fuzz::record::Kind::Tool, 1).unwrap().detail()
        else {
            panic!("the continuation tool did not run");
        };
        let continuation_item = continuation_item.clone();
        assert_ne!(continuation_item, original_call.id);
        if reused_call_ids {
            assert_eq!(
                continuation, &original_call.call_id,
                "the fake did not reuse the provider ID"
            );
        }
        let owner = run.registry.session().unwrap().clone();
        let captured = binding.restore().unwrap().unwrap();
        // The body-start event above proves the actual callback received this canonical
        // call. Memory-only conversation history contains committed calls, not a live body's
        // private preparation; do not wait for or invent a premature history checkpoint.
        let canonical = callbacks
            .calls
            .lock()
            .unwrap()
            .iter()
            .find(|call| call.id == continuation_item)
            .cloned()
            .expect("the started body did not receive its canonical call");
        let offered =
            lk_agents_core::delegation::remote::routing::captured_offers(&canonical).unwrap();
        let request = native_receipt(
            &mut run,
            owner.routing_context(
                REMOTE,
                REMOTE,
                json!({"label":"the model rewrote Friday"}),
                None,
                &offered,
            ),
            "remote routing context",
        )
        .await
        .unwrap();
        let verdict = lk_agents_core::delegation::remote::routing::parse(
            r#"{"action":"respond","candidate":0,"response":0}"#,
            &request,
        )
        .unwrap();
        let candidate: Arc<dyn RemoteTaskDriver> = Arc::new(AdmissionCandidate);
        let admission = native_receipt(
            &mut run,
            owner.admit_call(
                &request,
                Some(Ok(verdict.clone())),
                Default::default(),
                Arc::downgrade(&candidate),
            ),
            "remote call admission",
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(admission.driver.as_ref().unwrap(), &candidate));
        let response_claim =
            admission.response_claim.expect("an unsent answer needs its owned claim");
        let lk_agents_abi::delegation::routing::CallAdmission::Respond {
            request: reference,
            message,
        } = admission.admission
        else {
            panic!("the native admission did not retain the human response")
        };
        assert_eq!(
            message.raw_text_content().as_deref(),
            Some("Friday"),
            "model arguments replaced the person's actual words"
        );
        let task = reference.remote().unwrap().1.clone();
        let origin = lk_agents_abi::session::ToolCallRef {
            agent_id: captured.root_frame_id.value.clone(),
            call_id: continuation_item.clone(),
        };
        let command = lk_agents_abi::delegation::remote::RemoteCommand {
            remote_id: REMOTE.into(),
            endpoint: REMOTE.into(),
            input: lk_agents_abi::delegation::remote::RemoteInput::Prepare {
                request_id: "lost-native-response".into(),
                request: lk_agents_abi::delegation::remote::A2ARequest::Send(
                    serde_json::from_value(json!({"message":{
                        "messageId":"native-response", "role":"ROLE_USER",
                        "contextId":task.context_id, "taskId":task.task_id,
                        "parts":[{"text":message.raw_text_content().unwrap()}]
                    }}))
                    .unwrap(),
                ),
                streaming: true,
                origin: Some(origin.clone()),
                input_request: Some(reference),
                control_scope: None,
            },
        };
        // This is the journal command used by the native transport driver. Enqueue it, then
        // lose its local receipt before the owner commits; the original operation still owns
        // the response, and subsequent tool calls must observe that admission rather than fork.
        use futures_util::FutureExt as _;
        let mut receipt = owner.remote(command);
        assert!(receipt.as_mut().now_or_never().is_none());
        drop(receipt);
        run.apply(&Input::Advance { ms: Millis(0) }).await;
        drop(response_claim);
        for _ in 0..2 {
            assert_eq!(
                native_receipt(
                    &mut run,
                    owner.admit_call(
                        &request,
                        Some(Ok(verdict.clone())),
                        Default::default(),
                        Arc::downgrade(&candidate)
                    ),
                    "duplicate remote call admission",
                )
                .await
                .unwrap()
                .admission,
                lk_agents_abi::delegation::routing::CallAdmission::AlreadyForwarded {
                    task: task.clone()
                }
            );
        }
        let accepted = binding.restore().unwrap().unwrap();
        let remote = &accepted.remote_agents[REMOTE];
        assert_eq!(remote.requests.len(), 1, "a lost receipt duplicated native response admission");
        assert_eq!(remote.requests["lost-native-response"].origin.as_ref(), Some(&origin));
        let recovered = native_receipt(
            &mut run,
            owner.recover(
                REMOTE,
                REMOTE,
                &lk_agents_abi::session::SessionRecovery::Call(origin.clone()),
            ),
            "remote response recovery",
        )
        .await
        .unwrap()
        .unwrap();
        let lk_agents_abi::delegation::remote::RemoteRecovery::Dispatch { request, .. } = recovered
        else {
            panic!("the response with a lost receipt was not recoverable")
        };
        let lk_agents_abi::delegation::remote::A2ARequest::Send(send) = *request else {
            panic!("response recovery changed operation")
        };
        assert_eq!(send.message.message_id, "native-response");
        assert_eq!(send.message.text(), Some("Friday"));
        assert_eq!(
            requests.lock().len(),
            3,
            "the pending foreground tool already owns a planning turn; do not wake another"
        );
        assert!(
            callbacks.expert_answers.lock().unwrap().is_empty(),
            "native remote admission must not revive the host AnswerExpert continuation path"
        );
        // The new caller reports dispatch while the original body still owns the eventual
        // result. Reused provider IDs must not redirect that result to this newer call.
        for input in [
            Input::ToolAnswered {
                tool: Ref::new(1),
                answer: lk_agents_fuzz::input::ToolAnswer::Value,
                text: AgentText("The response is with the existing task.".into()),
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(4), text: AgentText("I will let you know.".into()) },
            Input::LlmEnd { chat: Ref::new(4) },
            Input::Advance { ms: Millis(500) },
        ] {
            run.apply(&input).await;
        }
        // The completed original task must become the presentation's subject even though a
        // newer tool output says only that the response was forwarded.
        run.apply(&Input::ToolAnswered {
            tool: Ref::new(0),
            answer: lk_agents_fuzz::input::ToolAnswer::Value,
            text: AgentText("Booked for Friday.".into()),
        })
        .await;
        run.apply(&Input::Advance { ms: Millis(100) }).await;
        let presentation = requests.lock().last().unwrap().clone();
        assert_eq!(
            presentation.tool_choice,
            Some(lk_agents_abi::ToolChoice::Forbidden),
            "a result presentation must not start new remote work"
        );
        assert!(
            presentation.chat_ctx.items.iter().filter_map(|item| item.as_message()).any(
                |message| message.role == Role::System
                    && message
                        .raw_text_content()
                        .is_some_and(|text| text.contains("Booked for Friday."))
            ),
            "the presentation lost the result behind a newer handoff acknowledgement"
        );
        let accepted = binding.restore().unwrap().unwrap();
        let original_output = accepted.frames[&accepted.root_frame_id]
            .history
            .iter()
            .find_map(|item| match item {
                lk_agents_abi::ChatItem::FunctionCallOutput(output)
                    if output.id == original_output_id =>
                {
                    Some(output)
                }
                _ => None,
            })
            .expect("the original tool output disappeared");
        assert_eq!(
            original_output.output, "Booked for Friday.",
            "the remote result replaced a later call instead of its own pending output"
        );
        let continuation_output = accepted.frames[&accepted.root_frame_id]
            .history
            .iter()
            .find_map(|item| match item {
                lk_agents_abi::ChatItem::FunctionCallOutput(output)
                    if output.id == format!("{continuation_item}_out") =>
                {
                    Some(output)
                }
                _ => None,
            })
            .expect("the continuation tool output disappeared");
        assert_eq!(continuation_output.output, "The response is with the existing task.");
        run.apply(&Input::Close { drain: false }).await;
        run.apply(&Input::Advance { ms: Millis(5_000) }).await;
        run.apply(&Input::AwaitExit {}).await;
        let (failures, _) = check_all(&run.record);
        assert!(failures.is_empty(), "{failures:?}");
    });
}
