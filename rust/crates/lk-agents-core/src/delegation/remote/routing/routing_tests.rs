use super::*;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::{A2AResponse, RemoteCommand, RemoteInput};
use lk_agents_abi::session::SessionInputCommand;
use lk_agents_abi::{ChatMessage, FunctionCall, Role};
use serde_json::json;
use std::time::UNIX_EPOCH;

const REMOTE: &str = "travel";
const ENDPOINT: &str = "https://travel.test";

fn message(id: &str, role: Role, text: &str) -> ChatMessage {
    ChatMessage::text(id, role, text, UNIX_EPOCH)
}

fn apply(state: &mut SessionSnapshot, input: RemoteInput) -> Option<InputRequestRef> {
    crate::session::remote::apply(
        state,
        RemoteCommand { remote_id: REMOTE.into(), endpoint: ENDPOINT.into(), input },
    )
    .unwrap()
}

fn send(id: &str, text: &str, reference: Option<InputRequestRef>) -> RemoteInput {
    let target = reference.as_ref().map(|reference| reference.remote().unwrap().1);
    RemoteInput::Prepare {
        request_id: id.into(),
        request: A2ARequest::Send(
            serde_json::from_value(json!({
                "message":{"messageId":id, "role":"ROLE_USER", "parts":[{"text":text}],
                    "taskId":target.map(|task| &task.task_id),
                    "contextId":target.map(|task| &task.context_id)}
            }))
            .unwrap(),
        ),
        streaming: true,
        origin: None,
        input_request: reference,
    }
}

fn observe(state: &mut SessionSnapshot, value: serde_json::Value) -> Option<InputRequestRef> {
    apply(
        state,
        RemoteInput::Response {
            request_id: "start".into(),
            response: Box::new(A2AResponse::Stream(serde_json::from_value(value).unwrap())),
        },
    )
}

fn question(source: &str, text: &str) -> serde_json::Value {
    json!({"task":{"id":"flight", "contextId":"trip", "status":{
        "state":"TASK_STATE_INPUT_REQUIRED", "message":{
            "messageId":source, "role":"ROLE_AGENT", "parts":[{"text":text}]}
    }}})
}

fn reserved() -> (SessionSnapshot, InputRequestRef) {
    let mut state = SessionSnapshot::new("session".into(), "execution".into(), "desk".into());
    apply(&mut state, send("start", "Find a flight to Rome", None));
    apply(&mut state, RemoteInput::DispatchStarted { request_id: "start".into() });
    let reference = observe(&mut state, question("question_1", "Which day?")).unwrap();
    state.chat_history.push(message("presentation", Role::Assistant, "Which day would you fly?"));
    crate::session::input::apply(
        &mut state,
        vec![SessionInputCommand::Presented {
            requests: vec![reference.clone()],
            message_id: "presentation".into(),
        }],
    )
    .unwrap();
    state.chat_history.push(message("human", Role::User, "Friday. Also find a hotel."));
    crate::session::input::apply(
        &mut state,
        vec![SessionInputCommand::Respond {
            requests: vec![reference.clone()],
            message_id: "human".into(),
        }],
    )
    .unwrap();
    (state, reference)
}

fn routing(state: &SessionSnapshot, text: &str) -> CallRoutingRequest {
    request(state, REMOTE, ENDPOINT, json!({"request":text}), None, &offered(state)).unwrap()
}

#[test]
fn one_human_turn_can_answer_a_flight_and_start_a_hotel_without_substituting_the_reserved_answer() {
    let (state, reference) = reserved();
    let original = state.clone();
    let hotel = routing(&state, "Find a hotel in Rome");
    assert_eq!(admit(&state, &hotel, Some(Ok(CallRoute::New))).unwrap(), CallAdmission::New);
    let flight = routing(&state, "Friday");
    assert_eq!(
        admit(&state, &flight, Some(Ok(CallRoute::Respond(reference.clone())))).unwrap(),
        CallAdmission::Respond {
            request: reference.clone(),
            message: Box::new(message("human", Role::User, "Friday. Also find a hotel.")),
        }
    );
    let task = reference.remote().unwrap().1.clone();
    assert_eq!(
        admit(&state, &flight, Some(Ok(CallRoute::Continue(task.clone())))).unwrap(),
        CallAdmission::Continue(task)
    );
    assert_eq!(state, original, "classification does not itself consume a held answer");
    assert_eq!(hotel.proposed, json!({"request":"Find a hotel in Rome"}));
}

#[test]
fn a_side_question_and_working_task_correction_continue_without_claiming_input() {
    let (mut state, reference) = reserved();
    let task = reference.remote().unwrap().1.clone();
    let request = routing(&state, "Why do you need the date?");
    assert_eq!(
        admit(&state, &request, Some(Ok(CallRoute::Continue(task.clone())))).unwrap(),
        CallAdmission::Continue(task.clone())
    );
    observe(
        &mut state,
        json!({"task":{"id":"flight", "contextId":"trip", "status":{
            "state":"TASK_STATE_WORKING"
        }}}),
    );
    let request = routing(&state, "Saturday, not Friday");
    assert!(request.candidates[0].responses.is_empty());
    assert_eq!(request.candidates[0].state, a2a::TaskState::Working);
    assert!(
        request.candidates[0]
            .messages
            .iter()
            .any(|message| { message.text() == Some("Find a flight to Rome") }),
        "a limited peer snapshot retains the initial request's intent"
    );
    assert_eq!(
        admit(&state, &request, Some(Ok(CallRoute::Continue(task.clone())))).unwrap(),
        CallAdmission::Continue(task)
    );
}

#[test]
fn explicit_task_selection_does_not_implicitly_consume_its_latest_held_answer() {
    let (state, reference) = reserved();
    let task = reference.remote().unwrap().1.clone();
    let request = request(
        &state,
        REMOTE,
        ENDPOINT,
        json!({"request":"Why this date?"}),
        Some("flight"),
        &offered(&state),
    )
    .unwrap();
    assert_eq!(
        admit(&state, &request, Some(Ok(CallRoute::Continue(task.clone())))).unwrap(),
        CallAdmission::Continue(task)
    );
    assert!(admit(&state, &request, Some(Ok(CallRoute::New))).is_err());
    assert!(
        admit(
            &state,
            &request,
            Some(Ok(CallRoute::Continue(RemoteTaskRef {
                task_id: "foreign".into(),
                context_id: "trip".into(),
            })))
        )
        .is_err()
    );
    assert!(super::request(&state, REMOTE, ENDPOINT, json!({}), Some("unknown"), &[]).is_err());
    assert!(super::request(&state, REMOTE, "https://foreign.test", json!({}), None, &[]).is_err());
}

#[test]
fn a_stale_model_turn_cannot_answer_a_replacement_question_or_revised_human_words() {
    let (mut state, reference) = reserved();
    let captured = routing(&state, "Friday");
    state.chat_history.last_mut().unwrap().content =
        message("human", Role::User, "Saturday").content;
    assert!(admit(&state, &captured, Some(Ok(CallRoute::Respond(reference.clone())))).is_err());
    state.chat_history.last_mut().unwrap().content =
        message("human", Role::User, "Friday. Also find a hotel.").content;
    let replacement = observe(&mut state, question("question_2", "What budget?")).unwrap();
    assert_ne!(reference, replacement);
    assert!(admit(&state, &captured, Some(Ok(CallRoute::Respond(reference)))).is_err());
    let task = &state.remote_agents[REMOTE].tasks["flight"];
    assert_eq!(task.input_requests[replacement.remote().unwrap().2].state, InputRequestState::Open);
    let before_question =
        request(&state, REMOTE, ENDPOINT, json!({"request":"Friday"}), None, &[]).unwrap();
    assert!(
        before_question.candidates[0].responses.is_empty(),
        "execution-time state cannot manufacture a model-turn input offer"
    );
}

#[test]
fn a_question_presented_while_the_detector_waits_is_not_offered_to_its_frontend_turn() {
    let (mut state, _) = reserved();
    let captured = offered(&state);
    let human = state.chat_history.last().unwrap().clone();
    let later = observe(&mut state, question("question_2", "What budget?")).unwrap();
    state.chat_history.push(message("later_prompt", Role::Assistant, "What is your budget?"));
    crate::session::input::apply(
        &mut state,
        vec![SessionInputCommand::Presented {
            requests: vec![later.clone()],
            message_id: "later_prompt".into(),
        }],
    )
    .unwrap();
    assert!(offered(&state).iter().any(|candidate| candidate.reference == later));
    let admitted = crate::session::input::route(
        &mut state,
        &human,
        &captured,
        Some(Ok(lk_agents_abi::session::InputRoute::NoMatch)),
    )
    .unwrap();
    assert!(admitted.offered.is_empty(), "an older human turn must not gain the new occurrence");
    assert!(refresh_offered(&state, &captured).is_empty());
}

#[test]
fn a_forwarded_answer_is_recognized_once_while_the_same_turn_can_still_start_new_work() {
    let (mut state, reference) = reserved();
    let captured = routing(&state, "Friday");
    apply(&mut state, send("answer", "Friday. Also find a hotel.", Some(reference.clone())));
    let expected = CallAdmission::AlreadyForwarded { task: reference.remote().unwrap().1.clone() };
    assert_eq!(
        admit(&state, &captured, Some(Ok(CallRoute::Respond(reference.clone())))).unwrap(),
        expected
    );
    let fresh_turn = routing(&state, "Friday");
    assert_eq!(
        admit(&state, &fresh_turn, Some(Ok(CallRoute::Respond(reference.clone())))).unwrap(),
        expected
    );
    let hotel = routing(&state, "Find a hotel in Rome");
    assert_eq!(admit(&state, &hotel, Some(Ok(CallRoute::New))).unwrap(), CallAdmission::New);
    apply(&mut state, RemoteInput::DispatchStarted { request_id: "answer".into() });
    apply(
        &mut state,
        RemoteInput::Response {
            request_id: "answer".into(),
            response: Box::new(A2AResponse::Stream(
                serde_json::from_value(json!({"task":{
                    "id":"flight", "contextId":"trip", "status":{"state":"TASK_STATE_COMPLETED"}
                }}))
                .unwrap(),
            )),
        },
    );
    let after_completion = routing(&state, "Friday");
    assert_eq!(after_completion.candidates[0].state, a2a::TaskState::Completed);
    assert_eq!(
        admit(&state, &after_completion, Some(Ok(CallRoute::Respond(reference.clone())))).unwrap(),
        expected
    );
    assert!(
        admit(
            &state,
            &after_completion,
            Some(Ok(CallRoute::Continue(reference.remote().unwrap().1.clone())))
        )
        .is_err()
    );
    assert!(super::request(&state, REMOTE, ENDPOINT, json!({}), Some("flight"), &[]).is_err());
    state.chat_history.push(message("later", Role::User, "Another trip"));
    assert!(
        routing(&state, "Find another flight").candidates.is_empty(),
        "forwarded-response offers do not leak into later human turns"
    );
}

#[test]
fn model_verdicts_can_select_only_offered_task_and_response_indices() {
    let (state, reference) = reserved();
    let request = routing(&state, "Friday");
    assert_eq!(
        parse(r#"{"action":"respond","candidate":0,"response":0}"#, &request).unwrap(),
        CallRoute::Respond(reference)
    );
    assert_eq!(
        parse(r#"{"action":"new","candidate":null,"response":null}"#, &request).unwrap(),
        CallRoute::New
    );
    assert_eq!(
        parse(r#"{"action":"unclear","candidate":null,"response":null}"#, &request).unwrap(),
        CallRoute::Unclear
    );
    for invalid in [
        r#"{"action":"new","candidate":0,"response":null}"#,
        r#"{"action":"continue","candidate":0,"response":0}"#,
        r#"{"action":"respond","candidate":0,"response":null}"#,
        r#"{"action":"respond","candidate":1,"response":0}"#,
        r#"{"action":"respond","candidate":0,"response":1}"#,
        r#"{"action":"respond","candidate":-1,"response":0}"#,
        r#"{"action":"respond","candidate":0,"response":0,"input_request_id":"flight/input_0"}"#,
        r#"{"action":"continue","task_id":"flight"}"#,
    ] {
        assert!(parse(invalid, &request).is_err(), "{invalid}");
    }
    let prompt: serde_json::Value = serde_json::from_str(&prompt(&request)).unwrap();
    assert_eq!(
        prompt["candidates"][0]["responses"][0]["held_response"],
        "Friday. Also find a hotel."
    );
    assert!(!prompt.to_string().contains("input_0"));
    assert!(admit(&state, &request, Some(Ok(CallRoute::Unclear))).is_err());
    assert!(admit(&state, &request, None).is_err());
}

#[test]
fn owner_authored_captures_overwrite_model_metadata_and_survive_function_call_roundtrips() {
    let (state, _) = reserved();
    let offered = offered(&state);
    let mut call = FunctionCall {
        id: "call".into(),
        call_id: "wire".into(),
        arguments: "{}".into(),
        name: "travel".into(),
        created_at: UNIX_EPOCH,
        extra: [(INPUT_OFFERS_KEY.into(), json!({"request_id":"invented"}))].into_iter().collect(),
        group_id: None,
    };
    capture_offers(&mut call, &offered);
    let restored: FunctionCall =
        serde_json::from_value(serde_json::to_value(&call).unwrap()).unwrap();
    assert_eq!(captured_offers(&restored).unwrap(), offered);
    call.extra.insert(INPUT_OFFERS_KEY.into(), json!({"request_id":"invented"}));
    assert!(captured_offers(&call).is_err());
}
