use super::*;
use crate::session::SessionCommand;
use futures_util::FutureExt as _;
use lk_agents_abi::session::{InputRoute, SessionCheckpoint};
use lk_agents_abi::{BoxFuture, ChatItem, FunctionCall, Maybe, Role};
use lk_agents_net::signal;
use lk_agents_voice::agent_session::{
    ActivityCommand, AgentUpdate, Close, Command, Sent, VoiceCommandOrigin,
};
use lk_agents_voice::trace::{Step, ToolStep};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

struct HeldCheckpoint {
    entered: signal::Producer<bool>,
    release: signal::Cell<bool>,
}
impl SessionCheckpoint for HeldCheckpoint {
    fn history(&self) -> Option<lk_agents_abi::storage::voice::VoiceHistory> {
        None
    }
    fn commit(
        &self,
    ) -> BoxFuture<'_, Result<serde_json::Value, lk_agents_abi::storage::StorageError>> {
        Box::pin(async move {
            self.entered.set(true);
            self.release.wait_map(|release| release.then_some(())).await.map_err(|_| {
                lk_agents_abi::storage::StorageError::Cancelled("the held checkpoint ended".into())
            })?;
            Ok(serde_json::Value::Null)
        })
    }
}

fn select(runtime: &mut VoiceRuntime, agent: &str, history: Option<Vec<ChatItem>>) {
    let update = AgentUpdate {
        agent: agent.into(),
        instructions: Default::default(),
        dynamic_instructions: None,
        toolset: Default::default(),
        chat_ctx: history,
        llm: Maybe::NotGiven,
        tts: Maybe::NotGiven,
        stt: Maybe::NotGiven,
        vad: Maybe::NotGiven,
        options: Default::default(),
        expressive: None,
        output_nodes: Default::default(),
        call: None,
        activation: None,
    };
    runtime.handle(
        VoiceEvent::Host(Sent {
            origin: VoiceCommandOrigin::Host,
            command: Command::Activity(ActivityCommand::Update(Box::new(update))),
        })
        .into(),
        runtime.shared.time.read(),
    );
}

#[tokio::test(start_paused = true)]
async fn a_local_response_checkpoint_keeps_its_owner_across_an_agent_switch() {
    held_local_response(false).await;
}

#[tokio::test(start_paused = true)]
async fn a_successful_held_checkpoint_cannot_restore_approval_after_a_human_revision() {
    held_local_response(true).await;
}

async fn held_local_response(revise: bool) {
    let (mut runtime, handle, steps) = crate::voice::tests::session("local-owner-checkpoint").await;
    let mut call = FunctionCall {
        id: "canonical-call".into(),
        call_id: "provider-call".into(),
        name: "book".into(),
        arguments: "{}".into(),
        created_at: UNIX_EPOCH,
        group_id: None,
        extra: Default::default(),
    };
    lk_agents_core::approval::set_mark(
        &mut call,
        lk_agents_abi::llm::tool_executor::Approval::Asked,
        "{}",
    );
    let output = lk_agents_voice::tool_executor::answer(
        &call,
        lk_agents_abi::llm::tool_executor::ToolOutcome::InputRequired {
            question: "Confirm the booking?".into(),
            verbatim: true,
        }
        .into(),
        UNIX_EPOCH,
    )
    .output;
    let question =
        ChatMessage::text("heard-question", Role::Assistant, "Confirm the booking?", UNIX_EPOCH);
    let response = ChatMessage::text("human-answer", Role::User, "Yes, book it.", UNIX_EPOCH);
    select(&mut runtime, "agent", None);
    let history = vec![
        call.clone().into(),
        output.clone().into(),
        question.clone().into(),
        response.clone().into(),
    ];
    // A private model context is not human delivery. Load the prior canonical readback and
    // response through the owner's storage event, which publishes their conversation history.
    runtime.handle(
        VoiceEvent::StorageReloaded {
            read: Vec::new(),
            history: lk_agents_abi::storage::voice::VoiceHistory {
                items: history.clone(),
                stored: history,
                revision: (1, 1),
                ..Default::default()
            },
        }
        .into(),
        runtime.shared.time.read(),
    );
    let mut state = runtime.conversation_snapshot().unwrap();
    assert!(state.chat_history.contains(&question));
    assert!(state.chat_history.contains(&response));
    let agent_id = state.root_frame_id.value.clone();
    let reference =
        InputRequestRef::Agent { agent_id: agent_id.clone(), request_id: "question".into() };
    input::apply(
        &mut state,
        vec![
            SessionInputCommand::Retain {
                agent_id: agent_id.clone(),
                request_id: "question".into(),
                input: Box::new(AgentInputRequest {
                    state: InputRequestState::Open,
                    source_message: question.clone(),
                    response_message: None,
                    origin: Some(ToolCallRef { agent_id, call_id: "canonical-call".into() }),
                    presented_message_id: None,
                }),
            },
            SessionInputCommand::Presented {
                requests: vec![reference.clone()],
                message_id: question.id.clone(),
            },
        ],
    )
    .unwrap();
    runtime.session_sink.as_ref().unwrap().stage_voice(state.clone()).unwrap();
    let request = input::routing_request(&state, "route".into(), response.clone()).unwrap();
    let (entered, writing) = signal::signal(false);
    let (release, released) = signal::signal(false);
    let (done, receipt) = signal::once();
    runtime.apply_conversation_command(
        SessionCommand::AdmitInput {
            request: Box::new(request),
            verdict: Some(Ok(InputRoute::Matched(vec![reference.clone()]))),
            done,
        },
        Arc::new(HeldCheckpoint { entered, release: released }),
        Vec::new(),
    );
    writing.wait_map(|entered| entered.then_some(())).await.unwrap();
    assert!(receipt.wait().now_or_never().is_none());
    let revision = ChatMessage::text("human-answer", Role::User, "No, do not book it.", UNIX_EPOCH);
    if revise {
        select(
            &mut runtime,
            "agent",
            Some(vec![
                call.into(),
                output.clone().into(),
                question.into(),
                revision.clone().into(),
            ]),
        );
    } else {
        select(&mut runtime, "another-agent", None);
        assert_eq!(runtime.shared.view.get().agent, "another-agent");
    }
    assert!(
        !steps
            .lock()
            .iter()
            .any(|step| matches!(step, Step::Tool(ToolStep::InputResponseDelivered { .. })))
    );

    release.finish(true);
    let written = crate::voice::conversation::checkpointed(&runtime.conversation_work).await;
    runtime.conversation_advanced(written, runtime.shared.time.read());
    receipt.wait().await.unwrap().unwrap();
    let notice_id = "human-answer/approval/canonical-call";
    assert!(
        !runtime.shared.chat_ctx.get().items.iter().any(|item| item.id() == notice_id),
        "the new agent cannot receive another agent's approval"
    );
    if revise {
        assert!(
            runtime
                .shared
                .chat_ctx
                .get()
                .items
                .iter()
                .any(|item| { matches!(item, ChatItem::Message(message) if message == &revision) }),
            "the canonical revised words survive the older successful receipt"
        );
        let state = runtime.conversation_snapshot().unwrap();
        let request = &state.frames[&state.root_frame_id].input_requests["question"];
        assert_eq!(
            request.state,
            InputRequestState::Responded,
            "the accepted checkpoint retains its exact delivery callback"
        );
        assert_eq!(
            request.response_message.as_ref(),
            Some(&response),
            "the accepted frozen response is not reinterpreted as the revision"
        );
    } else {
        select(&mut runtime, "agent", None);
        assert!(
            runtime.shared.chat_ctx.get().items.iter().any(|item| item.id() == notice_id),
            "the accepted owner receives its private approval observation"
        );
    }
    assert!(
        runtime.shared.chat_ctx.get().items.iter().any(
            |item| matches!(item, ChatItem::FunctionCallOutput(recorded) if recorded == &output)
        )
    );
    let delivered: Vec<_> = steps
        .lock()
        .iter()
        .filter_map(|step| match step {
            Step::Tool(ToolStep::InputResponseDelivered { request, call_id, message_id }) => {
                Some((request.clone(), call_id.clone(), message_id.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(delivered, vec![(reference, Some("canonical-call".into()), response.id)]);
    handle.api.send(Command::Close(Close::default())).unwrap();
    runtime.run().await;
}
