//! Reloading stored history preserves exact response admission and the original read-backs.

use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
use lk_agents_abi::session::InputRequestState;
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, StorageSetup, simulate};
use lk_agents_fuzz::input::{
    AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
};
use lk_agents_fuzz::record::Kind;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{Step, ToolStep};

/// The person's turn: `words`, heard and ended.
fn says(utterance: u32, words: &str) -> [Input; 7] {
    [
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(utterance),
            text: UserText(words.into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(utterance) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(3_000) },
    ]
}

fn asks(chat: u32, tool: u32, question: &str) -> [Input; 5] {
    [
        Input::LlmToolCall {
            chat: Ref::new(chat),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(chat) },
        Input::Advance { ms: Millis(100) },
        Input::ToolAnswered {
            tool: Ref::new(tool),
            answer: ToolAnswer::AskVerbatim,
            text: AgentText(question.into()),
        },
        Input::Advance { ms: Millis(3_000) },
    ]
}

/// There is exactly one presented open question on each of these model requests.
fn recognizes_response(chat: u32) -> [Input; 3] {
    [
        Input::LlmToolCall {
            chat: Ref::new(chat),
            name: "route_input_response".into(),
            args: ToolArgs(r#"{"candidates":[0],"unclear":false}"#.into()),
        },
        Input::LlmEnd { chat: Ref::new(chat) },
        Input::Advance { ms: Millis(100) },
    ]
}

async fn wait_for_resource(run: &mut Run, kind: Kind, index: u32) {
    for _ in 0..100 {
        if run.record.live().of(kind).contains(&index) {
            return;
        }
        crate::playout::apply_with_playout(run, &Input::Advance { ms: Millis(100) }).await;
    }
    panic!("the expected {kind:?} request {index} never became ready");
}

// Catches a reload putting an already answered question back in the classifier's candidates.
// Original read-backs remain tool outcomes; canonical human words and response state are separate.
#[test]
fn answered_occurrences_survive_a_stored_history_readback() {
    let setup = Setup {
        vad: true,
        stt: true,
        turn: TurnMode::Stt,
        input_routing: true,
        tools: vec!["lookup".into()],
        storage: Some(StorageSetup::default()),
        ..Setup::default()
    };
    simulate(async {
        let mut binding = None;
        let mut run = Run::start_with(&setup, 0, Default::default(), |args, _| {
            binding = args.session_sink.clone();
        })
        .await;
        let binding = binding.expect("stored voice has its native session binding");
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
        ]
        .into_iter()
        .chain(asks(0, 0, "Which day?"))
        .chain(says(0, "Monday"))
        .chain(recognizes_response(1))
        .chain(asks(2, 1, "Which city?"))
        .chain(says(1, "Paris"))
        .chain(recognizes_response(3))
        .chain([
            Input::LlmText { chat: Ref::new(4), text: AgentText("All set.".into()) },
            Input::LlmEnd { chat: Ref::new(4) },
            Input::Advance { ms: Millis(1_000) },
        ])
        .chain(says(2, "Thanks"));
        for input in inputs {
            match &input {
                Input::LlmToolCall { chat, .. }
                | Input::LlmEnd { chat }
                | Input::LlmText { chat, .. } => {
                    wait_for_resource(&mut run, Kind::Chat, chat.nth).await;
                }
                Input::ToolAnswered { tool, .. } => {
                    wait_for_resource(&mut run, Kind::Tool, tool.nth).await;
                }
                _ => {}
            }
            crate::playout::apply_with_playout(&mut run, &input).await;
        }

        let snapshot = binding.restore().unwrap().unwrap();
        let requests = &snapshot.frames[&snapshot.root_frame_id].input_requests;
        let chats: Vec<_> = run
            .record
            .resources(lk_agents_fuzz::record::Kind::Chat)
            .filter_map(|resource| match resource.detail() {
                lk_agents_fuzz::record::Detail::Chat { tools, input_candidates, .. } => {
                    Some((tools, input_candidates))
                }
                _ => None,
            })
            .collect();
        let outcomes: Vec<_> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, event)| {
                if let SessionEvent::ConversationItemAdded { item, .. } = event
                    && let ChatItem::FunctionCallOutput(output) = &**item
                {
                    Some((output.id.as_str(), output.name.as_str(), output.output.as_str()))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            requests.len(),
            2,
            "both questions must actually be retained; requests: {requests:?}; model offers: {chats:?}; outcomes: {outcomes:?}"
        );
        for (question, response) in [("Which day?", "Monday"), ("Which city?", "Paris")] {
            let request = requests
                .values()
                .find(|request| {
                    request.source_message.raw_text_content().as_deref() == Some(question)
                })
                .unwrap();
            assert_eq!(request.state, InputRequestState::Responded, "{question}");
            assert_eq!(
                request
                    .response_message
                    .as_ref()
                    .and_then(|message| message.raw_text_content())
                    .as_deref(),
                Some(response)
            );
        }

        let deliveries: Vec<_> = run
            .record
            .journal()
            .iter()
            .filter_map(|(at, step)| {
                matches!(step, Step::Tool(ToolStep::InputResponseDelivered { .. })).then_some(*at)
            })
            .collect();
        assert_eq!(deliveries.len(), 2, "each response reaches its occurrence once");
        let first = deliveries[0];
        let read_back_after_response = run.record.host_view().storage().iter().any(|(at, step)| {
            first < *at
                && matches!(
                    step,
                    lk_agents_fuzz::record::StorageStep::ConversationCheckpoint { .. }
                )
        });
        assert!(
            read_back_after_response,
            "the story must read stored history after accepting a response"
        );
        let mut history = Vec::<ChatItem>::new();
        for (_, event) in run.record.host_view().events() {
            match event {
                SessionEvent::HistoryReloaded { items } => {
                    history = items.clone();
                }
                SessionEvent::ConversationItemAdded { item, .. } => {
                    match history.iter_mut().find(|seen| seen.id() == item.id()) {
                        Some(seen) => *seen = (**item).clone(),
                        None => history.push((**item).clone()),
                    }
                }
                _ => {}
            }
        }
        for response in ["Monday", "Paris"] {
            assert!(history.iter().any(|item| matches!(item, ChatItem::Message(message)
                if message.role == Role::User
                    && message.raw_text_content().as_deref() == Some(response))));
        }
        let outputs: Vec<_> = history
            .iter()
            .filter_map(|item| match item {
                ChatItem::FunctionCallOutput(output) => Some(output.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            outputs,
            [format!("{WAITING_PREFIX}Which day?"), format!("{WAITING_PREFIX}Which city?")]
        );
    });
}
