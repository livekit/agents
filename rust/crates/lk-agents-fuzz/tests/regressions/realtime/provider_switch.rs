use lk_agents_abi::{ChatItem, Role, llm::realtime::RealtimeProvider};
use lk_agents_fuzz::{
    drive::{RealtimeSetup, Run, Setup, StorageSetup, simulate},
    input::{AgentText, Input, Millis, Ref, RtEvent, ToolAnswer, UserText},
    record::{Kind, StorageStep},
};

async fn advance(run: &mut Run, millis: u64) {
    crate::playout::apply_with_playout(run, &Input::Advance { ms: Millis(millis) }).await;
}

/// Exercise the production adapter and both production dialect machines, over independent
/// servers: foreign IDs must survive two swaps and the real SQLite checkpoint boundary.
#[test]
fn xai_history_survives_openai_and_a_return_to_xai() {
    check_switch(false);
}

#[test]
fn foreign_tool_calls_and_outputs_survive_two_provider_switches() {
    check_switch(true);
}

fn check_switch(with_tools: bool) {
    let run = simulate(async {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                provider: RealtimeProvider::Xai,
                fallback_provider: Some(RealtimeProvider::Openai),
                turn_detection: false,
                tool_calls: Some(if with_tools { 100 } else { 0 }),
                ..Default::default()
            }),
            tools: if with_tools { vec!["lookup".into()] } else { vec![] },
            max_tool_steps: Some(0),
            storage: Some(StorageSetup::default()),
            ..Default::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for (nth, text) in
            ["first question", "second question", "third question"].into_iter().enumerate()
        {
            run.apply(&Input::GenerateReply {
                user_input: Some(UserText(text.into())),
                instructions: None,
                allow: None,
            })
            .await;
            advance(&mut run, 15_000).await;
            if with_tools {
                let tools = run.record.live().of(Kind::Tool);
                assert_eq!(tools.len(), 1, "one tool from this model");
                run.apply(&Input::ToolAnswered {
                    tool: Ref::new(tools[0]),
                    answer: ToolAnswer::Value,
                    text: AgentText("booking recorded".into()),
                })
                .await;
                advance(&mut run, 15_000).await;
            }
            if nth < 2 {
                let sockets = run.record.live().of(Kind::Realtime);
                assert_eq!(sockets.len(), 1);
                run.apply(&Input::RealtimeEvent {
                    socket: Ref::new(sockets[0]),
                    event: RtEvent::FatalError,
                })
                .await;
                advance(&mut run, 2_000).await;
            }
        }
        run.apply(&Input::Close { drain: true }).await;
        run.apply(&Input::AwaitExit {}).await;
        run
    });
    let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
    let rejected: Vec<_> = run.record.inference_view().realtime_breaches().iter().filter(|(_, _, breach)| {
        matches!(breach, lk_agents_fuzz::record::Breach::Refused { event, .. } if event == "conversation.item.create")
    }).collect();
    assert!(rejected.is_empty(), "every replayed history item must be accepted: {rejected:#?}");
    let saved = run
        .record
        .host_view()
        .storage()
        .iter()
        .rev()
        .find_map(|(_, s)| match s {
            StorageStep::Saved { items } => Some(items),
            _ => None,
        })
        .expect("SQLite history readback");
    let messages: Vec<_> = saved.iter().filter_map(ChatItem::as_message).collect();
    assert_eq!(messages.iter().filter(|m| m.role == Role::User).count(), 3, "{messages:#?}");
    assert_eq!(
        messages.iter().filter(|m| m.role == Role::Assistant).count(),
        if with_tools { 6 } else { 3 },
        "{messages:#?}"
    );
    let assistants: Vec<_> = messages.iter().filter(|m| m.role == Role::Assistant).collect();
    assert_eq!(assistants[0].id.len(), 36, "originating xAI ID remains public");
    assert!(
        assistants[if with_tools { 2 } else { 1 }].id.starts_with("item_msg_"),
        "the second response came from OpenAI"
    );
    assert_eq!(assistants.last().unwrap().id.len(), 36, "the last response came from xAI");
    if with_tools {
        let calls: Vec<_> = saved
            .iter()
            .filter_map(|item| match item {
                ChatItem::FunctionCall(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 3);
        for call in calls {
            assert_eq!(saved.iter().filter(|item| matches!(item, ChatItem::FunctionCallOutput(o) if o.call_id == call.call_id && o.output == "booking recorded")).count(), 1);
        }
    }
    assert_eq!(
        saved.iter().map(ChatItem::id).collect::<std::collections::BTreeSet<_>>().len(),
        saved.len()
    );
}

/// The user finished speaking, but ASR is still pending when the provider fails before its
/// delayed response starts. The replacement must answer the preserved partial exactly once.
#[test]
fn a_pending_user_turn_keeps_its_partial_transcript_across_a_provider_switch() {
    use lk_agents_fuzz::{fakes::RealtimeDelays, input::Count};
    let run = simulate(async {
        let setup = Setup {
            stt: false,
            vad: false,
            realtime: Some(RealtimeSetup {
                provider: RealtimeProvider::Xai,
                fallback_provider: Some(RealtimeProvider::Openai),
                turn_detection: false,
                tool_calls: Some(0),
                delays: RealtimeDelays {
                    created: Some(2_000),
                    transcription: Some(25_000),
                    transcription_partial: Some(10),
                    ..Default::default()
                },
                ..Default::default()
            }),
            storage: Some(StorageSetup::default()),
            ..Default::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        run.apply(&Input::MicFrames { mic: Ref::new(0), chunks: Count(30) }).await;
        advance(&mut run, 300).await;
        run.apply(&Input::CommitTurn { skip_reply: false }).await;
        advance(&mut run, 500).await;
        run.apply(&Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::FatalError }).await;
        advance(&mut run, 15_000).await;
        run.apply(&Input::Close { drain: true }).await;
        run.apply(&Input::AwaitExit {}).await;
        run
    });
    let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
    let saved = run
        .record
        .host_view()
        .storage()
        .iter()
        .rev()
        .find_map(|(_, s)| match s {
            StorageStep::Saved { items } => Some(items),
            _ => None,
        })
        .unwrap();
    let messages: Vec<_> = saved.iter().filter_map(ChatItem::as_message).collect();
    assert_eq!(
        messages
            .iter()
            .filter(|m| m.role == Role::User && m.raw_text_content().is_some_and(|t| !t.is_empty()))
            .count(),
        1,
        "the original accepted words survive: {messages:#?}"
    );
    assert_eq!(
        messages.iter().filter(|m| m.role == Role::Assistant).count(),
        1,
        "one answer from the replacement: {messages:#?}"
    );
    assert_eq!(messages[0].role, Role::User);
}
