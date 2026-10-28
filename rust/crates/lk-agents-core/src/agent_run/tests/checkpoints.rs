//! Managed checkpoints settle before model admission and accepted answer delivery.

use super::*;

fn managed(tools: Vec<FunctionToolInfo>) -> Driver {
    let input = RunInput {
        invocation_id: "managed".into(),
        message: user("message", "Find a room"),
        history: Vec::new(),
        answer_tool: None,
        delegated: false,
        mode: crate::tool_outcome::Mode::Text,
        llm: Some(LlmRef { id: LLM, provider: ProviderId::of_label("fake") }),
        wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
        cleanup_timeout: Duration::from_secs(5),
    };
    let mut driver = Driver {
        run: AgentRun::new(text_agent(tools), input).unwrap().with_checkpoints(),
        now: Ts::ZERO,
        effects: Vec::new(),
        settled: Vec::new(),
    };
    driver.send(Event::Start);
    driver
}

fn checkpoint(driver: &Driver) -> CheckpointId {
    driver
        .effects
        .iter()
        .rev()
        .find_map(|effect| match effect {
            Effect::HostCheckpoint { checkpoint } => Some(*checkpoint),
            _ => None,
        })
        .expect("a managed checkpoint")
}

fn verdict(driver: &mut Driver, yielded: bool) -> Vec<Effect> {
    driver.send(Event::HostCheckpointDone { checkpoint: checkpoint(driver), result: Ok(yielded) })
}

#[test]
fn a_waiter_after_a_tool_preserves_its_result_without_starting_the_next_model() {
    let mut driver = managed(vec![string_tool("lookup", "room")]);
    assert_eq!(driver.requests(), 0);
    let first = checkpoint(&driver);
    assert!(
        driver
            .send(Event::HostCheckpointDone {
                checkpoint: CheckpointId(first.0 + 1),
                result: Ok(false),
            })
            .is_empty(),
        "a stale verdict cannot launch a request"
    );
    verdict(&mut driver, false);
    driver.reply("", &[("lookup", json!({"room":"suite"}))]);
    driver.run_tools(|_| value("suite found"));
    assert_eq!(driver.requests(), 1, "the next model still waits for admission");
    let effects = verdict(&mut driver, true);
    let result = driver.result();
    assert!(result.interrupted);
    assert_eq!(result.usage.requests, 1);
    assert_eq!(result.usage.tool_calls, 1);
    assert_eq!(result.outcome, RunOutcome::Completed { output: Output::Nothing });
    assert!(result.new_messages.iter().any(|item| matches!(item,
        ChatItem::FunctionCallOutput(output) if output.output == "suite found"
    )));
    assert!(!effects.iter().any(|effect| matches!(
        effect,
        Effect::LlmRequestStart { .. } | Effect::Emit(StreamEvent::TextDelta { .. })
    )));
}

#[test]
fn the_accepted_result_waits_for_its_delivery_verdict() {
    let mut driver = managed(Vec::new());
    verdict(&mut driver, false);
    driver.reply("the answer", &[]);
    assert!(driver.finished().is_none());
    verdict(&mut driver, true);
    assert!(driver.result().interrupted);
    assert!(driver.result().new_messages.iter().any(|item| matches!(item,
        ChatItem::Message(message) if message.raw_text_content().as_deref() == Some("the answer")
    )));

    let mut continued = managed(Vec::new());
    verdict(&mut continued, false);
    continued.reply("the accepted answer", &[]);
    verdict(&mut continued, false);
    assert!(!continued.result().interrupted);
    assert_eq!(
        continued.result().outcome,
        RunOutcome::Completed { output: Output::Text("the accepted answer".into()) }
    );
}

#[test]
fn cancellation_waits_for_a_pending_checkpoint_callback_to_settle() {
    let mut driver = managed(Vec::new());
    let pending = checkpoint(&driver);
    let effects = driver.send(Event::Cancel);
    assert!(effects.contains(&Effect::HostCheckpointCancel { checkpoint: pending }));
    assert!(driver.finished().is_none());
    verdict(&mut driver, false);
    assert!(matches!(driver.finished(), Some(Err(RunError::Cancelled))));
    assert_eq!(driver.requests(), 0);
    assert!(
        driver
            .send(Event::HostCheckpointDone { checkpoint: pending, result: Ok(false) })
            .is_empty()
    );
}
