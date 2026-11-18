use super::super::tests::*;
use super::*;
use serde_json::json;

#[test]
fn exact_tool_output_is_recorded_once_only_after_its_local_boundary_accepts_it() {
    let mut driver =
        Driver::new(text_agent(vec![string_tool("lookup", "id")]), "Find booking", vec![]);
    driver.reply("", &[("lookup", json!({"id":"ABC"}))]);
    for (call, arguments) in driver.prepares() {
        driver.prepared(call, arguments.into());
    }
    let call = driver.executes()[0].0;
    let output = ToolOutputId(1);
    let effects = driver.send(AgentRunEvent::ToolOutputRequested {
        output,
        call: Some(call),
        text: "Reference ABC.".into(),
        stream: true,
    });
    let message = effects
        .iter()
        .find_map(|effect| match effect {
            AgentRunEffect::DeliverToolOutput { message, .. } => Some(message.clone()),
            _ => None,
        })
        .unwrap();
    assert!(!driver.run.progress_history().iter().any(|item| item.id() == message.id));
    assert!(
        matches!(effects.as_slice(), [AgentRunEffect::DeliverToolOutput { .. }, AgentRunEffect::Emit(StreamEvent::TextDelta { message_id, delta })] if message_id == &message.id && delta == "Reference ABC.")
    );
    driver.send(AgentRunEvent::ToolOutputDelivered { output, outcome: SayOutcome::Delivered });
    assert!(
        driver
            .send(AgentRunEvent::ToolOutputDelivered { output, outcome: SayOutcome::Delivered })
            .is_empty()
    );
    driver.send(AgentRunEvent::ToolSettled { call, report: value("Booked").into() });
    driver.reply("Done", &[]);
    let result = driver.result();
    assert_eq!(result.all_messages.iter().filter(|item| item.id() == message.id).count(), 1);
    assert_eq!(result.new_messages.iter().filter(|item| item.id() == message.id).count(), 1);
    assert_eq!(result.verbatim, None, "Working speech cannot become terminal replay instructions");
}

#[test]
fn an_interrupted_exact_handoff_does_not_enter_perceived_history_or_cancel_the_business_tool() {
    let mut driver =
        Driver::new(text_agent(vec![string_tool("lookup", "id")]), "Find booking", vec![]);
    driver.reply("", &[("lookup", json!({"id":"ABC"}))]);
    for (call, arguments) in driver.prepares() {
        driver.prepared(call, arguments.into());
    }
    let call = driver.executes()[0].0;
    let output = ToolOutputId(1);
    driver.send(AgentRunEvent::ToolOutputRequested {
        output,
        call: Some(call),
        text: "Unheard reference".into(),
        stream: false,
    });
    let effects = driver
        .send(AgentRunEvent::ToolOutputDelivered { output, outcome: SayOutcome::Interrupted });
    assert!(matches!(
        &effects[..],
        [AgentRunEffect::ToolOutputSettled { outcome: SayOutcome::Interrupted, .. }]
    ));
    driver.send(AgentRunEvent::ToolSettled { call, report: value("Booked").into() });
    driver.reply("Done", &[]);
    assert!(
        !driver
            .result()
            .all_messages
            .iter()
            .filter_map(ChatItem::as_message)
            .any(|message| message.raw_text_content().as_deref() == Some("Unheard reference"))
    );
}
