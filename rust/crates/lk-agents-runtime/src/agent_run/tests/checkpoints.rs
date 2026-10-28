//! A host-observed waiter stops the native runner after an already completed tool.

use super::*;

#[tokio::test(start_paused = true)]
async fn a_waiter_preserves_the_tool_result_and_prevents_the_next_model() {
    let mut setup = setup(
        vec![Reply::Call("find_slot", json!({"day": "Monday"})), Reply::Text("never sent")],
        Body::YieldAfterTool,
        OutputSpec::Text { validated: false },
        Arc::new(Registry::default()),
    );
    let updates = drain(&mut setup.updates).await;
    let settled = finished(&updates).expect("a managed yield settles successfully");
    assert!(settled.result.interrupted);
    assert_eq!(settled.result.usage.requests, 1);
    assert_eq!(settled.result.usage.tool_calls, 1);
    assert!(settled.result.new_messages.iter().any(|item| matches!(item,
        lk_agents_abi::ChatItem::FunctionCallOutput(output) if output.output == "9:00 is free"
    )));
    assert!(
        !updates
            .iter()
            .any(|update| matches!(update, RunUpdate::Stream(StreamEvent::TextDelta { .. })))
    );
}
