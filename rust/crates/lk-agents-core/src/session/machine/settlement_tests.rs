use super::*;

#[test]
fn an_accepted_child_survives_lost_release_and_propagates_without_another_child_run() {
    let config = config(false, ChatHistory::Fresh, true);
    let mut driver = Driver::new(config.clone(), None, "Book");
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let child = driver.session.state.active_frame_id.clone();
    let child_invocation = driver.invocation();
    let parent_call = driver.session.state.frames[&child].parent.clone().unwrap();
    driver.auto_release = false;
    driver.session.recording = true;
    driver.accepted(json!({"reference":"accepted"}));
    assert!(driver.effects.iter().any(|effect| matches!(effect,
        SessionEffect::Checkpoint(state)
            if state.frames[&child].completed
                && state.frames[&child].output == Some(json!({"reference":"accepted"}))
    )));
    let mut heard = message("heard-child", Role::Assistant, "A quiet room is avail");
    heard.interrupted = true;
    driver.send(SessionEvent::DialogueDelivered {
        invocation_id: child_invocation,
        message: heard.clone(),
    });
    driver.send(SessionEvent::Cancel);
    let saved = driver.session.snapshot();
    assert!(saved.frames[&child].completed);
    assert_eq!(saved.frames[&child].output, Some(json!({"reference":"accepted"})));
    assert!(saved.frames[&child].error.is_none());
    assert!(
        saved.frames[&parent_call.frame_id]
            .pending_calls
            .iter()
            .any(|pending| pending.parent == parent_call)
    );
    drop(driver);

    let resumed = Driver::new(config, Some(saved), "Thanks");
    assert_eq!(resumed.session.state.frames.len(), 1);
    let root = &resumed.session.state.frames[&resumed.session.state.root_frame_id];
    assert!(root.pending_calls.is_empty());
    let call = root
        .history
        .iter()
        .find_map(|item| match item {
            ChatItem::FunctionCall(call) if call.id == parent_call.call_id => Some(call),
            _ => None,
        })
        .unwrap();
    assert_eq!(root.history.iter().filter(|item| matches!(item,
        ChatItem::FunctionCallOutput(output)
            if output.id == call.output_id() && output.output == "{\"reference\":\"accepted\"}"
    )).count(), 1);
    assert_eq!(root.history.iter().filter(|item| item.id() == heard.id).count(), 1);
    assert!(root.history.iter().any(|item| matches!(item,
        ChatItem::Message(message) if message == &heard
    )));
    assert_eq!(
        resumed.session.state.chat_history.iter().filter(|message| message.id == heard.id).count(),
        1
    );
    assert!(
        resumed
            .effects
            .iter()
            .filter_map(|effect| match effect {
                SessionEffect::InvocationStarted { frame_id, .. } => Some(frame_id),
                _ => None,
            })
            .all(|id| id == &resumed.session.state.root_frame_id)
    );
    assert!(
        !resumed.effects.iter().any(|effect| matches!(
            effect,
            SessionEffect::Run { effect: agent_run::AgentRunEffect::ToolPrepare { .. }, .. }
        )),
        "the retained accepted output never starts the business tool again"
    );
}

#[test]
fn a_parked_call_map_uses_chat_order_after_reconstruction_and_defers_new_input_once() {
    let config = config(false, ChatHistory::Fresh, false);
    let mut driver = Driver::new(config.clone(), None, "Book three rooms");
    driver.auto_release = false;
    driver.reply(
        "",
        &[
            ("book", json!({"nights":2})),
            ("book", json!({"nights":4})),
            ("book", json!({"nights":7})),
        ],
    );
    let calls: Vec<_> = driver
        .effects
        .iter()
        .filter_map(|effect| match effect {
            SessionEffect::Run {
                effect: agent_run::AgentRunEffect::ToolPrepare { call, arguments, .. },
                ..
            } => Some((*call, Value::Object(arguments.clone()))),
            _ => None,
        })
        .collect();
    for (call, arguments) in calls.into_iter().rev() {
        driver.run(agent_run::AgentRunEvent::ToolPrepared {
            call,
            prepared: ResourceId(100 + call.0),
            identity: ArgumentIdentity::Json(arguments),
        });
    }
    let mut saved = driver.session.snapshot();
    assert_eq!(saved.active_frame_id, saved.root_frame_id);
    let root = saved.frames.get_mut(&saved.root_frame_id).unwrap();
    assert_eq!(root.pending_calls.len(), 3);
    root.pending_calls.reverse();
    drop(driver);

    let mut resumed = Driver::new(config, Some(saved), "Also add breakfast");
    for (nights, reference) in [(2, "first"), (4, "second"), (7, "third")] {
        assert_eq!(
            resumed
                .request()
                .1
                .items
                .last()
                .unwrap()
                .as_message()
                .unwrap()
                .raw_text_content()
                .as_deref(),
            Some(json!({"nights":nights}).to_string().as_str())
        );
        resumed.accepted(json!({"reference":reference}));
    }
    let root = &resumed.session.state.frames[&resumed.session.state.root_frame_id];
    assert!(root.pending_calls.is_empty());
    assert_eq!(
        root.history
            .iter()
            .filter(|item| item
                .as_message()
                .is_some_and(
                    |message| message.raw_text_content().as_deref() == Some("Also add breakfast")
                ))
            .count(),
        1
    );
    assert_eq!(resumed.session.state.frames.len(), 1);
    assert_eq!(
        resumed
            .request()
            .1
            .items
            .iter()
            .filter_map(|item| match item {
                ChatItem::FunctionCallOutput(output) => Some(output.output.clone()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        ["{\"reference\":\"first\"}", "{\"reference\":\"second\"}", "{\"reference\":\"third\"}"]
    );
}
