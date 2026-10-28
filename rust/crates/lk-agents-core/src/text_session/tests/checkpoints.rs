//! Yielding a child preserves its originating parent call and active frame.

use super::*;

fn verdict(driver: &mut Driver, yielded: bool) {
    let checkpoint = driver
        .effects
        .iter()
        .rev()
        .find_map(|effect| match effect {
            Effect::Run { effect: agent_run::Effect::HostCheckpoint { checkpoint }, .. } => {
                Some(*checkpoint)
            }
            _ => None,
        })
        .expect("the active frame awaits a checkpoint");
    driver.run(agent_run::Event::HostCheckpointDone { checkpoint, result: Ok(yielded) });
}

#[test]
fn a_yielded_child_remains_active_without_resolving_its_parent_call() {
    let config = config(false, ChatHistory::Fresh, false);
    let mut driver = Driver {
        session: TextSession::new(config.clone(), input(None)).unwrap().with_checkpoints(),
        config,
        effects: Vec::new(),
        auto_release: true,
    };
    driver.send(Event::Start(message("human", Role::User, "Book a room")));
    verdict(&mut driver, false);
    driver.reply("", &[("book", json!({"nights": 2}))]);
    driver.prepare(json!({"nights": 2}));
    verdict(&mut driver, true);

    let turn = driver.turn();
    assert!(turn.result.interrupted);
    assert_ne!(turn.state.active_frame_id, turn.state.root_frame_id);
    let active = &turn.state.frames[&turn.state.active_frame_id];
    assert!(!active.completed);
    let parent_call = active.parent.as_ref().expect("the originating call remains");
    let parent = &turn.state.frames[&parent_call.frame_id];
    assert!(parent.pending_calls.iter().any(|pending| pending.parent == *parent_call
        && pending.status == ChildCallStatus::Running { frame_id: active.id.clone() }));
    assert!(!parent.history.iter().any(|item| matches!(item,
        ChatItem::FunctionCallOutput(output) if output.call_id == parent_call.call_id
    )));
    assert_eq!(
        driver
            .effects
            .iter()
            .filter(|effect| matches!(
                effect,
                Effect::Run { effect: agent_run::Effect::LlmRequestStart { .. }, .. }
            ))
            .count(),
        1
    );
}
