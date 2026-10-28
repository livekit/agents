use super::*;

/// A conversation that keeps every rule and reached what `needs` names.
fn reached_run(needs: &[&str]) -> Trace {
    (0..600)
        .map(|seed| run(seed, Sampling::BasicPos))
        .find(|o| o.violations.is_empty() && needs.iter().all(|r| o.reached.contains(r)))
        .unwrap_or_else(|| panic!("a seed reaches {needs:?}"))
        .trace
}

fn breaks(needs: &[&str], rule: fn(&Trace) -> Vec<String>, mutate: impl FnOnce(&mut Trace)) {
    let mut trace = reached_run(needs);
    assert!(rule(&trace).is_empty(), "{:?}", rule(&trace));
    mutate(&mut trace);
    assert!(!rule(&trace).is_empty(), "the counterexample passed");
}

fn finished_turn(t: &mut Trace) -> &mut Turn {
    t.turns.iter_mut().find(|t| t.after.is_some()).expect("a successful turn")
}

#[test]
fn a_turn_ending_twice_or_succeeding_after_a_cancel_is_rejected() {
    breaks(&["TextSessionCompleted"], finishes_once, |t| {
        let turn = finished_turn(t);
        let end = turn.effects.iter().find(|e| matches!(e, Effect::Finished(_))).unwrap().clone();
        turn.effects.push(end);
    });
    breaks(&["TextSessionCompleted"], finishes_once, |t| finished_turn(t).cancelled = true);
}

#[test]
fn an_invocation_ending_twice_is_rejected() {
    breaks(&["TextSessionChildStarted"], invocations_end_once, |t| {
        let turn = t
            .turns
            .iter_mut()
            .find(|t| t.effects.iter().any(|e| matches!(e, Effect::InvocationSettled { .. })))
            .unwrap();
        let settled = turn
            .effects
            .iter()
            .find(|e| matches!(e, Effect::InvocationSettled { .. }))
            .unwrap()
            .clone();
        turn.effects.push(settled);
    });
}

#[test]
fn a_chain_deeper_than_the_bound_is_rejected() {
    breaks(&["TextSessionChildCompleted"], depth_bounded, |t| {
        let turn = t
            .turns
            .iter_mut()
            .find(|t| t.after.as_ref().is_some_and(|s| s.frames.len() > 1))
            .unwrap();
        let state = turn.after.as_mut().unwrap();
        let child = state.frames.values().find(|f| f.parent.is_some()).unwrap().clone();
        // nine frames, each the child of the one before
        let mut parent = child.parent.clone().unwrap();
        for k in 0..MAX_DEPTH {
            let mut frame = child.clone();
            frame.id.value = format!("{}-deep{k}", child.id.value);
            frame.parent = Some(parent.clone());
            parent.frame_id = frame.id.clone();
            state.frames.insert(frame.id.clone(), frame);
        }
    });
}

#[test]
fn a_parent_call_resolved_twice_is_rejected() {
    breaks(&["TextSessionChildCompleted"], parent_resolved_once, |t| {
        let turn = t
            .turns
            .iter_mut()
            .find(|t| t.after.as_ref().is_some_and(|s| s.frames.values().any(|f| f.completed)))
            .unwrap();
        let state = turn.after.as_mut().unwrap();
        let call = state.frames.values().find(|f| f.completed).unwrap().parent.clone().unwrap();
        let parent = state.frames.get_mut(&call.frame_id).unwrap();
        let output = parent
            .history
            .iter()
            .find(|i| matches!(i, ChatItem::FunctionCallOutput(o) if o.call_id == call.call_id))
            .unwrap()
            .clone();
        parent.history.push(output);
    });
}

#[test]
fn a_child_started_without_the_persons_approval_is_rejected() {
    breaks(
        &["TextSessionApprovalAsked", "TextSessionChildStarted"],
        approval_precedes_child,
        |t| {
            let started = t.turns.iter().position(|turn| {
            turn.before.as_ref().is_some_and(|s| s.frames.values().any(|f| f.pending_calls.iter().any(|c| c.status == ChildCallStatus::Approval)))
                && turn.effects.iter().any(|e| matches!(e, Effect::InvocationStarted { frame_id, .. } if !frame_id.value.ends_with("/frame_0")))
        });
            let turn = &mut t.turns[started.expect("an approved child")];
            for frame in turn.before.as_mut().unwrap().frames.values_mut() {
                frame.pending_calls.clear();
            }
        },
    );
}

#[test]
fn a_fresh_child_with_the_parents_dialogue_is_rejected() {
    let mut trace = (0..600)
        .map(|seed| run(seed, Sampling::BasicPos))
        .find(|o| {
            o.violations.is_empty()
                && o.trace.setup.history == ChatHistory::Fresh
                && o.reached.contains("TextSessionChildStarted")
        })
        .expect("a fresh child")
        .trace;
    assert!(history_mode(&trace).is_empty());
    let (n, turn) = trace.turns.iter_mut().enumerate().find(|(_, t)| t.effects.iter().any(|e| matches!(e, Effect::InvocationStarted { frame_id, .. } if !frame_id.value.ends_with("/frame_0")))).unwrap();
    for e in &mut turn.effects {
        if let Effect::InvocationStarted { frame_id, history, .. } = e
            && !frame_id.value.ends_with("/frame_0")
        {
            history.insert(
                0,
                ChatItem::Message(ChatMessage::text(
                    format!("human-{n}"),
                    Role::User,
                    "hello",
                    UNIX_EPOCH,
                )),
            );
            break;
        }
    }
    assert!(!history_mode(&trace).is_empty(), "the counterexample passed");
}

#[test]
fn a_seed_replays_its_run() {
    let (a, b) = (run(3, Sampling::BasicPos), run(3, Sampling::BasicPos));
    assert_eq!(
        a.trace.turns.iter().map(|t| t.effects.clone()).collect::<Vec<_>>(),
        b.trace.turns.iter().map(|t| t.effects.clone()).collect::<Vec<_>>()
    );
}
