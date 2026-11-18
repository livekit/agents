use super::*;
use lk_agents_abi::text_session::LiveTurnOrigin;

fn driver() -> Driver {
    let mut config = config(false, ChatHistory::Fresh, false);
    config.agents.get_mut("desk").unwrap().agent_tools.clear();
    Driver {
        session: Session::new(config.clone(), input(None)).unwrap().with_live(LiveTurnOrigin {
            request_id: "native_original".into(),
            execution_id: "AST_original".into(),
            task_ids: vec!["AST_original".into()],
        }),
        config,
        effects: Vec::new(),
        auto_release: false,
    }
}

fn side() -> Vec<ChatMessage> {
    vec![
        message("opaque_human", Role::User, "When is check-in?"),
        message("opaque_assistant", Role::Assistant, "Check-in is at 3pm."),
    ]
}

#[test]
fn opaque_dialogue_stays_provisional_until_its_exact_write_acknowledgment() {
    let mut driver = driver();
    driver.send(SessionEvent::Start(message("original", Role::User, "Reserve a room.")));
    let preview = driver.session.prepare_live_dialogue("opaque_source".into(), side()).unwrap();
    assert!(preview.chat_history.iter().any(|item| item.id == "opaque_assistant"));
    assert!(!driver.session.state.chat_history.iter().any(|item| item.id == "opaque_assistant"));
    assert!(driver.session.live_dialogue_pending());
    assert!(driver.session.commit_live_dialogue("another_source", Ts::ZERO).is_err());
    assert!(driver.session.live_dialogue_pending());
    driver.session.commit_live_dialogue("opaque_source", Ts::ZERO).unwrap();
    assert!(!driver.session.live_dialogue_pending());
    assert!(driver.session.state.chat_history.iter().any(|item| item.id == "opaque_assistant"));
    assert!(
        driver.session.input_candidates().is_empty(),
        "observed words grant no input occurrence authority"
    );
}

#[test]
fn foreign_context_keeps_own_new_messages_and_cannot_be_credited_to_the_original_task() {
    let mut driver = driver();
    driver.send(SessionEvent::Start(message("original", Role::User, "Reserve a room.")));
    driver.reply("Own reply.", &[]);
    driver.session.prepare_live_dialogue("opaque_source".into(), side()).unwrap();
    driver.session.commit_live_dialogue("opaque_source", Ts::ZERO).unwrap();
    driver.present_text();
    let turn = driver
        .effects
        .iter()
        .rev()
        .find_map(|effect| match effect {
            SessionEffect::LiveTurnFinished { result, .. } => {
                Some(result.as_ref().as_ref().unwrap())
            }
            _ => None,
        })
        .unwrap();
    assert!(
        matches!(&turn.result.outcome,RunOutcome::Completed {output:Output::Text(text)} if text=="Own reply.")
    );
    assert!(turn.result.new_messages.iter().any(|item| item.id() == "original"));
    assert!(turn.result.new_messages.iter().all(|item| !item.id().starts_with("opaque_")));
    assert!(turn.result.all_messages.iter().any(|item| item.id() == "opaque_assistant"));
    assert!(turn.state.chat_history.iter().any(|item| item.id == "opaque_human"));
}

#[test]
fn closing_the_owner_aborts_provisional_dialogue_without_claiming_it_was_delivered() {
    let mut driver = driver();
    driver.send(SessionEvent::Start(message("original", Role::User, "Reserve a room.")));
    driver.session.prepare_live_dialogue("opaque_source".into(), side()).unwrap();
    driver.send(SessionEvent::Cancel);
    assert!(!driver.session.live_dialogue_pending());
    assert!(driver.session.state.chat_history.iter().all(|item| !item.id.starts_with("opaque_")));
    assert!(driver.session.commit_live_dialogue("opaque_source", Ts::ZERO).is_err());
}
