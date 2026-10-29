//! `on_user_turn_completed` edits a copy of the driving agent's own context, not the session's
//! history of every agent, and a second turn waits for the first turn's hook.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, HookAnswer, Input, Millis, Ref, TurnMode, UserText};
use lk_agents_fuzz::record::{Host, Kind};
use lk_agents_voice::agent_session::HookKind;
use lk_agents_voice::events::SessionEvent;

fn turn(utterance: u32, words: &str, skip_reply: bool) -> Vec<Input> {
    vec![
        Input::SttStart { stt: Ref::new(0) },
        Input::SttFinal {
            utterance: Ref::new(utterance),
            text: UserText(words.into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(utterance) },
        Input::Advance { ms: Millis(20) },
        Input::CommitUserTurn { transcript_timeout: Millis(100), stt_flush: Millis(0), skip_reply },
        Input::Advance { ms: Millis(1_000) },
    ]
}

/// Every hook still waited on, answered as returning nothing.
async fn answer_hooks(run: &mut Run) {
    for nth in run.record.live().of(Kind::Hook) {
        let answer = Input::HookDone {
            hook: Ref::new(nth),
            answer: HookAnswer::Returned,
            text: AgentText(String::new()),
        };
        run.apply(&answer).await;
    }
    run.apply(&Input::Advance { ms: Millis(100) }).await;
}

fn turn_hooks(run: &Run) -> Vec<Vec<String>> {
    run.record
        .host()
        .iter()
        .filter_map(|(_, h)| match h {
            Host::Hook { kind: HookKind::OnUserTurnCompleted, ctx_ids, .. } => {
                Some(ctx_ids.clone())
            }
            _ => None,
        })
        .collect()
}

// Catches: the host built the hook's `turn_ctx` from `session.history` (every agent's items,
// the switch's records included) and sent all of it back as the reply's context, so a RAG edit
// after a handoff put the session's records into the new agent's prompt.
#[test]
fn the_turn_hook_gets_the_agents_own_context() {
    let setup =
        Setup { stt: true, turn: TurnMode::Manual, hook_bound_ms: Some(1_000), ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        answer_hooks(&mut run).await;
        for input in turn(0, "first agent hears this", true) {
            run.apply(&input).await;
        }
        answer_hooks(&mut run).await;
        let handoff = Input::Handoff {
            agent: "next".into(),
            instructions: AgentText("You are next.".into()),
            tools: vec![],
        };
        run.apply(&handoff).await;
        run.apply(&Input::Advance { ms: Millis(100) }).await;
        answer_hooks(&mut run).await;
        for input in turn(1, "second agent hears this", false) {
            run.apply(&input).await;
        }
        run
    });
    let mut user = Vec::new();
    let mut records = Vec::new();
    for (_, ev) in run.record.host_view().events() {
        if let SessionEvent::ConversationItemAdded { item, .. } = ev {
            match &**item {
                ChatItem::Message(m) if m.role == Role::User => user.push(m.id.clone()),
                ChatItem::AgentHandoff(_) => records.push(item.id().to_owned()),
                _ => {}
            }
        }
    }
    assert!(!user.is_empty(), "the first turn was committed");
    let hooks = turn_hooks(&run);
    assert_eq!(hooks.len(), 1, "the first turn asks for no reply, so no hook");
    // The switch carries the conversation forward, so the first turn is in the agent's view; the
    // session's record of the switch is history, not the agent's context.
    assert!(hooks[0].contains(&user[0]), "the carried conversation: {:?}", hooks[0]);
    assert!(!records.is_empty(), "the switch was recorded");
    assert!(
        records.iter().all(|r| !hooks[0].contains(r)),
        "the hook was handed the session's handoff record: {:?}",
        hooks[0]
    );
}

// Catches: a second turn decided while the first turn's hook ran ended that wait with no edits
// and answered the first turn anyway; the reference awaits the old task ("we never cancel user
// code").
#[test]
fn a_second_turn_waits_for_the_first_turns_hook() {
    let setup =
        Setup { stt: true, turn: TurnMode::Manual, hook_bound_ms: Some(1_000), ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        answer_hooks(&mut run).await;
        for input in turn(0, "first words", false) {
            run.apply(&input).await;
        }
        for input in turn(1, "second words", false) {
            run.apply(&input).await;
        }
        run.apply(&Input::Advance { ms: Millis(20_000) }).await;
        run
    });
    assert_eq!(turn_hooks(&run).len(), 1, "the second turn's hook waits for the first's");
    let replies = run.record.host_view().speeches().len();
    assert_eq!(replies, 0, "no turn is answered while the first hook runs");
}
