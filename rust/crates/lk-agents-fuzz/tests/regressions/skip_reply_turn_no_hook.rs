//! A turn committed with `skip_reply` keeps its words without `on_user_turn_completed`.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, TurnMode, UserText};
use lk_agents_voice::agent_session::HookKind;
use lk_agents_voice::trace::{HandoffStep, Step};

// Catches: a turn that asks for no reply (`commit_user_turn(skip_reply=True)`, or one AMD
// consumed) was parked on the developer's `on_user_turn_completed` like any other, its item
// committed only once the hook answered and stamped with its delay. The reference commits the
// item and returns before the hook (found by `parity.py`'s payload comparison: the AMD scenario's
// user item carried an `on_user_turn_completed_delay` the reference's has not).
#[test]
fn a_skip_reply_turn_runs_no_hook() {
    let setup =
        Setup { stt: true, turn: TurnMode::Manual, hook_bound_ms: Some(1_000), ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let inputs = [
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("yeah hi".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(20) },
            Input::CommitUserTurn {
                transcript_timeout: Millis(100),
                stt_flush: Millis(0),
                skip_reply: true,
            },
            Input::Advance { ms: Millis(2_000) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let hooks = run
        .record
        .journal()
        .iter()
        .filter(|(_, s)| {
            matches!(
                s,
                Step::Handoff(HandoffStep::HookIssued { kind: HookKind::OnUserTurnCompleted, .. })
            )
        })
        .count();
    assert_eq!(hooks, 0, "on_user_turn_completed ran for a turn that asks for no reply");
    let users = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter(|(_, ev)| {
            matches!(ev, lk_agents_voice::events::SessionEvent::ConversationItemAdded { item, .. }
                if matches!(&**item, ChatItem::Message(m) if m.role == Role::User))
        })
        .count();
    assert_eq!(users, 1, "the turn's words reach the record");
}
