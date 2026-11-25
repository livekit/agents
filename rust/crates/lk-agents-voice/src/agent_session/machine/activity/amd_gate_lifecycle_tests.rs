//! The AMD gate across a run's edges: a second run, activation while a turn is in its hook,
//! the session's own options, deferred tool replies and the close. On the harness of the parent
//! module.

use super::*;
use crate::agent_session::commands::{ActivityCommand, ToolUpdate, TurnCommand};
use crate::agent_session::options::{AgentSessionOptions, OverrideLayer};
use crate::trace::AnswerDrop;

/// Every observation that names a turn, with the run it went to.
fn turn_observations(g: &Gate) -> Vec<(AmdRunId, TurnId)> {
    g.out
        .iter()
        .filter_map(|e| match e {
            VoiceEffect::Amd {
                run,
                observation:
                    AmdObservation::TurnCommitted { turn, .. }
                    | AmdObservation::TurnEnded { turn, .. }
                    | AmdObservation::ReplyCreated { turn, .. }
                    | AmdObservation::SilenceHold { turn, .. },
            } => Some((*run, *turn)),
            _ => None,
        })
        .collect()
}

/// A turn's observations go only to the run that gated it, and to none once that run detached:
/// a turn `Detach` settled, still in its hook when a second run attaches, reports nothing to the
/// second run.
#[test]
fn turn_observations_go_only_to_the_run_that_gated_the_turn() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let hook = g.last_hook();
    g.detach(allow(), DetachCancel::Keep);
    let second = AmdRunId(2);
    g.attach(second);
    g.answer(AmdAnswer::Activated { run: second, opening: None });
    g.hook_done(hook, HookOutcome::Returned);
    g.only_reply_to("hello");
    g.advance_to(100);
    g.commit(2, "hello again");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    g.answer(AmdAnswer::ReplyDecided { run: second, turn: TurnId(2), decision: allow() });
    g.only_reply_to("hello again");

    let seen = turn_observations(&g);
    assert!(
        seen.iter().all(|(run, turn)| {
            (*run == RUN && *turn == TurnId(1)) || (*run == second && *turn == TurnId(2))
        }),
        "{seen:?}"
    );
    assert!(seen.contains(&(second, TurnId(2))));
    let detached =
        g.observed(RUN).iter().position(|o| matches!(o, AmdObservation::Detached { .. }));
    assert_eq!(detached, Some(g.observed(RUN).len() - 1), "nothing reaches a run after Detached");
}

/// A turn committed before `Activated` is not bound: its allow keeps the entry hold even when
/// the hook returns after activation (`detector.py:1047`: no resume for a turn it did not bind).
/// The decision may come before the hook returns or after.
#[test]
fn an_allow_for_a_turn_committed_before_activation_keeps_the_entry_hold() {
    for decided_in_hook in [true, false] {
        let mut g = Gate::hooked();
        g.attach(RUN);
        g.commit(1, "hello");
        let hook = g.last_hook();
        g.activate(None);
        if decided_in_hook {
            g.decide(1, allow());
            g.hook_done(hook, HookOutcome::Returned);
        } else {
            g.hook_done(hook, HookOutcome::Returned);
            g.decide(1, allow());
        }
        let speech = g.only_reply_to("hello");
        g.advance_to(1_000);
        assert!(!g.authorized(speech), "decided_in_hook={decided_in_hook}: Hold::Entry stays");
        g.detach(allow(), DetachCancel::Keep);
        assert!(g.authorized(speech), "decided_in_hook={decided_in_hook}");
    }
}

/// While attached, AMD's interruptions win over every host layer, a later `override_options`
/// included; at detach the remaining layers decide again.
#[test]
fn a_host_override_entered_during_amd_does_not_turn_interruptions_off() {
    let enabled = |g: &Gate| g.d.session.config.effective().turn.interruption.enabled();
    let mut g = Gate::listening();
    let mut options = AgentSessionOptions::default();
    options.turn.interruption.enabled = Some(false);
    let layer = OverrideLayer { id: 1, options };
    g.send(Command::Turn(TurnCommand::Override(Box::new(layer))));
    assert!(enabled(&g), "AMD keeps interruptions on over a later host layer");
    g.detach(allow(), DetachCancel::Keep);
    assert!(!enabled(&g), "after detach the host layer decides");
    g.send(Command::Turn(TurnCommand::ReleaseOverride(1)));
    assert!(enabled(&g), "and the session's own value comes back with it");
}

impl Gate {
    /// A detached tool reports progress that asks for a reply.
    fn tool_progress(&mut self, text: &str) {
        let mut tool = crate::agent_session::machine::activity::tools::tests::running("run", 1);
        tool.updates = 1;
        self.d.activity.tools.running.push(tool);
        let update =
            ToolUpdate::Progress { message: text.into(), raw: text.into(), reply_required: true };
        self.d.on_tool_update(ts(self.at), "run".into(), 1, "reused".into(), update);
        self.step(None);
    }

    fn tool_renders(&self) -> Vec<crate::tool_executor::Rendering> {
        self.out
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::RenderToolHandlingTemplate { rendering, .. } => Some(*rendering),
                _ => None,
            })
            .collect()
    }
}

/// A tool update that asks for a reply while AMD is attached keeps its output on the record and
/// drops the reply, even when the floor frees only after detach.
#[test]
fn a_tool_update_under_amd_never_gets_its_deferred_reply() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    g.tool_progress("the order shipped");
    assert!(
        g.d.session.chat_ctx.items().any(|item| matches!(item, ChatItem::FunctionCallOutput(_)))
    );
    g.detach(deny(), DetachCancel::Keep);
    g.advance_to(1_000);
    assert!(g.tool_renders().is_empty(), "no deferred reply is rendered after detach");
    assert!(g.d.activity.tools.pending.is_empty());
}

/// A deferred reply whose instructions were rendering when AMD attached is not scheduled when
/// they come back under AMD.
#[test]
fn a_deferred_reply_rendered_before_attach_is_dropped_under_amd() {
    let mut g = Gate::new();
    g.tool_progress("the order shipped");
    let [crate::tool_executor::Rendering::ToolReply { render }] = g.tool_renders()[..] else {
        panic!("the deferred reply renders: {:?}", g.tool_renders());
    };
    g.attach(RUN);
    g.activate(None);
    let from = g.out.len();
    g.d.tool_reply_rendered(ts(g.at), render, Some("Say the order shipped.".into()));
    g.step(None);
    assert!(g.generations_since(from).is_empty(), "no reply under AMD");
    g.detach(allow(), DetachCancel::Keep);
    g.advance_to(1_000);
    assert!(g.generations_since(from).is_empty(), "nor after it");
}

/// A draining close releases the gate locally: a reply held for silence plays during the drain,
/// AMD hears `Detached`, and a turn still undecided keeps its words with no reply.
#[test]
fn a_draining_close_releases_the_gate() {
    let mut g = Gate::listening();
    g.commit(1, "please leave a message");
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("please leave a message");
    assert!(!g.authorized(speech));
    let close = crate::agent_session::commands::Close { drain: true, ..Default::default() };
    let from = g.send(Command::Close(close));
    assert!(g.authorized_since(from, speech), "the held reply plays in the drain");
    assert!(
        g.observed_since(from).iter().any(|o| matches!(o, AmdObservation::Detached { run: RUN })),
        "the run sees its gate released"
    );
    assert!(g.d.activity.gate.is_none());
}

/// A draining close with a turn parked on its decision keeps the turn's words and answers
/// nothing.
#[test]
fn a_draining_close_abandons_a_parked_turn() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    let close = crate::agent_session::commands::Close { drain: true, ..Default::default() };
    g.send(Command::Close(close));
    g.advance_to(30_000);
    assert!(g.generations().is_empty());
    assert_eq!(g.ended(1), [TurnEnd::Abandoned]);
    assert!(g.saved_user_words().contains(&"hello".to_string()));
}

/// A decision for a turn that already ended without one (`skip_reply`, `StopResponse`, a drop)
/// is dropped as `TurnEnded`, not as a duplicate.
#[test]
fn a_decision_after_the_turn_ended_is_dropped_as_ended() {
    let mut g = Gate::listening();
    let mut turn = info(1, "one moment");
    turn.skip_reply = true;
    g.commit_info(turn);
    g.decide(1, allow());
    let why = g.gate_steps().into_iter().find_map(|s| match s {
        GateStep::AnswerDropped { turn: Some(TurnId(1)), why, .. } => Some(*why),
        _ => None,
    });
    assert_eq!(why, Some(AnswerDrop::TurnEnded));

    g.commit(2, "hello");
    g.decide(2, allow());
    g.decide(2, deny());
    let why = g.gate_steps().into_iter().find_map(|s| match s {
        GateStep::AnswerDropped { turn: Some(TurnId(2)), why, .. } => Some(*why),
        _ => None,
    });
    assert_eq!(why, Some(AnswerDrop::Duplicate), "a second decision is a duplicate");
}

/// An opening the scheduler refuses (scheduling paused by a host drain) does not bypass the hold
/// and is not recorded as the opening.
#[test]
fn a_refused_opening_records_no_bypass() {
    let mut g = Gate::new();
    g.attach(RUN);
    g.send(Command::Activity(ActivityCommand::Drain { call: None }));
    g.activate(Some(Opening::Text("Hi, this is Ada.".into())));
    assert!(
        !g.gate_steps().iter().any(|s| matches!(s, GateStep::OpeningBypass { .. })),
        "{:?}",
        g.gate_steps()
    );
    assert!(g.gate_steps().iter().any(|s| matches!(s, GateStep::OpeningDropped { .. })));
}
