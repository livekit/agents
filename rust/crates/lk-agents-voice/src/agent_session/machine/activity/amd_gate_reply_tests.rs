//! The AMD gate's activation, opening, completion, attached-session rules, reply payload and
//! observations, on the harness of the parent module.

use super::*;

// ---- Activation and entry (proposal "Apply a decision", "Activation and the opening") ----

/// Row 1 while `Pending` allows the reply but keeps `Hold::Entry` (`detector.py:1047`): the
/// reply is made and held until the gate frees authorization.
#[test]
fn an_allow_before_activation_keeps_the_entry_hold() {
    let mut g = Gate::new();
    g.attach(RUN);
    g.commit(1, "hello");
    g.decide(1, allow());
    let speech = g.only_reply_to("hello");
    assert!(!g.authorized(speech), "Hold::Entry stays");
    g.advance_to(5_000);
    assert!(!g.authorized(speech));
    g.detach(allow(), DetachCancel::Keep);
    assert!(g.authorized(speech), "Detach frees authorization");
}

/// The opening plays through the hold (spec §10.6). Python:
/// `test_opening_text_plays_while_other_replies_stay_held`.
#[test]
fn the_opening_plays_while_turns_stay_held() {
    use crate::agent_session::machine::activity::scheduling::SpeechKind;
    let mut g = Gate::new();
    g.attach(RUN);
    g.activate(Some(Opening::Text("Hi, this is Ada from the clinic.".into())));
    let opening = g.created(SpeechKind::Say);
    assert_eq!(opening.len(), 1, "the opening is created at activation");
    assert!(g.authorized(opening[0]), "the opening bypasses the hold");
    g.commit(1, "hello");
    assert!(g.reply_to("hello").is_none(), "a turn during the opening waits for its decision");
}

/// Opening instructions generate with `tool_choice = none` (spec §10.6). Python:
/// `test_opening_instructions_generate_a_reply_without_tools`.
#[test]
fn opening_instructions_generate_without_tools() {
    let mut g = Gate::new();
    g.attach(RUN);
    let from = g.activate(Some(Opening::Instructions("Greet the callee.".into())));
    let generated = g.generations_since(from);
    assert_eq!(generated.len(), 1, "the opening generates once");
    let (_, ctx) = g.generations().into_iter().find(|(id, _)| *id == generated[0]).unwrap();
    assert_eq!(ctx.tool_choice, Some(lk_agents_abi::ToolChoice::Forbidden));
    assert!(g.request_texts(generated[0]).iter().any(|t| t.contains("Greet the callee.")));
    assert!(g.authorized(generated[0]));
}

/// A participant speaking when `Activated` is applied drops the opening (spec §10.6): it is a
/// local fact, so a late `Activated` cannot let a stale opening play. Participant audio is
/// discarded while AMD is pending (spec §5.2), so the participant can be speaking at activation
/// only if they already were when AMD attached. Python:
/// `test_participant_speech_before_the_opening_plays_cancels_it`.
#[test]
fn an_opening_is_dropped_while_the_participant_speaks() {
    use crate::agent_session::machine::activity::scheduling::SpeechKind;
    for speaking in [false, true] {
        let mut g = Gate::new();
        g.advance_to(100);
        if speaking {
            g.vad_start(100);
        }
        g.attach(RUN);
        g.activate(Some(Opening::Text("Hi, this is Ada.".into())));
        let created = g.created(SpeechKind::Say).len();
        assert_eq!(created, usize::from(!speaking), "speaking={speaking}");
    }
}

/// Start of speech before the opening's first playout cancels it; after it, it does not.
#[test]
fn speech_before_the_openings_first_frame_cancels_it() {
    let mut g = Gate::new();
    g.attach(RUN);
    g.activate(Some(Opening::Text("Hi, this is Ada.".into())));
    let opening = g.one_opening();
    g.advance_to(100);
    g.vad_start(50);
    assert!(g.cancelled(opening));

    let mut g = Gate::new();
    g.attach(RUN);
    g.activate(Some(Opening::Text("Hi, this is Ada.".into())));
    let opening = g.one_opening();
    g.first_frame(opening);
    g.advance_to(100);
    g.vad_start(50);
    assert!(
        !g.out.iter().any(|e| matches!(e, VoiceEffect::CancelSpeech(id) if *id == opening)),
        "after its first frame the opening is an ordinary speech"
    );
}

// ---- Detach (proposal "Detach", steps 1-5; spec §12.3) ----

/// Steps 3-5: `pending` decides a parked turn now; the gate goes and `Detached` is emitted in the
/// same transition.
#[test]
fn detach_settles_a_parked_turn_and_acknowledges_at_once() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    let from = g.detach(allow(), DetachCancel::Keep);
    let speech = g.only_reply_to("hello");
    assert!(g.authorized(speech));
    assert!(
        g.observed_since(from).iter().any(|o| matches!(o, AmdObservation::Detached { run: RUN })),
        "Detached in the same step"
    );
    assert!(g.gate_steps().iter().any(|s| matches!(s, GateStep::Settled { turn: TurnId(1), .. })));
}

/// `detached_is_prompt`: completion does not wait for a customer hook. The turn in its hook
/// carries the settled decision to `finish_turn`.
#[test]
fn detach_with_a_turn_in_its_hook_does_not_wait_for_it() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let hook = g.last_hook();
    let from = g.detach(allow(), DetachCancel::Keep);
    assert!(
        g.observed_since(from).iter().any(|o| matches!(o, AmdObservation::Detached { run: RUN })),
        "Detached before the hook returns"
    );
    assert!(g.generations().is_empty());
    g.hook_done(hook, HookOutcome::Returned);
    let speech = g.only_reply_to("hello");
    assert!(g.authorized(speech), "the carried allow replies without a gate");
}

/// Completion after a decision is stored but before it is applied replaces it with the "AMD
/// completed" row: here a stage reply becomes a plain reply.
#[test]
fn detach_replaces_a_stored_decision() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let hook = g.last_hook();
    g.decide(1, ReplyDecision { dtmf: true, ..held(SILENCE) });
    g.detach(allow(), DetachCancel::Keep);
    assert!(g.observed(RUN).iter().any(|o| matches!(o, AmdObservation::Detached { run: RUN })));
    g.hook_done(hook, HookOutcome::Returned);
    let speech = g.only_reply_to("hello");
    assert!(!g.request_texts(speech).iter().any(|t| t == STAGE), "the stage row is replaced");
    assert!(!g.tools_of(speech).contains(&SEND_DTMF_TOOL.to_string()));
    assert!(g.authorized(speech), "`pending` never asks for silence");
}

/// A `wait` category at completion denies the turn in its hook (`detector.py:1060`).
/// Python: `test_wait_does_not_block_completion_or_allow_a_reply_afterward`.
#[test]
fn a_denying_pending_settles_the_turn_in_its_hook() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "please hold");
    let hook = g.last_hook();
    g.decide(1, allow());
    g.detach(deny(), DetachCancel::Keep);
    g.hook_done(hook, HookOutcome::Returned);
    assert!(g.generations().is_empty());
    assert_eq!(g.saved_user_words(), ["please hold"]);
}

/// Step 4: a reply held for silence plays at once, for every completion reason. Python:
/// `test_completion_cancels_silence_wait_and_releases_the_held_reply`.
#[test]
fn detach_releases_a_reply_held_for_silence() {
    let mut g = Gate::listening();
    g.advance_to(1_000);
    g.commit(1, "please leave a message");
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("please leave a message");
    g.advance_to(1_200);
    let from = g.detach(allow(), DetachCancel::Keep);
    assert!(g.authorized_since(from, speech), "released by the detach, not the timer");
}

/// Steps 1 and 4: a `cancel` that forbids replies removes the held reply.
#[test]
fn detach_forbidding_replies_removes_a_held_reply() {
    let mut g = Gate::listening();
    g.commit(1, "please leave a message");
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("please leave a message");
    g.detach(deny(), DetachCancel::ForbidReplies);
    assert!(g.cancelled(speech));
    g.advance_to(5_000);
    assert!(!g.authorized(speech));
}

/// Step 1: forbidding replies cancels an authorized reply still generating and an unplayed
/// opening, and keeps a speech that is playing. Python:
/// `test_cancel_pending_speech_includes_preemptive_and_queued_speech`.
#[test]
fn detach_forbidding_replies_keeps_only_what_is_playing() {
    let mut g = Gate::new();
    g.attach(RUN);
    g.activate(Some(Opening::Text("Hi, this is Ada.".into())));
    let opening = g.one_opening();
    g.first_frame(opening);
    g.commit(1, "hello");
    g.decide(1, allow());
    let reply = g.only_reply_to("hello");
    assert!(!g.cancelled(opening));
    g.detach(deny(), DetachCancel::ForbidReplies);
    assert!(g.cancelled(reply), "an unplayed reply is cancelled");
    assert!(
        !g.out.iter().any(|e| matches!(e, VoiceEffect::CancelSpeech(id) if *id == opening))
            && !g.out.iter().any(|e| matches!(e, VoiceEffect::Speech(id, SpeechUpdate::View(v)) if *id == opening && v.interrupted())),
        "a playing speech keeps playing, as `_cancel_pending_speeches` does"
    );
}

/// Step 5: commits after `Detach` are not gated (spec §9.5) and AMD observes nothing more.
#[test]
fn commits_after_detach_are_not_gated() {
    let mut g = Gate::listening();
    g.commit(1, "hello");
    assert!(g.generations().is_empty(), "gated while attached");
    g.detach(allow(), DetachCancel::Keep);
    let first = g.only_reply_to("hello");
    g.advance_to(100);
    let from = g.commit(2, "who is this");
    g.only_reply_to("who is this");
    assert!(g.authorized(first), "the settled turn plays");
    assert!(g.observed_since(from).is_empty(), "no observation after Detached");
}

/// A new run may attach while an old turn is still in its hook; that turn keeps the decision
/// the old run settled and is not the new run's turn.
#[test]
fn a_new_run_attaches_while_an_old_turn_is_in_its_hook() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let hook = g.last_hook();
    g.detach(allow(), DetachCancel::Keep);
    let second = AmdRunId(2);
    g.attach(second);
    g.hook_done(hook, HookOutcome::Returned);
    g.only_reply_to("hello");
    assert!(
        !g.observed(second)
            .iter()
            .any(|o| matches!(o, AmdObservation::TurnCommitted { turn: TurnId(1), .. })),
        "turn 1 belongs to the first run"
    );
    // The new run gates its own turns.
    g.answer(AmdAnswer::Activated { run: second, opening: None });
    g.commit(2, "hello again");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    assert!(g.reply_to("hello again").is_none(), "the new run holds its own turn");
    g.answer(AmdAnswer::ReplyDecided { run: second, turn: TurnId(2), decision: allow() });
    g.only_reply_to("hello again");
}

// ---- Session rules while attached ----

/// `say()` and `generate_reply()` are refused while the gate is attached, except the opening.
#[test]
fn host_say_and_generate_reply_are_refused_while_attached() {
    use crate::agent_session::machine::activity::scheduling::SpeechKind;
    use crate::agent_session::{GenerateReplyRequest, SayText, SpeechCommand};
    let mut g = Gate::listening();
    g.send(Command::Speech(SpeechCommand::Say {
        text: SayText::Text("Hello there.".into()),
        allow_interruptions: None,
        add_to_chat_ctx: true,
        call: None,
    }));
    g.send(Command::Speech(SpeechCommand::GenerateReply(GenerateReplyRequest {
        user_input: None,
        instructions: Some("Say hi.".into()),
        allow_interruptions: None,
        tool_choice: None,
        chat_ctx: None,
        toolset: None,
        tool_names: None,
        on_enter: false,
        call: None,
    })));
    assert!(g.created(SpeechKind::Say).is_empty(), "say() refused");
    assert!(g.created(SpeechKind::GenerateReply).is_empty(), "generate_reply() refused");
}

/// AMD forces interruptions on while attached and restores them at detach.
#[test]
fn interruptions_are_forced_on_and_restored() {
    let interruptible = |g: &Gate, speech: SpeechId| {
        g.out.iter().rev().find_map(|e| match e {
            VoiceEffect::Speech(id, SpeechUpdate::View(view)) if *id == speech => {
                Some(view.allow_interruptions)
            }
            _ => None,
        })
    };
    let mut g = Gate::with(|d| {
        d.session.config.edit(|o, _| o.turn.interruption.enabled = Some(false));
    });
    g.attach(RUN);
    g.activate(None);
    g.commit(1, "hello");
    g.decide(1, allow());
    let gated = g.only_reply_to("hello");
    assert_eq!(interruptible(&g, gated), Some(true), "forced on while attached");
    g.detach(allow(), DetachCancel::Keep);
    g.advance_to(100);
    g.commit(2, "who is this");
    let after = g.only_reply_to("who is this");
    assert_eq!(interruptible(&g, after), Some(false), "restored at detach");
}

// ---- Reply payload (proposal "Reply payload") ----

/// Stage instructions reach the reply's request and never the saved conversation (spec §14
/// invariant 8). Python: `test_stage_instructions_expire_after_first_human_reply`.
#[test]
fn stage_instructions_reach_the_request_and_are_never_saved() {
    let mut g = Gate::listening();
    g.commit(1, "you have reached the screener");
    g.decide(1, staged());
    let speech = g.only_reply_to("you have reached the screener");
    assert!(g.request_texts(speech).iter().any(|t| t == STAGE));
    assert!(!g.persisted_anywhere(STAGE), "stage instructions are never saved");

    // The next reply does not inherit them.
    g.commit(2, "go ahead");
    g.decide(2, allow());
    let next = g.only_reply_to("go ahead");
    assert!(!g.request_texts(next).iter().any(|t| t == STAGE));
}

/// `dtmf` adds the built-in tool to that reply only (spec §11.1). Python:
/// `test_menu_is_observability_only_and_dtmf_tool_is_temporary`.
#[test]
fn dtmf_permission_adds_the_built_in_tool_to_that_reply_only() {
    let mut g = Gate::listening();
    g.commit(1, "press one for sales");
    g.decide(1, ReplyDecision { dtmf: true, ..staged() });
    let ivr = g.only_reply_to("press one for sales");
    assert_eq!(g.tools_of(ivr).iter().filter(|t| *t == SEND_DTMF_TOOL).count(), 1);

    g.commit(2, "hello this is Sam");
    g.decide(2, allow());
    let human = g.only_reply_to("hello this is Sam");
    assert!(!g.tools_of(human).contains(&SEND_DTMF_TOOL.to_string()));
}

/// An agent tool with the same id is kept instead of the built-in one.
#[test]
fn an_agents_own_dtmf_tool_is_kept() {
    use std::sync::Arc;
    let mut g = Gate::with(|d| {
        d.activity.toolset = Arc::new(crate::tool_executor::Toolset {
            specs: vec![lk_agents_abi::ToolSpec {
                name: SEND_DTMF_TOOL.into(),
                description: "the agent's own".into(),
                json_schema: serde_json::json!({"type": "object"}),
                flags: Default::default(),
                provider_tool: None,
            }],
            host: None,
            agent: None,
        });
    });
    g.attach(RUN);
    g.activate(None);
    g.commit(1, "press one for sales");
    assert!(g.generations().is_empty(), "held for its decision");
    g.decide(1, ReplyDecision { dtmf: true, ..staged() });
    let ivr = g.only_reply_to("press one for sales");
    assert!(g.request_texts(ivr).iter().any(|t| t == STAGE));
    let (_, ctx) = g.generations().into_iter().find(|(id, _)| *id == ivr).unwrap();
    let dtmf: Vec<_> = ctx.request.tools.iter().filter(|t| t.name == SEND_DTMF_TOOL).collect();
    assert_eq!(dtmf.len(), 1, "one tool with that id");
    assert_eq!(dtmf[0].description, "the agent's own");
}

// ---- Preemptive generation (proposal "Preemptive generation"; Decisions) ----

impl Gate {
    fn preemptive(&mut self, transcript: &str) -> Vec<SpeechId> {
        let from = self.out.len();
        self.d.on_preemptive_generation(ts(self.at), transcript.into(), None);
        self.step(None);
        self.generations_since(from)
    }
}

/// A direct allow with no instructions and no DTMF adopts the preemptive generation: the head
/// start is kept. Python: `test_preemptive_generation_commits_only_the_accepted_reply`.
#[test]
fn a_plain_allow_adopts_the_preemptive_generation() {
    let mut g = Gate::listening();
    let early = g.preemptive("hello this is Sam");
    assert_eq!(early.len(), 1, "preemptive generation may run while authorization is held");
    assert!(!g.tools_of(early[0]).contains(&SEND_DTMF_TOOL.to_string()));
    assert!(!g.authorized(early[0]));
    g.commit(1, "hello this is Sam");
    assert!(!g.authorized(early[0]), "not before the decision");
    let from = g.decide(1, allow());
    assert!(g.generations_since(from).is_empty(), "adopted, not generated again");
    assert_eq!(g.reply_created_for(1), [early[0]]);
    assert!(g.authorized(early[0]));
}

/// A decision with stage instructions or DTMF discards the preemptive generation and generates
/// again.
#[test]
fn a_staged_allow_discards_the_preemptive_generation() {
    for decision in [staged(), ReplyDecision { dtmf: true, ..allow() }] {
        let mut g = Gate::listening();
        let early = g.preemptive("press one for sales");
        g.commit(1, "press one for sales");
        let from = g.decide(1, decision.clone());
        assert!(g.cancelled(early[0]), "{decision:?}: discarded");
        let fresh = g.generations_since(from);
        assert_eq!(fresh.len(), 1, "{decision:?}: generated again");
        assert_eq!(g.reply_created_for(1), fresh);
        assert!(!g.authorized(early[0]));
    }
}

/// A denied turn discards the preemptive generation.
#[test]
fn a_denied_turn_discards_the_preemptive_generation() {
    let mut g = Gate::listening();
    let early = g.preemptive("please hold");
    g.commit(1, "please hold");
    g.decide(1, deny());
    assert!(g.cancelled(early[0]));
    assert!(!g.authorized(early[0]));
}

// ---- Observations (proposal "Session to AMD") ----

/// Participant speech boundaries carry the VAD's backdated boundary times.
#[test]
fn user_speech_boundaries_are_backdated_vad_times() {
    let mut g = Gate::listening();
    g.advance_to(1_000);
    g.vad_start(300);
    g.advance_to(2_000);
    g.vad_end(400);
    let speech: Vec<_> = g
        .observed(RUN)
        .into_iter()
        .filter_map(|o| match o {
            AmdObservation::UserSpeech(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        speech,
        [
            UserSpeechObservation::Started { at: ts(700) },
            UserSpeechObservation::Ended { at: ts(1_600) },
        ]
    );
}

/// At attach the session reports `AgentBusy` and each speech already queued or playing.
#[test]
fn attach_reports_busy_and_the_speeches_already_live() {
    let mut g = Gate::new();
    g.commit(1, "hello");
    let before = g.only_reply_to("hello");
    let from = g.attach(RUN);
    let at_attach = g.observed_since(from);
    assert!(at_attach.iter().any(|o| matches!(o, AmdObservation::AgentBusy { .. })));
    assert!(
        at_attach.iter().any(|o| matches!(
            o,
            AmdObservation::Speech { speech, event: SpeechObservation::Created { .. }, .. }
                if *speech == before
        )),
        "{at_attach:?}"
    );
}

/// Every speech created while attached is reported with `add_to_chat_ctx`.
#[test]
fn speeches_are_reported_as_they_are_created() {
    let mut g = Gate::listening();
    g.commit(1, "hello");
    g.decide(1, allow());
    let speech = g.only_reply_to("hello");
    assert!(g.observed(RUN).iter().any(|o| matches!(
        o,
        AmdObservation::Speech { speech: s, event: SpeechObservation::Created { add_to_chat_ctx: true, .. }, .. }
            if *s == speech
    )));
}

/// Nothing is observed before attach, and a detached run observes nothing more.
#[test]
fn nothing_is_observed_without_a_gate() {
    let mut g = Gate::new();
    g.commit(1, "hello");
    g.only_reply_to("hello");
    assert!(g.out.iter().all(|e| !matches!(e, VoiceEffect::Amd { .. })));
}

/// The close abandons a parked turn with `Dropped { append_user_message }` semantics: its words
/// are saved, no reply, and `TurnEnded` fires.
#[test]
fn the_close_abandons_a_parked_turn() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    g.send(Command::Close(Default::default()));
    g.advance_to(30_000);
    assert!(g.generations().is_empty());
    assert_eq!(g.ended(1), [TurnEnd::Abandoned]);
    assert!(g.saved_user_words().contains(&"hello".to_string()));
}

/// A turn dropped behind a speech that cannot be interrupted ends as `Dropped` and is not
/// answered later (proposal "Turn exits without a decision applied": the uninterruptible drop).
/// The speech predates the attach, so AMD's forced interruptions do not reach it.
#[test]
fn a_turn_dropped_behind_an_uninterruptible_speech_ends_as_dropped() {
    use crate::agent_session::{SayText, SpeechCommand};
    let mut g = Gate::new();
    g.send(Command::Speech(SpeechCommand::Say {
        text: SayText::Text("Please hold while I connect you.".into()),
        allow_interruptions: Some(false),
        add_to_chat_ctx: true,
        call: None,
    }));
    g.attach(RUN);
    g.activate(None);
    g.commit(1, "hello");
    assert_eq!(g.ended(1), [TurnEnd::Dropped]);
    g.decide(1, allow());
    assert!(g.generations().is_empty(), "a dropped turn is not answered");
}
