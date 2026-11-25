//! Reply decisions: the spec §10.1 table, stage instructions (§10.2), the likely-human mark
//! (§9.4), stage corrections (§9.3), stages the application handles (§9.5), DTMF and menus
//! (§11). Python's tests are in `tests/test_amd.py`.

use super::harness::*;
use super::stage::LIKELY_HUMAN_STAGES;
use super::*;
use crate::amd::{AmdAnswer, DetachCancel, ReplyDecision};
use serde_json::json;

/// The pending decision of a completion that found a person after a machine stage.
fn human_after_machine(instructions: &str) -> ReplyDecision {
    ReplyDecision {
        allow: true,
        instructions: Some(instructions.to_owned()),
        dtmf: false,
        silence: None,
    }
}

fn instructions_for(stage: Stage) -> &'static str {
    match stage {
        Stage::MachineScreening => DEFAULT_SCREENING_INSTRUCTIONS,
        Stage::MachineVm => DEFAULT_VOICEMAIL_INSTRUCTIONS,
        Stage::MachineIvr => DEFAULT_IVR_INSTRUCTIONS,
        other => panic!("{other:?} has no stage instructions"),
    }
}

/// The allowed reply of a stage, for an `uncertain` prediction or the stage's own.
fn stage_reply(stage: Stage) -> ReplyDecision {
    match stage {
        Stage::Uncertain => plain(),
        Stage::MachineIvr => ivr(DEFAULT_IVR_INSTRUCTIONS),
        other => staged(instructions_for(other)),
    }
}

fn category(stage: Stage) -> Category {
    match stage {
        Stage::Uncertain => Category::Uncertain,
        Stage::MachineScreening => Category::MachineScreening,
        Stage::MachineVm => Category::MachineVm,
        Stage::MachineIvr => Category::MachineIvr,
        Stage::Human => Category::Human,
        Stage::MachineUnavailable => Category::MachineUnavailable,
    }
}

// Rows of spec §10.1.

#[test]
fn test_human_and_initial_uncertain_do_not_wait() {
    // replies outside a machine stage do not wait, including the initial stage (spec §10.3)
    let mut h = H::started(AmdOptions::default());
    let turn = h.turn("Hi.", Category::Uncertain);
    assert_eq!(h.decision(turn), plain());
    assert!(h.detaches().is_empty());

    // a direct human answer replies under the agent's own instructions
    let mut h = H::started(AmdOptions::default());
    let turn = h.turn("Hello, who is this?", Category::Human);
    assert!(h.decisions(turn).is_empty(), "a completing prediction sends no ReplyDecided");
    assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
    let outcome = h.outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::Human, AmdReason::Finished));
}

#[test]
fn test_a_human_at_call_start_completes_at_once() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Hello?", Category::Human);
    assert_eq!(h.outcome().category, Stage::Human);
}

#[test]
fn test_machine_prediction_updates_state_before_reply_silence() {
    for stage in
        [Stage::MachineScreening, Stage::MachineVm, Stage::MachineIvr, Stage::MachineUnavailable]
    {
        let mut h = H::started(AmdOptions::default());
        let turn = h.commit("hello");
        h.advance(ms(200));
        let result = Classification { category: category(stage), inference_duration: ms(200) };
        h.respond(h.request_for(turn).req, Ok(result));

        let prediction = h.last_prediction();
        assert_eq!(prediction.category, category(stage));
        assert_eq!(prediction.stage, stage);
        assert_eq!(prediction.reason, AmdReason::Prediction);
        assert_eq!(prediction.delay, ms(200));
        assert_eq!(prediction.inference_duration, Some(ms(200)));

        if stage == Stage::MachineUnavailable {
            // the call cannot complete: no reply, and every unplayed speech is cancelled
            assert!(h.decisions(turn).is_empty());
            assert_eq!(h.detach(), completed_with(deny(), DetachCancel::ForbidReplies));
            assert_eq!(h.outcome().category, Stage::MachineUnavailable);
        } else {
            // the machine reply waits for `machine_silence_threshold`; the session times it
            assert_eq!(h.decision(turn), stage_reply(stage), "{stage:?}");
            assert!(h.detaches().is_empty());
        }
    }
}

#[test]
fn test_a_zero_silence_threshold_disables_the_hold() {
    // spec §4.1: `machine_silence_threshold` 0 disables the silence hold
    let options = AmdOptions { machine_silence_threshold: Duration::ZERO, ..AmdOptions::default() };
    let mut h = H::started(options);
    let turn = h.turn("Please state your name.", Category::MachineScreening);
    let expected = ReplyDecision { silence: None, ..staged(DEFAULT_SCREENING_INSTRUCTIONS) };
    assert_eq!(h.decision(turn), expected);
}

#[test]
fn test_stage_without_instructions_completes_without_a_reply() {
    for stage in [Stage::MachineScreening, Stage::MachineVm, Stage::MachineIvr] {
        let mut options = AmdOptions::default();
        match stage {
            Stage::MachineScreening => options.screening_instructions = None,
            Stage::MachineVm => options.voicemail_instructions = None,
            _ => options.ivr_instructions = None,
        }
        let mut h = H::started(options);
        let turn = h.turn("hello", category(stage));

        assert!(h.decisions(turn).is_empty(), "{stage:?}");
        assert_eq!(h.detach(), completed_with(deny(), DetachCancel::ForbidReplies), "{stage:?}");
        assert!(h.menu_requests().is_empty(), "{stage:?}: no menu extraction");
        let outcome = h.outcome();
        assert_eq!(outcome.category, stage);
        assert_eq!(outcome.reason, AmdReason::Finished);
        assert_eq!(outcome.last_prediction.map(|p| p.turn_id), Some(1));
        assert!(!outcome.voicemail_message_played);
    }
}

#[test]
fn test_a_correction_into_a_stage_without_instructions_completes() {
    // spec §9.5: a stage correction that enters the stage completes too
    let options = AmdOptions { voicemail_instructions: None, ..AmdOptions::default() };
    let mut h = H::started(options);
    h.turn("Please state your name.", Category::MachineScreening);
    let turn = h.turn("Leave a message after the tone.", Category::MachineVm);
    assert!(h.decisions(turn).is_empty());
    assert_eq!(h.detach(), completed_with(deny(), DetachCancel::ForbidReplies));
    let outcome = h.outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::MachineVm, AmdReason::Finished));
}

// `human_instructions` (spec §10.2).

#[test]
fn test_human_instructions_are_skipped_without_a_machine_stage() {
    for uncertain_first in [false, true] {
        let options = AmdOptions {
            human_instructions: Some("The automated screening has finished.".to_owned()),
            ..AmdOptions::default()
        };
        let mut h = H::started(options);
        if uncertain_first {
            let turn = h.turn("Hi.", Category::Uncertain);
            assert_eq!(h.decision(turn), plain());
        }
        h.turn("Hello, who is this?", Category::Human);
        assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep), "{uncertain_first}");
        assert_eq!(h.outcome().category, Stage::Human);
    }
}

#[test]
fn test_human_instructions_none_keeps_agent_instructions() {
    let options = AmdOptions { human_instructions: None, ..AmdOptions::default() };
    let mut h = H::started(options);
    // voicemail has no likely-human turn, so the first human prediction completes AMD
    let turn = h.turn("Leave a message.", Category::MachineVm);
    assert_eq!(h.decision(turn), staged(DEFAULT_VOICEMAIL_INSTRUCTIONS));
    h.turn("Hello?", Category::Human);
    assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
    assert_eq!(h.outcome().category, Stage::Human);
}

#[test]
fn test_human_instructions_remember_machine_history() {
    for bridge in [Category::Uncertain, Category::Wait] {
        let mut h = H::started(AmdOptions::default());
        let first = h.turn("Please state your name.", Category::MachineScreening);
        assert_eq!(h.decision(first), staged(DEFAULT_SCREENING_INSTRUCTIONS));
        // wait blocks the reply; uncertain keeps the screening stage and its instructions
        let second = h.turn("One moment.", bridge);
        let expected =
            if bridge == Category::Wait { deny() } else { staged(DEFAULT_SCREENING_INSTRUCTIONS) };
        assert_eq!(h.decision(second), expected, "{bridge:?}");
        // the first human after screening is likely human, with no stage instructions
        let third = h.turn("Hello?", Category::Human);
        assert_eq!(h.decision(third), held(false), "{bridge:?}");

        h.turn("Hello, who is this?", Category::Human);
        let pending = human_after_machine(DEFAULT_HUMAN_INSTRUCTIONS);
        assert_eq!(h.detach(), completed_with(pending, DetachCancel::Keep), "{bridge:?}");
    }
}

// The likely-human mark (spec §9.4).

#[test]
fn test_a_first_human_after_a_menu_or_screener_waits_for_the_next_turn() {
    for stage in LIKELY_HUMAN_STAGES {
        let mut h = H::started(AmdOptions::default());
        h.turn("hello", category(stage));
        let second = h.turn("Hi.", Category::Human);

        let held_prediction = h.last_prediction();
        assert_eq!(held_prediction.category, Category::Uncertain, "{stage:?}");
        assert_eq!(held_prediction.stage, stage, "{stage:?}");
        assert_eq!(held_prediction.reason, AmdReason::Prediction);
        // allowed without stage instructions while it may be a person; DTMF stays in a menu
        assert_eq!(h.decision(second), held(stage == Stage::MachineIvr), "{stage:?}");
        assert!(h.detaches().is_empty(), "{stage:?}");

        let third = h.commit("Hello, who is this?");
        let request = h.request_for(third);
        assert_eq!(request.stage, stage);
        assert!(request.likely_human);
        let previous = PreviousPrediction {
            turn_id: 2,
            category: Category::Uncertain,
            reason: AmdReason::Prediction,
        };
        assert_eq!(request.previous_prediction, Some(previous));
        // the model is told the held turn is likely a person
        assert_eq!(state(&request)["previous_prediction"]["category"], json!(LIKELY_HUMAN));

        h.classify(third, Category::Human);
        assert!(h.decisions(third).is_empty());
        let pending = human_after_machine(DEFAULT_HUMAN_INSTRUCTIONS);
        assert_eq!(h.detach(), completed_with(pending, DetachCancel::Keep), "{stage:?}");
        let outcome = h.outcome();
        assert_eq!(outcome.category, Stage::Human);
        assert_eq!(outcome.last_prediction.map(|p| p.turn_id), Some(3));
    }
}

#[test]
fn test_the_likely_human_mark_survives_uncertain_and_clears_on_another_category() {
    // spec §9.4 rule 4: only a valid prediction other than `uncertain` clears the mark
    let mut h = H::started(AmdOptions::default());
    h.turn("Please state your name.", Category::MachineScreening);
    h.turn("Hi.", Category::Human);
    let uncertain = h.turn("Um.", Category::Uncertain);
    assert_eq!(h.decision(uncertain), held(false), "the mark keeps the reply without instructions");
    let probe = h.commit("Hello?");
    assert!(h.request_for(probe).likely_human);
    h.classify(probe, Category::MachineScreening);
    assert_eq!(h.decision(probe), staged(DEFAULT_SCREENING_INSTRUCTIONS));
    let next = h.commit("Who is calling?");
    assert!(!h.request_for(next).likely_human);
}

#[test]
fn test_a_held_human_followed_by_a_voicemail_greeting_is_voicemail() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for sales.", Category::MachineIvr);
    h.turn("Please hold.", Category::Wait);
    h.turn("Hi.", Category::Human);
    let voicemail = h.turn("Leave a message.", Category::MachineVm);
    assert_eq!(h.decision(voicemail), staged(DEFAULT_VOICEMAIL_INSTRUCTIONS));
    assert!(h.detaches().is_empty());
    // a person after the voicemail stage is not held
    h.turn("Hello?", Category::Human);
    assert_eq!(h.outcome().category, Stage::Human);
}

// `wait` (spec §4.4, §10.1).

#[test]
fn test_wait_skips_reply_and_allows_a_fresh_prediction() {
    for stage in [Stage::Uncertain, Stage::MachineScreening, Stage::MachineVm, Stage::MachineIvr] {
        let mut h = H::started(AmdOptions::default());
        if stage != Stage::Uncertain {
            h.turn("hello", category(stage));
        }
        let waiting = h.turn("While you wait, hear about our special offers.", Category::Wait);
        assert_eq!(h.decision(waiting), deny(), "{stage:?}");
        let prediction = h.last_prediction();
        assert_eq!((prediction.category, prediction.stage), (Category::Wait, stage));

        h.advance(secs(2));
        let following = h.commit("hello");
        let request = h.request_for(following);
        assert_eq!(request.stage, stage);
        assert_eq!(listed(&request), spec_recommended(stage));
        h.classify(following, Category::Uncertain);
        // a kept machine stage still answers with its rule
        assert_eq!(h.decision(following), stage_reply(stage), "{stage:?}");
    }
}

#[test]
fn test_fallback_and_reuse_leave_fsm_unchanged() {
    for first in [Category::MachineIvr, Category::Wait] {
        for reason in [AmdReason::InferenceError, AmdReason::InferenceTimeout, AmdReason::Reused] {
            let case = format!("{first:?} then {reason:?}");
            let mut h =
                H::started(AmdOptions { inference_timeout: ms(100), ..AmdOptions::default() });
            let turn = h.turn("hello", first);
            let stage = h.last_prediction().stage;
            let expected =
                if first == Category::Wait { deny() } else { ivr(DEFAULT_IVR_INSTRUCTIONS) };
            assert_eq!(h.decision(turn), expected, "{case}");

            let turn = match reason {
                AmdReason::Reused => h.commit(""),
                AmdReason::InferenceError => {
                    let turn = h.commit("hello");
                    h.respond(h.request_for(turn).req, Err(ClassifyError::InvalidOutput));
                    turn
                }
                _ => {
                    let turn = h.commit("hello");
                    h.advance(ms(100));
                    turn
                }
            };
            let prediction = h.last_prediction();
            assert_eq!((prediction.reason, prediction.stage), (reason, stage), "{case}");
            // only an empty turn repeats a wait; a turn without a prediction may be a person
            let expected = match (first, reason) {
                (Category::Wait, AmdReason::Reused) => deny(),
                (Category::Wait, _) => plain(),
                _ => ivr(DEFAULT_IVR_INSTRUCTIONS),
            };
            assert_eq!(h.decision(turn), expected, "{case}");
        }
    }
}

#[test]
fn test_inference_error_falls_back_and_uncertain_keeps_the_stage() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Leave a message.", Category::MachineVm);
    for (turn_id, result, reason) in [
        (2, Err(ClassifyError::InvalidOutput), AmdReason::InferenceError),
        (
            3,
            Ok(Classification { category: Category::Uncertain, inference_duration: ms(1) }),
            AmdReason::Prediction,
        ),
        (4, Err(ClassifyError::Api), AmdReason::InferenceError),
    ] {
        let turn = h.commit("hello");
        h.respond(h.request_for(turn).req, result);
        let prediction = h.last_prediction();
        assert_eq!((prediction.turn_id, prediction.reason), (turn_id, reason));
        assert_eq!(
            (prediction.category, prediction.stage),
            (Category::Uncertain, Stage::MachineVm)
        );
        // the stage instructions still apply while the stage is kept
        assert_eq!(h.decision(turn), staged(DEFAULT_VOICEMAIL_INSTRUCTIONS), "turn {turn_id}");
    }
    assert!(h.detaches().is_empty());
}

#[test]
fn test_wait_does_not_block_completion_or_allow_a_reply_afterward() {
    for reason in [AmdReason::Timeout, AmdReason::Cancelled] {
        let mut h = H::started(AmdOptions { timeout: secs(5), ..AmdOptions::default() });
        let turn = h.turn("Please hold.", Category::Wait);
        assert_eq!(h.decision(turn), deny());
        if reason == AmdReason::Timeout {
            h.advance(secs(5));
        } else {
            h.step(In::Close);
        }
        // `wait` denies before the "AMD completed" row (Python `detector.py:1060`)
        assert_eq!(h.detach(), completed_with(deny(), DetachCancel::Keep), "{reason:?}");
        assert_eq!(h.outcome().reason, reason);
    }
}

// Stage corrections (spec §9.3).

#[test]
fn test_correction_uses_current_turn_and_persists() {
    let cases = [
        (
            Stage::MachineVm,
            Stage::MachineScreening,
            "Please state your name and why you are calling.",
        ),
        (
            Stage::MachineIvr,
            Stage::MachineScreening,
            "Please state your name and why you are calling.",
        ),
        (Stage::MachineScreening, Stage::MachineIvr, "Press 1 for appointments."),
    ];
    for (initial, corrected, transcript) in cases {
        for bridge in [None, Some(Category::Wait), Some(Category::Uncertain)] {
            let case = format!("{initial:?} -> {bridge:?} -> {corrected:?}");
            let mut h = H::started(AmdOptions::default());
            let first = h.commit("Please record your name.");
            let request = h.request_for(first);
            assert_eq!(request.previous_prediction, None, "{case}");
            assert_eq!(listed(&request), spec_recommended(Stage::Uncertain), "{case}");
            h.classify(first, category(initial));
            assert_eq!(h.decision(first), stage_reply(initial), "{case}");

            let mut previous_turn = 1;
            if let Some(bridge) = bridge {
                let turn = h.turn("One moment.", bridge);
                assert_eq!(h.decisions(turn)[0].allow, bridge != Category::Wait, "{case}");
                previous_turn = 2;
            }
            let current_turn = previous_turn + 1;
            let turn = h.commit(transcript);
            let request = h.request_for(turn);
            assert_eq!(request.stage, initial, "{case}");
            let previous = PreviousPrediction {
                turn_id: previous_turn,
                category: bridge.unwrap_or(category(initial)),
                reason: AmdReason::Prediction,
            };
            assert_eq!(request.previous_prediction, Some(previous), "{case}");
            assert!(!request.recommended.contains(&category(corrected)), "{case}");

            let result =
                Classification { category: category(corrected), inference_duration: ms(30) };
            h.respond(request.req, Ok(result));
            // the correction applies to the current turn
            assert_eq!(h.decision(turn), stage_reply(corrected), "{case}");
            let prediction = h.last_prediction();
            assert_eq!(prediction.reason, AmdReason::Prediction);
            assert_eq!(prediction.stage, corrected, "{case}");
            assert_eq!(prediction.inference_duration, Some(ms(30)));

            // and to later turns
            let human = h.commit("Hello, Sam speaking.");
            let request = h.request_for(human);
            assert_eq!(request.stage, corrected, "{case}");
            let previous = PreviousPrediction {
                turn_id: current_turn,
                category: category(corrected),
                reason: AmdReason::Prediction,
            };
            assert_eq!(request.previous_prediction, Some(previous), "{case}");
            assert!(request.recommended.contains(&Category::Human), "{case}");
            h.classify(human, Category::Human);
            if LIKELY_HUMAN_STAGES.contains(&corrected) {
                assert_eq!(h.last_prediction().category, Category::Uncertain, "{case}");
                h.turn("Hello?", Category::Human);
            }
            assert_eq!(h.outcome().category, Stage::Human, "{case}");
        }
    }
}

#[test]
fn test_split_screening_rollover_retains_all_turns() {
    let mut h = H::started(AmdOptions::default());
    let history = [
        ("Please state your name and why you are calling.", Category::MachineScreening),
        ("Okay.", Category::Uncertain),
        ("They can't take the call.", Category::Uncertain),
        ("Feel free to leave a message.", Category::MachineVm),
    ];
    for (index, (transcript, category)) in history.iter().enumerate() {
        let turn = h.commit(transcript);
        let earlier: Vec<String> =
            user_turns(&h.request_for(turn)).into_iter().map(|(_, text)| text).collect();
        let expected: Vec<String> = history[..=index].iter().map(|(t, _)| t.to_string()).collect();
        assert_eq!(earlier, expected);
        h.classify(turn, *category);
    }
    let third = &h.predictions()[2];
    assert_eq!((third.category, third.stage), (Category::Uncertain, Stage::MachineScreening));
    assert_eq!(h.last_prediction().stage, Stage::MachineVm);
}

// Unbound turns and agent changes (spec §7.1, §10.1 rows 1 and 2).

#[test]
fn test_a_turn_before_listening_starts_gets_the_unbound_row() {
    // AMD is pending: the turn is not bound, and the first row allows it at once
    let mut h = H::new(AmdOptions::default());
    h.observe(crate::amd::AmdObservation::AgentBusy { busy: false, at: h.now });
    let early = h.commit("hello");
    assert_eq!(h.decision(early), plain());
    assert!(h.requests().is_empty());
    assert!(h.predictions().is_empty());

    h.step(In::Callee(Callee::Answered));
    let turn = h.commit("Please state your name.");
    assert_eq!(h.request_for(turn).turn_id, 1, "an unbound turn takes no AMD turn id");
}

#[test]
fn test_a_turn_the_gate_has_not_bound_gets_the_unbound_row() {
    // the run is active, but the gate committed the turn before it applied `Activated`: the
    // gate never holds for it, so AMD answers with the first row and keeps no turn
    let mut h = H::started(AmdOptions::default());
    let early = h.commit_with(Commit { unbound: true, ..Commit::text("hello") });
    assert_eq!(h.decision(early), plain());
    assert!(h.requests().is_empty());
    assert!(h.predictions().is_empty());

    let turn = h.commit("Please state your name.");
    assert_eq!(h.request_for(turn).turn_id, 1, "an unbound turn takes no AMD turn id");
    h.classify(turn, Category::MachineScreening);
    assert_eq!(h.decision(turn), staged(DEFAULT_SCREENING_INSTRUCTIONS));
    // the row still denies once replies are forbidden or the call waits
    let waiting = h.turn("Please hold.", Category::Wait);
    assert_eq!(h.decision(waiting), deny());
    let early = h.commit_with(Commit { unbound: true, ..Commit::text("hello") });
    assert_eq!(h.decision(early), deny());
}

#[test]
fn test_a_turn_after_the_overall_deadline_completes_with_timeout() {
    let mut h = H::started(AmdOptions { timeout: secs(5), ..AmdOptions::default() });
    // the deadline passed but its timer has not fired yet
    h.skip(secs(6));
    let turn = h.commit("hello");
    assert!(h.requests().is_empty());
    assert!(h.decisions(turn).iter().all(|d| *d == plain()), "row 1 allows the turn");
    assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
    assert_eq!(h.outcome().reason, AmdReason::Timeout);
}

#[test]
fn test_a_turn_under_another_agent_completes_with_agent_changed() {
    let mut h = H::started(AmdOptions::default());
    let turn = h.commit_with(Commit { agent: "transfer".to_owned(), ..Commit::text("hello") });
    // the turn is classified first: Python finds the change while it waits for the prediction
    h.classify(turn, Category::MachineScreening);
    assert_eq!(h.predictions().len(), 1);
    assert!(h.decisions(turn).is_empty());
    assert_eq!(h.detach(), completed_with(deny(), DetachCancel::Keep));
    let outcome = h.outcome();
    assert_eq!(outcome.reason, AmdReason::AgentChanged);
    assert!(outcome.is_error());
}

// Completion rows (spec §10.1 "AMD completed", §12).

#[test]
fn test_terminal_result_completes_once_and_releases_turn_hooks() {
    for terminal in [Category::Human, Category::MachineUnavailable] {
        let mut h = H::started(AmdOptions::default());
        h.turn("hello", terminal);
        let outcome = h.outcome();
        assert_eq!(outcome.reason, AmdReason::Finished);
        h.step(In::Close);
        h.finish();
        assert_eq!(h.completions().len(), 1, "{terminal:?}");
        let AmdAnswer::Detach { pending, .. } = h.detach() else { unreachable!() };
        assert_eq!(pending.allow, terminal == Category::Human);
    }
}

#[test]
fn test_late_unavailable_does_not_cancel_reply() {
    let mut h = H::started(AmdOptions { inference_timeout: ms(20), ..AmdOptions::default() });
    h.turn("Please state your name.", Category::MachineScreening);
    let turn = h.commit("hello");
    let req = h.request_for(turn).req;
    h.advance(ms(20));
    assert_eq!(h.last_prediction().reason, AmdReason::InferenceTimeout);
    assert_eq!(h.decision(turn), staged(DEFAULT_SCREENING_INSTRUCTIONS));

    let unavailable =
        Classification { category: Category::MachineUnavailable, inference_duration: ms(30) };
    h.respond(req, Ok(unavailable));
    assert!(h.detaches().is_empty());
    assert_eq!(h.predictions().len(), 2);
    let next = h.commit("hello");
    assert_eq!(h.request_for(next).stage, Stage::MachineScreening);
}

#[test]
fn test_nonterminal_finish_releases_a_waiter_without_a_prediction() {
    for reason in [AmdReason::Cancelled, AmdReason::ParticipantDisconnected, AmdReason::Timeout] {
        let options =
            AmdOptions { timeout: secs(3), inference_timeout: secs(10), ..AmdOptions::default() };
        let mut h = H::started(options);
        let turn = h.commit("hello");
        let req = h.request_for(turn).req;
        match reason {
            AmdReason::Cancelled => {
                h.step(In::Close);
            }
            AmdReason::ParticipantDisconnected => {
                h.step(In::Callee(Callee::Disconnected));
            }
            _ => h.advance(secs(3)),
        }
        assert_eq!(h.cancelled(), vec![req], "{reason:?}");
        assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep), "{reason:?}");
        assert!(h.armed_timers().is_empty(), "{reason:?}: {:?}", h.armed_timers());
        let outcome = h.outcome();
        assert_eq!(outcome.reason, reason);
        assert_eq!(outcome.category, Stage::Uncertain);
        assert_eq!(outcome.last_prediction, None);
        assert!(h.predictions().is_empty(), "{reason:?}");
    }
}

// DTMF (spec §11.1) and menus (spec §11.2).

#[test]
fn test_menu_is_observability_only_and_dtmf_tool_is_temporary() {
    let mut h = H::started(AmdOptions::default());
    let turn = h.turn("Press 1 for billing.", Category::MachineIvr);
    assert_eq!(h.decision(turn), ivr(DEFAULT_IVR_INSTRUCTIONS));
    let menus = h.menu_requests();
    assert_eq!(menus.len(), 1);
    assert_eq!((menus[0].turn_id, menus[0].transcript.as_str()), (1, "Press 1 for billing."));
    assert_eq!(h.armed(AmdTimer::Menu(menus[0].req)), Some(h.now.saturating_add(MENU_TIMEOUT)));

    h.advance(secs(1));
    let answers = h.answers().len();
    let billing = IvrMenuOption {
        label: "Billing".to_owned(),
        dtmf: "1".to_owned(),
        ..IvrMenuOption::default()
    };
    let menu = Menu { menu: "Choose a department".to_owned(), options: vec![billing.clone()] };
    h.step(In::MenuExtracted { req: menus[0].req, menu: Some(menu) });
    assert_eq!(
        h.menus(),
        vec![MenuObserved {
            session_id: SESSION.to_owned(),
            turn_id: 1,
            menu: "Choose a department".to_owned(),
            options: vec![billing],
            extraction_duration: secs(1),
        }]
    );
    assert_eq!(h.answers().len(), answers, "a menu never authorizes an action");

    // the DTMF tool belongs to `machine-ivr` replies only
    let screening = h.turn("Please state your name.", Category::MachineScreening);
    assert_eq!(h.decision(screening), staged(DEFAULT_SCREENING_INSTRUCTIONS));
}

#[test]
fn test_same_ivr_state_extracts_each_menu() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for billing.", Category::MachineIvr);
    h.turn("Press 2 for support.", Category::MachineIvr);
    let menus = h.menu_requests();
    assert_eq!(menus.len(), 2);
    assert_ne!(menus[0].req, menus[1].req);
    assert_eq!(h.cancelled_menus(), 1, "the second turn stops the first extraction");
    assert_eq!((menus[1].turn_id, menus[1].transcript.as_str()), (2, "Press 2 for support."));
}

#[test]
fn test_slow_menu_does_not_block_classification_or_next_turn() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for billing.", Category::MachineIvr);
    let req = h.menu_requests()[0].req;
    // a newer turn with text makes the extraction stale and stops it
    let next = h.commit("Hello?");
    assert_eq!(h.requests_for(next), 1);
    assert_eq!(h.cancelled_menus(), 1);
    h.step(In::MenuExtracted {
        req,
        menu: Some(Menu { menu: "Billing".to_owned(), options: vec![] }),
    });
    assert!(h.menus().is_empty());
}

#[test]
fn test_an_empty_turn_does_not_cancel_the_menu() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for billing.", Category::MachineIvr);
    let req = h.menu_requests()[0].req;
    h.commit("");
    assert_eq!(h.cancelled_menus(), 0);
    h.step(In::MenuExtracted {
        req,
        menu: Some(Menu { menu: "Billing".to_owned(), options: vec![] }),
    });
    assert_eq!(h.menus().len(), 1);
}

#[test]
fn test_unexpected_menu_error_does_not_end_detection() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for billing.", Category::MachineIvr);
    let req = h.menu_requests()[0].req;
    h.step(In::MenuExtracted { req, menu: None });
    assert!(h.menus().is_empty());
    assert!(h.detaches().is_empty());
    let next = h.commit("Hello?");
    assert_eq!(h.requests_for(next), 1);
}

#[test]
fn test_only_a_non_empty_menu_is_observed() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for billing.", Category::MachineIvr);
    let req = h.menu_requests()[0].req;
    h.step(In::MenuExtracted { req, menu: Some(Menu::default()) });
    assert!(h.menus().is_empty());

    // options without a menu sentence still make a menu
    h.turn("Billing, one. Support, two.", Category::MachineIvr);
    let req = h.menu_requests()[1].req;
    let options = vec![IvrMenuOption {
        label: "Billing".to_owned(),
        dtmf: "1".to_owned(),
        ..IvrMenuOption::default()
    }];
    h.step(In::MenuExtracted { req, menu: Some(Menu { menu: String::new(), options }) });
    assert_eq!(h.menus().len(), 1);
}

#[test]
fn test_a_menu_after_its_deadline_or_completion_is_dropped() {
    let menu = || Menu { menu: "Choose a department".to_owned(), options: vec![] };

    // at the deadline, the timer first or the result first
    for timer_first in [true, false] {
        let mut h = H::started(AmdOptions::default());
        h.turn("Press 1 for billing.", Category::MachineIvr);
        let req = h.menu_requests()[0].req;
        if timer_first {
            h.advance(MENU_TIMEOUT);
            assert_eq!(h.cancelled_menus(), 1, "the deadline stops the extraction");
        } else {
            // the executor has not fired the timer yet
            h.skip(MENU_TIMEOUT);
        }
        h.step(In::MenuExtracted { req, menu: Some(menu()) });
        h.advance(secs(1));
        assert!(h.menus().is_empty(), "timer first: {timer_first}");
        assert_eq!(h.armed(AmdTimer::Menu(req)), None, "timer first: {timer_first}");
        assert!(h.detaches().is_empty(), "a menu deadline is silent");
    }

    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for billing.", Category::MachineIvr);
    let req = h.menu_requests()[0].req;
    h.step(In::Close);
    assert_eq!(h.cancelled_menus(), 1, "completion stops the extraction");
    h.step(In::MenuExtracted { req, menu: Some(menu()) });
    assert!(h.menus().is_empty(), "after completion");
}

/// The options model allows at most 20 options (`_inference.py`): a menu with more fails
/// validation and is not observed.
#[test]
fn test_a_menu_with_more_than_twenty_options_is_not_observed() {
    let option = |n: usize| IvrMenuOption {
        label: format!("Option {n}"),
        dtmf: n.to_string(),
        ..IvrMenuOption::default()
    };
    for (count, observed) in [(MAX_MENU_OPTIONS, true), (MAX_MENU_OPTIONS + 1, false)] {
        let mut h = H::started(AmdOptions::default());
        h.turn("Press 1 for billing.", Category::MachineIvr);
        let req = h.menu_requests()[0].req;
        let options: Vec<_> = (1..=count).map(option).collect();
        let menu = Menu { menu: "Choose a department".to_owned(), options: options.clone() };
        h.step(In::MenuExtracted { req, menu: Some(menu) });
        let menus = h.menus();
        assert_eq!(menus.len(), usize::from(observed), "{count} options");
        if observed {
            assert_eq!(menus[0].options, options);
        }
    }
}

#[test]
fn test_decision_model_classifies_with_one_choice_and_skips_menus() {
    let setup = AmdSetup { extract_menus: false, ..setup() };
    let mut h = H::started_with(AmdOptions::default(), setup);
    let turn = h.turn("Press 1 for billing.", Category::MachineIvr);
    assert_eq!(h.decision(turn), ivr(DEFAULT_IVR_INSTRUCTIONS));
    assert!(h.menu_requests().is_empty());
}
