//! Options, counters, deadlines and completion (spec §4.1, §5.2, §12). Python's tests are in
//! `tests/test_amd.py`.

use super::harness::*;
use super::*;
use crate::amd::{AmdAnswer, AmdObservation, DetachCancel, Opening, TurnEnd};

// Options (spec §4.1).

#[test]
fn test_default_limits_are_ten_seconds_idle_and_one_minute_overall() {
    let options = AmdOptions::default();
    assert_eq!(options.inference_timeout, secs(2));
    assert_eq!((options.max_uncertain_turns, options.max_inference_timeouts), (3, 3));
    let h = H::started(options);
    assert_eq!(h.answers().first(), Some(&AmdAnswer::Activated { run: RUN, opening: None }));
    // the overall deadline starts at activation
    assert_eq!(h.armed(AmdTimer::Overall), Some(T0.saturating_add(secs(60))));
    assert_eq!(h.armed(AmdTimer::Idle), Some(T0.saturating_add(secs(10))));
}

#[test]
fn test_machine_silence_default_and_validation() {
    assert_eq!(AmdOptions::default().machine_silence_threshold, ms(1500));
    // zero disables the hold; it is not refused
    let zero = AmdOptions { machine_silence_threshold: Duration::ZERO, ..AmdOptions::default() };
    assert!(Amd::new(RUN, zero, setup()).is_ok());
}

#[test]
fn test_options_are_validated() {
    let refused = |options: AmdOptions| Amd::new(RUN, options, setup()).err();
    let base = AmdOptions::default;
    for zero in [
        AmdOptions { idle_timeout: Duration::ZERO, ..base() },
        AmdOptions { voicemail_idle_timeout: Duration::ZERO, ..base() },
        AmdOptions { timeout: Duration::ZERO, ..base() },
        AmdOptions { inference_timeout: Duration::ZERO, ..base() },
    ] {
        assert_eq!(refused(zero), Some(AmdOptionsError::NotPositive));
    }
    assert_eq!(
        refused(AmdOptions { max_uncertain_turns: 0, ..base() }),
        Some(AmdOptionsError::NotPositive)
    );
    assert_eq!(
        refused(AmdOptions { max_inference_timeouts: 0, ..base() }),
        Some(AmdOptionsError::MaxInferenceTimeouts)
    );
    assert_eq!(
        refused(AmdOptions { opening_text: Some("  ".to_owned()), ..base() }),
        Some(AmdOptionsError::EmptyOpeningText)
    );
    assert_eq!(
        refused(AmdOptions { opening_instructions: Some(String::new()), ..base() }),
        Some(AmdOptionsError::EmptyOpeningInstructions)
    );
    assert_eq!(refused(base()), None);
}

// Listening setup (spec §5.2) and activation.

#[test]
fn test_opening_comes_with_activation() {
    let opening = Opening::Text("Hi, this is Alex.".to_owned());
    let options =
        AmdOptions { opening_text: Some("Hi, this is Alex.".to_owned()), ..AmdOptions::default() };
    let mut h = H::started(options);
    h.step(In::Callee(Callee::Answered));
    let activations: Vec<_> =
        h.answers().into_iter().filter(|a| matches!(a, AmdAnswer::Activated { .. })).collect();
    assert_eq!(activations, vec![AmdAnswer::Activated { run: RUN, opening: Some(opening) }]);
}

#[test]
fn test_missing_audio_subscription_times_out_and_releases_session() {
    for (callee, reason) in [
        (Callee::Missing, AmdReason::ParticipantMissing),
        (Callee::Disconnected, AmdReason::ParticipantDisconnected),
    ] {
        let mut h = H::new(AmdOptions::default());
        h.step(In::Callee(callee));
        assert!(!h.answers().iter().any(|a| matches!(a, AmdAnswer::Activated { .. })));
        assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep), "{callee:?}");
        let outcome = h.outcome();
        assert_eq!(outcome.reason, reason);
        assert_eq!(outcome.is_error(), reason == AmdReason::ParticipantMissing);
        assert_eq!((outcome.category, outcome.last_prediction), (Stage::Uncertain, None));
    }
}

#[test]
fn test_completed_amd_cannot_be_entered_again() {
    let mut h = H::new(AmdOptions::default());
    h.step(In::Close);
    assert_eq!(h.outcome().reason, AmdReason::Cancelled);
    h.step(In::Callee(Callee::Answered));
    assert!(!h.answers().iter().any(|a| matches!(a, AmdAnswer::Activated { .. })));
    assert_eq!(h.completions().len(), 1);
}

// Completion (spec §12).

#[test]
fn test_completed_event_is_error_only_when_amd_stopped_without_detecting() {
    use AmdReason::*;
    for reason in [
        Finished,
        MaxUncertainTurns,
        InferenceTimeout,
        IdleTimeout,
        Timeout,
        ParticipantMissing,
        ParticipantDisconnected,
        AgentChanged,
        InternalError,
        Cancelled,
    ] {
        let outcome = Outcome {
            category: Stage::Uncertain,
            reason,
            voicemail_message_played: false,
            last_prediction: None,
        };
        let expected =
            matches!(reason, InternalError | ParticipantMissing | AgentChanged | Cancelled);
        assert_eq!(outcome.is_error(), expected, "{reason:?}");
    }
}

#[test]
fn test_completion_waits_for_detached_and_legs_closed() {
    // `amd_completed` fires after the session removed the gate and the legs closed (either order)
    for detached_first in [true, false] {
        let mut h = H::started(AmdOptions::default());
        h.turn("Hello?", Category::Human);
        assert_eq!(h.detaches().len(), 1);
        assert!(h.completions().is_empty());
        if detached_first {
            h.observe(AmdObservation::Detached { run: RUN });
        } else {
            h.step(In::LegsClosed);
        }
        assert!(h.completions().is_empty(), "detached first: {detached_first}");
        if detached_first {
            h.step(In::LegsClosed);
        } else {
            h.observe(AmdObservation::Detached { run: RUN });
        }
        assert_eq!(h.completions().len(), 1, "detached first: {detached_first}");
    }
}

#[test]
fn test_session_close_completes_amd_and_releases_resources() {
    let mut h = H::started(AmdOptions::default());
    let turn = h.turn("Please state your name.", Category::MachineScreening);
    h.replied(turn);
    h.silence_held(turn);
    h.step(In::Close);
    h.step(In::Close);
    // the held reply plays at once (spec §10.3)
    assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
    let outcome = h.outcome();
    assert_eq!((outcome.reason, outcome.category), (AmdReason::Cancelled, Stage::MachineScreening));
}

#[test]
fn test_unexpected_inference_errors_finish_with_internal_error() {
    let mut h = H::started(AmdOptions::default());
    h.turn("Press 1 for billing.", Category::MachineIvr);
    let turn = h.commit("Press 2 for support.");
    let req = h.request_for(turn).req;
    h.step(In::LegFailed);
    assert_eq!(h.cancelled(), vec![req]);
    let outcome = h.outcome();
    assert_eq!((outcome.reason, outcome.category), (AmdReason::InternalError, Stage::MachineIvr));
}

#[test]
fn test_invalid_model_output_falls_back_and_releases_the_reply() {
    let mut h = H::started(AmdOptions::default());
    let turn = h.commit("hello");
    h.respond(h.request_for(turn).req, Err(ClassifyError::InvalidOutput));
    let prediction = h.last_prediction();
    assert_eq!(prediction.reason, AmdReason::InferenceError);
    assert_eq!((prediction.category, prediction.stage), (Category::Uncertain, Stage::Uncertain));
    assert_eq!(prediction.inference_duration, None);
    assert_eq!(h.decision(turn), plain());
    assert!(h.detaches().is_empty());
    assert!(h.armed(AmdTimer::Idle).is_some());
}

// `max_uncertain_turns` (spec §12.1).

#[test]
fn test_three_uncertain_predictions_complete_detection() {
    let mut h = H::started(AmdOptions::default());
    for turn_id in 1..=3 {
        h.turn("Hi.", Category::Uncertain);
        assert_eq!(h.detaches().is_empty(), turn_id < 3, "turn {turn_id}");
    }
    assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
    let outcome = h.outcome();
    assert_eq!(
        (outcome.reason, outcome.category),
        (AmdReason::MaxUncertainTurns, Stage::Uncertain)
    );
}

#[test]
fn test_a_likely_human_counts_toward_max_uncertain_turns() {
    // spec §9.4: it can complete with the machine stage as the category
    let mut h = H::started(AmdOptions { max_uncertain_turns: 1, ..AmdOptions::default() });
    h.turn("Please state your name.", Category::MachineScreening);
    assert!(h.detaches().is_empty());
    h.turn("Hi.", Category::Human);
    let outcome = h.outcome();
    assert_eq!(outcome.reason, AmdReason::MaxUncertainTurns);
    assert_eq!(outcome.category, Stage::MachineScreening);
}

#[test]
fn test_wait_resets_inference_timeout_and_uncertain_turn_budgets() {
    for prior in ["uncertain", "timeout"] {
        let options = AmdOptions {
            inference_timeout: ms(50),
            max_inference_timeouts: 2,
            max_uncertain_turns: 2,
            ..AmdOptions::default()
        };
        let mut h = H::started(options);
        let respond = |h: &mut H, turn| {
            if prior == "uncertain" {
                h.classify(turn, Category::Uncertain);
            } else {
                h.advance(ms(50));
            }
        };
        let first = h.commit("Hi.");
        respond(&mut h, first);
        assert_eq!(h.decision(first), plain());
        let waiting = h.turn("Please hold.", Category::Wait);
        assert_eq!(h.decision(waiting), deny());
        let following = h.commit("Hi.");
        respond(&mut h, following);
        // a timeout after a wait is uncertain, so it does not block the reply
        assert_eq!(h.decision(following), plain(), "{prior}");
        assert!(h.detaches().is_empty(), "{prior}");
    }
}

// `inference_timeout` and `max_inference_timeouts` (spec §8.4, §12.1).

#[test]
fn test_inference_timeout_limit_completes_detection() {
    for (limit, expected) in [(None, 3), (Some(1), 1), (Some(4), 4)] {
        let mut options = AmdOptions { inference_timeout: ms(10), ..AmdOptions::default() };
        if let Some(limit) = limit {
            options.max_inference_timeouts = limit;
        }
        let mut h = H::started(options);
        for turn_id in 1..=expected {
            h.commit("hello");
            h.advance(ms(10));
            assert_eq!(h.last_prediction().reason, AmdReason::InferenceTimeout);
            assert_eq!(h.detaches().is_empty(), turn_id < expected, "{limit:?}: turn {turn_id}");
        }
        assert_eq!(h.outcome().reason, AmdReason::InferenceTimeout, "{limit:?}");
    }
}

#[test]
fn test_inference_timeout_limit_waits_for_silence_before_completion() {
    // the limit completes only once no request is pending and no reply is held
    for limit in [1, 3] {
        let options = AmdOptions {
            inference_timeout: ms(200),
            max_inference_timeouts: limit,
            ..AmdOptions::default()
        };
        let mut h = H::started(options);
        let first = h.turn("Please state your name.", Category::MachineScreening);
        h.reply_after_silence(first, "This is Alex.");
        for n in 1..=limit {
            let turn = h.commit("hello");
            h.advance(ms(200));
            assert_eq!(h.last_prediction().reason, AmdReason::InferenceTimeout);
            assert_eq!(h.decision(turn), staged(DEFAULT_SCREENING_INSTRUCTIONS));
            // the gate's order: the reply exists, the turn ends with it, then the hold starts
            h.reply(turn);
            assert!(h.detaches().is_empty(), "limit {limit}: turn {n}: reply created");
            h.ended(turn, TurnEnd::Replied);
            assert!(h.detaches().is_empty(), "limit {limit}: turn {n}: replied");
            h.silence_held(turn);
            assert!(h.detaches().is_empty(), "limit {limit}: turn {n}: held");
            h.advance(ms(700));
            h.silence_released(turn);
            assert_eq!(h.detaches().is_empty(), n < limit, "limit {limit}: turn {n} released");
        }
        assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
        assert_eq!(h.outcome().reason, AmdReason::InferenceTimeout);
    }
}

/// At the limit, `_update_idle` completes before the turn's decision unless the reply would wait
/// for silence (`_reply_held_at`): the participant speaks, or has been silent for less than the
/// threshold. `Detach` then settles the turn with the "AMD completed" row, without stage
/// instructions.
#[test]
fn test_the_timeout_limit_completes_before_deciding_unless_the_reply_would_wait() {
    // (case, threshold, held)
    for (case, threshold, held) in [
        ("no threshold", Duration::ZERO, false),
        ("silent long enough", SILENCE, false),
        ("silent long enough by the eot delay", SILENCE, false),
        ("silent too briefly", SILENCE, true),
        ("speaking", SILENCE, true),
    ] {
        let options = AmdOptions {
            inference_timeout: ms(200),
            max_inference_timeouts: 1,
            machine_silence_threshold: threshold,
            ..AmdOptions::default()
        };
        let mut h = H::started(options);
        let first = h.turn("Please state your name.", Category::MachineScreening);
        h.reply_played(first, "This is Alex.");
        h.advance(secs(1));
        let mut commit = Commit::text("hello");
        match case {
            "silent long enough" => {
                h.speech_started(h.now);
                h.speech_ended(h.now);
                h.advance(ms(1400));
            }
            // no boundary since the previous commit: the silence starts at the endpoint
            "silent long enough by the eot delay" => commit.eot_delay = Some(ms(1400)),
            "silent too briefly" => {
                h.speech_started(h.now);
                h.advance(ms(500));
                h.speech_ended(h.now);
            }
            "speaking" => {
                h.speech_started(h.now);
                commit.speaking = true;
            }
            _ => {}
        }
        let turn = h.commit_with(commit);
        h.advance(ms(200));
        assert_eq!(h.last_prediction().reason, AmdReason::InferenceTimeout, "{case}");
        if !held {
            assert!(h.decisions(turn).is_empty(), "{case}: no decision before completion");
            assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep), "{case}");
            assert_eq!(h.outcome().reason, AmdReason::InferenceTimeout, "{case}");
            continue;
        }
        assert_eq!(h.decision(turn), staged(DEFAULT_SCREENING_INSTRUCTIONS), "{case}");
        assert!(h.detaches().is_empty(), "{case}: the reply waits for silence");
        h.replied(turn);
        h.silence_held(turn);
        assert!(h.detaches().is_empty(), "{case}: held");
        h.silence_released(turn);
        assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep), "{case}");
    }
}

/// A turn that asks for no reply, or ends without one, has no reply to hold: the idle timer and
/// the timeout limit go on. The end can reach AMD before the prediction or after the decision.
#[test]
fn test_a_turn_without_a_reply_holds_nothing() {
    use TurnEnd::*;
    // (how the turn goes without a reply, whether the end comes after the decision)
    for (how, after) in [
        (Stopped, false),
        (Stopped, true),
        (Dropped, false),
        (Dropped, true),
        (SkipReply, false),
        // the commit asked for no reply; the end has not reached AMD yet
        (SkipReply, true),
    ] {
        let case = format!("{how:?}, after decision: {after}");
        let skip_reply = how == SkipReply;
        let end = |h: &mut H, turn| {
            if how != SkipReply || !after {
                h.ended(turn, how);
            }
        };

        // the idle timer
        let mut h = H::started(AmdOptions { idle_timeout: secs(1), ..AmdOptions::default() });
        let first = h.turn("Please state your name.", Category::MachineScreening);
        h.reply_played(first, "This is Alex.");
        let turn = h.commit_with(Commit { skip_reply, ..Commit::text("Who is calling?") });
        if !after {
            end(&mut h, turn);
        }
        h.classify(turn, Category::MachineScreening);
        if after {
            end(&mut h, turn);
        }
        assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(1))), "{case}");
        h.advance(secs(1));
        assert_eq!(h.outcome().reason, AmdReason::IdleTimeout, "{case}");

        // the timeout limit
        let options = AmdOptions {
            inference_timeout: ms(200),
            max_inference_timeouts: 1,
            ..AmdOptions::default()
        };
        let mut h = H::started(options);
        let first = h.turn("Please state your name.", Category::MachineScreening);
        h.reply_played(first, "This is Alex.");
        let turn = h.commit_with(Commit { skip_reply, ..Commit::text("Who is calling?") });
        if !after {
            end(&mut h, turn);
        }
        h.advance(ms(200));
        if after {
            end(&mut h, turn);
        }
        assert_eq!(h.detaches().len(), 1, "{case}");
        assert_eq!(h.outcome().reason, AmdReason::InferenceTimeout, "{case}");
    }
}

/// `Denied` follows a denied decision and `Abandoned` the session's close: neither changes what
/// AMD holds.
#[test]
fn test_denied_and_abandoned_turns_change_nothing_else() {
    let mut h = H::started(AmdOptions { idle_timeout: secs(1), ..AmdOptions::default() });
    let waiting = h.turn("Please hold.", Category::Wait);
    assert_eq!(h.decision(waiting), deny());
    h.ended(waiting, TurnEnd::Denied);
    assert_eq!(h.armed(AmdTimer::Idle), None, "wait keeps the idle timer off");
    let next = h.commit("Hello?");
    assert_eq!(h.request_for(next).turn_id, 2);

    // the close ends the turn in its hook, then reports `Detached`
    h.ended(next, TurnEnd::Abandoned);
    assert!(h.detaches().is_empty());
    h.observe(AmdObservation::Detached { run: RUN });
    assert_eq!(h.detach(), completed_with(deny(), DetachCancel::Keep));
    h.step(In::LegsClosed);
    assert_eq!(h.completions()[0].reason, AmdReason::Cancelled);
}

#[test]
fn test_valid_prediction_resets_timeout_budget_before_reply_is_allowed() {
    let options = AmdOptions {
        inference_timeout: ms(100),
        max_inference_timeouts: 2,
        ..AmdOptions::default()
    };
    let mut h = H::started(options);
    let first = h.commit("hello");
    h.advance(ms(100));
    assert_eq!(h.decision(first), plain());
    let second = h.turn("Please state your name.", Category::MachineScreening);
    h.replied(second);
    h.silence_held(second);

    // the commit cancels the held reply
    let third = h.commit("hello");
    h.advance(ms(100));
    h.replied(third);
    h.silence_held(third);
    h.silence_released(third);
    assert!(h.detaches().is_empty(), "the screening prediction reset the count");

    let fourth = h.commit("hello");
    h.advance(ms(100));
    h.replied(fourth);
    h.silence_held(fourth);
    h.silence_released(fourth);
    assert_eq!(h.outcome().reason, AmdReason::InferenceTimeout);
}

#[test]
fn test_result_after_deadline_is_ignored_before_timer_callback() {
    // a result after its deadline counts as a timeout, even before the timer fires
    let mut h = H::started(AmdOptions { inference_timeout: secs(1), ..AmdOptions::default() });
    let turn = h.commit("hello");
    let req = h.request_for(turn).req;
    h.skip(secs(2));
    h.classify(turn, Category::Human);
    assert_eq!(h.reasons(), vec![(1, AmdReason::InferenceTimeout)]);
    assert_eq!(h.last_prediction().category, Category::Uncertain);
    assert_eq!(h.decision(turn), plain());
    assert!(h.detaches().is_empty());

    h.step(In::Timeout(AmdTimer::Inference(req)));
    assert_eq!(h.predictions().len(), 1, "the late timer repeats nothing");
}

#[test]
fn test_timeout_rearms_and_late_result_cannot_change_a_newer_turn() {
    let mut h = H::started(AmdOptions { inference_timeout: ms(20), ..AmdOptions::default() });
    let first = h.commit("hello");
    let stale = h.request_for(first).req;
    h.advance(ms(20));
    assert_eq!(h.decision(first), plain());
    let second = h.commit("hello");
    h.respond(
        stale,
        Ok(Classification { category: Category::MachineVm, inference_duration: ms(30) }),
    );
    assert_eq!(h.reasons(), vec![(1, AmdReason::InferenceTimeout)]);
    h.classify(second, Category::MachineIvr);
    let prediction = h.last_prediction();
    assert_eq!((prediction.turn_id, prediction.stage), (2, Stage::MachineIvr));
}

#[test]
fn test_classifier_that_ignores_cancellation_cannot_change_state() {
    let outcomes = [
        Ok(Category::Human),
        Ok(Category::Wait),
        Ok(Category::MachineScreening),
        Err(ClassifyError::Api),
    ];
    for cancelled_by in ["timeout", "new_turn", "finish"] {
        for outcome in outcomes {
            let case = format!("{cancelled_by}, {outcome:?}");
            let mut h =
                H::started(AmdOptions { inference_timeout: ms(20), ..AmdOptions::default() });
            let first = h.commit("hello");
            let stale = h.request_for(first).req;
            let newer = match cancelled_by {
                "timeout" => {
                    h.advance(ms(20));
                    None
                }
                "new_turn" => Some(h.commit("a new prompt")),
                _ => {
                    h.step(In::Close);
                    None
                }
            };
            let predictions = h.predictions();
            let detaches = h.detaches().len();
            let result =
                outcome.map(|category| Classification { category, inference_duration: ms(30) });
            h.respond(stale, result);
            assert_eq!(h.predictions(), predictions, "{case}");
            assert_eq!(h.detaches().len(), detaches, "{case}");
            assert!(h.decisions(first).iter().all(|d| *d == plain()), "{case}");

            match (cancelled_by, newer) {
                ("new_turn", Some(newer)) => {
                    h.classify(newer, Category::MachineVm);
                    assert_eq!(h.decision(newer), staged(DEFAULT_VOICEMAIL_INSTRUCTIONS), "{case}");
                    assert_eq!(h.last_prediction().stage, Stage::MachineVm);
                }
                ("timeout", _) => {
                    assert_eq!(h.reasons(), vec![(1, AmdReason::InferenceTimeout)], "{case}");
                    let next = h.commit("hello");
                    assert_eq!(h.request_for(next).stage, Stage::Uncertain, "{case}");
                }
                _ => assert_eq!(h.outcome().reason, AmdReason::Cancelled, "{case}"),
            }
        }
    }
}

#[test]
fn test_late_human_cannot_replace_a_held_timeout() {
    let mut h = H::started(AmdOptions { inference_timeout: ms(200), ..AmdOptions::default() });
    let first = h.turn("Leave a message.", Category::MachineVm);
    let speech = h.replied(first);
    h.silence_held(first);
    h.silence_released(first);
    // the message was cut, so the next voicemail turn may try again
    h.speech(speech, crate::amd::SpeechObservation::Done { interrupted: true, failed: false });
    let turn = h.commit("hello");
    let req = h.request_for(turn).req;
    h.advance(ms(200));
    assert_eq!(h.decision(turn), staged(DEFAULT_VOICEMAIL_INSTRUCTIONS));
    h.replied(turn);
    h.silence_held(turn);

    h.respond(req, Ok(Classification { category: Category::Human, inference_duration: ms(250) }));
    assert!(h.detaches().is_empty());
    let categories: Vec<_> = h.predictions().iter().map(|p| p.category).collect();
    assert_eq!(categories, vec![Category::MachineVm, Category::Uncertain]);
    assert_eq!(h.last_prediction().stage, Stage::MachineVm);
}

// The overall deadline (spec §12.1 `timeout`).

#[test]
fn test_amd_enforces_overall_deadline_independently_of_fsm() {
    for pending in ["idle", "classification", "silence", "speech"] {
        let options =
            AmdOptions { timeout: secs(3), inference_timeout: secs(10), ..AmdOptions::default() };
        let mut h = H::started(options);
        let mut req = None;
        if pending != "idle" {
            let turn = h.commit("hello");
            req = Some(h.request_for(turn).req);
            if pending != "classification" {
                h.classify(turn, Category::MachineVm);
                h.replied(turn);
                h.silence_held(turn);
                if pending == "speech" {
                    h.speech_started(h.now);
                }
            }
        }
        h.advance(secs(3));
        // a held reply plays at once at completion, for every reason (spec §10.3)
        assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep), "{pending}");
        if pending == "classification" {
            assert_eq!(h.cancelled(), vec![req.unwrap()]);
        }
        assert!(h.armed_timers().is_empty(), "{pending}: {:?}", h.armed_timers());
        let outcome = h.outcome();
        assert_eq!(outcome.reason, AmdReason::Timeout, "{pending}");
        let expected = if matches!(pending, "silence" | "speech") {
            Stage::MachineVm
        } else {
            Stage::Uncertain
        };
        assert_eq!(outcome.category, expected, "{pending}");
    }
}

// Idle (spec §12.1 `idle_timeout`).

#[test]
fn test_untranscribed_speech_restarts_idle_timer() {
    let options = AmdOptions { idle_timeout: secs(2), timeout: secs(20), ..AmdOptions::default() };
    let mut h = H::started(options);
    h.advance(secs(1));
    h.speech_started(h.now);
    assert_eq!(h.armed(AmdTimer::Idle), None);
    h.advance(secs(3));
    assert!(h.detaches().is_empty());
    h.speech_ended(h.now);
    assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(2))));
    h.advance(secs(2));
    let outcome = h.outcome();
    assert_eq!(outcome.reason, AmdReason::IdleTimeout);
    assert_eq!(outcome.last_prediction, None);
}

#[test]
fn test_wait_pauses_the_idle_timer_until_the_next_prediction() {
    let mut h = H::started(AmdOptions { idle_timeout: secs(1), ..AmdOptions::default() });
    let waiting = h.turn("Please hold.", Category::Wait);
    assert_eq!(h.decision(waiting), deny());
    assert_eq!(h.armed(AmdTimer::Idle), None);
    h.advance(secs(3));
    let reused = h.commit("");
    assert_eq!(h.decision(reused), deny());
    h.advance(secs(3));
    assert!(h.detaches().is_empty());

    let turn = h.turn("Hi.", Category::Uncertain);
    assert_eq!(h.decision(turn), plain());
    assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(1))));
    h.advance(secs(1));
    assert_eq!(h.outcome().reason, AmdReason::IdleTimeout);
}

#[test]
fn test_speech_end_does_not_start_idle_while_customer_hook_runs() {
    let mut h = H::started(AmdOptions { idle_timeout: secs(1), ..AmdOptions::default() });
    let turn = h.commit("Please state your name.");
    // the turn is in its customer hook
    h.observe(AmdObservation::AgentBusy { busy: true, at: h.now });
    h.classify(turn, Category::MachineScreening);
    h.speech_started(h.now);
    h.speech_ended(h.now);
    h.advance(secs(2));
    // `StopResponse`: the turn ends without a reply
    h.ended(turn, TurnEnd::Stopped);
    assert_eq!(h.armed(AmdTimer::Idle), None);
    assert!(h.detaches().is_empty());
    h.observe(AmdObservation::AgentBusy { busy: false, at: h.now });
    assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(1))));
    h.advance(secs(1));
    assert_eq!(h.outcome().reason, AmdReason::IdleTimeout);
}

#[test]
fn test_false_interruption_settlement_rearms_idle_without_a_speech_handle() {
    let mut h = H::started(AmdOptions { idle_timeout: secs(1), ..AmdOptions::default() });
    h.observe(AmdObservation::AgentBusy { busy: true, at: h.now });
    h.advance(secs(2));
    assert!(h.detaches().is_empty());
    h.observe(AmdObservation::AgentBusy { busy: false, at: h.now });
    h.advance(secs(1));
    assert_eq!(h.outcome().reason, AmdReason::IdleTimeout);
}

#[test]
fn test_a_watched_speech_holds_the_idle_timer() {
    let mut h = H::started(AmdOptions { idle_timeout: secs(1), ..AmdOptions::default() });
    // the gate reports the speech as busy in the same step it reports the speech
    let speech = h.created(crate::agent_session::machine::activity::scheduling::SpeechKind::Say);
    h.busy(true);
    assert_eq!(h.armed(AmdTimer::Idle), None);
    h.advance(secs(2));
    assert!(h.detaches().is_empty());
    h.play(speech, "Hi, this is Alex.");
    h.busy(false);
    assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(1))));
}

#[test]
fn test_a_held_reply_holds_the_idle_timer_until_it_is_released() {
    // held from the allowed decision until `SilenceHold { held: false }`, or until a turn end
    // that prevents the reply
    for release in ["silence", "dropped"] {
        let mut h = H::started(AmdOptions { idle_timeout: secs(1), ..AmdOptions::default() });
        let turn = h.turn("Please state your name.", Category::MachineScreening);
        assert_eq!(h.armed(AmdTimer::Idle), None, "{release}");
        h.advance(secs(3));
        assert!(h.detaches().is_empty(), "{release}");
        if release == "silence" {
            // `Replied` keeps the hold: the session enters it after the turn ends
            h.replied(turn);
            assert_eq!(h.armed(AmdTimer::Idle), None, "{release}");
            h.silence_held(turn);
            assert_eq!(h.armed(AmdTimer::Idle), None, "{release}");
            h.silence_released(turn);
        } else {
            // the scheduler refused the reply
            h.ended(turn, TurnEnd::Dropped);
        }
        assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(1))), "{release}");
    }
}

#[test]
fn test_late_prediction_preserves_stage_and_idle_timer() {
    for (previous, late, idle) in [
        (Category::MachineIvr, Category::MachineVm, secs(1)),
        (Category::MachineVm, Category::MachineIvr, secs(2)),
    ] {
        let options = AmdOptions {
            idle_timeout: secs(1),
            voicemail_idle_timeout: secs(2),
            inference_timeout: ms(10),
            machine_silence_threshold: Duration::ZERO,
            ..AmdOptions::default()
        };
        let mut h = H::started(options);
        h.turn("hello", previous);
        let turn = h.commit("hello");
        let req = h.request_for(turn).req;
        h.advance(ms(10));
        assert_eq!(h.last_prediction().reason, AmdReason::InferenceTimeout);
        let deadline = h.armed(AmdTimer::Idle);
        assert_eq!(deadline, Some(h.now.saturating_add(idle)), "{previous:?}");

        h.respond(req, Ok(Classification { category: late, inference_duration: ms(20) }));
        assert_eq!(h.predictions().len(), 2);
        assert_eq!(h.armed(AmdTimer::Idle), deadline);
        let next = h.commit("hello");
        let stage =
            if previous == Category::MachineVm { Stage::MachineVm } else { Stage::MachineIvr };
        assert_eq!(h.request_for(next).stage, stage);
    }
}
