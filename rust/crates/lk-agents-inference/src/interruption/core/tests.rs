use super::*;
use crate::interruption::options::{
    AUDIO_PREFIX_DURATION, DETECTION_INTERVAL, InterruptionOptions, MAX_AUDIO_DURATION,
    MIN_INTERRUPTION_DURATION, REMOTE_INFERENCE_TIMEOUT,
};

const SR: u32 = 16_000;

/// The defaults, with the credentials already resolved.
fn core_with(threshold: Option<f64>) -> InterruptionCore {
    InterruptionCore::new(
        InterruptionOptions::new(
            threshold,
            MIN_INTERRUPTION_DURATION,
            MAX_AUDIO_DURATION,
            AUDIO_PREFIX_DURATION,
            DETECTION_INTERVAL,
            REMOTE_INFERENCE_TIMEOUT,
            "https://gateway".to_owned(),
            "k".to_owned(),
            "s".to_owned(),
        )
        .unwrap(),
    )
}

fn core() -> InterruptionCore {
    core_with(None)
}

/// `n` samples of mono 16 kHz, each equal to `value`, so a window's contents are identifiable.
fn frame(n: usize, value: i16) -> AudioFrame {
    AudioFrame {
        data: std::iter::repeat_n(value, n).flat_map(i16::to_le_bytes).collect(),
        sample_rate: SR,
        num_channels: 1,
        samples_per_channel: n as u32,
    }
}

fn ts(secs: f64) -> Ts {
    Ts::from_nanos((secs * 1e9) as u64)
}

fn end_overlap(
    c: &mut InterruptionCore,
    ended_at: f64,
    agent_ended: bool,
    now: f64,
) -> Option<InputOutcome> {
    c.on_input(InputEvent::OverlapSpeechEnded(OverlapEnd { ended_at, agent_ended }), now).unwrap()
}

/// The server's answer to the request stamped `sent`, arriving at `at`.
fn verdict(
    c: &mut InterruptionCore,
    detected: bool,
    sent: f64,
    prediction_duration: f64,
    probabilities: &[f32],
    at: f64,
) -> RecvOutcome {
    let v = Verdict {
        created_at: ts(sent).as_nanos(),
        prediction_duration,
        probabilities: probabilities.to_vec(),
    };
    let msg = if detected { ServerMessage::Detected(v) } else { ServerMessage::InferenceDone(v) };
    c.on_server_message(msg, ts(at), at).unwrap()
}

fn start_overlap(c: &mut InterruptionCore, speech_duration: f64, at: f64) {
    c.on_input(
        InputEvent::OverlapSpeechStarted(OverlapStart {
            speech_duration,
            started_at: at,
            user_speaking_span: None,
        }),
        at,
    )
    .unwrap();
}

// Catches: buffering audio before the agent speaks, or shipping it after it stops.
#[test]
fn audio_outside_an_agent_turn_is_dropped_entirely() {
    let mut c = core();
    assert_eq!(c.on_input(InputEvent::Audio(frame(1600, 1)), 0.0).unwrap(), None);
    assert_eq!(c.window.len(), 0, "buffered audio with no agent speaking");

    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    c.on_input(InputEvent::Audio(frame(1600, 1)), 0.0).unwrap();
    assert_eq!(c.window.len(), 1600);

    c.on_input(InputEvent::AgentSpeechEnded, 0.0).unwrap();
    assert_eq!(c.window.len(), 0, "the window must be reset when the agent stops");
    c.on_input(InputEvent::Audio(frame(1600, 1)), 0.0).unwrap();
    assert_eq!(c.window.len(), 0);
}

// Catches: an overlap-started sentinel opening an overlap while the agent is silent.
#[test]
fn an_overlap_cannot_open_while_the_agent_is_silent() {
    let mut c = core();
    start_overlap(&mut c, 0.0, 1.0);
    assert!(!c.overlap_started);
    assert_eq!(c.overlap_started_at, None);
    // ... and no event is produced when it "ends", because it never began.
    let out = end_overlap(&mut c, 2.0, false, 2.0);
    assert_eq!(out, None);
}

// Catches: a request on every frame instead of every `detection_interval`.
#[test]
fn requests_are_emitted_once_per_detection_interval_not_per_frame() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);

    let batch = (DETECTION_INTERVAL_SAMPLES) / 10;
    let mut requests = 0;
    for _ in 0..10 {
        if let Some(InputOutcome::Infer(_)) =
            c.on_input(InputEvent::Audio(frame(batch, 3)), 1.0).unwrap()
        {
            requests += 1;
        }
    }
    assert_eq!(requests, 1, "1600 samples of 160-sample frames is exactly one request");
}
const DETECTION_INTERVAL_SAMPLES: usize = 1600;

// Catches: dropping the audio prefix; the trim keeps the prefix plus the speech already
// detected.
#[test]
fn the_first_request_carries_the_configured_prefix_of_pre_overlap_audio() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    // 2 s of agent-only audio, then the user has been talking for 0.5 s.
    c.on_input(InputEvent::Audio(frame(2 * SR as usize, 7)), 0.0).unwrap();
    start_overlap(&mut c, 0.5, 2.0);
    // prefix 1.0 s + speech 0.5 s = 1.5 s kept out of the 2 s buffered.
    assert_eq!(c.window.len(), (1.5 * SR as f64) as usize);

    let out = c.on_input(InputEvent::Audio(frame(1600, 9)), 2.0).unwrap();
    let Some(InputOutcome::Infer(samples)) = out else { panic!("expected a request") };
    assert_eq!(samples.len(), (1.5 * SR as f64) as usize + 1600);
}

// Catches: re-trimming on a second overlap within one agent turn.
#[test]
fn a_second_overlap_in_the_same_agent_turn_does_not_re_trim_the_window() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    c.on_input(InputEvent::Audio(frame(2 * SR as usize, 7)), 0.0).unwrap();
    start_overlap(&mut c, 0.5, 2.0);
    let after_first = c.window.len();

    end_overlap(&mut c, 2.2, false, 2.2);
    start_overlap(&mut c, 0.0, 2.5);
    assert_eq!(c.window.len(), after_first, "the window was trimmed a second time");
    assert_eq!(c.overlap_count, 2);
}

// Catches: clearing the cache when the overlap ends.
#[test]
fn ending_an_overlap_keeps_in_flight_entries_but_starting_one_clears_them() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1, 2]);
    c.record_request(ts(1.1), vec![3, 4]);

    end_overlap(&mut c, 1.2, false, 1.2);
    assert_eq!(c.cache.iter().count(), 2, "in-flight requests were dropped");

    start_overlap(&mut c, 0.0, 1.3);
    assert_eq!(c.cache.iter().count(), 0, "a new overlap must start from a clean cache");
}

// Catches: reporting the first answered inference rather than the last, and counting requests
// cumulatively instead of per overlap.
#[test]
fn overlap_end_reports_the_newest_answered_inference_and_drains_the_counter() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    c.record_request(ts(1.1), vec![2]);
    // Both come back; the second one is slower.
    verdict(&mut c, false, 1.0, 0.01, &[0.1, 0.1], 1.05);
    verdict(&mut c, false, 1.1, 0.02, &[0.4, 0.3], 1.3);

    let out = end_overlap(&mut c, 1.4, false, 1.45);
    let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
    assert!(!ev.is_interruption);
    assert_eq!(ev.num_requests, 2);
    assert_eq!(ev.prediction_duration, 0.02, "reported the older inference");
    assert!((ev.total_duration - 0.2).abs() < 1e-9);
    assert_eq!(ev.overlap_started_at, Some(1.0));
    assert_eq!(ev.detected_at, 1.4);

    // And the counter is drained, not merely read.
    start_overlap(&mut c, 0.0, 2.0);
    let out = end_overlap(&mut c, 2.1, false, 2.1);
    let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
    assert_eq!(ev.num_requests, 0);
}

// Catches: `pop_if` accepting a zero-RTT entry created by the response path.
#[test]
fn an_unanswered_entry_never_describes_the_overlap() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    c.record_request(ts(1.1), vec![2]);
    verdict(&mut c, false, 1.0, 0.05, &[0.2, 0.2], 1.2);

    let out = end_overlap(&mut c, 1.3, false, 1.3);
    let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
    assert_eq!(ev.prediction_duration, 0.05);
    assert_eq!(ev.speech_input, Some(vec![1]));
}

// Catches: `pop_if` accepting a non-positive round trip from a bogus echoed stamp.
#[test]
fn a_verdict_echoing_a_stamp_we_never_sent_does_not_describe_the_overlap() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    // The server answers a request stamped in the future.
    verdict(&mut c, false, 99.0, 0.5, &[0.7, 0.7], 1.2);

    let out = end_overlap(&mut c, 1.3, false, 1.3);
    let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
    assert_eq!(
        (ev.total_duration, ev.prediction_duration, ev.probability),
        (0.0, 0.0, 0.0),
        "a zero round trip must not be reported as a real inference"
    );
}

// Catches: emitting nothing when no request was made, or stale durations instead of 0.0.
#[test]
fn an_overlap_with_no_requests_still_emits_a_zeroed_event() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    let out = end_overlap(&mut c, 1.1, true, 1.1);
    let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
    assert!(ev.agent_ended && !ev.is_interruption);
    assert_eq!((ev.total_duration, ev.probability, ev.num_requests), (0.0, 0.0, 0));
    assert_eq!(ev.speech_input, None);
}

// Catches: an overlap reported twice, by the verdict and by the end sentinel.
#[test]
fn a_detected_verdict_closes_the_overlap_so_the_end_sentinel_emits_nothing() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![5]);

    let out = verdict(&mut c, true, 1.0, 0.03, &[0.9, 0.8, 0.1], 1.15);
    let RecvOutcome::Detected { event, span_update } = out else { panic!("expected Detected") };
    assert!(event.is_interruption);
    assert_eq!(event.num_requests, 1);
    assert_eq!(event.probability, 0.8_f32 as f64);
    assert!((event.total_duration - 0.15).abs() < 1e-9);
    assert!((event.detection_delay - 0.15).abs() < 1e-9);
    assert_eq!(span_update, None, "no span was supplied");

    assert_eq!(end_overlap(&mut c, 1.2, false, 1.2), None);
}

// Catches: applying a verdict that arrives after the overlap closed.
#[test]
fn a_verdict_arriving_after_the_overlap_closed_is_ignored() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![5]);
    end_overlap(&mut c, 1.1, false, 1.1);

    let out = verdict(&mut c, true, 1.0, 0.03, &[0.9, 0.9], 1.2);
    assert_eq!(out, RecvOutcome::Ignored);
}

#[test]
fn interruption_verdict_identity_rejects_previous_overlap_and_unknown_requests() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    end_overlap(&mut c, 1.1, false, 1.1);
    start_overlap(&mut c, 0.0, 1.2);
    c.record_request(ts(1.2), vec![2]);

    assert_eq!(verdict(&mut c, true, 1.0, 0.1, &[0.9, 0.9], 1.3), RecvOutcome::Ignored);
    assert_eq!(verdict(&mut c, true, 99.0, 0.1, &[0.9, 0.9], 1.3), RecvOutcome::Ignored);
    assert_eq!(c.cache.iter().count(), 1, "unknown echoes must not create requests");
    assert!(matches!(
        verdict(&mut c, true, 1.2, 0.1, &[0.9, 0.9], 1.4),
        RecvOutcome::Detected { .. }
    ));
}

#[test]
fn interruption_verdict_identity_accepts_each_request_once() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    verdict(&mut c, false, 1.0, 0.1, &[0.1, 0.1], 1.1);
    assert_eq!(verdict(&mut c, true, 1.0, 0.1, &[0.9, 0.9], 1.2), RecvOutcome::Ignored);
    assert!(c.overlap_started, "duplicate verdict must leave current overlap alone");
}

// Catches: #6034 regressing (no threshold known must be an error).
#[test]
fn a_session_created_with_no_threshold_anywhere_is_fatal() {
    let mut c = core();
    let err = c
        .on_server_message(ServerMessage::SessionCreated { default_threshold: None }, ts(0.0), 0.0)
        .unwrap_err();
    assert_eq!(err.error.status_code, Some(500));
    assert!(!err.error.retryable, "reconnecting cannot conjure a threshold");

    // Either source alone is enough.
    let user = ServerMessage::SessionCreated { default_threshold: None };
    assert_eq!(
        core_with(Some(0.9)).on_server_message(user, ts(0.0), 0.0),
        Ok(RecvOutcome::Ignored)
    );
    let server = ServerMessage::SessionCreated { default_threshold: Some(0.5) };
    assert_eq!(core().on_server_message(server, ts(0.0), 0.0), Ok(RecvOutcome::Ignored));
}

// Answered windows cannot extend another request's absolute deadline.
#[test]
fn the_inference_deadline_skips_answered_entries_and_fires_at_expiry() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    c.record_request(ts(1.1), vec![2]);
    // The oldest is answered; the scan must move past it to the second one.
    verdict(&mut c, false, 1.0, 0.0, &[], 1.05);

    assert!(c.check_inflight_timeout(ts(1.79)).is_ok(), "0.69s is inside the 0.7s window");
    let err = c.check_inflight_timeout(ts(1.8)).unwrap_err();
    assert_eq!(err.error.status_code, Some(408));
    assert!(!err.error.retryable);
    assert_eq!(err.error.message, "interruption inference timed out after 0.7s (ws)");
}

#[test]
fn interruption_inference_deadline_cannot_affect_a_later_speech() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    assert_eq!(c.next_deadline(), Some(ts(1.7)));
    let error = c.handle(InterruptionEvent::Deadline, ts(1.7), 1.7).unwrap_err();
    assert_eq!(error.error.status_code, Some(408));
    assert_eq!(c.next_deadline(), None);
    c.on_input(InputEvent::AgentSpeechStarted, 2.0).unwrap();
    start_overlap(&mut c, 0.0, 2.0);
    assert_eq!(verdict(&mut c, true, 1.0, 0.1, &[0.9, 0.9], 2.1), RecvOutcome::Ignored);
    assert!(c.overlap_started);
}

#[test]
fn interruption_verdict_identity_rejects_other_connections_and_speeches() {
    let mut c = core();
    c.handle(InterruptionEvent::ConnectionOpened, ts(0.0), 0.0).unwrap();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    let infer = c
        .handle(InterruptionEvent::Input(InputEvent::Audio(frame(1600, 1))), ts(1.0), 1.0)
        .unwrap();
    let InterruptionEffect::InterruptionInference { identity, .. } = infer[0] else {
        panic!("expected authorized inference");
    };
    let message =
        ServerMessage::Detected(Verdict { created_at: identity.request_id, ..Default::default() });
    assert!(
        c.handle(
            InterruptionEvent::ServerMessage {
                connection: identity.connection + 1,
                message: message.clone()
            },
            ts(1.1),
            1.1
        )
        .unwrap()
        .is_empty()
    );
    assert!(
        c.request_eligible(identity),
        "a mismatched response cannot consume the registered request"
    );

    assert!(c.handle(InterruptionEvent::ConnectionOpened, ts(1.2), 1.2).is_err());
    c.on_input(InputEvent::AgentSpeechStarted, 1.2).unwrap();
    start_overlap(&mut c, 0.0, 1.2);
    assert!(!c.request_eligible(identity));
    assert!(
        c.handle(
            InterruptionEvent::ServerMessage { connection: identity.connection, message },
            ts(1.3),
            1.3
        )
        .unwrap()
        .is_empty()
    );
    assert!(c.overlap_started);

    let infer = c
        .handle(InterruptionEvent::Input(InputEvent::Audio(frame(1600, 1))), ts(1.4), 1.4)
        .unwrap();
    let InterruptionEffect::InterruptionInference { identity, .. } = infer[0] else {
        panic!("expected authorized inference");
    };
    c.on_input(InputEvent::AgentSpeechStarted, 1.5).unwrap();
    start_overlap(&mut c, 0.0, 1.5);
    assert!(!c.request_eligible(identity));
    assert_eq!(verdict(&mut c, true, 1.4, 0.1, &[0.9, 0.9], 1.6), RecvOutcome::Ignored);
    assert!(c.overlap_started);
}

#[test]
fn the_registered_overlap_request_survives_verdict_delivery() {
    let mut c = core();
    c.handle(InterruptionEvent::ConnectionOpened, ts(0.0), 0.0).unwrap();
    c.handle(InterruptionEvent::Input(InputEvent::AgentSpeechStarted), ts(0.0), 0.0).unwrap();
    c.handle(
        InterruptionEvent::Input(InputEvent::OverlapRequest {
            request: ReqId(42),
            start: OverlapStart { speech_duration: 0.0, started_at: 1.0, user_speaking_span: None },
        }),
        ts(1.0),
        1.0,
    )
    .unwrap();
    assert!(
        c.handle(
            InterruptionEvent::Input(InputEvent::OverlapRequestEnded {
                request: ReqId(41),
                end: OverlapEnd { ended_at: 1.05, agent_ended: false },
            }),
            ts(1.05),
            1.05
        )
        .unwrap()
        .is_empty()
    );
    let effects = c
        .handle(InterruptionEvent::Input(InputEvent::Audio(frame(1600, 1))), ts(1.1), 1.1)
        .unwrap();
    let InterruptionEffect::InterruptionInference { identity, .. } = effects[0] else {
        panic!("expected an inference request");
    };
    let verdict =
        ServerMessage::Detected(Verdict { created_at: identity.request_id, ..Default::default() });
    let effects = c
        .handle(
            InterruptionEvent::ServerMessage {
                connection: identity.connection,
                message: verdict.clone(),
            },
            ts(1.2),
            1.2,
        )
        .unwrap();
    assert!(
        matches!(effects.as_slice(), [InterruptionEffect::Overlap { request: Some(ReqId(42)), event }] if event.is_interruption)
    );
    assert!(
        c.handle(
            InterruptionEvent::ServerMessage { connection: identity.connection, message: verdict },
            ts(1.3),
            1.3
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn pending_inferences_cannot_be_evicted_to_escape_their_deadline() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    for _ in 0..CACHE_MAXSIZE {
        c.handle(InterruptionEvent::Input(InputEvent::Audio(frame(1600, 1))), ts(1.0), 1.0)
            .unwrap();
    }
    assert!(
        c.handle(InterruptionEvent::Input(InputEvent::Audio(frame(1600, 1))), ts(1.0), 1.0)
            .is_err()
    );
    assert!(!c.overlap_started);
}

// Catches: the span being decorated on every verdict instead of once.
#[test]
fn the_user_speech_span_is_handed_back_exactly_once() {
    use std::sync::Arc;
    struct Noop;
    impl lk_agents_abi::RefRegistry for Noop {
        fn release(&self, _id: u64) {}
    }
    let registry: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(Noop);

    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    c.on_input(
        InputEvent::OverlapSpeechStarted(OverlapStart {
            speech_duration: 0.0,
            started_at: 1.0,
            user_speaking_span: Some(HostRef::new(3, &registry)),
        }),
        1.0,
    )
    .unwrap();
    c.record_request(ts(1.0), vec![1]);

    let out = verdict(&mut c, true, 1.0, 0.0, &[0.9, 0.7], 1.1);
    let RecvOutcome::Detected { span_update: Some((span, attrs)), .. } = out else {
        panic!("expected a span update")
    };
    assert_eq!(span.id(), 3);
    assert_eq!(attrs.is_interruption, "true");
    assert_eq!(attrs.probability, 0.7_f32 as f64);
    assert!(c.user_speech_span.is_none(), "the span must be released as it is written");
}

// Catches: a server `error` frame losing its payload (`body`) or status code.
#[test]
fn a_server_error_carries_the_code_as_status_and_the_text_as_body() {
    let mut c = core();
    let err = c
        .on_server_message(
            ServerMessage::Error {
                message: "model unavailable".to_owned(),
                code: 503,
                session_id: "sess".to_owned(),
            },
            ts(0.0),
            0.0,
        )
        .unwrap_err();
    assert_eq!(err.error.status_code, Some(503));
    assert_eq!(err.error.message, "LiveKit Adaptive Interruption returned error: 503");
    assert_eq!(err.body.as_deref(), Some("model unavailable"));
    assert!(err.error.retryable, "a 5xx is worth reconnecting for");

    let err = c
        .on_server_message(
            ServerMessage::Error {
                message: "bad key".to_owned(),
                code: 401,
                session_id: "sess".to_owned(),
            },
            ts(0.0),
            0.0,
        )
        .unwrap_err();
    assert!(!err.error.retryable, "a 401 will keep failing (livekit/agents#5831)");
}

// Catches: `_reset_state` missing a field.
#[test]
fn agent_speech_start_re_arms_every_field_that_survives_a_turn() {
    let mut c = core();
    c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
    c.on_input(InputEvent::Audio(frame(800, 1)), 0.0).unwrap();
    start_overlap(&mut c, 0.0, 1.0);
    c.record_request(ts(1.0), vec![1]);
    assert!(c.overlap_started && c.num_requests == 1 && c.cache.iter().count() == 1);

    c.on_input(InputEvent::AgentSpeechStarted, 2.0).unwrap();
    assert!(c.agent_speech_started);
    assert!(!c.overlap_started);
    assert_eq!(c.overlap_count, 0);
    assert_eq!(c.accumulated_samples, 0);
    assert_eq!(c.num_requests, 0);
    assert_eq!(c.cache.iter().count(), 0);
    assert_eq!(c.window.len(), 0);
    assert!(c.user_speech_span.is_none());
}
