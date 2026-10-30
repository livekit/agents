//! Session decisions over a decision core no executor runs.

use super::*;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::lifecycle::SessionStartPlans;
use crate::events::SessionEvent;
use lk_agents_core::tts::markup::ChatMessageExt;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::session_effects::SessionStart as _;

#[test]
fn realtime_transport_timeout_keeps_its_phase_and_request_identity_on_session_error() {
    use crate::agent_activity::realtime::RealtimeWake;
    use crate::agent_session::effects::Effect;
    use lk_agents_abi::Ts;
    use lk_agents_abi::errors::{ApiError, ApiErrorKind, TimeoutPhase};
    use lk_agents_abi::llm::realtime as rt;

    let driver = &mut driver();
    let mut out = driver.out();
    let mut timeout = ApiError::timed_out(TimeoutPhase::Connect);
    timeout.request_id = Some("handshake-7".into());
    driver.on_realtime(
        Ts::ZERO,
        RealtimeWake::Event(rt::Event::Error(rt::RealtimeModelError {
            timestamp: Ts::ZERO,
            label: "realtime".into(),
            message: timeout.message.clone(),
            recoverable: true,
            req_id: None,
            code: None,
            provider_error: Some(timeout),
        })),
        &mut out,
    );
    let effects = driver.settle_step(out);
    let error = effects
        .into_iter()
        .find_map(|effect| match effect {
            Effect::Emit(SessionEvent::Error(error)) => Some(error),
            _ => None,
        })
        .expect("the session did not report the transport error");
    let ProviderError::Local(error) = error.error else {
        panic!("a native transport error changed to a host error");
    };
    assert_eq!(error.kind, ApiErrorKind::Timeout);
    assert_eq!(error.phase, Some(TimeoutPhase::Connect));
    assert_eq!(error.request_id.as_deref(), Some("handshake-7"));
}

#[test]
fn late_realtime_transcript_keeps_its_source_without_starting_another_speech() {
    use crate::agent_activity::realtime::RealtimeWake;
    use crate::agent_session::effects::{Effect, Planned};
    use lk_agents_abi::llm::realtime as rt;
    use lk_agents_abi::{ChatMessage, Role, Ts};

    let driver = &mut driver();
    let mut out = driver.out();
    let source =
        ChatMessage::text("old-response", Role::Assistant, "Already heard", std::time::UNIX_EPOCH);
    let source_time = source.created_at;
    driver.commit_item(
        source.into(),
        crate::agent::Source::Remote { item_id: "old-response".into() },
        crate::agent::Visibility::Both,
        &mut out,
    );
    driver.on_realtime(
        Ts::ZERO,
        RealtimeWake::Event(rt::Event::GenerationCreated(rt::GenerationCreated {
            response_id: "new-response".into(),
            req_id: None,
        })),
        &mut out,
    );
    let players = driver.activity.realtime_players.clone();
    out.take();
    for event in [
        rt::Event::TranscriptAttribution(rt::TranscriptAttribution {
            response_id: "late-transcript".into(),
            source_response_id: "old-response".into(),
        }),
        rt::Event::GenerationCreated(rt::GenerationCreated {
            response_id: "late-transcript".into(),
            req_id: None,
        }),
        rt::Event::MessageStarted(rt::MessageStarted {
            message_id: "late-transcript".into(),
            response_id: Some("late-transcript".into()),
        }),
        rt::Event::TextDelta(rt::TextDelta {
            message_id: "late-transcript".into(),
            response_id: Some("late-transcript".into()),
            delta: "A delayed transcription".into(),
            start_time: None,
        }),
        rt::Event::MessageDone(rt::MessageDone {
            message_id: "late-transcript".into(),
            response_id: Some("late-transcript".into()),
        }),
        rt::Event::GenerationDone(rt::GenerationDone {
            response_id: "late-transcript".into(),
            status: rt::GenerationStatus::Completed,
            usage: Default::default(),
        }),
    ] {
        driver.on_realtime(Ts::ZERO, RealtimeWake::Event(event), &mut out);
    }
    assert_eq!(driver.activity.realtime_players, players, "a correction took the voice floor");
    assert!(
        !out.take().iter().any(|effect| matches!(
            effect,
            Planned::Effect(Effect::RealtimePlayer { .. } | Effect::SpeechStart { .. })
        )),
        "a correction was sent through speech playback"
    );
    let correction =
        &driver.session.chat_ctx.entry("late-transcript").expect("late words were lost").item;
    let correction = correction.as_message().unwrap();
    assert_eq!(correction.text_content().as_deref(), Some("A delayed transcription"));
    assert_eq!(
        correction.metrics.provider_request_ids.as_deref(),
        Some(["old-response".to_owned()].as_slice())
    );
    assert_eq!(correction.created_at, source_time);
}

// Catches (voice-session.md "releases the lane within 30 seconds"): a speech whose tasks were
// aborted (the interruption watchdog, `cancel_tasks`) reports neither done nor failed, so its
// lane was never released and the store renewed its lease until the call closed.
#[test]
fn a_speech_ended_without_its_turn_releases_the_lane_within_30_s() {
    use crate::agent_session::effects::Effect;
    use crate::agent_session::storage::LANE_RELEASE_BOUND;
    use crate::agent_session::time::Now;
    use lk_agents_abi::Ts;

    let driver = &mut driver();
    driver.session.storage.enabled = true;
    let speech = SpeechId(7);
    driver.storage_started(speech);
    let mut out = driver.out();
    driver.release_speech(Ts::ZERO, speech, &mut out);
    let released = driver.settle_step(out);
    assert!(!released.iter().any(|e| matches!(e, Effect::StorageAbort(_))));
    let wall = std::time::UNIX_EPOCH;
    let due = Ts::ZERO.saturating_add(LANE_RELEASE_BOUND);
    assert_eq!(driver.wake_at(), Some(due), "the release is bounded");
    let effects = driver.handle(None, Now { at: due, wall });
    assert!(
        effects.iter().any(|e| matches!(e, Effect::StorageAbort(s) if *s == speech)),
        "the lane was never released: {effects:?}"
    );
}

// Catches: a `say` whose step reported spoken after the close's hard edge released it read its
// kind off the scheduler's entry, gone with the release, so its item was committed as a reply's,
// naming no voice (`items_carry_their_payload`, survey seed 114526; fixed in b1936ecfe).
#[test]
fn a_say_released_before_its_step_reports_keeps_its_source() {
    use crate::agent::Source;
    use crate::agent_activity::scheduling::SpeechKind;
    use crate::agent_activity::speech::SpeechSpec;
    use crate::agent_session::effects::Effect;
    use crate::generation::{GenEvent, ReplySource};
    use lk_agents_abi::{ChatMessage, Role, Ts};

    let driver = &mut driver();
    let mut out = driver.out();
    let spec = SpeechSpec::new(SpeechKind::Say, ReplySource::Text("One moment.".into()), true);
    let speech = driver.start_speech(Ts::ZERO, spec, &mut out);
    driver.release_speech(Ts::ZERO, speech, &mut out);
    let said = ChatMessage::text("said", Role::Assistant, "One moment.", std::time::UNIX_EPOCH);
    let spoken = GenEvent::StepSpoken {
        speech,
        step: 1,
        interrupted: true,
        has_tool_output: false,
        tools_running: false,
        item: Some(Box::new(said)),
    };
    driver.on_generation(Ts::ZERO, spoken, &mut out);
    let sources: Vec<Source> = driver
        .settle_step(out)
        .into_iter()
        .filter_map(|effect| match effect {
            Effect::Emit(SessionEvent::ConversationItemAdded { source, .. })
                if matches!(source, Source::Say { .. } | Source::Reply { .. }) =>
            {
                Some(source)
            }
            _ => None,
        })
        .collect();
    assert!(
        matches!(sources.as_slice(), [Source::Say { speech: s }] if *s == speech),
        "the say's item lost its source: {sources:?}"
    );
}

/// A session's decision core with no provider or leg, at the zero instant; nothing runs it.
fn driver() -> Driver {
    let legs = lk_agents_abi::voice::session_options::SessionLegs::none();
    let config = crate::agent_activity::DriverConfig::new(Default::default(), legs.into());
    let view = Arc::new(SessionView {
        launch_holds: Default::default(),
        config: config.clone(),
        agent: "agent".into(),
        vad: None,
        usage: Default::default(),
        usage_revision: 0,
    });
    Driver::new(InitialSession {
        rt_resolved: config.effective.turn.owner(),
        config,
        parts: SessionParts {
            agent: "agent".into(),
            max_unrecoverable_errors: 3,
            ..Default::default()
        },
        instructions: Default::default(),
        dynamic_instructions: None,
        output_nodes: Default::default(),
        toolset: Default::default(),
        hook_bound: None,
        expressive: Default::default(),
        sinks: Default::default(),
        video_instance: None,
        video_sampler: None,
        eot_thresholds: None,
        eot_provider: None,
        rt_session: None,
        rt_caps: Default::default(),
        rt_disabled: false,
        view,
        next_id: Box::new({
            let mut next = 0;
            move || {
                next += 1;
                next
            }
        }),
        now: Now { at: lk_agents_abi::Ts::ZERO, wall: std::time::UNIX_EPOCH },
        built_at: 0.0,
        journalled: false,
        forward_chat_context: true,
        keyterms: None,
    })
}

// Catches: a second session in one job recording, or hosting the remote session.
#[test]
fn only_the_primary_session_records_or_hosts() {
    let plan = |primary_taken| SessionStartPlans::plan_session_start(true, primary_taken);
    let first = plan(false);
    assert!(first.is_primary && first.hosting && first.records && first.recording.audio);
    let second = plan(true);
    assert!(!second.is_primary && !second.hosting && !second.records);
}

// Catches: interruption metrics never reaching `metrics_collected`. The detector's verdict carries
// them; the session reported only the overlap event.
#[test]
fn an_overlap_verdict_reports_its_interruption_metrics() {
    use crate::agent_session::effects::Effect;
    use lk_agents_abi::Ts;
    use lk_agents_abi::metrics::base::{AgentMetrics, InterruptionMetrics};
    use lk_agents_abi::turn::detector::LegEvent;
    use lk_agents_abi::turn::interruption::OverlapVerdict;

    let driver = &mut driver();
    let mut out = driver.out();
    let metrics = InterruptionMetrics {
        timestamp: 1.0,
        total_duration: 0.2,
        prediction_duration: 0.02,
        detection_delay: 0.3,
        num_interruptions: 1,
        num_backchannels: 0,
        num_requests: 1,
        metadata: None,
    };
    let verdict = OverlapVerdict {
        is_interruption: true,
        agent_ended: false,
        ignore_user_transcript_until: Ts::ZERO,
        reported: None,
        metrics: Some(metrics.clone()),
    };
    driver.on_overlap_verdict(Ts::ZERO, LegEvent::Verdict { req: None, verdict }, &mut out);
    let reported = driver.settle_step(out).into_iter().find_map(|effect| match effect {
        Effect::Emit(SessionEvent::MetricsCollected { metrics, .. }) => match *metrics {
            AgentMetrics::Interruption(m) => Some(m),
            _ => None,
        },
        _ => None,
    });
    assert_eq!(reported, Some(metrics));
}
