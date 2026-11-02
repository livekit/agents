//! Session decisions over a decision core no executor runs.

#[test]
fn terminal_provider_release_does_not_reacquire_resources_from_stale_selection_slots() {
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::resources::{ResourceId, TextOutputRef};

    let mut driver = driver();
    driver.take_outbox();
    let id = ResourceId(901);
    let mut effects = driver.handle(
        Some(VoiceEvent::Host(Sent {
            origin: VoiceCommandOrigin::Host,
            command: Command::Activity(ActivityCommand::SetTranscriptionOutput(Some(
                TextOutputRef { id, label: "owned transcript output".into() },
            ))),
        })),
        Now { at: lk_agents_abi::Ts::ZERO, wall: std::time::UNIX_EPOCH },
    );
    effects.extend(driver.finish(Vec::new()));
    // Runtime drop repeats cleanup after normal finish, while terminal state may still name
    // its old selections. A selection is not a second ownership admission.
    driver.release_all_providers();
    effects.extend(driver.take_outbox());
    assert_eq!(
        effects
            .iter()
            .filter(|effect| matches!(effect,
                VoiceEffect::ProviderRelease(released) if *released == id
            ))
            .count(),
        1
    );
}

use super::*;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::lifecycle::SessionStartPlans;
use crate::events::SessionEvent;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::session_effects::SessionStart as _;
use lk_agents_core::tts::markup::ChatMessageExt;
use std::sync::Arc;

#[test]
fn realtime_transport_timeout_keeps_its_phase_and_request_identity_on_session_error() {
    use crate::agent_activity::realtime::RealtimeWake;
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::Ts;
    use lk_agents_abi::errors::{ApiError, ApiErrorKind, TimeoutPhase};
    use lk_agents_abi::llm::realtime as rt;

    let driver = &mut driver();
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
    );
    let effects = driver.settle_step();
    let error = effects
        .into_iter()
        .find_map(|effect| match effect {
            VoiceEffect::Emit(SessionEvent::Error(error)) => Some(error),
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
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::llm::realtime as rt;
    use lk_agents_abi::{ChatMessage, Role, Ts};

    let driver = &mut driver();
    let source =
        ChatMessage::text("old-response", Role::Assistant, "Already heard", std::time::UNIX_EPOCH);
    let source_time = source.created_at;
    driver.commit_item(
        source.into(),
        crate::agent::Source::Remote { item_id: "old-response".into() },
        crate::agent::Visibility::Both,
    );
    driver.on_realtime(
        Ts::ZERO,
        RealtimeWake::Event(rt::Event::GenerationCreated(rt::GenerationCreated {
            response_id: "new-response".into(),
            req_id: None,
        })),
    );
    let players = driver.activity.realtime_players().clone();
    driver.take_outbox();
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
        driver.on_realtime(Ts::ZERO, RealtimeWake::Event(event));
    }
    assert_eq!(*driver.activity.realtime_players(), players, "a correction took the voice floor");
    assert!(
        !driver.take_outbox().iter().any(|effect| matches!(
            effect,
            VoiceEffect::RealtimePlayer { .. } | VoiceEffect::StartSpeech { .. }
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
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::storage::LANE_RELEASE_BOUND;
    use crate::agent_session::time::Now;
    use lk_agents_abi::Ts;

    let driver = &mut driver();
    driver.session.storage.enabled = true;
    let speech = SpeechId(7);
    driver.storage_started(speech);
    driver.release_speech(Ts::ZERO, speech);
    let released = driver.settle_step();
    assert!(!released.iter().any(|e| matches!(e, VoiceEffect::StorageAbort(_))));
    let wall = std::time::UNIX_EPOCH;
    let due = Ts::ZERO.saturating_add(LANE_RELEASE_BOUND);
    assert_eq!(driver.wake_at(), Some(due), "the release is bounded");
    let effects = driver.handle(None, Now { at: due, wall });
    assert!(
        effects.iter().any(|e| matches!(e, VoiceEffect::StorageAbort(s) if *s == speech)),
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
    use crate::agent_session::effects::VoiceEffect;
    use crate::generation::{GenEvent, ReplySource};
    use lk_agents_abi::{ChatMessage, Role, Ts};

    let driver = &mut driver();
    let spec = SpeechSpec::new(SpeechKind::Say, ReplySource::Text("One moment.".into()), true);
    let speech = driver.start_speech(Ts::ZERO, spec);
    driver.release_speech(Ts::ZERO, speech);
    let said = ChatMessage::text("said", Role::Assistant, "One moment.", std::time::UNIX_EPOCH);
    let spoken = GenEvent::StepSpoken {
        speech,
        step: 1,
        interrupted: true,
        has_tool_output: false,
        tools_running: false,
        item: Some(Box::new(said)),
    };
    driver.on_generation(Ts::ZERO, spoken);
    let sources: Vec<Source> = driver
        .settle_step()
        .into_iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Emit(SessionEvent::ConversationItemAdded { source, .. })
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
pub(crate) fn driver() -> VoiceDriver {
    let legs = lk_agents_abi::voice::session_options::SessionLegs::none();
    let config = crate::agent_activity::DriverConfig::new(Default::default(), legs);
    let view = Arc::new(SessionView {
        launch_holds: Default::default(),
        config: config.clone(),
        agent: "agent".into(),
        agent_scope: 0,
        vad: None,
        usage: Default::default(),
        usage_revision: 0,
    });
    VoiceDriver::new(InitialSession {
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
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::Ts;
    use lk_agents_abi::metrics::base::{AgentMetrics, InterruptionMetrics};
    use lk_agents_abi::turn::detector::LegEvent;
    use lk_agents_abi::turn::interruption::OverlapVerdict;

    let driver = &mut driver();
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
    driver.on_overlap_verdict(Ts::ZERO, LegEvent::Verdict { req: None, verdict });
    let reported = driver.settle_step().into_iter().find_map(|effect| match effect {
        VoiceEffect::Emit(SessionEvent::MetricsCollected { metrics, .. }) => match *metrics {
            AgentMetrics::Interruption(m) => Some(m),
            _ => None,
        },
        _ => None,
    });
    assert_eq!(reported, Some(metrics));
}

// Catches: a recognition stream's usage report arriving after a later stream also ended was
// counted as a request of its own (5168d8080).
#[test]
fn a_late_usage_report_never_creates_a_request() {
    use crate::audio::Tagged;
    use lk_agents_abi::Ts;
    use lk_agents_abi::resources::ResourceId;
    use lk_agents_abi::stt::events::{RecognitionUsage, SpeechEvent, SpeechEventType};
    use lk_agents_core::usage::ModelKind;

    let driver = &mut driver();
    for stream in [1, 2] {
        let request = driver.session.start_request(ModelKind::Stt, "deepgram", "nova-3");
        driver.session.stt_requests.insert(ResourceId(stream), request);
    }
    driver.stt_stream_ended(ResourceId(1));
    driver.stt_stream_ended(ResourceId(2));
    let mut report = SpeechEvent::new(SpeechEventType::RecognitionUsage);
    report.recognition_usage =
        Some(Box::new(RecognitionUsage { audio_duration: Some(2.0), ..Default::default() }));
    driver.on_stt(Ts::ZERO, Tagged { stream: ResourceId(1), item: report });
    driver.settle_step();
    assert_eq!(driver.session.usage.snapshot().requests, 2);
}

#[test]
fn startup_and_out_of_step_effects_are_drained_exactly_once() {
    use crate::agent_session::effects::VoiceEffect;
    let mut driver = driver();
    driver.emit_now(SessionEvent::BackchannelConfirmed);
    let now = Now { at: lk_agents_abi::Ts::ZERO, wall: std::time::UNIX_EPOCH };
    let startup = driver.open(now);
    assert_eq!(
        startup
            .iter()
            .filter(|effect| matches!(
                effect,
                VoiceEffect::Emit(SessionEvent::BackchannelConfirmed)
            ))
            .count(),
        1
    );
    assert!(driver.take_outbox().is_empty(), "open already returned this execution batch");
    assert!(driver.open(now).is_empty(), "opening twice cannot redeliver startup effects");
    driver.emit_now(SessionEvent::BackchannelConfirmed);
    let next = driver.handle(None, now);
    assert_eq!(
        next.iter()
            .filter(|effect| matches!(
                effect,
                VoiceEffect::Emit(SessionEvent::BackchannelConfirmed)
            ))
            .count(),
        1
    );
    assert!(driver.take_outbox().is_empty());
    assert!(
        !driver
            .handle(None, now)
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::Emit(SessionEvent::BackchannelConfirmed)))
    );
}

#[test]
fn deadlines_are_observational_and_due_at_their_exact_bound() {
    use crate::agent_session::effects::{Timer, VoiceEffect};
    use crate::trace::{CoreStep, Step};
    use lk_agents_abi::Ts;
    for nanos in [99, 100, 101] {
        let mut driver = driver();
        driver.journalled = true;
        let due = Ts::from_nanos(100);
        driver.set_deadline(Timer::DtmfDebounce, due);
        driver.settle_step(); // Finish the transition that installed the deadline.
        for _ in 0..3 {
            assert_eq!(driver.deadline(Timer::DtmfDebounce), Some(due));
            assert_eq!(driver.wake_at(), Some(due));
        }
        assert!(driver.take_outbox().is_empty(), "deadline queries must not add observations");
        let effects =
            driver.handle(None, Now { at: Ts::from_nanos(nanos), wall: std::time::UNIX_EPOCH });
        assert_eq!(effects.iter().filter(|effect| matches!(effect,
            VoiceEffect::Journal(step) if matches!(step.as_ref(), Step::Core(CoreStep::TimerFired(_, _)))
        )).count(), usize::from(nanos >= 100), "wake at {nanos} fired the wrong deadline");
        assert_eq!(driver.deadline(Timer::DtmfDebounce), (nanos < 100).then_some(due));
        driver.clear_deadline(Timer::DtmfDebounce);
        assert_eq!(driver.deadline(Timer::DtmfDebounce), None);
    }
}

#[test]
fn speech_created_by_a_callback_is_scheduled_without_another_input() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::coordination::VoiceCallback;
    use crate::trace::{SpeechStep, Step};
    let mut driver = driver();
    driver.journalled = true;
    driver.defer_callback(VoiceCallback::ReadBackDtmf("1 2".into()));
    let effects = driver.settle_step();
    assert_eq!(effects.iter().filter(|effect| matches!(effect,
        VoiceEffect::Journal(step) if matches!(step.as_ref(), Step::Speech(SpeechStep::Authorized { .. }))
    )).count(), 1, "callback-created speech must acquire the available floor in this transition");
    assert!(!driver.settle_step().iter().any(|effect| matches!(effect,
        VoiceEffect::Journal(step) if matches!(step.as_ref(), Step::Speech(SpeechStep::Authorized { .. }))
    )), "settling again cannot authorize a step twice");
}

#[test]
fn a_handoff_drains_accepted_callback_speech_and_holds_new_host_speech() {
    use crate::agent_activity::handoff::HandoffEvent;
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::coordination::VoiceCallback;
    use crate::trace::{SpeechStep, Step};
    use lk_agents_abi::Maybe;
    let mut driver = driver();
    driver.journalled = true;
    // This callback is ahead of the pause callback in FIFO, but the handoff has already
    // installed its draining phase. Advancement must consult that state, not scan the queue.
    driver.defer_callback(VoiceCallback::ReadBackDtmf("1 2".into()));
    driver.handoff_step(HandoffEvent::Asked {
        update: Box::new(AgentUpdate {
            agent: "next".into(),
            instructions: Default::default(),
            dynamic_instructions: None,
            toolset: Default::default(),
            chat_ctx: None,
            llm: Maybe::NotGiven,
            tts: Maybe::NotGiven,
            stt: Maybe::NotGiven,
            vad: Maybe::NotGiven,
            options: Default::default(),
            expressive: None,
            output_nodes: Default::default(),
            call: None,
        }),
        run: None,
    });
    let effects = driver.settle_step();
    assert!(effects.iter().any(|effect| matches!(effect,
        VoiceEffect::Journal(step) if matches!(step.as_ref(), Step::Speech(SpeechStep::SchedulingPaused))
    )), "handoff must install its admission pause");
    assert_eq!(effects.iter().filter(|effect| matches!(effect,
        VoiceEffect::Journal(step) if matches!(step.as_ref(), Step::Speech(SpeechStep::Authorized { .. }))
    )).count(), 1, "accepted callback speech must continue through the drain");
    driver.on_speech(
        lk_agents_abi::Ts::ZERO,
        SpeechCommand::Say {
            text: SayText::Text("For the arriving agent".into()),
            allow_interruptions: None,
            add_to_chat_ctx: true,
            call: None,
        },
    );
    assert!(
        !driver
            .settle_step()
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::StartSpeech { .. })),
        "new host speech must remain held for the arriving agent"
    );
}

#[test]
fn callback_speech_cannot_be_authorized_past_the_close_hard_edge() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::coordination::VoiceCallback;
    use crate::trace::{SpeechStep, Step};
    let mut driver = driver();
    driver.journalled = true;
    driver.begin_closing(lk_agents_abi::Ts::ZERO, Close::default());
    driver.settle_step();
    assert!(driver.session.closing_past_edge());
    driver.defer_callback(VoiceCallback::ReadBackDtmf("1 2".into()));
    assert!(!driver.settle_step().iter().any(|effect| matches!(effect,
        VoiceEffect::Journal(step) if matches!(step.as_ref(), Step::Speech(SpeechStep::Authorized { .. }))
    )));
}

#[test]
fn a_stale_dtmf_readback_cannot_restart_after_close_stops_collection() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::coordination::VoiceCallback;
    let mut driver = driver();
    driver.defer_callback(VoiceCallback::ReadBackDtmf("1 2".into()));
    // Close stops DTMF synchronously, before its deferred exit continuation. A callback
    // already queued by that stopped collection cannot recreate its speech during exit.
    driver.begin_closing(lk_agents_abi::Ts::ZERO, Close::default());
    assert!(
        !driver
            .settle_step()
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::StartSpeech { .. }))
    );
}
