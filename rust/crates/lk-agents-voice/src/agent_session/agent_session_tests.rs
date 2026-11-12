use super::*;
use crate::agent_session::lifecycle::SessionStartPlans;
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::events::SessionEvent;
use lk_agents_abi::errors::ProviderError;
use lk_agents_core::markup::ChatMessageExt;
use std::sync::Arc;

#[test]
fn pending_voice_progress_is_coalesced_until_the_writer_acknowledges() {
    use crate::agent_session::effects::VoiceEffect;
    let mut driver = driver();
    driver.take_outbox();
    driver.session.storage.enabled = true;
    let speech = SpeechId(10);
    driver.storage_started(speech);
    let partial = |text| {
        lk_agents_abi::ChatMessage::text(
            "reply",
            lk_agents_abi::Role::Assistant,
            text,
            std::time::UNIX_EPOCH,
        )
    };
    driver.storage_progress(speech, partial("first"));
    let checkpoint = driver
        .take_outbox()
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::StorageCheckpoint { checkpoint, .. } => Some(*checkpoint),
            _ => None,
        })
        .unwrap();
    for _ in 0..100 {
        driver.storage_progress(speech, partial("latest"));
    }
    assert!(
        !driver
            .take_outbox()
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::StorageCheckpoint { .. }))
    );
    driver.storage_checkpointed(lk_agents_abi::Ts::ZERO, checkpoint, Ok(()));
    let effects = driver.take_outbox();
    let histories: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::StorageCheckpoint { history, .. } => Some(history),
            _ => None,
        })
        .collect();
    assert_eq!(histories.len(), 1);
    assert_eq!(
        histories[0].last().unwrap().as_message().unwrap().raw_text_content().as_deref(),
        Some("latest")
    );
}

#[test]
fn a_failed_generation_drains_accepted_progress_before_abandoning_its_run() {
    use crate::agent_session::effects::VoiceEffect;
    let mut driver = driver();
    driver.take_outbox();
    driver.session.storage.enabled = true;
    let speech = SpeechId(9);
    driver.storage_started(speech);
    driver.storage_progress(
        speech,
        lk_agents_abi::ChatMessage::text(
            "reply",
            lk_agents_abi::Role::Assistant,
            "heard",
            std::time::UNIX_EPOCH,
        ),
    );
    let checkpoint = driver
        .take_outbox()
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::StorageCheckpoint { checkpoint, .. } => Some(*checkpoint),
            _ => None,
        })
        .unwrap();
    driver.storage_abandon(speech, Box::new(crate::generation::fatal("model failed")));
    assert!(
        !driver.take_outbox().iter().any(|effect| matches!(effect, VoiceEffect::StorageAbort(_)))
    );
    driver.storage_checkpointed(lk_agents_abi::Ts::ZERO, checkpoint, Ok(()));
    assert!(
        driver
            .take_outbox()
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::StorageAbort(id) if *id == speech))
    );
}

#[test]
fn a_final_voice_commit_waits_for_delivered_progress_checkpoint_receipts() {
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::{ChatMessage, Role, Ts};
    let mut driver = driver();
    driver.take_outbox();
    driver.session.storage.enabled = true;
    let speech = SpeechId(8);
    driver.storage_started(speech);
    let mut partial = ChatMessage::text("reply", Role::Assistant, "heard", std::time::UNIX_EPOCH);
    partial.interrupted = true;
    driver.storage_progress(speech, partial);
    let effects = driver.take_outbox();
    let checkpoint = effects
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::StorageCheckpoint { checkpoint, history, .. } => {
                assert!(history.iter().any(|item| item.id() == "reply"));
                Some(*checkpoint)
            }
            _ => None,
        })
        .unwrap();
    driver.storage_settle(
        Ts::ZERO,
        speech,
        crate::generation::TurnOutcome {
            forwarded_text: "heard".into(),
            assistant_message: None,
            cut_at: None,
        },
        false,
    );
    assert!(
        !driver
            .take_outbox()
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::StorageCommit { .. }))
    );
    driver.storage_checkpointed(Ts::ZERO, checkpoint, Ok(()));
    assert_eq!(
        driver
            .take_outbox()
            .iter()
            .filter(|effect| matches!(effect, VoiceEffect::StorageCommit { .. }))
            .count(),
        1
    );
    driver.storage_checkpointed(Ts::ZERO, checkpoint, Ok(()));
    assert!(
        !driver
            .take_outbox()
            .iter()
            .any(|effect| matches!(effect, VoiceEffect::StorageCommit { .. }))
    );
}

#[test]
fn bound_storage_loss_and_failed_commit_close_the_call_with_a_storage_error() {
    use lk_agents_abi::storage::StorageError;
    use lk_agents_abi::voice::events::{CloseReason, ErrorSource};
    for event in [
        VoiceEvent::StorageLost(SpeechId(7)),
        VoiceEvent::StorageFailed {
            speech: SpeechId(7),
            error: StorageError::Unavailable("write failed".into()),
        },
    ] {
        let mut driver = driver();
        driver.session.storage.enabled = true;
        driver.handle(
            Some(event),
            crate::agent_session::time::Now {
                at: lk_agents_abi::Ts::ZERO,
                wall: std::time::UNIX_EPOCH,
            },
        );
        let closing =
            driver.session.closing.as_ref().expect("durability failure closes the bound call");
        assert_eq!(closing.reason, CloseReason::Error);
        assert_eq!(closing.error.as_ref().unwrap().source, ErrorSource::Storage);
    }
}

#[test]
fn conversation_updates_publish_once_per_cycle_and_include_view_only_edits() {
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::{ChatMessage, Role};

    let mut driver = driver();
    driver.take_outbox();
    let message = |id| ChatMessage::text(id, Role::User, id, std::time::UNIX_EPOCH).into();
    for id in ["first", "second"] {
        driver.commit_item(
            message(id),
            crate::agent::Source::HostCtx,
            crate::agent::Visibility::Both,
        );
    }
    let contexts = |effects: Vec<VoiceEffect>| {
        effects
            .into_iter()
            .filter_map(|effect| match effect {
                VoiceEffect::ChatContext(ctx) => Some(ctx),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let updates = contexts(driver.settle_step());
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].items.len(), 2);
    assert!(contexts(driver.settle_step()).is_empty());

    let version = driver.session.chat_ctx.version();
    driver.session.chat_ctx.update_agent_chat_ctx(vec![message("replacement")], "agent");
    let updates = contexts(driver.settle_step());
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].items[0].id(), "replacement");
    assert_eq!(driver.session.chat_ctx.version(), version);
}

#[test]
fn terminal_provider_release_does_not_reacquire_resources_from_stale_selection_slots() {
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::resources::{ResourceId, TranscriptionOutputRef};

    let mut driver = driver();
    driver.take_outbox();
    let id = ResourceId(901);
    let mut effects = driver.handle(
        Some(VoiceEvent::Host(Sent {
            origin: VoiceCommandOrigin::Host,
            command: Command::Activity(ActivityCommand::SetTranscriptionOutput(Some(
                TranscriptionOutputRef { id, label: "owned transcript output".into() },
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

#[test]
fn realtime_transport_timeout_keeps_its_limit_and_request_identity_on_session_error() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::realtime::RealtimeWake;
    use lk_agents_abi::Ts;
    use lk_agents_abi::errors::{ApiError, ApiErrorKind, Timeout};
    use lk_agents_abi::llm::realtime as rt;

    let driver = &mut realtime_driver();
    let mut timeout = ApiError::timed_out(Timeout::Connect);
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
    assert_eq!(error.timeout, Some(Timeout::Connect));
    assert_eq!(error.request_id.as_deref(), Some("handshake-7"));
}

#[test]
fn realtime_end_delivers_the_pending_speech_error_before_closing_its_player() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::realtime::{RealtimeWake, Request};
    use crate::agent_session::machine::activity::scheduling::VoiceRunParticipation;
    use lk_agents_abi::{Ts, llm::realtime as rt};

    let mut driver = realtime_driver();
    let speech = driver
        .realtime_request(
            Ts::ZERO,
            Request::GenerateReply { instructions: None, tool_choice: None, read_back: false },
            None,
            None,
            VoiceRunParticipation::Outside,
        )
        .unwrap();
    driver.take_outbox();
    driver.on_realtime(Ts::ZERO, RealtimeWake::Ended);
    let effects = driver.settle_step();
    let failed = effects
        .iter()
        .position(|effect| {
            matches!(effect,
                VoiceEffect::RealtimePlayer { speech: id, event }
                if *id == speech && matches!(**event, rt::Event::GenerateReplyFailed(_))
            )
        })
        .expect("the pending speech lost its terminal error");
    let closed = effects
        .iter()
        .position(|effect| {
            matches!(effect,
                VoiceEffect::CloseRealtimePlayer(id) if *id == speech
            )
        })
        .expect("the failed player was not closed");
    assert!(failed < closed);
}

#[test]
fn late_realtime_transcript_keeps_its_source_without_starting_another_speech() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::realtime::RealtimeWake;
    use lk_agents_abi::llm::realtime as rt;
    use lk_agents_abi::{ChatMessage, Role, Ts};

    let driver = &mut realtime_driver();
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
    let players: Vec<_> = driver.activity.realtime_players().copied().collect();
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
    assert_eq!(
        driver.activity.realtime_players().copied().collect::<Vec<_>>(),
        players,
        "a correction took the voice floor"
    );
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

// Catches: a speech whose tasks were
// aborted (the interruption watchdog, `cancel_tasks`) reports neither done nor failed, so its
// partition was never released and the store renewed its lease until the call closed.
#[test]
fn a_speech_ended_without_its_turn_releases_the_partition_within_30_s() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::storage::PARTITION_RELEASE_BOUND;
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
    let due = Ts::ZERO.saturating_add(PARTITION_RELEASE_BOUND);
    assert_eq!(driver.wake_at(), Some(due), "the release is bounded");
    let effects = driver.handle(None, Now { at: due, wall });
    assert!(
        effects.iter().any(|e| matches!(e, VoiceEffect::StorageAbort(s) if *s == speech)),
        "the partition_key was never released: {effects:?}"
    );
}

// Catches: a `say` whose step reported spoken after the close's hard edge released it read its
// kind off the scheduler's entry, gone with the release, so its item was committed as a reply's,
// naming no voice (`items_carry_their_payload`, survey seed 114526).
#[test]
fn a_say_released_before_its_step_reports_keeps_its_source() {
    use crate::agent::Source;
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::scheduling::SpeechKind;
    use crate::agent_session::machine::activity::speech::SpeechSpec;
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
        cut_at: None,
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

/// A session's decision core with no provider or leg, at the zero instant; nothing runs it, so
/// nothing exports its spans.
pub(crate) fn driver() -> VoiceDriver {
    let parts =
        SessionParts { agent: "agent".into(), max_unrecoverable_errors: 3, ..Default::default() };
    let now = Now { at: lk_agents_abi::Ts::ZERO, wall: std::time::UNIX_EPOCH };
    driver_with(parts, None, now)
}

/// [`driver`] with these parts, hooks hosted and bounded by `hook_bound`, built at `now`.
pub(crate) fn driver_with(
    parts: SessionParts,
    hook_bound: Option<std::time::Duration>,
    now: Now,
) -> VoiceDriver {
    VoiceDriver::new(initial_session(parts, hook_bound, now))
}

pub(crate) fn realtime_driver() -> VoiceDriver {
    let parts = SessionParts { agent: "agent".into(), ..Default::default() };
    let now = Now { at: lk_agents_abi::Ts::ZERO, wall: std::time::UNIX_EPOCH };
    let mut initial = initial_session(parts, None, now);
    initial.rt_session = Some(lk_agents_abi::resources::RealtimeRef {
        id: lk_agents_abi::resources::ResourceId(1),
        provider: lk_agents_abi::ProviderId::of_label("realtime"),
        capabilities: initial.rt_caps,
    });
    VoiceDriver::new(initial)
}

fn initial_session(
    parts: SessionParts,
    hook_bound: Option<std::time::Duration>,
    now: Now,
) -> InitialSession {
    let legs = lk_agents_abi::voice::session_options::SessionLegs::none();
    let config =
        crate::agent_session::machine::activity::DriverConfig::new(Default::default(), legs);
    let view = Arc::new(SessionView {
        launch_holds: Default::default(),
        config: config.clone(),
        agent: parts.agent.clone(),
        agent_scope: 0,
        vad: None,
        usage: Default::default(),
        usage_revision: 0,
    });
    InitialSession {
        rt_resolved: config.effective().turn.owner(),
        config,
        parts,
        instructions: Default::default(),
        dynamic_instructions: None,
        output_nodes: Default::default(),
        toolset: Default::default(),
        hook_bound,
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
        now,
        journalled: false,
        forward_chat_context: true,
        keyterms: None,
    }
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
// counted as a request of its own.
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
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::handoff::HandoffEvent;
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
            activation: None,
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

// Catches: `session_start` not ending with the opening, not reaching the exporter on the event
// stream, or not the span the runtime's start work nests under.
#[test]
fn the_opening_ends_session_start_on_the_event_stream() {
    use crate::agent_session::effects::VoiceEffect;
    use lk_agents_abi::telemetry::record::Record;

    let driver = &mut driver();
    let start = driver.session_start_span().expect("session_start runs from construction");
    let now = Now { at: lk_agents_abi::Ts::ZERO, wall: std::time::UNIX_EPOCH };
    let ended: Vec<_> = driver
        .open(now)
        .into_iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Emit(SessionEvent::Telemetry(records)) => Some(records),
            _ => None,
        })
        .flat_map(|records| records.as_ref().clone())
        .filter_map(|record| match record {
            Record::Span(span) => Some(span),
            _ => None,
        })
        .collect();
    let [span] = &ended[..] else { panic!("{ended:?}") };
    assert_eq!((span.name.as_ref(), span.context), ("session_start", start));
    assert_eq!(driver.session_start_span(), None);
}

// Catches: the terminal admission pass dropping the speech updates and stream events of a
// say()/generate_reply() it refuses, so the runtime answers `created` with no handle behind it.
#[test]
fn a_speech_refused_at_finish_is_created_viewed_released_and_answered() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::speech_handle::SpeechUpdate;

    let host = |command| VoiceEvent::Host(Sent { origin: VoiceCommandOrigin::Host, command });
    let (say, reply) = ((RequestId(1), SpeechId(71)), (RequestId(2), SpeechId(72)));
    let call = |(id, speech)| Some(SpeechCall { id, speech });
    let mut driver = driver();
    driver.take_outbox();
    let effects = driver.finish(vec![
        host(Command::Speech(SpeechCommand::Say {
            text: SayText::Text("hello".into()),
            allow_interruptions: None,
            add_to_chat_ctx: true,
            call: call(say),
        })),
        host(Command::Speech(SpeechCommand::GenerateReply(GenerateReplyRequest {
            user_input: None,
            instructions: None,
            allow_interruptions: None,
            tool_choice: None,
            chat_ctx: None,
            toolset: None,
            tool_names: None,
            on_enter: false,
            call: call(reply),
        }))),
    ]);
    for (request, speech) in [say, reply] {
        let updates: Vec<_> = effects
            .iter()
            .filter_map(|effect| match effect {
                VoiceEffect::Speech(id, update) if *id == speech => Some(update),
                _ => None,
            })
            .collect();
        assert!(
            matches!(
                updates[..],
                [SpeechUpdate::Created, SpeechUpdate::View(view), SpeechUpdate::Released]
                    if view.done()
            ),
            "{speech:?}: {updates:?}"
        );
        assert!(effects.iter().any(|effect| matches!(effect,
            VoiceEffect::Answer(id, Answer::Speech) if *id == request
        )));
        for announced in [true, false] {
            assert!(effects.iter().any(|effect| match effect {
                VoiceEffect::Emit(SessionEvent::SpeechCreated { speech: created, .. }) => {
                    announced && *created == speech
                }
                VoiceEffect::Emit(SessionEvent::SpeechDone { speech: done, .. }) => {
                    !announced && *done == speech
                }
                _ => false,
            }));
        }
    }
}

/// An agent with no instructions, tools or providers of its own.
pub(crate) fn agent_update(agent: &str, call: Option<RequestId>) -> AgentUpdate {
    use lk_agents_abi::Maybe;
    AgentUpdate {
        agent: agent.into(),
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
        call,
        activation: None,
    }
}

/// Two handoffs asked in one tool batch both land, in the order asked, each answered once.
/// https://github.com/livekit/agents/issues/4729 https://github.com/livekit/agents/issues/5150
/// https://github.com/livekit/agents/issues/7320 https://github.com/livekit/agents/issues/8477
#[test]
fn parallel_handoffs_land_in_order_and_are_each_answered_once() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::handoff::HandoffEvent;
    let mut driver = driver();
    driver.take_outbox();
    for (agent, call) in [("billing", 1), ("shipping", 2)] {
        let update = Box::new(agent_update(agent, Some(RequestId(call))));
        driver.handoff_step(HandoffEvent::Asked { update, run: None });
    }
    let effects = driver.settle_step();
    let answers: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Answer(call, Answer::Handoff(result)) => Some((call.0, *result)),
            _ => None,
        })
        .collect();
    assert_eq!(answers, [(1, Ok(())), (2, Ok(()))]);
    let handoffs: Vec<_> = driver
        .session
        .chat_ctx
        .items()
        .filter_map(|item| match item {
            lk_agents_abi::ChatItem::AgentHandoff(h) => {
                Some((h.old_agent_id.clone(), h.new_agent_id.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        handoffs,
        [
            (None, "agent".to_owned()),
            (Some("agent".into()), "billing".to_owned()),
            (Some("billing".into()), "shipping".to_owned()),
        ]
    );
    assert_eq!(driver.activity.agent_name(), "shipping");
    assert!(!driver.activity.in_window(), "no switch is left holding new turns");
}

/// A cut realtime reply is cancelled, and truncated to what was heard only where supported.
/// https://github.com/livekit/agents/issues/1016 https://github.com/livekit/agents/issues/1050
/// https://github.com/livekit/agents/pull/3655 https://github.com/livekit/agents/issues/3237
#[test]
fn a_cut_realtime_reply_is_truncated_only_when_the_model_supports_it() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::realtime::RealtimeWake;
    use lk_agents_abi::{RealtimeCommand, Ts, llm::realtime as rt};

    for truncation in [false, true] {
        let parts = SessionParts { agent: "agent".into(), ..Default::default() };
        let now = Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH };
        let mut initial = initial_session(parts, None, now);
        initial.rt_caps.message_truncation = truncation;
        initial.rt_session = Some(lk_agents_abi::resources::RealtimeRef {
            id: lk_agents_abi::resources::ResourceId(1),
            provider: lk_agents_abi::ProviderId::of_label("realtime"),
            capabilities: initial.rt_caps,
        });
        let mut driver = VoiceDriver::new(initial);
        let created = rt::GenerationCreated { response_id: "resp_1".into(), req_id: None };
        driver.on_realtime(Ts::ZERO, RealtimeWake::Event(rt::Event::GenerationCreated(created)));
        let speech = *driver.activity.realtime_players().next().expect("the reply plays");
        driver.take_outbox();
        let cut = rt::Truncate {
            message_id: "msg_1".into(),
            has_audio: true,
            audio_end_ms: 400,
            audio_transcript: None,
        };
        driver.on_realtime(Ts::ZERO, RealtimeWake::Heard { speech, cuts: vec![cut.clone()] });
        let sent: Vec<_> = driver
            .take_outbox()
            .into_iter()
            .filter_map(|effect| match effect {
                VoiceEffect::Realtime { command, .. } => Some(command),
                _ => None,
            })
            .collect();
        let shape: Vec<_> = sent
            .iter()
            .map(|command| match command {
                RealtimeCommand::Interrupt(i) => format!("interrupt {:?}", i.response_id),
                RealtimeCommand::Truncate(t) => {
                    format!("truncate {} {}", t.message_id, t.audio_end_ms)
                }
                other => format!("{other:?}"),
            })
            .collect();
        let mut want = vec![format!("interrupt {:?}", Some("resp_1"))];
        if truncation {
            want.push(format!("truncate {} {}", cut.message_id, cut.audio_end_ms));
        }
        assert_eq!(shape, want, "message_truncation={truncation}");
    }
}

/// A late realtime user transcript is placed where its turn began, before the reply to it.
/// https://github.com/livekit/agents/pull/6616 https://github.com/livekit/agents/issues/2148
#[test]
fn delayed_realtime_transcripts_preserve_onset_order() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::realtime::RealtimeWake;
    use lk_agents_abi::{ChatMessage, Role, Ts, llm::realtime as rt};
    use std::time::{Duration, UNIX_EPOCH};

    let driver = &mut realtime_driver();
    let reply =
        ChatMessage::text("reply", Role::Assistant, "Sure.", UNIX_EPOCH + Duration::from_secs(5));
    driver.commit_item(
        reply.into(),
        crate::agent::Source::Remote { item_id: "reply".into() },
        crate::agent::Visibility::Both,
    );
    driver.take_outbox();
    let mut told = 0;
    for transcript in ["book a table", "book a table for two"] {
        let final_ = rt::InputTranscription {
            item_id: "user_1".into(),
            transcript: transcript.into(),
            is_final: true,
            confidence: None,
            turn_started_at: Some(2.0),
        };
        driver.on_realtime(Ts::ZERO, RealtimeWake::Event(rt::Event::InputTranscription(final_)));
        told += driver
            .settle_step()
            .iter()
            .filter(|effect| {
                matches!(effect, VoiceEffect::Emit(SessionEvent::UserInputTranscribed(t))
                    if t.is_final && t.ev.alternatives.first().is_some_and(|a| a.text == transcript))
            })
            .count();
    }
    assert_eq!(told, 2, "each final is told once");
    let order: Vec<(String, Option<String>)> = driver
        .session
        .chat_ctx
        .items()
        .filter_map(|item| item.as_message())
        .map(|m| (m.id.clone(), m.text_content()))
        .collect();
    assert_eq!(
        order,
        [
            ("user_1".to_owned(), Some("book a table for two".to_owned())),
            ("reply".to_owned(), Some("Sure.".to_owned())),
        ]
    );
    let user = driver.session.chat_ctx.entry("user_1").unwrap().item.as_message().unwrap();
    assert_eq!(user.created_at, UNIX_EPOCH + Duration::from_secs(2));
    assert_eq!(user.metrics.started_speaking_at, Some(2.0));
}

/// Each realtime response is metered once as one request, also when its usage lands mid-handoff.
/// https://github.com/livekit/agents/pull/2097 https://github.com/livekit/agents/pull/3168
/// https://github.com/livekit/agents/issues/2222 https://github.com/livekit/agents/pull/1080
#[test]
fn realtime_usage_is_metered_once_per_response_across_handoffs() {
    use crate::agent_session::machine::activity::handoff::HandoffEvent;
    use crate::agent_session::machine::activity::realtime::RealtimeWake;
    use lk_agents_abi::metrics::base::{AgentMetrics, RealtimeModelMetrics};
    use lk_agents_abi::usage::{ModelUsage, TokenUsage};
    use lk_agents_abi::{Ts, llm::realtime as rt};

    let metrics = |response: &str| {
        rt::Event::Metrics(rt::RealtimeMetrics {
            metrics: Box::new(AgentMetrics::RealtimeModel(RealtimeModelMetrics {
                label: "realtime".into(),
                request_id: response.into(),
                timestamp: 0.0,
                duration: 1.0,
                connection_duration: None,
                ttft: None,
                cancelled: false,
                tokens: TokenUsage { input_tokens: 100, output_tokens: 20, ..Default::default() },
                tokens_per_second: 20.0,
                acquire_time: None,
                connection_reused: false,
                metadata: None,
            })),
        })
    };
    let created = |response: &str| {
        rt::Event::GenerationCreated(rt::GenerationCreated {
            response_id: response.into(),
            req_id: None,
        })
    };
    let driver = &mut realtime_driver();
    let update = Box::new(agent_update("billing", Some(RequestId(1))));
    driver.handoff_step(HandoffEvent::Asked { update, run: None });
    driver.settle_step();
    assert_eq!(driver.activity.agent_name(), "billing", "the first switch landed");
    driver.on_realtime(Ts::ZERO, RealtimeWake::Event(created("resp_1")));
    driver.on_realtime(Ts::ZERO, RealtimeWake::Event(metrics("resp_1")));
    // The next switch drains behind the second reply, whose usage arrives mid-switch.
    driver.on_realtime(Ts::ZERO, RealtimeWake::Event(created("resp_2")));
    let update = Box::new(agent_update("shipping", Some(RequestId(2))));
    driver.handoff_step(HandoffEvent::Asked { update, run: None });
    driver.on_realtime(Ts::ZERO, RealtimeWake::Event(metrics("resp_2")));
    driver.settle_step();

    let usage = driver.session.usage.snapshot();
    let [ModelUsage::Llm(llm)] = usage.models.as_slice() else { panic!("{:?}", usage.models) };
    assert_eq!((llm.requests, llm.unmeasured_requests), (2, 0));
    assert_eq!((llm.tokens.input_tokens, llm.tokens.output_tokens), (200, 40));
    assert_eq!(usage.requests, 2);
}

/// A realtime transcript's `user_input_transcribed` names the item the server transcribed.
#[test]
fn a_realtime_transcript_names_the_servers_item() {
    use crate::agent_session::effects::VoiceEffect;
    use crate::agent_session::machine::activity::realtime::RealtimeWake;
    use lk_agents_abi::{Ts, llm::realtime as rt};

    let driver = &mut realtime_driver();
    let heard = rt::InputTranscription {
        item_id: "item_7".into(),
        transcript: "book a table".into(),
        is_final: true,
        confidence: None,
        turn_started_at: None,
    };
    driver.on_realtime(Ts::ZERO, RealtimeWake::Event(rt::Event::InputTranscription(heard)));

    let named: Vec<Option<String>> = driver
        .settle_step()
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Emit(SessionEvent::UserInputTranscribed(t)) => Some(t.item_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(named, [Some("item_7".to_owned())]);
}
