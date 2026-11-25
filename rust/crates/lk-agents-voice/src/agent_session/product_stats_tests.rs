use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::machine::activity::handoff::HandoffEvent;
use crate::agent_session::tests::{
    agent_update, driver, driver_with, realtime_driver, realtime_driver_on,
};
use crate::agent_session::time::Now;
use crate::agent_session::{SessionParts, VoiceDriver};
use crate::audio::Tagged;
use crate::audio_recognition::{EndOfTurnInfo, TurnDetectionMode, TurnId, TurnSnapshot};
use crate::events::SessionEvent;
use lk_agents_abi::metrics::base::{AgentMetrics, RealtimeModelMetrics};
use lk_agents_abi::productstats::models::EndpointedBy;
use lk_agents_abi::resources::{LlmRef, ResourceId, SttRef, TtsRef};
use lk_agents_abi::stt::events::SpeechData;
use lk_agents_abi::telemetry::record::Record;
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::voice::session_options::{DetectorKind, SessionLegs};
use lk_agents_abi::{Maybe, agent::AgentDefinition};
use lk_agents_abi::{ProviderId, RecognitionUsage, SpeechEvent, SpeechEventType, Ts};
use lk_agents_telemetry::productstats::testkit::{Stat, stat};
use std::sync::Arc;
use std::time::Duration;

/// The product statistics among `records`.
fn stats_of(records: impl IntoIterator<Item = Record>) -> Vec<Stat> {
    let stats = records.into_iter().filter_map(|record| match record {
        Record::ProductStat(measured) => Some(stat(measured.into_measurement())),
        _ => None,
    });
    stats.collect()
}

/// The product statistics the session measured since the last call, handed over or not.
fn taken(driver: &mut VoiceDriver) -> Vec<Stat> {
    let mut stats = handed_over(driver.take_outbox());
    stats.extend(stats_of(driver.telemetry.drain()));
    stats
}

/// The product statistics a step handed to the runtime with its telemetry records.
fn handed_over(effects: Vec<VoiceEffect>) -> Vec<Stat> {
    let records = effects.into_iter().filter_map(|effect| match effect {
        VoiceEffect::Emit(SessionEvent::Telemetry(records)) => Some(records),
        _ => None,
    });
    stats_of(records.flat_map(|records| records.to_vec()))
}

/// One statistic, with the four model labels and then `more`.
fn model_stat(
    metric: &'static str,
    [kind, plugin, provider, name]: [&str; 4],
    more: &[(&str, &str)],
    value: f64,
) -> Stat {
    let labels = [
        ("lk.product.agents.model.kind", kind),
        ("lk.product.agents.model.plugin", plugin),
        ("lk.product.agents.model.provider", provider),
        ("lk.product.agents.model.name", name),
    ];
    let labels = labels.iter().chain(more).map(|(k, v)| ((*k).to_owned(), (*v).to_owned()));
    (metric, labels.collect(), value)
}

/// [`driver`] on a session given these legs, with its default options resolved over them.
/// Its audio output pauses when `legs.sink_can_pause`.
fn driver_on(legs: lk_agents_abi::voice::session_options::SessionLegs) -> VoiceDriver {
    let audio = lk_agents_abi::resources::AudioOutputRef {
        id: lk_agents_abi::resources::ResourceId(90),
        label: "audio".into(),
        capabilities: lk_agents_abi::voice::rtc::AudioOutputCapabilities {
            pause: legs.sink_can_pause,
        },
    };
    let parts = SessionParts {
        agent: "agent".into(),
        max_unrecoverable_errors: 3,
        outputs: crate::agent_session::Outputs::new(Some(audio), None),
        ..Default::default()
    };
    let now = Now { at: lk_agents_abi::Ts::ZERO, wall: std::time::UNIX_EPOCH };
    VoiceDriver::new(crate::agent_session::tests::initial_session_on(legs, parts, None, now))
}

/// A session whose STT is LiveKit Inference's `deepgram/nova-3:en`.
fn inference_stt_session() -> VoiceDriver {
    let stt = SttRef {
        id: ResourceId(7),
        provider: ProviderId {
            label: "livekit.agents.inference.stt.STT".into(),
            model: "deepgram/nova-3:en".into(),
            provider: "livekit".into(),
        },
        capabilities: Default::default(),
        instance: 0,
    };
    let parts = SessionParts { agent: "agent".into(), stt: Some(stt), ..Default::default() };
    driver_with(parts, None, Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH })
}

// Catches: the audio an STT heard not counted, counted under the Inference gateway instead of
// the vendor and model it runs (`livekit`, `deepgram/nova-3:en`), or a turn's transcription
// delay recorded without what ended the speech, or recorded for a turn that has none.
#[test]
fn the_session_stt_is_counted_by_what_it_heard_and_how_late_it_transcribed() {
    let mut session = inference_stt_session();
    taken(&mut session);
    let nova = ["stt", "livekit-inference", "deepgram", "nova-3"];

    let mut usage = SpeechEvent::new(SpeechEventType::RecognitionUsage);
    usage.recognition_usage =
        Some(Box::new(RecognitionUsage { audio_duration: Some(2.5), ..Default::default() }));
    session.on_stt(Ts::ZERO, Tagged { stream: ResourceId(7), item: usage });
    assert_eq!(
        taken(&mut session),
        [model_stat("lk.product.agents.model.audio_duration", nova, &[], 2.5)]
    );

    let mut turn = EndOfTurnInfo {
        turn_id: TurnId(1),
        owner: TurnDetectionMode::Vad,
        stt_enabled: true,
        skip_reply: false,
        new_transcript: "book a table".into(),
        transcript_confidence: 0.9,
        metrics: Default::default(),
        backchannel_over_agent: false,
        late: false,
        snapshot: Default::default(),
        promoted: None,
    };
    session.count_transcription_delay(&turn);
    assert_eq!(taken(&mut session), [], "a turn with no delay to measure records nothing");
    // The VAD ended the speech at 1.0 s; the final arrived at 1.3 s.
    turn.metrics.transcription_delay = Some(Duration::from_millis(300));
    turn.snapshot = TurnSnapshot {
        last_speaking_time: Some(Ts::from_nanos(1_000_000_000)),
        speech_ended_by: Some(EndpointedBy::Vad),
        last_final_transcript_time: Some(Ts::from_nanos(1_300_000_000)),
        speech_start_time: Some(Ts::ZERO),
    };
    session.count_transcription_delay(&turn);
    let endpointing = [("lk.product.agents.model.transcription_delay.endpointing", "vad")];
    assert_eq!(
        taken(&mut session),
        [model_stat("lk.product.agents.model.transcription_delay", nova, &endpointing, 0.3)]
    );
}

// Catches: a realtime response not counted as one request with its tokens and first token, a
// cancelled response timed, the labels taken from what a plugin wrote on its metrics instead of
// the model installed, or a duplex model counted as realtime.
#[test]
fn a_realtime_response_is_one_request_of_the_installed_model() {
    let response = |cancelled: bool| {
        AgentMetrics::RealtimeModel(RealtimeModelMetrics {
            label: "written.by.the.Plugin".into(),
            ttft: Some(0.4),
            cancelled,
            tokens: TokenUsage { input_tokens: 30, output_tokens: 9, ..TokenUsage::default() },
            ..lk_agents_abi::metrics::testkit::realtime()
        })
    };
    let mut session = realtime_driver();
    taken(&mut session);
    let model = ["realtime", "custom", "unknown", "unknown"];
    session.count_realtime_response(&response(false));
    assert_eq!(
        taken(&mut session),
        [
            model_stat("lk.product.agents.model.requests", model, &[], 1.0),
            model_stat("lk.product.agents.model.input_tokens", model, &[], 30.0),
            model_stat("lk.product.agents.model.output_tokens", model, &[], 9.0),
            model_stat("lk.product.agents.model.ttft", model, &[], 0.4),
        ]
    );
    session.count_realtime_response(&response(true));
    let counted: Vec<_> = taken(&mut session).into_iter().map(|(metric, ..)| metric).collect();
    assert!(!counted.contains(&"lk.product.agents.model.ttft"), "{counted:?}");

    let mut duplex = realtime_driver_on(|model| model.supports_overlapping_speech = true);
    taken(&mut duplex);
    duplex.count_realtime_response(&response(false));
    let kinds: Vec<_> =
        taken(&mut duplex).into_iter().map(|(_, labels, _)| labels[0].1.clone()).collect();
    assert!(kinds.iter().all(|kind| kind == "duplex"), "{kinds:?}");
}

/// The statistics named `metric`, each as its label values in declaration order.
fn counted(stats: &[Stat], metric: &str) -> Vec<Vec<String>> {
    let named = stats.iter().filter(|(name, ..)| *name == metric);
    named.map(|(_, labels, _)| labels.iter().map(|(_, v)| v.clone()).collect()).collect()
}

// Catches: a voice session not counted once when it starts, counted again when it opens twice,
// or its pipeline taken from the wrong model (a duplex model counted as realtime).
#[test]
fn a_voice_session_is_counted_once_at_its_start_with_its_configuration() {
    let now = Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH };
    let started = |mut session: VoiceDriver| {
        taken(&mut session);
        let mut stats = handed_over(session.open(now));
        stats.extend(handed_over(session.open(now)));
        counted(&stats, "lk.product.agents.voice.sessions")
    };
    // [pipeline, turn detection, interruption, false interruption]
    assert_eq!(started(driver()), [["stt_llm_tts", "manual", "vad", "off"]]);
    let pipeline = SessionLegs {
        stt: true,
        vad: true,
        llm: true,
        tts: true,
        sink_can_pause: true,
        ..SessionLegs::none()
    };
    assert_eq!(started(driver_on(pipeline)), [["stt_llm_tts", "vad", "vad", "on"]]);
    // A turn detection model and the interruption model, both given to the session.
    let models = SessionLegs {
        detector: Some(DetectorKind::Streaming),
        adaptive_detector: true,
        ..pipeline
    };
    assert_eq!(started(driver_on(models)), [["stt_llm_tts", "model", "adaptive", "on"]]);
    // No VAD: the STT ends the turns; a sink that cannot pause resumes nothing.
    let stt_only = SessionLegs { vad: false, sink_can_pause: false, ..pipeline };
    assert_eq!(started(driver_on(stt_only)), [["stt_llm_tts", "stt", "vad", "off"]]);
    assert_eq!(started(realtime_driver())[0][0], "realtime");
    let duplex = realtime_driver_on(|model| model.supports_overlapping_speech = true);
    assert_eq!(started(duplex)[0][0], "duplex");
}

// Catches: a session not counted once when it ends, counted when it starts, or an agent replacing
// another not reported as a handoff.
#[test]
fn a_session_is_counted_once_at_its_end_with_its_topology() {
    let ended = |mut session: VoiceDriver| {
        assert_eq!(counted(&taken(&mut session), "lk.product.agents.sessions"), [[""; 0]; 0]);
        session.close_spans();
        session.close_spans();
        counted(&taken(&mut session), "lk.product.agents.sessions")
    };
    // [entrypoint, topology, delegation]
    assert_eq!(ended(driver()), [["rtc_session", "single", "none"]]);
    let mut handed = driver();
    let update = Box::new(agent_update("billing", None));
    handed.handoff_step(HandoffEvent::Asked { update, run: None });
    handed.settle_step();
    assert_eq!(ended(handed), [["rtc_session", "handoff", "none"]]);
}

fn model(label: &str, name: &str) -> ProviderId {
    ProviderId { label: label.into(), model: name.into(), provider: "acme".into() }
}

fn given<T>(model: T) -> Maybe<Option<T>> {
    Maybe::Given(Some(model))
}

fn tts(id: u64, name: &str) -> TtsRef {
    TtsRef {
        id: ResourceId(id),
        provider: model("livekit.plugins.acme.tts.TTS", name),
        capabilities: Default::default(),
        num_channels: 1,
    }
}

fn llm(id: u64, name: &str) -> LlmRef {
    LlmRef { id: ResourceId(id), provider: model("livekit.plugins.acme.llm.LLM", name) }
}

/// The features counted among `stats`: each metric's last name, with its labels' values.
fn features(stats: &[Stat]) -> Vec<(String, Vec<String>)> {
    let named = stats.iter().filter_map(|(metric, labels, value)| {
        let feature = metric.strip_prefix("lk.product.agents.feature.")?;
        assert_eq!(*value, 1.0, "a session adds 1 to a feature, never more");
        Some((feature.to_owned(), labels.iter().map(|(_, v)| v.clone()).collect()))
    });
    named.collect()
}

// Catches: a model re-installed unchanged (every handoff names the agent's models) counted as a
// switch, a kind switched twice counted twice, a switch of one kind hiding another's, or the
// switch counted under the wrong kind.
#[test]
fn a_model_switch_is_counted_once_for_each_kind_changed() {
    let parts = SessionParts {
        agent: "agent".into(),
        tts: Some(tts(1, "sonic-2")),
        llm: Some(llm(2, "gpt-4o-mini")),
        ..Default::default()
    };
    let mut session = driver_with(parts, None, Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH });
    taken(&mut session);
    // The same models, registered again under new ids.
    session.update_models(
        Maybe::NotGiven,
        Maybe::NotGiven,
        given(llm(4, "gpt-4o-mini")),
        given(tts(3, "sonic-2")),
    );
    assert_eq!(features(&taken(&mut session)), []);
    session.update_models(Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven, given(tts(5, "v3")));
    session.update_models(Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven, given(tts(6, "v4")));
    // An agent that brings its own LLM.
    let mut update = agent_update("billing", None);
    update.llm = given(llm(7, "gpt-4.1"));
    session.handoff_step(HandoffEvent::Asked { update: Box::new(update), run: None });
    let mut stats = taken(&mut session);
    stats.extend(handed_over(session.settle_step()));
    let switched = |kind: &str| ("model_switch".to_owned(), vec![kind.to_owned()]);
    assert_eq!(features(&stats), [switched("tts"), switched("llm")]);
}

// Catches: the person's first language counted as a switch, a region or an unspecific code
// (`multi`) taken for another language, an STT that reports no language counted, or a second
// switch counted again.
#[test]
fn a_language_switch_is_counted_once_when_the_stt_reports_another_language() {
    let mut session = inference_stt_session();
    session.input.live_stt = Some(ResourceId(7));
    taken(&mut session);
    let hear = |session: &mut VoiceDriver, language: &str| {
        let mut heard = SpeechEvent::new(SpeechEventType::FinalTranscript);
        heard.alternatives.push(SpeechData::new(language, "a sentence long enough to tell"));
        session.on_stt(Ts::ZERO, Tagged { stream: ResourceId(7), item: heard });
        features(&taken(session))
    };
    for same in ["", "en", "en-US", "multi", "en-GB"] {
        assert_eq!(hear(&mut session, same), [], "{same}");
    }
    assert_eq!(hear(&mut session, "fr"), [("language_switch".to_owned(), vec![])]);
    assert_eq!(hear(&mut session, "en"), []);
}

// Catches: an agent's MCP toolset not counted when it runs, counted for an agent without one, or
// counted again for the next agent that has one.
#[test]
fn an_agent_with_an_mcp_toolset_counts_the_feature_once() {
    let mut session = driver();
    taken(&mut session);
    let hand_to = |session: &mut VoiceDriver, agent: &str, mcp: bool| {
        let mut update = agent_update(agent, None);
        let definition = AgentDefinition {
            mcp,
            name: agent.into(),
            instructions: String::new(),
            dynamic_instructions: false,
            tools: Vec::new(),
            output: lk_agents_abi::agent::OutputSpec::Text { validated: false },
            retries: Default::default(),
        };
        update.toolset.agent = Some(Arc::new(Frame(definition)));
        session.handoff_step(HandoffEvent::Asked { update: Box::new(update), run: None });
        let mut stats = handed_over(session.settle_step());
        stats.extend(taken(session));
        features(&stats)
    };
    assert_eq!(hand_to(&mut session, "plain", false), []);
    assert_eq!(hand_to(&mut session, "tools", true), [("mcp".to_owned(), vec![])]);
    assert_eq!(hand_to(&mut session, "more", true), []);
}

struct Frame(AgentDefinition);

impl lk_agents_abi::agent::AgentFrame for Frame {
    fn definition(&self) -> &AgentDefinition {
        &self.0
    }
    fn host(&self) -> &Arc<dyn lk_agents_abi::agent::AgentCallbacks> {
        unreachable!("no callback runs")
    }
    fn host_context(&self) -> &lk_agents_abi::HostRef {
        unreachable!("no callback runs")
    }
}
