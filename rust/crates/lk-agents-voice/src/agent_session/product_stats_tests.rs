use crate::agent_session::tests::{driver_with, realtime_driver, realtime_driver_on};
use crate::agent_session::time::Now;
use crate::agent_session::{SessionParts, VoiceDriver};
use crate::audio::Tagged;
use crate::audio_recognition::{EndOfTurnInfo, TurnDetectionMode, TurnId, TurnSnapshot};
use lk_agents_abi::metrics::base::{AgentMetrics, RealtimeModelMetrics};
use lk_agents_abi::productstats::models::EndpointedBy;
use lk_agents_abi::resources::{ResourceId, SttRef};
use lk_agents_abi::telemetry::record::Record;
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{ProviderId, RecognitionUsage, SpeechEvent, SpeechEventType, Ts};
use lk_agents_telemetry::productstats::testkit::{Stat, stat};
use std::time::Duration;

/// The product statistics the session measured since the last call.
pub(crate) fn taken(driver: &mut VoiceDriver) -> Vec<Stat> {
    let records = driver.telemetry.drain().into_iter();
    let stats = records.filter_map(|record| match record {
        Record::ProductStat(measured) => Some(stat(measured.into_measurement())),
        _ => None,
    });
    stats.collect()
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
