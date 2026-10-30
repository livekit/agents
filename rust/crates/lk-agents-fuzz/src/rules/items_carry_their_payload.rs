//! What the host is told about a turn is whole: the payloads, not just that something happened.
//!
//! An assistant message of a step cut short says it was interrupted; a pipeline reply's timed at the LLM (`llm_node_ttft`) names that model
//! (`llm_metadata`), one timed at the voice (`tts_node_ttfb`) names the voice (`tts_metadata`); a
//! user turn transcribed while a recognition stream was open names the STT (`stt_metadata`), as
//! 9024425d9 restored. Every item was created within the session and no later than announced.
//! Each model's metrics record carries its label, a wall-clock timestamp within the session,
//! durations that are not negative, and the speech id of the speech it is attributed to.
use std::{
    collections::BTreeMap,
    time::{Duration, SystemTime},
};

use lk_agents_abi::{
    ChatItem, ChatMessage, MetricsMetadata, Role,
    metrics::base::{AgentMetrics, RealtimeModelMetrics},
};
use lk_agents_voice::{
    agent::Source,
    agent_activity::scheduling::SpeechKind,
    events::SessionEvent,
    trace::{CoreStep, HandoffStep, SpeechStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Kind, Observed, Recording},
};

/// The clock's nudges (distinct stamps for one instant) stay well inside this.
const SLACK: Duration = Duration::from_millis(1);

pub struct ItemsCarryTheirPayload;

impl Rule for ItemsCarryTheirPayload {
    const CONTRACT: Contract = Contract {
        id: "items_carry_their_payload",
        guarantee: "An assistant message of a step cut short is marked interrupted; messages name the models that timed them; items and metrics are stamped on the wall clock within the session; metrics carry their label, non-negative durations and the speech id they are attributed to.",
        examples: &[a_named_reply],
        counterexamples: &[
            a_reply_naming_no_model,
            an_interrupted_reply_marked_whole,
            metrics_on_another_clock,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let journal = r.journal();
        let Some(epoch) = journal.iter().find_map(|(_, s)| match s {
            Step::Core(CoreStep::WallEpoch { at }) => Some(*at),
            _ => None,
        }) else {
            return;
        };
        let wall = |at: &Observed| epoch + Duration::from_millis(at.at_ms) + SLACK;
        // How each step of each speech ended: (speech, step) -> interrupted.
        let mut spoken: BTreeMap<(u64, u32), bool> = BTreeMap::new();
        let mut speech_ids: BTreeMap<u64, String> = BTreeMap::new();
        let mut realtime: std::collections::BTreeSet<u64> = Default::default();
        for (at, step) in journal {
            match step {
                Step::Emitted(SessionEvent::SpeechCreated { speech, kind, .. }) => {
                    speech_ids.insert(speech.0, speech.to_string());
                    if *kind == SpeechKind::Realtime {
                        realtime.insert(speech.0);
                    }
                }
                Step::Speech(SpeechStep::StepSpoken { speech, step, interrupted }) => {
                    spoken.insert((speech.0, *step), *interrupted);
                }
                Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => {
                    let created = item.created_at();
                    check.expect_with(
                        Subject::Session,
                        epoch <= created + SLACK && created <= wall(at),
                        || format!("item {} created at {created:?}, announced at {} ms of a session from {epoch:?}", item.id(), at.at_ms),
                    );
                    let ChatItem::Message(m) = &**item else { continue };
                    let step = match source {
                        Source::Reply { speech, step } => Some((speech.0, *step)),
                        Source::Say { speech } => Some((speech.0, 1)),
                        _ => None,
                    };
                    // A pipeline reply's message, which the reference stamps with its models (a
                    // realtime one, and a `say`, it does not).
                    let pipeline = matches!(source, Source::Reply { speech, .. } if !realtime.contains(&speech.0));
                    message_names_models(r, check, m, at, pipeline);
                    // A step cut short is recorded as cut (one whose voice failed may be too).
                    if let Some((speech, _)) = step.filter(|k| spoken.get(k) == Some(&true))
                        && m.role == Role::Assistant
                    {
                        check.expect_with(Subject::Speech(speech), m.interrupted, || {
                            format!("item {} of a step cut short says it was not interrupted", m.id)
                        });
                    }
                }
                Step::Emitted(SessionEvent::MetricsCollected { metrics, speech }) => {
                    let id = speech.and_then(|s| speech_ids.get(&s.0).map(String::as_str));
                    metrics_are_whole(check, metrics, id, epoch, wall(at));
                }
                _ => {}
            }
        }
    }
}

/// A message names each model its metrics timed.
fn message_names_models(
    r: &Recording,
    check: &mut Check,
    m: &ChatMessage,
    at: &Observed,
    pipeline: bool,
) {
    let named =
        |meta: &Option<MetricsMetadata>| meta.as_ref().is_some_and(|m| m.model_name.is_some());
    let mut expect = |holds: bool, what: &str| {
        check.expect_with(Subject::Session, holds, || {
            format!("{:?} item {} names no {what} model", m.role, m.id)
        });
    };
    match m.role {
        Role::Assistant if pipeline => {
            if m.metrics.llm_node_ttft.is_some() {
                expect(named(&m.metrics.llm_metadata), "llm");
            }
            if m.metrics.tts_node_ttfb.is_some() {
                expect(named(&m.metrics.tts_metadata), "tts");
            }
        }
        Role::User if m.metrics.transcription_delay.is_some() && stt_open(r, at) => {
            expect(named(&m.metrics.stt_metadata), "stt");
        }
        _ => {}
    }
}

/// Whether the session had an STT at `at`: a recognition stream open, and no agent since taken
/// over without one (its turn, decided as the old one's stream closes, names no STT).
fn stt_open(r: &Recording, at: &Observed) -> bool {
    // The session's recogniser: the answering-machine detector's own transcribes for it alone.
    let open = r.opened().iter().filter(|o| o.kind == Kind::Stt && o.detail != crate::record::Detail::AmdStt && o.at.seq <= at.seq).any(|o| {
        !r.closed().iter().any(|(c, k, n)| *k == Kind::Stt && *n == o.nth && c.seq <= at.seq)
    });
    let agent_has_stt = r
        .journal()
        .iter()
        .take_while(|(a, _)| a.seq <= at.seq)
        .filter_map(|(_, s)| match s {
            Step::Handoff(HandoffStep::AgentUpdated { legs, .. }) => Some(legs.stt),
            _ => None,
        })
        .last()
        .unwrap_or(true);
    open && agent_has_stt
}

/// A model's metrics record is whole: labelled, stamped on the wall clock within the session,
/// durations not negative (a missing first token or audio duration is `None`), attributed by id.
fn metrics_are_whole(
    check: &mut Check,
    metrics: &AgentMetrics,
    speech: Option<&str>,
    epoch: SystemTime,
    wall: SystemTime,
) {
    let secs =
        |t: SystemTime| t.duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
    let (from, to) = (secs(epoch - SLACK), secs(wall));
    let non_negative = |t: Option<f64>| t.is_none_or(|t| t >= 0.0);
    let (kind, label, request, timestamp, durations, id) = match metrics {
        AgentMetrics::Llm(m) => (
            "llm",
            &m.label,
            &m.request_id,
            m.timestamp,
            m.duration >= 0.0 && non_negative(m.ttft),
            Some(&m.speech_id),
        ),
        AgentMetrics::Tts(m) => (
            "tts",
            &m.label,
            &m.request_id,
            m.timestamp,
            m.duration >= 0.0 && m.audio_duration >= 0.0 && non_negative(m.ttfb),
            Some(&m.speech_id),
        ),
        AgentMetrics::Stt(m) => (
            "stt",
            &m.label,
            &m.request_id,
            m.timestamp,
            m.duration >= 0.0 && non_negative(m.audio_duration),
            None,
        ),
        AgentMetrics::RealtimeModel(m) => (
            "realtime",
            &m.label,
            &m.request_id,
            m.timestamp,
            m.duration >= 0.0 && non_negative(m.ttft),
            None,
        ),
        _ => return,
    };
    check.expect_with(Subject::Session, (from..=to).contains(&timestamp), || {
        format!("{kind} metrics {request:?} stamped {timestamp}, outside the session's wall clock [{from}, {to}]")
    });
    check.expect_with(Subject::Session, durations, || {
        format!("{kind} metrics {request:?} with a negative duration: {metrics:?}")
    });
    // A connection's record (no request) is the model's, not a request's: it may carry no label.
    if !request.is_empty() {
        check.expect_with(Subject::Session, !label.is_empty(), || {
            format!("{kind} metrics {request:?} carry no label")
        });
    }
    if let (Some(id), Some(speech)) = (id, speech) {
        check.expect_with(Subject::Session, id.as_deref() == Some(speech), || {
            format!("{kind} metrics {request:?} attributed to {speech} carry speech_id {id:?}")
        });
    }
}

fn epoch() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_704_067_200)
}

fn reply(interrupted: bool, llm: Option<MetricsMetadata>) -> Step {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    let mut m = ChatMessage::text("item_msg_1", Role::Assistant, "Sure.", epoch());
    m.interrupted = interrupted;
    m.metrics.llm_node_ttft = Some(0.3);
    m.metrics.llm_metadata = llm;
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(ChatItem::Message(m)),
        agent: "agent".into(),
        source: Source::Reply { speech: SpeechId(3), step: 1 },
        visibility: lk_agents_voice::agent::Visibility::Both,
        run: None,
    })
}

fn named() -> Option<MetricsMetadata> {
    Some(MetricsMetadata {
        model_name: Some("gpt-4.1-mini".into()),
        model_provider: Some("openai".into()),
    })
}

fn recorded(interrupted: bool) -> Authored {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Authored::new()
        .journal(Step::Core(CoreStep::WallEpoch { at: epoch() }))
        .journal(Step::Speech(SpeechStep::StepSpoken { speech: SpeechId(3), step: 1, interrupted }))
}

/// A whole reply, naming its model, flagged as its step ended.
fn a_named_reply() -> Recording {
    recorded(false).journal(reply(false, named())).done()
}

/// The LLM timed the reply, and the item names no model: 9024425d9's loss.
fn a_reply_naming_no_model() -> Recording {
    recorded(false).journal(reply(false, None)).done()
}

/// The reply was cut, and its item says it was heard whole.
fn an_interrupted_reply_marked_whole() -> Recording {
    recorded(true).journal(reply(false, named())).done()
}

/// A realtime response stamped on a clock that started with the model, not the wall's.
fn metrics_on_another_clock() -> Recording {
    let metrics = RealtimeModelMetrics {
        label: "lk_agents_realtime".into(),
        request_id: "resp_1".into(),
        timestamp: 5.1,
        duration: 1.0,
        connection_duration: None,
        ttft: Some(0.5),
        cancelled: false,
        tokens: Default::default(),
        tokens_per_second: 0.0,
        acquire_time: 0.0,
        connection_reused: false,
        metadata: None,
    };
    Authored::new()
        .journal(Step::Core(CoreStep::WallEpoch { at: epoch() }))
        .later(6_000)
        .journal(Step::Emitted(SessionEvent::MetricsCollected {
            metrics: Box::new(AgentMetrics::RealtimeModel(metrics)),
            speech: None,
        }))
        .done()
}
