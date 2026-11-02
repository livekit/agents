//! Answering-machine decisions and the native detector setup accepted by a runtime.

pub mod effects;
pub mod machine;

use crate::agent_session::SystemTap;
use crate::events::SessionEvent;
use lk_agents_abi::voice::rtc::{ParticipantKind, RtcRoom};
use lk_agents_abi::{HostRef, Llm, Stt};
use lk_agents_abi::{SpeechEvent, SpeechEventType, VadEventType};
use machine::classifier::{
    AmdCategory, AmdConfig, AmdReason, Human, In, Start, TranscriptSource, Verdict,
};
use std::{fmt, sync::Arc};

pub struct AmdSetup {
    pub config: AmdConfig,
    /// `detection_options["prompt"]`, else the reference's [`AMD_PROMPT`](lk_agents_abi::voice::amd::AMD_PROMPT).
    pub prompt: String,
    /// The classifier LLM: `AMD(llm=..)`, else the session's own (`detector.AMD._resolve_classifier`).
    pub llm: Arc<dyn Llm>,
    pub tools_host: Option<HostRef>,
    /// `AMD(stt=..)`: AMD's own recogniser; `None` reads the session's finals.
    pub stt: Option<Arc<dyn Stt>>,
    /// The call's room, where the callee's answer is awaited; `None` for a session with no room
    /// io (a console's), which starts listening at once.
    pub room: Option<Arc<dyn RtcRoom>>,
    /// `AMD(participant_identity=..)`: the one participant whose audio is awaited.
    pub participant: Option<String>,
}

impl fmt::Debug for AmdSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AmdSetup")
            .field("config", &self.config)
            .field("llm", &self.llm.label())
            .field("stt", &self.stt.as_ref().map(|s| s.label()))
            .finish()
    }
}

/// One item off the session feed, as the reference's `AudioRecognition` hands it to the
/// classifier: VAD bounds, the session's non-blank finals, the end of turn (`_on_end_of_turn`,
/// before the commit verdict) and the session's close (`self._amd.aclose()`).
pub fn feed_input(item: SystemTap) -> Option<In> {
    match item {
        SystemTap::Vad(ev) => match ev.kind {
            VadEventType::StartOfSpeech => Some(In::Human(Human::SpeechStarted)),
            VadEventType::EndOfSpeech => {
                let silence =
                    std::time::Duration::try_from_secs_f64(ev.silence_duration).unwrap_or_default();
                Some(In::Human(Human::SpeechEnded { silence }))
            }
            VadEventType::InferenceDone => None,
        },
        // The empty clearing final is no words.
        SystemTap::Stt(ev) => final_text(&ev).filter(|text| !text.trim().is_empty()).map(|text| {
            In::Human(Human::Transcript { text, source: TranscriptSource::SessionStt })
        }),
        SystemTap::Session(SessionEvent::UserTurnCompleted(_)) => Some(In::Human(Human::EndOfTurn)),
        SystemTap::Session(_) | SystemTap::Ivr(_) => None,
        SystemTap::CloseRequested => Some(In::Close),
    }
}

/// `_receive`: a non-empty final of AMD's own STT reaches the machine; everything else is dropped.
pub fn tap_input(ev: &SpeechEvent) -> Option<In> {
    final_text(ev)
        .filter(|text| !text.is_empty())
        .map(|text| In::Human(Human::Transcript { text, source: TranscriptSource::AmdStt }))
}

fn final_text(ev: &SpeechEvent) -> Option<String> {
    if ev.kind != SpeechEventType::FinalTranscript {
        return None;
    }
    ev.best_alternative().map(|a| a.text.clone())
}

/// `AMD._setup`'s outcome: listening opens, or the call ended before detection could run. Without
/// a room nothing is awaited and the callee counts as answered.
pub enum Callee {
    Answered,
    Missing(String),
}

impl Callee {
    /// A callee whose audio track is subscribed has answered, except a SIP callee, who has
    /// answered once its `sip.callStatus` is active.
    pub fn awaits_sip_answer(kind: ParticipantKind) -> bool {
        kind == ParticipantKind::Sip
    }

    /// `_start_listening`: the detection budget from the answer, then the input gate. A missing
    /// callee settles `uncertain` rather than listening for the whole no-speech threshold.
    pub fn inputs(&self) -> Vec<In> {
        match self {
            Callee::Answered => {
                vec![In::Start(Start::DetectionTimer), In::Start(Start::Listening)]
            }
            Callee::Missing(_) => vec![In::Verdict(Verdict::Settle {
                category: AmdCategory::Uncertain,
                reason: AmdReason::ParticipantMissing,
                speech_duration: None,
            })],
        }
    }
}
