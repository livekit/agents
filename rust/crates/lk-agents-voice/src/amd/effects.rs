//! What the AMD machine asks of its executor.
//!
//! Producer: [`Amd`](super::machine::Amd). Handler: the AMD executor in `lk-agents-runtime`,
//! which sends answers to the session, runs the classifier and menu legs, arms deadlines and
//! hands events to the host. The machine owns every state these change.

use super::AmdAnswer;
use super::events::{AmdEvent, AmdReason, Category, Stage};
use crate::audio_recognition::TurnId;
use lk_agents_abi::{FunctionCall, FunctionCallOutput, Ts};
use std::time::Duration;

/// In transition order: a cancellation precedes its replacement, and an answer precedes the
/// events of the same transition only where Python's order puts it there.
#[derive(Debug, Clone, PartialEq)]
pub enum AmdEffect {
    /// Deliver to the session's reply gate.
    Answer(AmdAnswer),
    /// Run the classifier; answer with `In::Classified` for the same `req`.
    Classify(ClassifyRequest),
    /// Abandon the classifier request in flight; its result, if any, is stale.
    CancelClassify,
    /// Extract the menu from a `machine-ivr` turn; answer with `In::MenuExtracted`.
    ExtractMenu(MenuRequest),
    /// Stop the menu extraction in flight; its result, if any, is stale.
    CancelMenu,
    /// Replace the timer's deadline; it fires as `In::Timeout`.
    SetDeadline(AmdTimer, Ts),
    /// Remove the timer's deadline, if armed.
    ClearDeadline(AmdTimer),
    /// Hand an event to the host.
    Emit(AmdEvent),
}

/// AMD's deadlines. A firing that no longer matches the machine's state is ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AmdTimer {
    /// The request's `inference_timeout`.
    Inference(ReqId),
    /// The turn's wait for its late AMD final (spec §6.2).
    LateFinal(TurnId),
    /// `idle_timeout`, or `voicemail_idle_timeout` in `machine-vm`.
    Idle,
    /// The overall `timeout`, from activation.
    Overall,
    /// The menu extraction's 5 s deadline.
    Menu(ReqId),
}

/// One classifier or menu request of a run. A result names it, so a stale one is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReqId(pub u64);

/// A turn's transcripts: the session's and AMD's, each `None` for an STT not in use.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnTranscript {
    pub session: Option<String>,
    pub amd: Option<String>,
}

impl TurnTranscript {
    /// The one transcript for events and the agent: the session's, else AMD's.
    pub fn transcript(&self) -> &str {
        [&self.session, &self.amd]
            .into_iter()
            .flatten()
            .find(|text| !text.is_empty())
            .map_or("", String::as_str)
    }

    /// What the classifier reads (spec §6.1): both transcripts as JSON when AMD has its own STT
    /// next to the session's, else the one transcript.
    pub fn for_classifier(&self) -> String {
        match (&self.session, &self.amd) {
            (Some(session), Some(amd)) => {
                python_json(&DualTranscript { transcript_1: session, transcript_2: amd })
            }
            _ => self.transcript().to_owned(),
        }
    }
}

#[derive(serde::Serialize)]
struct DualTranscript<'a> {
    transcript_1: &'a str,
    transcript_2: &'a str,
}

/// The agent spoke, words hidden (spec §8.2).
pub const AGENT_REPLY: &str = "(the agent spoke)";

/// One item of AMD's own classifier history (spec §8.2). It never copies the agent's history.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassifierItem {
    /// A committed turn with text, as a user message.
    User { turn_id: u64, transcript: TurnTranscript },
    /// An agent reply, as an assistant message: its words, or [`AGENT_REPLY`].
    Agent { text: String },
    /// A successful `send_dtmf_events` call ...
    DtmfCall(FunctionCall),
    /// ... and its output.
    DtmfOutput(FunctionCallOutput),
}

/// The previous accepted prediction, as a request shows it (spec §8.1).
#[derive(Debug, Clone, PartialEq)]
pub struct PreviousPrediction {
    pub turn_id: u64,
    pub category: Category,
    pub reason: AmdReason,
}

/// Everything one classification sends (spec §8.1). The classifier leg renders it for an LLM
/// (prompt, [`ClassifyRequest::state_json`], history) or a decision model (one `choice`).
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifyRequest {
    pub req: ReqId,
    /// The session's turn it classifies.
    pub turn: TurnId,
    /// AMD's id for that turn.
    pub turn_id: u64,
    pub stage: Stage,
    pub previous_prediction: Option<PreviousPrediction>,
    /// The likely-human mark is set: the previous category shows as `uncertain (likely human)`.
    pub likely_human: bool,
    /// The stage's recommended list, sorted by value.
    pub recommended: Vec<Category>,
    pub speech_duration: Duration,
    /// At most the 20 latest items, never starting with a DTMF call or output.
    pub history: Vec<ClassifierItem>,
}

/// How a request shows the previous category while the likely-human mark is set.
pub const LIKELY_HUMAN: &str = "uncertain (likely human)";

#[derive(serde::Serialize)]
struct RequestState<'a> {
    stage: &'a str,
    previous_prediction: Option<PreviousState<'a>>,
    recommended_next_categories: Vec<&'a str>,
    speech_duration: f64,
}

#[derive(serde::Serialize)]
struct PreviousState<'a> {
    category: &'a str,
    reason: &'a str,
    turn_id: u64,
}

impl ClassifyRequest {
    /// The request's system message: stage, previous prediction, recommended list and speech
    /// duration, as `_inference.classify` encodes them.
    pub fn state_json(&self) -> String {
        let previous = self.previous_prediction.as_ref().map(|previous| PreviousState {
            category: if self.likely_human { LIKELY_HUMAN } else { previous.category.as_str() },
            reason: previous.reason.as_str(),
            turn_id: previous.turn_id,
        });
        let state = RequestState {
            stage: self.stage.as_str(),
            previous_prediction: previous,
            recommended_next_categories: self.recommended.iter().map(|c| c.as_str()).collect(),
            speech_duration: self.speech_duration.as_secs_f64(),
        };
        python_json(&state)
    }
}

/// A menu extraction from one `machine-ivr` turn's transcript (spec §11.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuRequest {
    pub req: ReqId,
    pub turn_id: u64,
    pub transcript: String,
}

#[derive(serde::Serialize)]
struct MenuInput<'a> {
    transcript: &'a str,
}

impl MenuRequest {
    /// The extraction's user message, as `extract_ivr_menu` encodes the transcript.
    pub fn user_json(&self) -> String {
        python_json(&MenuInput { transcript: &self.transcript })
    }
}

/// `json.dumps` with its defaults: `", "` and `": "` separators, non-ASCII escaped. The model
/// reads these strings, so they match Python's byte for byte.
fn python_json(value: &impl serde::Serialize) -> String {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonFormatter);
    match value.serialize(&mut serializer) {
        Ok(()) => String::from_utf8(out).unwrap_or_default(),
        Err(_) => String::new(),
    }
}

struct PythonFormatter;

impl serde_json::ser::Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        for c in fragment.chars() {
            if c.is_ascii() {
                writer.write_all(&[c as u8])?;
            } else {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}
