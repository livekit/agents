//! The AMD executor's trace steps, journalled beside the session's own (`trace::Step::Amd`).

use super::AmdRunId;
use super::effects::ReqId;
use super::events::{AmdReason, Category, Stage};
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::audio_recognition::TurnId;

/// The AMD executor's steps (`lk-agents-runtime`'s `voice::amd`): the oracle's view of one run.
/// The DST rules read them next to what the fake models were asked and answered, and what the
/// host and the audio sink received.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum AmdStep {
    /// A turn committed under the run's gate became one of AMD's turns.
    TurnBound {
        run: AmdRunId,
        turn: TurnId,
        /// AMD's turn id: local to the run, from 1.
        turn_id: u64,
        /// The turn had a transcript at commit.
        has_text: bool,
        /// The turn waits for its late AMD final (spec §6.2).
        late_wait: bool,
    },
    /// A classifier request was sent to the model for `turn`.
    RequestIssued { run: AmdRunId, req: ReqId, turn: TurnId },
    /// The request in flight was abandoned; its result, if any, is stale.
    RequestCancelled { run: AmdRunId, req: ReqId },
    /// A turn's prediction.
    Predicted {
        run: AmdRunId,
        turn: TurnId,
        turn_id: u64,
        /// The request whose result decided the turn. For a `reused` prediction it belongs to an
        /// older turn; `None` for a prediction no request decided (an inference timeout, or a
        /// reuse of the latest prediction with no request pending).
        source_req: Option<ReqId>,
        /// What the classifier answered, before the likely-human downgrade; `None` without an
        /// answer.
        raw: Option<Category>,
        category: Category,
        stage: Stage,
        reason: AmdReason,
        /// The likely-human mark after this prediction.
        likely_human: bool,
    },
    /// The session made a reply for the turn: `ReplyCreated` reached the run.
    ReplyBound { run: AmdRunId, turn: TurnId, speech: SpeechId },
    /// The voicemail message played to its end while the run was active (spec §10.4).
    VoicemailPlayed { run: AmdRunId, speech: SpeechId },
    /// `amd_completed`: the run's outcome, after the session's `Detached` and the legs closed.
    Completed { run: AmdRunId, category: Stage, reason: AmdReason },
}
