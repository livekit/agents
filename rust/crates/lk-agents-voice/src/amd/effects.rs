//! Work requested by answering-machine classification.
use super::machine::classifier::{
    AmdPrediction, AmdTimer, Authorization, ClassifyRequest, Refusal,
};
use lk_agents_abi::{Ts, Vocabulary};

#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum AmdEffect {
    SetDeadline(AmdTimer, Ts),
    ClearDeadline(AmdTimer),
    Prediction(AmdPrediction),
    RequestClassification(ClassifyRequest),
    CancelClassification,
    Refused(Refusal),
    /// `_activity._pause_authorization()` / `_resume_authorization()`, performed by the driver.
    Authorization(Authorization),
    /// `session.interrupt(force=True)`: a greeting being spoken over must be cut.
    Interrupt {
        force: bool,
    },
    /// [`Amd::skips_turn_reply`](super::machine::classifier::Amd::skips_turn_reply) changed: the session's turns skip their reply, or reply again.
    SkipReplies(bool),
}
