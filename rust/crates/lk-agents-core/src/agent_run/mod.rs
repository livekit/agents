pub mod effects;
pub mod execution;
pub mod machine;
pub mod planning;
#[cfg(test)]
mod planning_tests;

pub use effects::AgentRunEffect;
pub use machine::{
    AgentRun, AgentRunContinuation, AgentRunEvent, DelegationCheckpoint, Opening, OutputVerdict,
    RunInput,
};
macro_rules! ids {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u64);
    )*};
}

ids! {
    /// One model request.
    RequestId,
    /// One tool call the model made, from preparation to settlement.
    CallId,
    /// One rendering of the dynamic instructions.
    RenderId,
    /// One managed-turn admission or delivery checkpoint.
    CheckpointId,
    /// One settled model step a coordinator decides the continuation of.
    StepId,
    /// One source-owned exact tool output, independent of its optional receipt reader.
    ToolOutputId,
}
