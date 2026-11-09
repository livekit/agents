pub mod effects;
pub mod execution;
pub mod machine;
pub mod planning;

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
    /// One tool call the model made (or an answer handed to a tool), from preparation to settlement.
    CallId,
    /// One rendering of the dynamic instructions.
    RenderId,
    /// One managed-turn admission or delivery checkpoint.
    CheckpointId,
    /// One settled model step a coordinator decides the continuation of.
    StepId,
}
