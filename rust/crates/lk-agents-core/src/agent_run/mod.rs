//! Shared invocation domain: [`machine::AgentRun`] owns transitions; [`effects`] declares
//! runtime work. Execution identity and result records remain ordinary domain helpers.

pub mod effects;
pub mod execution;
pub mod machine;
pub mod result;

pub use effects::AgentRunEffect;
pub use machine::{
    AgentRun, AgentRunContinuation, AgentRunEvent, DelegationCheckpoint, Opening, OutputVerdict,
    RunInput,
};
pub use result::{CallId, CheckpointId, RenderId, RequestId, StepId};
