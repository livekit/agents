//! Shared tool-call domain: call admission/execution state in [`machine`], host work in [`effects`].

pub mod effects;
pub mod machine;

pub use effects::ToolCallEffect;
pub(crate) use machine::FinalCandidate;
pub use machine::{
    Admission, PrepareFailure, Recorded, RunningToolExecution, Settled, ToolCalls, admit,
};
