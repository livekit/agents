//! One durable A2A arrival. [`machine::TaskMachine`] owns its synchronous lifecycle;
//! [`effects`] declares runtime storage/handler work without changing the ABI storage contracts.

pub mod effects;
pub mod machine;

pub use effects::A2aTaskEffect;
pub use machine::{
    A2aTaskEvent, Access, Answer, Completion, Handle, HandlerEvent, LaneRun, Operation, RunEnd,
    Stage, TaskMachine, rejected_commit,
};
