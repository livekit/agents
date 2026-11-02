//! Backend conversation domain. [`machine::TextSession`] owns child continuation;
//! [`effects`] declares runtime work. Routing definitions and history helpers remain here.

pub mod effects;
mod history;
pub mod machine;
mod routing;

pub use effects::TextSessionEffect;
pub use lk_agents_abi::text_session::{
    ChildCallStatus, FrameId, ParentCall, PendingChildCall, SessionFrame, TextSessionError,
    TextSessionState, TextSessionTurn,
};
pub use machine::{TextSession, TextSessionEvent, TextSessionInput};
pub use routing::{ChatHistory, LocalAgentTool, RoutingError, SessionAgent, TextSessionConfig};
