//! `AgentSession`: the Node handle on the core session, [`SessionApi`].
//!
//! The class is hand-written; its command methods are not. `SessionApi`'s `#[host(command)]`s
//! (`lk-agents-voice/src/agent_session/api/host.rs`) generate one `#[napi] impl` block onto this
//! class (`generated/objects.rs`), the same declarations `lk-agents-py`'s `AgentSession` gets its
//! pymethods from. Assembling a session from JS providers is not bound yet, so there is no
//! constructor: the class is where that goes, and where the generated methods already land.
//!
//! [`SessionApi`]: lk_agents_voice::agent_session::SessionApi

use napi_derive::napi;

/// The core session, opened and commanded from JS.
#[napi]
pub struct AgentSession {
    pub(crate) inner: lk_agents_voice::agent_session::SessionApi,
}
