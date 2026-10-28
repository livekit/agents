//! Answering-machine decisions and the native detector setup accepted by a runtime.

pub mod classifier;

use classifier::AmdConfig;
use lk_agents_abi::voice::amd::{AMD_PROMPT, AmdPort};
use lk_agents_abi::{HostRef, Llm, Stt};
use lk_agents_abi::voice::rtc::RtcRoom;
use std::{fmt, sync::Arc};

pub type AmdFactory = fn(crate::agent_session::PeerHandle, AmdSetup) -> Arc<dyn AmdPort>;

pub struct AmdSetup {
    pub config: AmdConfig,
    /// `detection_options["prompt"]`, else the reference's [`AMD_PROMPT`].
    pub prompt: String,
    /// The classifier LLM: `AMD(llm=..)`, else the session's own (`detector.AMD._resolve_classifier`).
    pub llm: Arc<dyn Llm>,
    pub tools_host: Option<HostRef>,
    /// `AMD(stt=..)`: AMD's own recogniser; `None` reads the session's finals.
    pub stt: Option<Arc<dyn Stt>>,
    /// The call's room, where the callee's answer is awaited; `None` for a session with no room
    /// io (a console's), which starts listening at once.
    pub room: Option<Arc<dyn RtcRoom>>,
    /// `AMD(participant_identity=..)`: the one participant whose audio is awaited.
    pub participant: Option<String>,
}

impl AmdSetup {
    pub fn new(config: AmdConfig, llm: Arc<dyn Llm>) -> Self {
        AmdSetup {
            config,
            prompt: AMD_PROMPT.to_owned(),
            llm,
            tools_host: None,
            stt: None,
            room: None,
            participant: None,
        }
    }
}

impl fmt::Debug for AmdSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AmdSetup")
            .field("config", &self.config)
            .field("llm", &self.llm.label())
            .field("stt", &self.stt.as_ref().map(|s| s.label()))
            .finish()
    }
}
