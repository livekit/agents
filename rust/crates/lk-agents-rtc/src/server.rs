//! Room administration and SIP capabilities used by session workflows.

use crate::ParticipantIdentity;
use std::collections::HashMap;
use std::time::Duration;

/// `api.CreateSIPParticipantRequest`, as `warm_transfer.WarmTransferTask._dial_human_agent` fills it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SipParticipantReq {
    pub sip_trunk_id: Option<String>,
    pub sip_call_to: String,
    pub room_name: String,
    pub participant_identity: ParticipantIdentity,
    pub wait_until_answered: bool,
    pub sip_number: Option<String>,
    pub headers: HashMap<String, String>,
    pub dtmf: String,
    pub ringing_timeout: Option<Duration>,
}
