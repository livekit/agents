//! Work requested by workflow lifecycle owners.
use lk_agents_abi::voice::workflows::IoEnableState;

/// Resource-specific effects of a lifecycle transition. The runtime supplies their handles.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TransferEffects {
    pub close_human: Option<bool>,
    pub stop_hold: bool,
    pub io: Option<IoEnableState>,
}

pub enum TransferToolEffect {
    MoveParticipant,
    Settled(TransferEffects),
}
