use super::BindingId;

/// Cleanup requested by the room owner. Retirement acknowledges completed cleanup;
/// cancellation or requesting close alone does not release a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomBindingEffect {
    Close(BindingId),
}
