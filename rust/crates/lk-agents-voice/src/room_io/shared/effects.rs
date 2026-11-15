use crate::room_io::bindings::effects::RoomBindingEffect;
use crate::room_io::legacy::effects::LegacyPrimaryEffect;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedRoomEffect {
    Legacy(LegacyPrimaryEffect),
    Binding(RoomBindingEffect),
}
