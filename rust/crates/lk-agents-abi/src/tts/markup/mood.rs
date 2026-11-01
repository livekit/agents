//! Mood values attached to transcript observations.

/// `AgentMood`: the closed set a delivery label normalizes to, in declaration order. The wire
/// strings are the exact `AgentMood` literals, a wire contract (`lk.expression`).
#[crate::host(wire_enum)]
#[crate::host(codec(extract, emit, host = "str"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mood {
    Excited,
    Happy,
    Playful,
    Curious,
    Surprised,
    Hopeful,
    Empathetic,
    Sad,
    Angry,
    Anxious,
    Calm,
}
