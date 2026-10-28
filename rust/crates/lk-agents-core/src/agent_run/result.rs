//! The correlation ids an invocation's effects and events carry. What it produces (its result
//! and stream events) is shared with the host: `lk_agents_abi::agent::run`.

macro_rules! ids {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u64);
    )*};
}

ids! {
    /// One model request.
    RequestId,
    /// One tool call the model made (or an answer handed to a tool), from preparation to settlement.
    CallId,
    /// One rendering of the dynamic instructions.
    RenderId,
    /// One managed-turn admission or delivery checkpoint.
    CheckpointId,
}
