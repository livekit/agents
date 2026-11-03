//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `handle = Manual` span declarations of `lk-agents-abi`.

/// An entity-owned span: what the span rule checks it against.
pub(crate) struct ManualSpan {
    pub name: &'static str,
    pub parents: &'static [&'static str],
    pub outlives_parent: bool,
}

/// Every `handle = Manual` span declaration, by name.
pub(crate) const MANUAL_SPANS: &[ManualSpan] = &[
    ManualSpan {
        name: "agent_session",
        parents: &["<current>", "job_entrypoint"],
        outlives_parent: false,
    },
    ManualSpan { name: "agent_speaking", parents: &["agent_turn"], outlives_parent: false },
    ManualSpan { name: "agent_turn", parents: &["agent_session"], outlives_parent: false },
    ManualSpan { name: "amd", parents: &["agent_session"], outlives_parent: false },
    ManualSpan {
        name: "drain_agent_activity",
        parents: &["session_close", "update_agent"],
        outlives_parent: false,
    },
    ManualSpan { name: "eou_detection", parents: &["eou_wait"], outlives_parent: false },
    ManualSpan { name: "eou_wait", parents: &["user_turn"], outlives_parent: false },
    ManualSpan {
        name: "keyterm_detection",
        parents: &["agent_session", "agent_turn"],
        outlives_parent: true,
    },
    ManualSpan { name: "on_enter", parents: &["start_agent_activity"], outlives_parent: true },
    ManualSpan { name: "on_exit", parents: &["drain_agent_activity"], outlives_parent: false },
    ManualSpan { name: "on_user_turn_completed", parents: &["user_turn"], outlives_parent: false },
    ManualSpan { name: "pause_agent_activity", parents: &["update_agent"], outlives_parent: false },
    ManualSpan {
        name: "resume_agent_activity",
        parents: &["update_agent"],
        outlives_parent: false,
    },
    ManualSpan { name: "session_close", parents: &["agent_session"], outlives_parent: false },
    ManualSpan { name: "session_start", parents: &["agent_session"], outlives_parent: false },
    ManualSpan {
        name: "start_agent_activity",
        parents: &["session_start", "update_agent"],
        outlives_parent: false,
    },
    ManualSpan { name: "update_agent", parents: &["agent_session"], outlives_parent: false },
    ManualSpan { name: "user_speaking", parents: &["user_turn"], outlives_parent: false },
    ManualSpan { name: "user_turn", parents: &["agent_session"], outlives_parent: false },
];
