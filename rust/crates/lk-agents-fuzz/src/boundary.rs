//! What the DST simulates and what it leaves outside, stated once and printed by every campaign
//! (owner decision 28, K-1, K-9).
//!
//! The DST exercises sans-IO cores with realistic fakes: BasicPOS (and the arrival baseline) over
//! the core's event deliveries. It never claims runtime-task interleavings: runtime adapters are
//! verified by conformance tests through the real adapters with cancellation at every await point,
//! stress tests, real-peer adapter tests and a regression test per bug (K-30). What the voice
//! harness still runs as tasks is listed as transitional: it leaves as the voice policy tasks
//! become core machines (E-1, E-3, E-13), and the task bridge with them.

/// One campaign's boundary.
pub struct Boundary {
    /// The cores the run steps itself, one event at a time.
    pub cores: &'static [&'static str],
    /// The fakes standing in for every provider and peer, by module.
    pub fakes: &'static [&'static str],
    /// Product code the run still executes as tasks or adapters, until it is a core or leaves.
    pub transitional: &'static [&'static str],
    /// Runtime adapters outside the DST, with what verifies them.
    pub outside: &'static [&'static str],
}

/// The voice campaigns (`tests/survey.rs`, `tests/soak.rs`, `tests/pos.rs`).
pub const VOICE: Boundary = Boundary {
    cores: &[
        "voice decision core (agent_session::Driver), stepped in the real runtime owner by the \
         deterministic harness per delivered event, with effects performed against fake ports",
        "voice data plane (data_plane forwarder), stepped by the real runtime per microphone frame",
        "realtime conversation machine and connection policy (lk_agents_inference::realtime), \
         stepped by fakes::realtime's pump per command, server event and deadline",
        "the runtime's room and avatar executors (lk_agents_runtime::room_io, ::avatar, ::rtc) \
         over FakeRoom, on the run's scheduler (decision 92)",
    ],
    fakes: &[
        "fakes::stt",
        "fakes::vad",
        "fakes::eot",
        "fakes::interruption",
        "fakes::llm",
        "fakes::tts",
        "fakes::tools",
        "fakes::nodes",
        "fakes::output",
        "fakes::avatar",
        "fakes::realtime (simulated OpenAI Realtime server, in the protocol's typed records; the \
         fallback adapter over two of them); a duplex model over the inference duplex segmenter \
         and gate",
        "lk_agents_test_support::rtc::FakeRoom",
    ],
    transitional: &[
        "voice policy tasks the core spawns (generation, playout, tools, hooks): scheduled as the \
         pool's turns, or one process each under the task bridge (E-1, E-3, E-13)",
        "the avatar runner over FakeRoom; the console's remote session over a socket",
    ],
    outside: &[
        "the voice runtime arrival-order loop (the harness drives its same actor and effect execution)",
        "inference's realtime pump, wire codec and fallback adapter (lk_agents_inference::realtime::native; \
         K-30 conformance; the fake pump and fallback mirror their contracts)",
        "native provider HTTP/WebSocket clients (F-18 real-peer tests)",
        "real LiveKit rooms through the rtc SDK (H-13, K-18)",
        "Python host bindings, lk-agents-py (B-19 conformance)",
        "worker, IPC and job processes (lk-agents-worker's own tests, decision 70)",
        "agent-db storage and the A2A door's transport (I-27 backend contracts; the A2A DST fakes the storage port)",
    ],
};

/// The A2A campaign (`tests/a2a_survey.rs`, `a2a` module).
pub const A2A: Boundary = Boundary {
    cores: &["A2A task machine (lk_agents_core::a2a::task::TaskMachine: deduplication, arrival \
         policy, claim, drain, the turn as separate steps (pending question, checkpoints, \
         output, settlement), commit, every typed storage failure) under every inbox policy, \
         one operation at a time"],
    fakes: &[
        "a2a::port: one lane in memory with the storage contract's semantics (arrival-ordered \
         lease, waiter mark, acquire snapshots, conditional pickups and orphan failures, \
         append-only concurrent history, resend under the fence), injecting Unavailable, \
         Uncertain before and after the commit applied, Rejected, LeaseLost, VersionConflict, \
         NotFound, a refused statement and a cancelled or failing handler",
        "a2a::handler: checkpoints, a yield, a text answer, a tool's question, a raise",
        "the clients: out-of-order arrivals on three channels under door options mixing the \
         policies, resends at once and after a failed answer or a dead worker, client cancels, \
         answers to questions with or without the waiting task's id",
    ],
    transitional: &[
        "none: the task machine and its turn are core steps the simulation steps itself",
    ],
    outside: &[
        "the runtime door's execution of the operations (runtime a2a_turns and a2a_door suites)",
        "lk-agents-storage's backends (real SQLite and a fake AgentDB server, its own suites)",
        "SubscribeToTask and resubscribe from a cursor: runtime door and response streams",
    ],
};

/// The TextSession campaign (`tests/text_session_survey.rs`, `text_session` module).
pub const TEXT_SESSION: Boundary = Boundary {
    cores: &["TextSession routing core (lk_agents_core::text_session::TextSession over \
         lk_agents_core::agent_run::AgentRun and the shared approval gate), reconstructed from \
         its returned state every turn and stepped per delivered model, host, delivery or \
         client event and due wake"],
    fakes: &[
        "text_session::World: a model that answers in text, delegates through the agent tool \
         (once or twice per reply, with refused arguments), answers as a child with a reference \
         or a question, repeats, changes or drops an approved call, fails or is cancelled; a \
         recursive desk down to the depth bound",
        "the host: argument binding, read-backs, output validation and checkpoints after their \
         own delays; the endpoint delivering each settled invocation's dialogue before \
         releasing it; a client cancelling turns midway",
    ],
    transitional: &["none: TextSession and AgentRun are core machines the simulation steps itself"],
    outside: &[
        "the runtime's TextSession runner, its stream and state codec, and the Python \
         text_session module (K-30 conformance, the generated state codec's tests)",
        "the application's context loads and saves around each turn (context.md; I-27)",
    ],
};

impl Boundary {
    /// Prints the boundary with what the campaign actually delivered, and fails on an empty
    /// inventory: a campaign that cannot say what it covered proves nothing.
    pub fn report(&self, inputs: &std::collections::BTreeSet<&str>, opened: &[String]) {
        println!("\nboundary (what this campaign simulates and what it leaves outside):");
        for (title, rows) in [
            ("cores stepped", self.cores),
            ("fakes", self.fakes),
            ("transitional, still tasks or adapters", self.transitional),
            ("outside the DST", self.outside),
        ] {
            assert!(!rows.is_empty(), "the boundary lists no {title}");
            println!("  {title}:");
            for row in rows {
                println!("    {row}");
            }
        }
        assert!(!inputs.is_empty(), "the campaign delivered no input");
        let frames = inputs.contains("MicFrames");
        let names: Vec<&str> = inputs.iter().copied().collect();
        println!("  inputs delivered ({}; microphone frames: {frames}):", names.len());
        println!("    {}", names.join(" "));
        println!("  resources opened: {}", opened.join(" "));
    }
}
