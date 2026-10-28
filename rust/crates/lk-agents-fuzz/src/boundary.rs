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
        "voice decision core (agent_session::Driver), stepped by Runtime::handle per delivered event",
        "voice data plane (data_plane forwarder), stepped by Runtime::on_frame per microphone frame",
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
        "fakes::realtime (simulated OpenAI Realtime server)",
        "lk_agents_rtc::FakeRoom",
    ],
    transitional: &[
        "voice policy tasks the core spawns (generation, playout, tools, hooks): scheduled as the \
         pool's turns, or one process each under the task bridge (E-1, E-3, E-13)",
        "lk-agents-inference NativeRealtime wire machine over the fake server's socket",
        "room io and the avatar runner over FakeRoom; the console's remote session over a socket",
    ],
    outside: &[
        "Runtime::run and the Tokio session loop (K-30 conformance and stress)",
        "native provider HTTP/WebSocket clients (F-18 real-peer tests)",
        "real LiveKit rooms through the rtc SDK (H-13, K-18)",
        "Python host bindings, lk-agents-py (B-19 conformance)",
        "worker, IPC and job processes (G-12 stress; worker DST is K-2)",
        "agent-db storage and the A2A door (I-27 backend contracts; DST K-3, K-4)",
    ],
};

/// The storage campaign (`tests/storage_survey.rs`, `storage` module).
pub const STORAGE: Boundary = Boundary {
    cores: &["storage core (lk_agents_storage::db plans and decodings, run_commit, merge_write, \
         Renewal, LanePoll, the commit Resolution), stepped by simulated workers one plan at a time"],
    fakes: &[
        "storage::db: the thread database as a real SQLite in memory (rusqlite, test tooling) \
         whose unixepoch() is the simulated clock",
        "the AgentDB faults: commit acknowledgment lost before or after it applied, unreachable \
         windows, paused owners whose lease lapses, workers lost mid-run, client cancels",
    ],
    transitional: &[
        "storage::worker mirrors the runtime engine's and the A2A turn's step order by hand \
         until the lane/turn orchestration is a core machine (I-1)",
    ],
    outside: &[
        "lk_agents_runtime::storage engine, SqliteStorage files and the AgentDB wsdb client \
         (I-27 backend contract suite, conformance with injected faults)",
        "the A2A turn driver and the host's handler (runtime a2a_turns suite)",
    ],
};

/// The inference campaign (`tests/inference_survey.rs`, `inference` module).
pub const INFERENCE: Boundary = Boundary {
    cores: &[
        "LLM request core (lk_agents_inference::llm::lifecycle::LlmRequest over attempts::Attempts), \
         stepped per delivered transport or consumer event and due wake",
        "avatar session core (lk_agents_inference::avatar::lifecycle::AvatarSessionLifecycle), \
         stepped per delivered transport, caller or program event and due wake",
    ],
    fakes: &[
        "inference::llm::Gateway: connections, rejections, recorded gpt-4.1-mini chunk timing, \
         drops, stalls, malformed and unterminated bodies, late or missing attempt releases",
        "inference::llm::Consumer: demand with backpressure, cancellation",
        "inference::avatar::Gateway: idempotent creates applied on receipt, lost answers, \
         rejections, sessions without terminate ids, late or missing attempt releases",
        "inference::avatar::Program: the creating caller, a racing caller, refusals, closes",
    ],
    transitional: &["none: both machines are core machines the simulation steps itself"],
    outside: &[
        "the runtime's HTTP exchange tasks and the Python avatar/LLM adapters (K-30 conformance, \
         F-18 real-peer tests)",
    ],
};

/// The worker campaign (`tests/worker_survey.rs`, `worker` module).
pub const WORKER: Boundary = Boundary {
    cores: &["process pool allocation core (lk_agents_worker::ipc::pool::PoolState), stepped per \
         delivered process, driver or server event and due wake"],
    fakes: &[
        "worker::Os: job processes that spawn or fail to, initialize, crash or hang, run a job \
         and exit, close late, and whose exit watches and reaps fail and are retried",
        "the driver: admissions with duplicate dispatches, cancels, a drain, the close, launches \
         of reserved jobs, forgets of settled requests, the cleanup report",
    ],
    transitional: &["the driver mirrors lk_agents_worker::ipc::pool's task loop by hand"],
    outside: &[
        "the pool's async task, the supervisor's process I/O, IPC and the server connection \
         (G-12 stress, K-30 conformance)",
        "registration, job assignment and the server lifecycle (WorkerState: K-2 remainder)",
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
