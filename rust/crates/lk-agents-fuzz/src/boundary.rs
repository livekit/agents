//! What the DST simulates and what it leaves outside, stated once and printed by every campaign
//! (K-1, K-9).
//!
//! Each campaign states which production decisions and executors it actually exercises. Voice
//! includes the real runtime loop and its injected tasks; provider/SDK network implementations
//! remain outside its fake ports. Backend A2A and TextSession execution is outside the DST.

/// One campaign's boundary.
pub struct Boundary {
    /// The cores the run steps itself, one event at a time.
    pub cores: &'static [&'static str],
    /// The fakes standing in for every provider and peer, by module.
    pub fakes: &'static [&'static str],
    /// Production execution scheduled alongside core event delivery.
    pub scheduled: &'static [&'static str],
    /// VoiceRuntime adapters outside the DST, with what verifies them.
    pub outside: &'static [&'static str],
}

/// The voice campaigns (`src/survey_tests.rs`, `src/soak_tests.rs`, `src/pos_tests.rs`).
pub const VOICE: Boundary = Boundary {
    cores: &[
        "voice decision core (agent_session::VoiceDriver), stepped by the runtime's real loop \
         (lk_agents_runtime::voice::VoiceRuntime::serve, then ::close with its task drain) per event \
         the deterministic harness delivers to its queue, with effects performed against fake ports",
        "voice data plane (data_plane forwarder), stepped by the real runtime per microphone frame",
        "the runtime's room and avatar executors (lk_agents_runtime::room_io, ::avatar, ::rtc) \
         over FakeRoom, on the run's injected scheduler and clock",
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
        "fakes::avatar (generator/media/RTC ports under the real runtime runner and stream executors)",
        "fakes::realtime (a realtime model at the Realtime port: the server model drawn from \
         measured timings, its happenings as the port's events, a fallback setup's swap as one; no \
         session or fallback logic); fakes::realtime::duplex (a duplex model's events at \
         the Realtime boundary, as its adapter's segmenter makes them)",
        "fakes::room (the LiveKit SDK room in memory, on the run's clock)",
    ],
    scheduled: &[
        "production runtime tasks (generation, playout, tools, hooks, room and avatar I/O) \
         on the injected scheduler and clock; the task bridge samples their execution turns",
    ],
    outside: &[
        "inference's realtime session (machine, pump, wire codec) and fallback adapter \
         (lk_agents_inference::realtime; its own tests over the simulated OpenAI Realtime server, \
         the fakes mirror their outcomes)",
        "native provider HTTP/WebSocket clients (F-18 real-peer tests)",
        "real LiveKit rooms through the rtc SDK (H-13, K-18)",
        "Python host bindings, lk-agents-py (B-19 conformance)",
        "worker, IPC and job processes (lk-agents-worker's own tests; the DST fakes the worker)",
        "backend A2A/TextSession execution and storage (core, runtime and backend contracts)",
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
            ("scheduled production execution", self.scheduled),
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
