//! Dispatch tracing from the job process timeline.

/// A job's dispatch timeline, as the host's root span records it.
pub struct JobTimeline;

impl lk_agents_abi::worker::job::JobTrace for JobTimeline {
    fn job_timeline(
        received_at: f64,
        accepted_at: f64,
        assigned_at: f64,
        launched_at: f64,
        entrypoint_started_at: f64,
        server_started_at: i64,
    ) -> (Vec<(String, i64)>, Vec<(String, f64)>) {
        let ns = |ts: f64| (ts * 1e9) as i64;
        let stages = [
            ("job_received", received_at),
            ("job_accepted", accepted_at),
            ("job_assigned", assigned_at),
            ("process_assigned", launched_at),
            ("entrypoint_started", entrypoint_started_at),
        ];
        let mut events: Vec<_> = stages
            .iter()
            .filter(|(_, ts)| *ts != 0.0)
            .map(|(name, ts)| ((*name).to_owned(), ns(*ts)))
            .collect();
        let gaps = [
            ("lk.job.accept_latency", received_at, accepted_at),
            ("lk.job.assignment_latency", accepted_at, assigned_at),
            ("lk.job.launch_latency", assigned_at, launched_at),
            ("lk.job.entrypoint_latency", launched_at, entrypoint_started_at),
            ("lk.job.dispatch_latency", received_at, entrypoint_started_at),
        ];
        let attributes = gaps
            .iter()
            .filter(|(_, start, end)| *start != 0.0 && *end != 0.0)
            .map(|(attr, start, end)| ((*attr).to_owned(), (end - start).max(0.0)))
            .collect();
        if server_started_at > 0 {
            let v = server_started_at as f64;
            let started = if v > 1e17 {
                v / 1e9
            } else if v > 1e11 {
                v / 1e3
            } else {
                v
            };
            events.push(("job_started_on_server".to_owned(), ns(started)));
        }
        (events, attributes)
    }
}
