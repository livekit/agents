//! Event-loop stalls (docs/internals/telemetry.md "Event-loop stalls"): each one's measurement,
//! and its `event_loop_blocked` span within a rate limit.

use lk_agents_abi::telemetry::decl::Seconds;
use lk_agents_abi::telemetry::errors::{ErrorType, Failure};
use lk_agents_abi::telemetry::instruments::EventLoopBlockedDuration;
use lk_agents_abi::telemetry::spans::session::{BlockingSeverity, EventLoopBlocked};
use lk_agents_net::telemetry::{self, Cx};
use parking_lot::Mutex;

/// A stall's report decides its severity from these, as 1.8.10's monitor did.
const BLOCKED_WARNING: f64 = 0.1;
const BLOCKED_ERROR: f64 = 0.5;
/// At most this many `event_loop_blocked` spans a minute; the rest are counted.
const BLOCKED_SPANS_PER_MINUTE: usize = 30;

/// The spans reported in the last minute, and the stalls suppressed since the last one.
#[derive(Default)]
pub struct Blocked(Mutex<(std::collections::VecDeque<u64>, u32)>);

/// One stall: its measurement always, its span within the rate limit.
pub fn event_loop_blocked(cx: &Cx, blocked: &Blocked, duration: f64, task: Option<String>) {
    let severity =
        if duration >= BLOCKED_ERROR { BlockingSeverity::Error } else { BlockingSeverity::Warning };
    telemetry::measure(cx, &EventLoopBlockedDuration { severity }, duration);
    let now = cx.now_nanos();
    let suppressed = {
        let (sent, suppressed) = &mut *blocked.0.lock();
        while sent.front().is_some_and(|at| now.saturating_sub(*at) >= 60_000_000_000) {
            sent.pop_front();
        }
        if sent.len() >= BLOCKED_SPANS_PER_MINUTE {
            *suppressed += 1;
            return;
        }
        sent.push_back(now);
        std::mem::take(suppressed)
    };
    let start = now.saturating_sub((duration * 1e9) as u64);
    let span = telemetry::start_at(
        cx,
        EventLoopBlocked {
            duration: Seconds(duration),
            threshold: Seconds(BLOCKED_WARNING),
            gc_time: None,
            cpu_time: None,
            severity,
            task,
            stack: None,
            import: None,
            suppressed: (suppressed > 0).then_some(suppressed),
        },
        start,
    );
    if severity == BlockingSeverity::Error {
        let message = format!("the event loop was blocked for {duration:.3}s");
        telemetry::end(span, Failure { kind: ErrorType::EventLoopBlocked, message });
    } else {
        telemetry::end(span, ());
    }
}
