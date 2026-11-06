//! Event-loop stalls (docs/internals/telemetry.md "Event-loop stalls"): each one's measurement,
//! and its `event_loop_blocked` span (with the loop's stack) and warning line, within a rate limit.

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
pub fn event_loop_blocked(
    cx: &Cx,
    blocked: &Blocked,
    duration: f64,
    stack: Option<String>,
    import: Option<String>,
) {
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
    log_stall(duration, stack.as_deref(), import.as_deref());
    let start = now.saturating_sub((duration * 1e9) as u64);
    let span = telemetry::start_at(
        cx,
        EventLoopBlocked {
            duration: Seconds(duration),
            threshold: Seconds(BLOCKED_WARNING),
            severity,
            stack,
            import,
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

/// The stall's warning: one line naming the innermost frame (the code that blocked); from
/// [`BLOCKED_ERROR`] on, the whole stack as well, most recent call last, as a traceback reads.
fn log_stall(duration: f64, stack: Option<&str>, import: Option<&str>) {
    let message = stall_message("event loop blocked for", duration, stack, import);
    log::warn!(target: "livekit.agents", "{message}");
}

/// `{lead} {duration}ms at {innermost frame}[, importing {file}]`, and the stack below it when the
/// stall lasted [`BLOCKED_ERROR`] or longer.
pub fn stall_message(
    lead: &str,
    duration: f64,
    stack: Option<&str>,
    import: Option<&str>,
) -> String {
    let mut message = format!("{lead} {}ms", (duration * 1e3).round());
    let frames: Vec<&str> = stack.map(|stack| stack.lines().collect()).unwrap_or_default();
    if let Some(innermost) = frames.first() {
        message.push_str(&format!(" at {innermost}"));
    }
    if let Some(file) = import {
        message.push_str(&format!(", importing {file}"));
    }
    if duration >= BLOCKED_ERROR && frames.len() > 1 {
        message.push_str("\nStack (most recent call last):");
        for frame in frames.iter().rev() {
            message.push_str(&format!("\n  {frame}"));
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use super::stall_message;

    // Catches: a short stall logging its whole stack, or a long one hiding the caller frames.
    #[test]
    fn a_long_stall_logs_its_whole_stack_a_short_one_its_frame() {
        let stack = "requests_get (http.py:9)\nlookup (tools.py:42)\nmain (app.py:7)";
        assert_eq!(
            stall_message("event loop blocked for", 0.2, Some(stack), None),
            "event loop blocked for 200ms at requests_get (http.py:9)"
        );
        assert_eq!(
            stall_message("event loop blocked for", 0.8, Some(stack), Some("slow.py")),
            "event loop blocked for 800ms at requests_get (http.py:9), importing slow.py\n\
             Stack (most recent call last):\n  main (app.py:7)\n  lookup (tools.py:42)\n  \
             requests_get (http.py:9)"
        );
        assert_eq!(
            stall_message("event loop blocked for", 0.8, None, None),
            "event loop blocked for 800ms"
        );
    }
}
