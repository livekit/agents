//! Event-loop stalls (docs/internals/telemetry.md "Event-loop stalls"): each one's measurement,
//! and its `event_loop_blocked` span (with the loop's stack) and warning line, within a rate limit
//! that a loop still stuck is exempt from.

use lk_agents_abi::telemetry::decl::Seconds;
use lk_agents_abi::telemetry::errors::{ErrorType, Failure};
use lk_agents_abi::telemetry::instruments::EventLoopBlockedDuration;
use lk_agents_abi::telemetry::policy;
use lk_agents_abi::telemetry::record::{StackFrame, StallState};
use lk_agents_abi::telemetry::spans::session::{BlockingSeverity, EventLoopBlocked};
use lk_agents_net::telemetry::{self, Cx};
use parking_lot::Mutex;

use std::collections::VecDeque;
use std::time::Duration;

const STACK_BYTES: usize = 4096;
/// At most this many `event_loop_blocked` spans a minute; the rest are counted.
const BLOCKED_SPANS_PER_MINUTE: usize = 30;

/// The spans reported in the last minute, and the stalls suppressed since the last one.
#[derive(Default)]
pub struct Blocked(Mutex<RateLimit>);

#[derive(Default)]
struct RateLimit {
    sent: VecDeque<u64>,
    suppressed: u32,
}

/// One stall: its measurement always, its span within the rate limit.
pub fn event_loop_blocked(
    cx: &Cx,
    blocked: &Blocked,
    duration: Duration,
    stack: Option<Vec<StackFrame>>,
    import: Option<String>,
    state: StallState,
) {
    let severity = if duration >= policy::BLOCKED_ERROR {
        BlockingSeverity::Error
    } else {
        BlockingSeverity::Warning
    };
    telemetry::measure(cx, &EventLoopBlockedDuration { severity }, duration.as_secs_f64());
    let now = cx.now_nanos();
    let suppressed = {
        let RateLimit { sent, suppressed } = &mut *blocked.0.lock();
        while sent.front().is_some_and(|at| now.saturating_sub(*at) >= 60_000_000_000) {
            sent.pop_front();
        }
        // A loop still stuck is reported once per stall and never suppressed: it may be the
        // one report of a deadlock.
        if sent.len() >= BLOCKED_SPANS_PER_MINUTE && state == StallState::Ended {
            *suppressed += 1;
            return;
        }
        sent.push_back(now);
        std::mem::take(suppressed)
    };
    let frames = stack.as_deref().map(format_frames);
    let message =
        formatted_message("event loop blocked for", duration, frames.as_deref(), import.as_deref());
    log::warn!(target: "livekit.agents", "{message}");
    let stack = frames.map(|frames| frames.join("\n"));
    let start = now.saturating_sub(u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX));
    let span = telemetry::start_at(
        cx,
        EventLoopBlocked {
            duration: Seconds(duration.as_secs_f64()),
            threshold: Seconds(policy::BLOCKED_WARNING.as_secs_f64()),
            severity,
            stack,
            import,
            suppressed: (suppressed > 0).then_some(suppressed),
        },
        start,
    );
    if severity == BlockingSeverity::Error {
        let message = format!("the event loop was blocked for {:.3}s", duration.as_secs_f64());
        telemetry::end(span, Failure { kind: ErrorType::EventLoopBlocked, message });
    } else {
        telemetry::end(span, ());
    }
}

fn format_frames(frames: &[StackFrame]) -> Vec<String> {
    let mut bytes = 0;
    frames
        .iter()
        .map(|frame| format!("{} ({}:{})", frame.function, frame.file, frame.line))
        .take_while(|frame| {
            if bytes + frame.len() + 1 > STACK_BYTES {
                return false;
            }
            bytes += frame.len() + usize::from(bytes > 0);
            true
        })
        .collect()
}

/// `{lead} {duration}ms at {innermost frame}[, importing {file}]`, and the stack below it when the
/// stall lasted [`policy::BLOCKED_ERROR`] or longer.
pub fn stall_message(
    lead: &str,
    duration: Duration,
    stack: Option<&[StackFrame]>,
    import: Option<&str>,
) -> String {
    let frames = stack.map(format_frames);
    formatted_message(lead, duration, frames.as_deref(), import)
}

fn formatted_message(
    lead: &str,
    duration: Duration,
    frames: Option<&[String]>,
    import: Option<&str>,
) -> String {
    let mut message = format!("{lead} {}ms", (duration.as_secs_f64() * 1e3).round());
    let frames = frames.unwrap_or_default();
    if let Some(innermost) = frames.first() {
        message.push_str(&format!(" at {innermost}"));
    }
    if let Some(file) = import {
        message.push_str(&format!(", importing {file}"));
    }
    if duration >= policy::BLOCKED_ERROR && frames.len() > 1 {
        message.push_str("\nStack (most recent call last):");
        for frame in frames.iter().rev() {
            message.push_str(&format!("\n  {frame}"));
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a short stall logging its whole stack, or a long one hiding the caller frames.
    #[test]
    fn a_long_stall_logs_its_whole_stack_a_short_one_its_frame() {
        let stack = [
            StackFrame { function: "requests_get".into(), file: "http.py".into(), line: 9 },
            StackFrame { function: "lookup".into(), file: "tools.py".into(), line: 42 },
            StackFrame { function: "main".into(), file: "app.py".into(), line: 7 },
        ];
        assert_eq!(
            stall_message("event loop blocked for", Duration::from_millis(200), Some(&stack), None),
            "event loop blocked for 200ms at requests_get (http.py:9)"
        );
        assert_eq!(
            stall_message(
                "event loop blocked for",
                Duration::from_millis(800),
                Some(&stack),
                Some("slow.py")
            ),
            "event loop blocked for 800ms at requests_get (http.py:9), importing slow.py\n\
             Stack (most recent call last):\n  main (app.py:7)\n  lookup (tools.py:42)\n  \
             requests_get (http.py:9)"
        );
        assert_eq!(
            stall_message("event loop blocked for", Duration::from_millis(800), None, None),
            "event loop blocked for 800ms"
        );
    }

    #[test]
    fn the_span_stack_keeps_the_original_byte_bound() {
        let frame = StackFrame { function: "f".repeat(1000), file: "x.py".into(), line: 1 };
        let frames = format_frames(&vec![frame; 20]);
        assert_eq!(frames.len(), 4);
        assert!(frames.join("\n").len() < STACK_BYTES);
    }
}
