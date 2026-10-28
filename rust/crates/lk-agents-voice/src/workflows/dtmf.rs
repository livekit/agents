//! DTMF collection: the digits the caller presses on the keypad, debounced, with an optional
//! read-back to confirm them; the collector, and the driver running it.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::{Out, Signal, Timer};
use crate::agent_session::{Answer, Driver, DtmfCommand, RequestId};
use crate::events::SessionEvent;
use crate::trace::{InterruptCause, InterruptStep};
use crate::workflows::instructions::render;
use crate::workflows::prompts::dtmf as p;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::{AgentState, UserState};
use std::sync::Arc;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// `DtmfEvent(str, Enum)` (`utils.DtmfEvent`). Declared in code order: the discriminant is
/// [`dtmf_event_to_code`]'s answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DtmfEvent {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Star,
    Pound,
    A,
    B,
    C,
    D,
}

impl DtmfEvent {
    const ALL: [DtmfEvent; 16] = {
        use DtmfEvent::*;
        [Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Star, Pound, A, B, C, D]
    };
    const VALUES: [&str; 16] =
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#", "A", "B", "C", "D"];

    pub fn as_str(self) -> &'static str {
        Self::VALUES[self as usize]
    }

    /// `DtmfEvent(value)`; `None` where the reference raises `ValueError`.
    pub fn parse(value: &str) -> Option<DtmfEvent> {
        Self::VALUES.iter().position(|v| *v == value).map(|i| Self::ALL[i])
    }
}

/// `dtmf_event_to_code`. The reference comment says letters are 10-15; they are 12-15 (`*` 10, `#` 11).
pub fn dtmf_event_to_code(event: DtmfEvent) -> u8 {
    event as u8
}

/// `format_dtmf` (`utils.format_dtmf`).
pub fn format_dtmf(events: &[DtmfEvent]) -> String {
    events.iter().map(|e| e.as_str()).collect::<Vec<_>>().join(" ")
}

pub use lk_agents_abi::voice::workflows::GetDtmfResult;

/// `GetDtmfResult.from_dtmf_inputs(events)`: the pressed digits, formatted.
pub fn dtmf_result(inputs: &[DtmfEvent]) -> GetDtmfResult {
    GetDtmfResult { user_input: format_dtmf(inputs) }
}

/// Each symbol a key, or the first that is not one, refused by name.
fn parse_all(
    symbols: &[String],
    refusal: impl Fn(&str) -> String,
) -> Result<Vec<DtmfEvent>, String> {
    symbols.iter().map(|s| DtmfEvent::parse(s).ok_or_else(|| refusal(s))).collect()
}

/// `dtmf_event_to_code` (`beta/workflows/utils.py:30`): digits are themselves, `*` is 10, `#` is 11
/// and `A`-`D` are 12-15.
#[lk_agents_abi::host(export(name = "dtmf_code", pid = false))]
pub fn dtmf_code(symbol: &str) -> Result<u8, String> {
    let event =
        DtmfEvent::parse(symbol).ok_or_else(|| format!("unknown DTMF symbol {symbol:?}"))?;
    Ok(dtmf_event_to_code(event))
}

/// `format_dtmf` (`beta/workflows/utils.py:44`): the space-joined digits that fill
/// `prompts::dtmf`'s `{digits}` hole.
#[lk_agents_abi::host(export(name = "format_dtmf", pid = false))]
pub fn format_dtmf_symbols(symbols: Vec<String>) -> Result<String, String> {
    Ok(format_dtmf(&parse_all(&symbols, |s| format!("unknown DTMF symbol {s:?}"))?))
}

/// `GetDtmfResult` as a `_core` value: the record plus its constructor from the keys pressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtmfResult(pub GetDtmfResult);

/// GetDtmfResult(user_input: 'str')
#[lk_agents_abi::host(object(
    name = "GetDtmfResult",
    cell = "value",
    pid = false,
    pyclass(eq, from_py_object),
    derive(Clone, PartialEq)
))]
impl DtmfResult {
    #[host(new(kw_only))]
    pub fn new(user_input: String) -> Self {
        DtmfResult(GetDtmfResult { user_input })
    }

    #[host(getter)]
    pub fn user_input(&self) -> String {
        self.0.user_input.clone()
    }

    #[host(method(name = "__repr__"))]
    pub fn host_repr(&self) -> String {
        format!(
            "GetDtmfResult(user_input={})",
            lk_agents_abi::utils::strings::py_repr(&self.0.user_input)
        )
    }

    /// `GetDtmfResult.from_dtmf_inputs(events)`: the digits pressed, formatted. An unknown event
    /// raises rather than being dropped.
    #[host(method)]
    pub fn from_dtmf_inputs(events: Vec<String>) -> Result<Self, String> {
        Ok(DtmfResult(dtmf_result(&parse_all(&events, |e| format!("{e:?} is not a DTMF key"))?)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtmfAction {
    /// Nothing. Either a guard rejected the event or there was nothing pending to cancel.
    Idle,
    /// `_generate_dtmf_reply.schedule()`: arm the timer for `at`, **replacing** any pending one.
    Schedule { at: Ts },
    /// `_generate_dtmf_reply.cancel()`.
    Cancel,
    /// `Debounced.__call__`: cancel the timer and run the body now ([`DtmfCollector::generate_reply`]).
    RunNow,
    /// Not a DTMF digit. See [`DtmfCollector::on_sip_dtmf`].
    Ignored,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtmfReply {
    /// `self.complete(ToolError(message))`: the timeout fired before `num_digits` arrived.
    Incomplete(String),
    /// `self.complete(GetDtmfResult(...))`.
    Complete(GetDtmfResult),
    /// `generate_reply(user_input=...)`, then [`DtmfCollector::finish_reply`].
    Confirm { user_input: String },
}

/// What a driver-run collection came to. `Complete` completes with the result, `Incomplete` with a
/// `ToolError` (the debounce fired early); `Stopped` completes nothing. Confirm is not an outcome:
/// the driver speaks the read-back and the task completes via `confirm_inputs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtmfOutcome {
    Complete(GetDtmfResult),
    Incomplete { message: String },
    Stopped,
}

impl DtmfOutcome {
    /// The record a host reads (`lk_agents_abi::voice::workflows::DtmfOutcomeWire`).
    pub fn wire(&self) -> lk_agents_abi::voice::workflows::DtmfOutcomeWire {
        use lk_agents_abi::voice::workflows as w;
        match self {
            DtmfOutcome::Complete(r) => {
                w::DtmfOutcomeWire::Complete(w::DtmfComplete { user_input: r.user_input.clone() })
            }
            DtmfOutcome::Incomplete { message } => {
                w::DtmfOutcomeWire::Incomplete(w::DtmfIncomplete { message: message.clone() })
            }
            DtmfOutcome::Stopped => w::DtmfOutcomeWire::Stopped(w::DtmfStopped {}),
        }
    }
}

/// The event/timer half of DTMF collection. Not `Clone`: a copy would share the guard.
#[derive(Debug)]
pub struct DtmfCollector {
    num_digits: usize,
    ask_for_confirmation: bool,
    timeout: Duration,
    stop_event: DtmfEvent,
    inputs: Vec<DtmfEvent>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `self._dtmf_reply_running`, shared with the host's `dtmf_reply_running()`.
    reply_running: Arc<AtomicBool>,
}

impl DtmfCollector {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(
        num_digits: usize,
        ask_for_confirmation: bool,
        timeout: Duration,
        stop_event: DtmfEvent,
    ) -> Result<Self, &'static str> {
        if num_digits == 0 {
            return Err("num_digits must be greater than 0");
        }
        Ok(DtmfCollector {
            num_digits,
            ask_for_confirmation,
            timeout,
            stop_event,
            inputs: Vec::new(),
            reply_running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// `dtmf_reply_running()`.
    pub fn reply_running(&self) -> bool {
        self.reply_running.load(Ordering::SeqCst)
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The guard, for the host to read at any time.
    pub fn reply_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.reply_running)
    }

    /// `_on_sip_dtmf_received`. Takes the raw digit: the stop key is tested before parsing, and an
    /// unknown digit is [`DtmfAction::Ignored`] (the reference's dropped `ValueError`).
    pub fn on_sip_dtmf(&mut self, digit: &str, now: Ts) -> DtmfAction {
        if self.reply_running() {
            return DtmfAction::Idle;
        }

        // immediately kick off the DTMF reply generation if matches the stop event
        if digit == self.stop_event.as_str() {
            return DtmfAction::RunNow;
        }

        let Some(event) = DtmfEvent::parse(digit) else { return DtmfAction::Ignored };
        self.inputs.push(event);
        DtmfAction::Schedule { at: now.saturating_add(self.timeout) }
    }

    /// `_on_user_state_changed`.
    pub fn on_user_state_changed(&mut self, new_state: &str, now: Ts) -> DtmfAction {
        self.on_state_changed(new_state == "speaking", now)
    }

    /// `_on_agent_state_changed`. Thinking counts as busy; speaking does too.
    pub fn on_agent_state_changed(&mut self, new_state: &str, now: Ts) -> DtmfAction {
        self.on_state_changed(matches!(new_state, "speaking" | "thinking"), now)
    }

    fn on_state_changed(&mut self, busy: bool, now: Ts) -> DtmfAction {
        if self.reply_running() {
            return DtmfAction::Idle;
        }
        if busy {
            // clear any pending DTMF reply generation
            DtmfAction::Cancel
        } else if !self.inputs.is_empty() {
            // A full debounce restart (`schedule()`), not a resume.
            DtmfAction::Schedule { at: now.saturating_add(self.timeout) }
        } else {
            DtmfAction::Idle
        }
    }

    /// The debounce body (`._generate_dtmf_reply`) without its `finally`; the interrupt is the driver's.
    pub fn generate_reply(&mut self) -> DtmfReply {
        self.reply_running.store(true, Ordering::SeqCst);

        // if input not fully received (i.e. timeout), return None
        if self.inputs.len() != self.num_digits {
            return DtmfReply::Incomplete(render(
                p::NOT_FULLY_RECEIVED,
                &[
                    ("expected", &self.num_digits.to_string()),
                    ("got", &self.inputs.len().to_string()),
                ],
            ));
        }

        // if not asking for confirmation, return the DTMF inputs
        if !self.ask_for_confirmation {
            return DtmfReply::Complete(dtmf_result(&self.inputs));
        }

        DtmfReply::Confirm {
            user_input: render(p::CONFIRM_PROMPT, &[("digits", &format_dtmf(&self.inputs))]),
        }
    }

    /// The `finally`, on every arm. Split out so the confirm arm holds `reply_running` across the
    /// read-back: digits pressed meanwhile must not append.
    pub fn finish_reply(&mut self) {
        self.reply_running.store(false, Ordering::SeqCst);
        self.inputs.clear();
    }
}

/// The collection while it runs.
#[derive(Default)]
pub(crate) struct Dtmf {
    running: Option<Collection>,
}

struct Collection {
    collector: DtmfCollector,
    /// Answered once with the first outcome, or `Stopped`.
    call: Option<RequestId>,
    /// Whether `Timer::DtmfDebounce` is armed, so a cancel is never a cancel of nothing.
    armed: bool,
    /// The confirm arm's read-back: the collector's guard is held until this speech is done.
    confirm: Option<SpeechId>,
}

pub(crate) enum DtmfTaskEvent {
    Start {
        collector: DtmfCollector,
        call: Option<RequestId>,
    },
    Digit(String),
    /// DTMF collection stopped, or the session closes: an owed answer is `Stopped`.
    Stop,
    AgentState(AgentState),
    UserState(UserState),
    SpeechDone(SpeechId),
    Debounced,
    /// The read-back asked for: the speech holding the guard, or `None` when it was refused.
    ReadBack(Option<SpeechId>),
}

impl From<DtmfCommand> for DtmfTaskEvent {
    fn from(cmd: DtmfCommand) -> Self {
        match cmd {
            DtmfCommand::Start { collector, call } => DtmfTaskEvent::Start { collector, call },
            DtmfCommand::Digit { digit } => DtmfTaskEvent::Digit(digit),
            DtmfCommand::Stop => DtmfTaskEvent::Stop,
        }
    }
}

impl Dtmf {
    pub(crate) fn handle(&mut self, ev: DtmfTaskEvent, now: Ts, out: &mut Out) {
        let action = match ev {
            DtmfTaskEvent::Start { collector, call } => {
                self.stop(out);
                self.running = Some(Collection { collector, call, armed: false, confirm: None });
                return;
            }
            DtmfTaskEvent::Stop => return self.stop(out),
            DtmfTaskEvent::Debounced => {
                let Some(dtmf) = &mut self.running else { return };
                dtmf.armed = false;
                return self.reply(out);
            }
            DtmfTaskEvent::ReadBack(speech) => {
                let Some(dtmf) = &mut self.running else { return };
                match speech {
                    Some(speech) => dtmf.confirm = Some(speech),
                    None => dtmf.collector.finish_reply(),
                }
                return;
            }
            ev => {
                let Some(dtmf) = &mut self.running else { return };
                match ev {
                    DtmfTaskEvent::Digit(digit) => dtmf.collector.on_sip_dtmf(&digit, now),
                    DtmfTaskEvent::AgentState(new) => {
                        dtmf.collector.on_agent_state_changed(new.as_str(), now)
                    }
                    DtmfTaskEvent::UserState(new) => {
                        dtmf.collector.on_user_state_changed(new.as_str(), now)
                    }
                    // The read-back's end releases its guard.
                    DtmfTaskEvent::SpeechDone(speech) if dtmf.confirm == Some(speech) => {
                        dtmf.confirm = None;
                        dtmf.collector.finish_reply();
                        return;
                    }
                    _ => return,
                }
            }
        };
        self.perform(action, out);
    }

    /// The call a collection still owes, when the loop ends.
    pub(crate) fn take_call(&mut self) -> Option<RequestId> {
        self.running.take().and_then(|d| d.call)
    }

    /// The collection stops: its guard released (the host reading `dtmf_reply_running()` must not
    /// see a guard nobody will release), its debounce cancelled, an owed answer `Stopped`.
    fn stop(&mut self, out: &mut Out) {
        let Some(mut dtmf) = self.running.take() else { return };
        dtmf.collector.finish_reply();
        if dtmf.armed {
            out.machine().cancel(Timer::DtmfDebounce);
        }
        out.answer(dtmf.call, Answer::Dtmf(DtmfOutcome::Stopped));
    }

    fn perform(&mut self, action: DtmfAction, out: &mut Out) {
        let Some(dtmf) = &mut self.running else { return };
        match action {
            DtmfAction::Idle | DtmfAction::Ignored => {}
            // `schedule()` replaces a pending timer: one arm, which the timer set replaces.
            DtmfAction::Schedule { at } => {
                dtmf.armed = true;
                out.machine().arm(Timer::DtmfDebounce, at);
            }
            // `Debounced.run()`: the pending timer cancelled, the body now.
            DtmfAction::Cancel | DtmfAction::RunNow => {
                if std::mem::take(&mut dtmf.armed) {
                    out.machine().cancel(Timer::DtmfDebounce);
                }
                if matches!(action, DtmfAction::RunNow) {
                    self.reply(out);
                }
            }
        }
    }

    /// `_generate_dtmf_reply`: interrupt, then the collector's decision; the confirm arm holds the
    /// guard until the read-back is done.
    fn reply(&mut self, out: &mut Out) {
        let Some(dtmf) = &mut self.running else { return };
        out.signal(Signal::InterruptForDtmf);
        let outcome = match dtmf.collector.generate_reply() {
            DtmfReply::Incomplete(message) => DtmfOutcome::Incomplete { message },
            DtmfReply::Complete(result) => DtmfOutcome::Complete(result),
            DtmfReply::Confirm { user_input } => {
                return out.signal(Signal::ReadBackDtmf(user_input));
            }
        };
        // The guard released; the call answered once (`if not self.done()`: a task already
        // completed takes no second result).
        dtmf.collector.finish_reply();
        out.answer(dtmf.call.take(), Answer::Dtmf(outcome));
    }
}

impl Driver {
    pub(crate) fn dtmf_step(&mut self, ev: DtmfTaskEvent, out: &mut Out) {
        let now = self.step.now;
        self.activity.dtmf.handle(ev, now, out);
    }

    /// The edges a collection reads, off the events on their way to the host.
    pub(crate) fn dtmf_on_event(&mut self, ev: &SessionEvent, out: &mut Out) {
        let ev = match ev {
            SessionEvent::AgentStateChanged { new, .. } => DtmfTaskEvent::AgentState(*new),
            SessionEvent::UserStateChanged { new, .. } => DtmfTaskEvent::UserState(*new),
            SessionEvent::SpeechDone { speech, .. } => DtmfTaskEvent::SpeechDone(*speech),
            _ => return,
        };
        self.dtmf_step(ev, out);
    }

    /// `self.session.interrupt()`: the host's non-forced interrupt, from the task's code.
    pub(crate) fn interrupt_for_dtmf(&mut self, now: Ts, out: &mut Out) {
        out.trace(InterruptStep::HostInterrupt { speech: None, force: false });
        if let Ok(speeches) = self.interrupt_all(now, false, InterruptCause::Host, out) {
            let waiter =
                crate::agent_activity::false_interruption::InterruptWaiter { speeches, call: None };
            self.activity.interrupt_waiters.push(waiter);
        }
    }

    /// The confirm arm's read-back; one refused at creation (a draining or closing queue)
    /// releases the guard now.
    pub(crate) fn read_back_dtmf(&mut self, now: Ts, user_input: String, out: &mut Out) {
        // `session.generate_reply()` answers on a realtime model's own socket; the read-back's
        // speech is named when the server creates it.
        if self.activity.rt_session.is_some() {
            self.realtime_reply(now, Some(user_input), None, None, None, None, true, out);
            return;
        }
        let h = self.generate_reply_speech(
            now,
            Some(user_input),
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            out,
        );
        let queued = self.activity.speech_scheduler.contains(h.id());
        self.dtmf_step(DtmfTaskEvent::ReadBack(queued.then(|| h.id())), out);
    }
}

/// The pause between two published keypad tones.
pub const DTMF_PUBLISH_DELAY: Duration = Duration::from_millis(300);

/// `send_dtmf_events(events)`: each key published as a keypad tone in `room`, 0.3 s apart; the
/// answer to the model names the keys sent, or the first that failed and why.
#[lk_agents_abi::host(export)]
pub async fn send_dtmf_events(
    room: std::sync::Arc<dyn lk_agents_rtc::RtcRoom>,
    events: Vec<String>,
    #[host(ambient)] clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> String {
    for event in &events {
        let sent = match DtmfEvent::parse(event) {
            Some(key) => room
                .local()
                .publish_dtmf(u32::from(dtmf_event_to_code(key)), event.clone())
                .await
                .map_err(|err| match err {
                    lk_agents_rtc::RtcError::Host(host) => host.message,
                    other => other.to_string(),
                }),
            None => Err(format!("unknown DTMF symbol {event:?}")),
        };
        if let Err(error) = sent {
            return format!("Failed to send DTMF event: {event}. Error: {error}");
        }
        clock.wait_until(clock.now().saturating_add(DTMF_PUBLISH_DELAY)).await;
    }
    format!("Successfully sent DTMF events: {}", events.join(", "))
}

/// Why [`receive_dtmf`] stopped short.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "livekit.agents.llm.tool_context.ToolError"))]
pub enum ReceiveDtmfError {
    #[error("num_digits must be greater than 0")]
    #[host(kind = "ValueError")]
    NoDigits,
    /// `DtmfEvent(digit)` refused the key, quoted as Python's `repr`.
    #[error("{0} is not a valid DtmfEvent")]
    #[host(kind = "ValueError")]
    UnknownKey(String),
    #[error("Digits input not fully received. Expect {0} digits, got {1}")]
    Incomplete(usize, usize),
}

#[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
/// `receive_dtmf(room, num_digits, timeout, stop_event)`: the keys a SIP caller presses in
/// `room`, until `num_digits` of them, the stop key, or `timeout` seconds without a key once the
/// first is in. Fewer than `num_digits` is [`ReceiveDtmfError::Incomplete`].
#[lk_agents_abi::host(export)]
pub async fn receive_dtmf(
    room: std::sync::Arc<dyn lk_agents_rtc::RtcRoom>,
    num_digits: usize,
    timeout: f64,
    stop_event: String,
    #[host(ambient)] clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> Result<DtmfResult, ReceiveDtmfError> {
    use lk_agents_abi::utils::strings::py_repr;
    if num_digits == 0 {
        return Err(ReceiveDtmfError::NoDigits);
    }
    let stop = DtmfEvent::parse(&stop_event)
        .ok_or_else(|| ReceiveDtmfError::UnknownKey(py_repr(&stop_event)))?;
    let gap = Duration::try_from_secs_f64(timeout.max(0.0)).unwrap_or(Duration::MAX);
    let mut events = room.events();
    let mut digits = Vec::new();
    let mut deadline = None;
    loop {
        let next = match deadline {
            None => events.recv().await,
            Some(at) => tokio::select! {
                next = events.recv() => next,
                _ = clock.wait_until(at) => break,
            },
        };
        let Some(event) = next else { break };
        let lk_agents_rtc::RoomEvent::SipDtmf { digit, .. } = event else { continue };
        if digit == stop.as_str() {
            break;
        }
        digits.push(
            DtmfEvent::parse(&digit)
                .ok_or_else(|| ReceiveDtmfError::UnknownKey(py_repr(&digit)))?,
        );
        if digits.len() == num_digits {
            break;
        }
        deadline = Some(clock.now().saturating_add(gap));
    }
    if digits.len() != num_digits {
        return Err(ReceiveDtmfError::Incomplete(num_digits, digits.len()));
    }
    Ok(DtmfResult(dtmf_result(&digits)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector(num_digits: usize, timeout_secs: f64) -> DtmfCollector {
        DtmfCollector::new(
            num_digits,
            false,
            Duration::from_secs_f64(timeout_secs),
            DtmfEvent::Pound,
        )
        .unwrap()
    }

    // Catches: a digit not re-arming the debounce from now, the stop key buffered, or an unknown
    // symbol buffered rather than dropped.
    #[test]
    fn a_digit_schedules_the_timeout_and_the_stop_key_runs_now() {
        let now = Ts::from_nanos(7_000_000_000);
        let mut c = collector(4, 2.5);
        assert_eq!(
            c.on_sip_dtmf("1", now),
            DtmfAction::Schedule { at: now.saturating_add(Duration::from_secs_f64(2.5)) }
        );
        assert_eq!(c.on_sip_dtmf("#", now), DtmfAction::RunNow);
        assert_eq!(c.on_sip_dtmf("E", now), DtmfAction::Ignored);
        assert_eq!(c.inputs, &[DtmfEvent::One]);
    }

    // Catches: treating `thinking` as idle (a half-entered sequence reported).
    #[test]
    fn agent_thinking_cancels_the_debounce() {
        let now = Ts::ZERO;
        let mut c = collector(2, 1.0);
        c.on_sip_dtmf("1", now);
        assert_eq!(c.on_agent_state_changed("thinking", now), DtmfAction::Cancel);
        assert!(
            matches!(c.on_agent_state_changed("listening", now), DtmfAction::Schedule { .. }),
            "and re-arms when idle"
        );
    }

    // Catches: a short sequence completing with a result instead of a `ToolError`.
    #[test]
    fn a_short_sequence_is_an_error_not_a_result() {
        let now = Ts::ZERO;
        let mut c = collector(3, 1.0);
        c.on_sip_dtmf("1", now);
        c.on_sip_dtmf("2", now);
        let DtmfReply::Incomplete(e) = c.generate_reply() else { panic!("expected an error") };
        assert!(e.contains('3') && e.contains('2'), "got {e:?}");
        assert!(c.reply_running(), "and the guard is held until finish_reply");
        c.finish_reply();
        assert!(!c.reply_running());
        assert!(c.inputs.is_empty(), "the buffer clears on every arm");
    }
}
