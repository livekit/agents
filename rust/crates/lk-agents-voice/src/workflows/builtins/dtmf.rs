use crate::workflows::builtins::ConfigError;
use crate::workflows::builtins::instructions::render;
use crate::workflows::builtins::prompts::dtmf as p;
use crate::workflows::builtins::schema::{Param, tool};
use crate::workflows::{Task, TaskCtx, TaskOptions, ToolError, ToolReply, args};
use async_trait::async_trait;
use lk_agents_abi::{ToolSpec, Ts};
use lk_agents_voice::generation::Instructions;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Order matters: it is the tool schema's `enum` order, which the model sees.
pub const DTMF_VALUES: [&str; 16] =
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "*", "#", "A", "B", "C", "D"];

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
fn parse_all(symbols: &[String], refusal: impl Fn(&str) -> String) -> Result<Vec<DtmfEvent>, String> {
    symbols.iter().map(|s| DtmfEvent::parse(s).ok_or_else(|| refusal(s))).collect()
}

/// `dtmf_event_to_code` (`beta/workflows/utils.py:30`): digits are themselves, `*` is 10, `#` is 11
/// and `A`-`D` are 12-15.
#[lk_agents_abi::host(export(name = "dtmf_code", pid = false))]
pub fn dtmf_code(symbol: &str) -> Result<u8, String> {
    let event = DtmfEvent::parse(symbol).ok_or_else(|| format!("unknown DTMF symbol {symbol:?}"))?;
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

// A workflow's collected value reaches the host (`CoreObject`) as its class, over a clone.
#[cfg(feature = "python")]
lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::CoreObjectClass { to_host: lk_agents_abi::python::core_object_to_host::<DtmfResult> }
}

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
        format!("GetDtmfResult({:?})", self.0)
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
    /// `self.complete(ToolError(...))` -- the timeout fired before `num_digits` arrived.
    Incomplete(ToolError),
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

/// The event/timer half of `GetDtmfTask`. Not `Clone`: a copy would share the guard.
#[derive(Debug)]
pub struct DtmfCollector {
    num_digits: usize,
    ask_for_confirmation: bool,
    timeout: Duration,
    stop_event: DtmfEvent,
    inputs: Vec<DtmfEvent>,
    /// `self._dtmf_reply_running`, shared with the host's `dtmf_reply_running()`.
    reply_running: Arc<AtomicBool>,
}

impl DtmfCollector {
    pub fn new(
        num_digits: usize,
        ask_for_confirmation: bool,
        timeout: Duration,
        stop_event: DtmfEvent,
    ) -> Result<Self, ConfigError> {
        if num_digits == 0 {
            return Err(ConfigError::NonPositiveDigits);
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

    pub fn pending(&self) -> &[DtmfEvent] {
        &self.inputs
    }

    /// `dtmf_reply_running()`.
    pub fn reply_running(&self) -> bool {
        self.reply_running.load(Ordering::SeqCst)
    }

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
            return DtmfReply::Incomplete(ToolError::plain(render(
                p::NOT_FULLY_RECEIVED,
                &[
                    ("expected", &self.num_digits.to_string()),
                    ("got", &self.inputs.len().to_string()),
                ],
            )));
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

#[derive(Debug, Clone)]
pub struct GetDtmfConfig {
    pub num_digits: usize,
    pub ask_for_confirmation: bool,
    /// The **per-digit** timeout, not a total budget: every digit re-arms it.
    pub dtmf_input_timeout: Duration,
    pub dtmf_stop_event: DtmfEvent,
    pub extra_instructions: Option<String>,
}

impl Default for GetDtmfConfig {
    fn default() -> Self {
        GetDtmfConfig {
            // Required in the reference; zero makes `GetDtmfTask::new` reject a forgotten value.
            num_digits: 0,
            ask_for_confirmation: false,
            dtmf_input_timeout: Duration::from_secs_f64(4.0),
            dtmf_stop_event: DtmfEvent::Pound,
            extra_instructions: None,
        }
    }
}

/// A task to collect DTMF inputs from the user.
///
/// Return a string of DTMF inputs if collected successfully, otherwise None.
pub struct GetDtmfTask {
    instructions: Instructions,
    ask_for_confirmation: bool,
}

impl GetDtmfTask {
    pub fn new(cfg: GetDtmfConfig, _options: TaskOptions) -> Result<Self, ConfigError> {
        // The event half (the collector) runs in the driver; only its validation is shared.
        if cfg.num_digits == 0 {
            return Err(ConfigError::NonPositiveDigits);
        }

        let mut instructions = String::from(p::BASE);
        if cfg.ask_for_confirmation {
            instructions.push_str(p::WITH_CONFIRMATION);
        } else {
            instructions.push_str(p::WITHOUT_CONFIRMATION);
        }
        if let Some(extra) = &cfg.extra_instructions {
            instructions.push('\n');
            instructions.push_str(extra);
        }

        Ok(GetDtmfTask {
            // A plain `str`, not an `Instructions` -- this workflow has no modality variants.
            instructions: Instructions::plain(instructions),
            ask_for_confirmation: cfg.ask_for_confirmation,
        })
    }
}

#[async_trait]
impl Task for GetDtmfTask {
    type Output = GetDtmfResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    /// Exactly one tool, by the flag: both would let the model skip the confirmation.
    fn tools(&self) -> Vec<ToolSpec> {
        let (name, desc) = if self.ask_for_confirmation {
            (p::CONFIRM_TOOL, p::CONFIRM_DESC)
        } else {
            (p::RECORD_TOOL, p::RECORD_DESC)
        };
        vec![tool(
            name,
            desc,
            &[Param::str_enum_list("inputs", &DTMF_VALUES, "")],
            Default::default(),
        )]
    }

    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<GetDtmfResult>,
    ) -> ToolReply {
        let expected = if self.ask_for_confirmation { p::CONFIRM_TOOL } else { p::RECORD_TOOL };
        if name != expected {
            return ToolReply::unknown_tool(name);
        }
        let raw = match args::string_list(&args_json, name, "inputs") {
            Ok(v) => v,
            Err(e) => return ToolReply::Error(e),
        };
        let mut inputs = Vec::with_capacity(raw.len());
        for value in &raw {
            match DtmfEvent::parse(value) {
                Some(e) => inputs.push(e),
                None => {
                    // What pydantic's enum validation produces before the body ever runs.
                    return ToolReply::Error(ToolError::plain(format!(
                        "invalid arguments for `{name}`: `{value}` is not a DTMF event"
                    )));
                }
            }
        }
        // A second call is a no-op here (the reference raises).
        ctx.complete(dtmf_result(&inputs));
        ToolReply::Silent
    }
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
        assert_eq!(c.pending(), &[DtmfEvent::One]);
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
        assert!(e.message.contains('3') && e.message.contains('2'), "got {:?}", e.message);
        assert!(c.reply_running(), "and the guard is held until finish_reply");
        c.finish_reply();
        assert!(!c.reply_running());
        assert!(c.pending().is_empty(), "the buffer clears on every arm");
    }
}
