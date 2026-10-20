//! Every telemetry string this workspace emitted before the vocabulary moved to the protocol
//! registry, pinned against the constant that now carries it. A registry change that renames,
//! drops or retypes one of them fails here, before it changes a byte on the wire.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use proc_macro2::{Delimiter, TokenStream, TokenTree};

use super::{attrs, gen_ai::events, semconv};
use crate::metrics::otel::INSTRUMENTS;
use crate::telemetry::report::{TAG_FAIL, TAG_SUCCESS};
use crate::utils::frame::InboundFrame;
use crate::{ATTRIBUTE_REDACTION_ENABLED, ATTRIBUTE_SIMULATION_ENABLED};

/// `(literal as it was in the source before the change, the constant that replaced it)`.
const PINS: &[(&str, &str)] = &[
    // telemetry/mod.rs `attrs`
    ("lk.speech_id", attrs::SPEECH_ID),
    ("lk.provider_request_ids", attrs::PROVIDER_REQUEST_IDS),
    ("lk.tts.label", attrs::TTS_LABEL),
    ("lk.tts.streaming", attrs::TTS_STREAMING),
    ("lk.is_interruption", attrs::IS_INTERRUPTION),
    ("lk.interruption.probability", attrs::INTERRUPTION_PROBABILITY),
    ("lk.interruption.total_duration", attrs::INTERRUPTION_TOTAL_DURATION),
    ("lk.interruption.prediction_duration", attrs::INTERRUPTION_PREDICTION_DURATION),
    ("lk.interruption.detection_delay", attrs::INTERRUPTION_DETECTION_DELAY),
    ("lk.metric", attrs::node::METRIC),
    ("lk.frame.resolution", attrs::node::FRAME_RESOLUTION),
    ("lk.tts.no_duplex_path", attrs::node::TTS_NO_DUPLEX_PATH),
    ("lk.tts.frames", attrs::node::TTS_FRAMES),
    ("lk.tts.bytes", attrs::node::TTS_BYTES),
    ("lk.error.retryable", attrs::node::ERROR_RETRYABLE),
    // `lk.frame.resolution` values (`InboundFrame::kind`)
    ("root", semconv::lk_frame_resolution::ROOT),
    ("live", semconv::lk_frame_resolution::LIVE),
    ("stale", semconv::lk_frame_resolution::STALE),
    ("unknown", semconv::lk_frame_resolution::UNKNOWN),
    // telemetry/gen_ai.rs `events`
    ("gen_ai.system.message", events::SYSTEM_MESSAGE),
    ("gen_ai.user.message", events::USER_MESSAGE),
    ("gen_ai.assistant.message", events::ASSISTANT_MESSAGE),
    ("gen_ai.tool.message", events::TOOL_MESSAGE),
    ("gen_ai.choice", events::CHOICE),
    // telemetry/report.rs tags
    ("lk.success", TAG_SUCCESS),
    ("lk.fail", TAG_FAIL),
    // telemetry/attributes.rs metadata keys
    ("lk.simulation.enabled", ATTRIBUTE_SIMULATION_ENABLED),
    ("lk.redaction.enabled", ATTRIBUTE_REDACTION_ENABLED),
    // voice/agent_activity/events.rs
    ("lk.pii.user_input", semconv::LK_PII_USER_INPUT),
    // lk-agents-node/src/lib.rs span names (and the two the tests use)
    ("metrics", semconv::span::METRICS),
    ("tts_synthesize", semconv::span::TTS_SYNTHESIZE),
    ("tts_stream", semconv::span::TTS_STREAM),
    ("llm_node", semconv::span::LLM_NODE),
    ("tts_node", semconv::span::TTS_NODE),
    // metrics/otel.rs instrument names, in `INSTRUMENTS` order
    ("lk.agents.turn.e2e_latency", semconv::metric::LK_AGENTS_TURN_E2E_LATENCY),
    ("lk.agents.turn.llm_ttft", semconv::metric::LK_AGENTS_TURN_LLM_TTFT),
    ("lk.agents.turn.tts_ttfb", semconv::metric::LK_AGENTS_TURN_TTS_TTFB),
    ("lk.agents.turn.transcription_delay", semconv::metric::LK_AGENTS_TURN_TRANSCRIPTION_DELAY),
    ("lk.agents.turn.end_of_turn_delay", semconv::metric::LK_AGENTS_TURN_END_OF_TURN_DELAY),
    (
        "lk.agents.turn.on_user_turn_completed_delay",
        semconv::metric::LK_AGENTS_TURN_ON_USER_TURN_COMPLETED_DELAY,
    ),
    ("lk.agents.usage.llm_input_tokens", semconv::metric::LK_AGENTS_USAGE_LLM_INPUT_TOKENS),
    (
        "lk.agents.usage.llm_input_cached_tokens",
        semconv::metric::LK_AGENTS_USAGE_LLM_INPUT_CACHED_TOKENS,
    ),
    ("lk.agents.usage.llm_output_tokens", semconv::metric::LK_AGENTS_USAGE_LLM_OUTPUT_TOKENS),
    (
        "lk.agents.usage.llm_input_audio_tokens",
        semconv::metric::LK_AGENTS_USAGE_LLM_INPUT_AUDIO_TOKENS,
    ),
    (
        "lk.agents.usage.llm_input_text_tokens",
        semconv::metric::LK_AGENTS_USAGE_LLM_INPUT_TEXT_TOKENS,
    ),
    (
        "lk.agents.usage.llm_output_audio_tokens",
        semconv::metric::LK_AGENTS_USAGE_LLM_OUTPUT_AUDIO_TOKENS,
    ),
    (
        "lk.agents.usage.llm_output_text_tokens",
        semconv::metric::LK_AGENTS_USAGE_LLM_OUTPUT_TEXT_TOKENS,
    ),
    ("lk.agents.usage.llm_session_duration", semconv::metric::LK_AGENTS_USAGE_LLM_SESSION_DURATION),
    ("lk.agents.usage.tts_characters", semconv::metric::LK_AGENTS_USAGE_TTS_CHARACTERS),
    ("lk.agents.usage.tts_audio_duration", semconv::metric::LK_AGENTS_USAGE_TTS_AUDIO_DURATION),
    ("lk.agents.usage.stt_audio_duration", semconv::metric::LK_AGENTS_USAGE_STT_AUDIO_DURATION),
    (
        "lk.agents.usage.interruption_num_requests",
        semconv::metric::LK_AGENTS_USAGE_INTERRUPTION_NUM_REQUESTS,
    ),
    ("lk.agents.connection.acquire_time", semconv::metric::LK_AGENTS_CONNECTION_ACQUIRE_TIME),
];

// Catches: a registry rename, removal or re-spelling reaching the wire through the vendored copy.
#[test]
fn every_pre_registry_literal_is_what_its_constant_says() {
    for (old, now) in PINS {
        assert_eq!(old, now, "the wire string changed: `{old}` is now `{now}`");
    }
}

// Catches: an instrument renamed in the registry while `INSTRUMENTS` keeps its old entry count.
#[test]
fn the_instrument_table_is_pinned_in_order() {
    let pinned: Vec<&str> =
        PINS.iter().map(|(old, _)| *old).filter(|s| s.starts_with("lk.agents.")).collect();
    let declared: Vec<&str> = INSTRUMENTS.iter().map(|(name, _, _)| *name).collect();
    assert_eq!(pinned, declared);
}

/// `(instrument, unit the Python host created it with before the change; "" = none)`.
const UNITS: &[(&str, &str)] = &[
    ("lk.agents.turn.e2e_latency", "s"),
    ("lk.agents.turn.llm_ttft", "s"),
    ("lk.agents.turn.tts_ttfb", "s"),
    ("lk.agents.turn.transcription_delay", "s"),
    ("lk.agents.turn.end_of_turn_delay", "s"),
    ("lk.agents.turn.on_user_turn_completed_delay", "s"),
    ("lk.agents.usage.llm_input_tokens", ""),
    ("lk.agents.usage.llm_input_cached_tokens", ""),
    ("lk.agents.usage.llm_output_tokens", ""),
    ("lk.agents.usage.llm_input_audio_tokens", ""),
    ("lk.agents.usage.llm_input_text_tokens", ""),
    ("lk.agents.usage.llm_output_audio_tokens", ""),
    ("lk.agents.usage.llm_output_text_tokens", ""),
    ("lk.agents.usage.llm_session_duration", "s"),
    ("lk.agents.usage.tts_characters", ""),
    ("lk.agents.usage.tts_audio_duration", "s"),
    ("lk.agents.usage.stt_audio_duration", "s"),
    ("lk.agents.usage.interruption_num_requests", ""),
    ("lk.agents.connection.acquire_time", "s"),
];

// Catches: the registry declaring a unit the host does not create the instrument with.
#[test]
fn every_instrument_keeps_the_unit_it_shipped_with() {
    let declared: Vec<(&str, &str)> =
        INSTRUMENTS.iter().map(|(name, _, unit)| (*name, *unit)).collect();
    assert_eq!(declared, UNITS);
}

// Catches: `InboundFrame::kind` (the emitter of `lk.frame.resolution`) drifting from the registry.
#[test]
fn the_frame_resolution_values_are_the_ones_the_bindings_report() {
    assert_eq!(InboundFrame::Root.kind(), "root");
    assert_eq!(InboundFrame::Unknown.kind(), "unknown");
    assert_eq!(semconv::lk_frame_resolution::LIVE, "live");
    assert_eq!(semconv::lk_frame_resolution::STALE, "stale");
}

/// `"lk.…"` literals that are not telemetry, or are deliberately literal, per source file
/// (relative to `rust/crates`), each with its reason. Everything else must come from [`semconv`].
const LITERAL_ALLOWLIST: &[(&str, &[&str], &str)] = &[
    (
        "lk-agents-abi/src/telemetry/attributes.rs",
        &["*"],
        "participant attributes, data-channel topics, frame userdata keys: wire protocol, not telemetry",
    ),
    (
        "lk-agents-abi/src/telemetry/mod.rs",
        &["*"],
        "the `marks_pii` tests spell near-misses on purpose",
    ),
    (
        "lk-agents-abi/src/telemetry/report.rs",
        &["lk.judge.{name}:{}", "lk.judge.booked:fail"],
        "a dynamic tag and its tests",
    ),
    (
        "lk-agents-abi/src/llm/tool_arguments.rs",
        &["lk.pii.raw_arguments", "lk.pii.repaired", "lk.pii.error"],
        "`log::warn!` key-values take literal keys; its test asserts the same keys",
    ),
    (
        "lk-agents-inference/src/mcp/session.rs",
        &["lk.pii.progress_message"],
        "`log::debug!` key-value",
    ),
    (
        "lk-agents-abi/src/evals/llm_judge.rs",
        &["lk.judge.system", "lk.judge.prompt"],
        "chat message ids",
    ),
    ("lk-agents-abi/src/tts/markup/chat.rs", &["lk.expressive.instructions"], "chat message id"),
    ("lk-agents-voice/src/generation/mod.rs", &["lk.agent_task.instructions"], "chat message id"),
    ("lk-agents-voice/src/remote_session.rs", &["lk.agent.session"], "data-channel topic"),
    (
        "lk-agents-voice/src/room_io/types.rs",
        &["lk.agent.pre-connect-audio-buffer"],
        "byte-stream topic",
    ),
    (
        "lk-agents-storage/src/chat_item.rs",
        &["lk.delegation.task_id", "lk.delegation.agent"],
        "test fixtures of chat-item extras",
    ),
    (
        "lk-agents-rtc/src/fake/tests.rs",
        &["*"],
        "fake-room tests of protocol topics and participant attributes",
    ),
];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable source tree") {
        let path = entry.expect("readable entry").path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target" || n == "generated") {
                continue;
            }
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The decoded text of a string, byte-string, char or byte literal token (escapes, raw strings
/// and `\u{..}` resolved), or the digits of an integer / float literal as `concat!` renders them
/// (suffix dropped), or `None` for a boolean or any other token.
fn string_value(lit: &proc_macro2::Literal) -> Option<String> {
    match syn::parse_str::<syn::Lit>(&lit.to_string()).ok()? {
        syn::Lit::Str(s) => Some(s.value()),
        syn::Lit::ByteStr(b) => Some(String::from_utf8_lossy(&b.value()).into_owned()),
        syn::Lit::Char(c) => Some(c.value().to_string()),
        syn::Lit::Byte(b) => Some((b.value() as char).to_string()),
        syn::Lit::Int(i) => Some(i.base10_digits().to_owned()),
        syn::Lit::Float(f) => Some(f.base10_digits().to_owned()),
        _ => None,
    }
}

/// `#[doc = "..."]`: prose, not an emitter site.
fn is_doc_attr(tokens: &[TokenTree], i: usize) -> bool {
    let TokenTree::Group(g) = &tokens[i] else { return false };
    g.delimiter() == Delimiter::Bracket
        && i > 0
        && punct_is(&tokens[i - 1], '#')
        && matches!(g.stream().into_iter().next(), Some(TokenTree::Ident(id)) if id == "doc")
}

fn punct_is(tt: &TokenTree, c: char) -> bool {
    matches!(tt, TokenTree::Punct(p) if p.as_char() == c)
}

/// The identifier's text without a raw prefix (`r#start` is `start`).
fn ident_text(tt: &TokenTree) -> Option<String> {
    let TokenTree::Ident(id) = tt else { return None };
    Some(id.to_string().trim_start_matches("r#").to_owned())
}

fn ident_is(tt: &TokenTree, name: &str) -> bool {
    ident_text(tt).is_some_and(|t| t == name)
}

fn tokens_of(g: &proc_macro2::Group) -> Vec<TokenTree> {
    g.stream().into_iter().collect()
}

/// A folded value: a string, or a tuple of folded values (for `let (a, (b, c)) = (…)`).
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Str(String),
    Tuple(Vec<Value>),
}

/// What one file decides statically: the qualified names it binds to a decidable value
/// (`const`, `static`, `let`; inside `mod m { … }` as `m::NAME`, inside `impl T { … }` as
/// `T::NAME`), resolved transitively to a fixed point with the one `fold` the call sites use,
/// and the names bound directly to `SpanBuilder` by one `type X = SpanBuilder;` or
/// `use … SpanBuilder as X;`.
#[derive(Default)]
struct Scope {
    values: HashMap<String, Value>,
    span_builder: HashSet<String>,
}

impl Scope {
    fn is_span_builder(&self, name: &str) -> bool {
        name == "SpanBuilder" || self.span_builder.contains(name)
    }

    /// A path as seen from `prefix` (`""` at the top, `"m::"` inside `mod m`, `"T::"` inside
    /// `impl T`): `Self::X` means the enclosing impl's `X`; the enclosing scope is tried first,
    /// then the file top level, so `names::SPAN` and `Names::SPAN` resolve from anywhere.
    fn get(&self, path: &str, prefix: &str) -> Option<&Value> {
        // `Self::X` is the enclosing impl's `X`; `self::X` is the enclosing module's `X`
        let resolved = match path.strip_prefix("Self::").or_else(|| path.strip_prefix("self::")) {
            Some(rest) => format!("{prefix}{rest}"),
            None => path.to_owned(),
        };
        self.values.get(&format!("{prefix}{resolved}")).or_else(|| self.values.get(&resolved))
    }
}

/// The names a `let` / `const` / `static` pattern binds, nested as the pattern is, and the token
/// index its value starts at (`=` plus one). `None` for a pattern this scan does not read (a
/// struct pattern, a binding without a value).
fn binding(tokens: &[TokenTree], i: usize) -> Option<(Pattern, usize)> {
    let mut j = i + 1;
    if tokens.get(j).is_some_and(|t| ident_is(t, "mut")) {
        j += 1;
    }
    let pattern = pattern_of(tokens.get(j)?)?;
    let stop =
        (j..tokens.len()).find(|&k| punct_is(&tokens[k], '=') || punct_is(&tokens[k], ';'))?;
    punct_is(&tokens[stop], '=').then_some((pattern, stop + 1))
}

/// A binding pattern: one name, or a parenthesised tuple of patterns (`(a, (b, c))`), with
/// `mut`, `ref`, `_` and type ascriptions ignored.
#[derive(Debug)]
enum Pattern {
    Name(String),
    Tuple(Vec<Pattern>),
}

fn pattern_of(tt: &TokenTree) -> Option<Pattern> {
    match tt {
        TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
            let inner = tokens_of(g);
            let mut out = Vec::new();
            for part in split_commas(&inner) {
                let part: Vec<&TokenTree> =
                    part.iter().filter(|t| !ident_is(t, "mut") && !ident_is(t, "ref")).collect();
                // `a: T` inside a tuple pattern never occurs; the ascription sits after the `)`
                if let Some(first) = part.first() {
                    out.push(pattern_of(first)?);
                }
            }
            Some(Pattern::Tuple(out))
        }
        tt => ident_text(tt).map(Pattern::Name),
    }
}

/// The comma-separated pieces of a token list, each as a slice of its tokens.
fn split_commas(tokens: &[TokenTree]) -> Vec<&[TokenTree]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (k, t) in tokens.iter().enumerate() {
        if punct_is(t, ',') {
            out.push(&tokens[start..k]);
            start = k + 1;
        }
    }
    if start < tokens.len() {
        out.push(&tokens[start..]);
    }
    out
}

/// Bind one pattern to one folded value, recursing through tuples; a mismatch binds nothing.
fn bind(pattern: &Pattern, value: &Value, prefix: &str, scope: &mut Scope) -> bool {
    match (pattern, value) {
        (Pattern::Name(n), v) => {
            let key = format!("{prefix}{n}");
            // first binding wins: a name rebound to a different literal later in the file would
            // otherwise flip on every pass and the fixed point would never settle
            if let std::collections::hash_map::Entry::Vacant(slot) = scope.values.entry(key) {
                slot.insert(v.clone());
                return true;
            }
            false
        }
        (Pattern::Tuple(ps), Value::Tuple(vs)) if ps.len() == vs.len() => {
            ps.iter().zip(vs).fold(false, |acc, (p, v)| bind(p, v, prefix, scope) | acc)
        }
        // `let (n,) = ("x",)` and `let (n) = "x"`: a one-element pattern takes a bare value
        (Pattern::Tuple(ps), v) if ps.len() == 1 => bind(&ps[0], v, prefix, scope),
        _ => false,
    }
}

/// One pass over `tokens` under `prefix`. Returns whether it learned a new binding, so the
/// caller can iterate to a fixed point (`const A: &str = B; const B: &str = C; …`).
fn collect_scope(tokens: &[TokenTree], prefix: &str, scope: &mut Scope) -> bool {
    let mut learned = false;
    for (i, tt) in tokens.iter().enumerate() {
        if let TokenTree::Group(g) = tt {
            let inner_prefix = inner_scope_prefix(tokens, i, prefix);
            learned |=
                collect_scope(&tokens_of(g), inner_prefix.as_deref().unwrap_or(prefix), scope);
        }
        if ident_is(tt, "const") || ident_is(tt, "static") || ident_is(tt, "let") {
            let Some((pattern, at)) = binding(tokens, i) else { continue };
            if let Some(value) = fold(tokens, at, prefix, scope) {
                learned |= bind(&pattern, &value, prefix, scope);
            }
        }
        let alias = if ident_is(tt, "type")
            && tokens.get(i + 2).is_some_and(|t| punct_is(t, '='))
            && tokens.get(i + 3).is_some_and(|t| ident_is(t, "SpanBuilder"))
            && tokens.get(i + 4).is_some_and(|t| punct_is(t, ';'))
        {
            tokens.get(i + 1).and_then(ident_text)
        } else if ident_is(tt, "SpanBuilder")
            && tokens.get(i + 1).is_some_and(|t| ident_is(t, "as"))
        {
            tokens.get(i + 2).and_then(ident_text)
        } else {
            None
        };
        if let Some(alias) = alias {
            learned |= scope.span_builder.insert(alias);
        }
    }
    learned
}

/// The path expression at `tokens[i]` (`a::b::C`, `Self::C`, or a bare `C`) and the index
/// just past it.
fn path_at(tokens: &[TokenTree], i: usize) -> Option<(String, usize)> {
    let mut path = ident_text(tokens.get(i)?)?;
    let mut j = i + 1;
    while tokens.get(j).is_some_and(|t| punct_is(t, ':'))
        && tokens.get(j + 1).is_some_and(|t| punct_is(t, ':'))
        && let Some(seg) = tokens.get(j + 2).and_then(ident_text)
    {
        path = format!("{path}::{seg}");
        j += 3;
    }
    Some((path, j))
}

/// THE evaluator, used for every binding and every call-site argument alike: the value the
/// expression at `tokens[i]` has when it is decidable from literals and same-file constants.
///
/// Folds: a string, byte-string, char or byte literal; a parenthesised tuple of foldables
/// (`("a", ("b", "c"))`, element-wise); `concat!` / `stringify!` of foldables, nesting included;
/// `format!` of one literal without a placeholder; a qualified path the scope binds
/// (`NAME`, `m::NAME`, `T::NAME`, `Self::NAME`); a parenthesised or block expression holding one.
/// Does not fold: an `if` (obfuscation, out of scope), any other call or macro.
fn fold(tokens: &[TokenTree], i: usize, prefix: &str, scope: &Scope) -> Option<Value> {
    let tt = tokens.get(i)?;
    if let TokenTree::Literal(l) = tt {
        return string_value(l).map(Value::Str);
    }
    if ident_text(tt).is_some() {
        let (path, after) = path_at(tokens, i)?;
        let is_macro = tokens.get(after).is_some_and(|t| punct_is(t, '!'));
        let Some(TokenTree::Group(g)) = tokens.get(after + 1).filter(|_| is_macro) else {
            return scope.get(&path, prefix).cloned();
        };
        let inner = tokens_of(g);
        return match path.as_str() {
            "concat" => {
                let mut out = String::new();
                for part in split_commas(&inner) {
                    match fold(part, 0, prefix, scope)? {
                        Value::Str(s) => out.push_str(&s),
                        Value::Tuple(_) => return None,
                    }
                }
                Some(Value::Str(out))
            }
            "stringify" => Some(Value::Str(g.stream().to_string().split_whitespace().collect())),
            "format" => {
                let Value::Str(s) = fold(&inner, 0, prefix, scope)? else { return None };
                let placeholders = s.replace("{{", "").replace("}}", "").contains('{');
                (inner.len() == 1 && !placeholders).then_some(Value::Str(s))
            }
            _ => None,
        };
    }
    if let TokenTree::Group(g) = tt
        && matches!(g.delimiter(), Delimiter::Parenthesis | Delimiter::Brace)
    {
        let inner = tokens_of(g);
        if inner.first().is_some_and(|t| ident_is(t, "if")) {
            return None; // ponytail: a conditional is obfuscation, out of scope
        }
        let parts = split_commas(&inner);
        let trailing_comma = inner.last().is_some_and(|t| punct_is(t, ','));
        if g.delimiter() == Delimiter::Parenthesis && (parts.len() > 1 || trailing_comma) {
            // a tuple: every element must fold
            let elems: Option<Vec<Value>> =
                parts.iter().map(|p| fold(p, 0, prefix, scope)).collect();
            return elems.map(Value::Tuple);
        }
        return (0..inner.len()).find_map(|k| fold(&inner, k, prefix, scope));
    }
    None
}

/// `fold`, as a string only.
fn fold_str(tokens: &[TokenTree], i: usize, prefix: &str, scope: &Scope) -> Option<String> {
    match fold(tokens, i, prefix, scope)? {
        Value::Str(s) => Some(s),
        Value::Tuple(_) => None,
    }
}

/// `SpanBuilder :: start ( <decidable string>`, whatever the layout, alias or raw identifier.
fn span_start_name(tokens: &[TokenTree], i: usize, prefix: &str, scope: &Scope) -> Option<String> {
    if !ident_is(&tokens[i], "start") || i < 3 {
        return None;
    }
    let path_ok = punct_is(&tokens[i - 1], ':')
        && punct_is(&tokens[i - 2], ':')
        && ident_text(&tokens[i - 3]).is_some_and(|n| scope.is_span_builder(&n));
    let Some(TokenTree::Group(args)) = tokens.get(i + 1) else { return None };
    if !path_ok || args.delimiter() != Delimiter::Parenthesis {
        return None;
    }
    fold_str(&tokens_of(args), 0, prefix, scope)
}

fn is_allowed(text: &str, allowed: &[&str]) -> bool {
    !text.starts_with("lk.") || allowed.contains(&"*") || allowed.contains(&text)
}

fn walk(
    tokens: &[TokenTree],
    rel: &str,
    allowed: &[&str],
    prefix: &str,
    scope: &Scope,
    out: &mut Vec<String>,
) {
    let mut i = 0;
    while i < tokens.len() {
        let tt = &tokens[i];
        let line = tt.span().start().line;
        // a composing macro is judged by what it composes, not by its pieces
        let composes = ident_is(tt, "concat") || ident_is(tt, "stringify");
        if composes
            && tokens.get(i + 1).is_some_and(|t| punct_is(t, '!'))
            && let Some(text) = fold_str(tokens, i, prefix, scope)
        {
            if !is_allowed(&text, allowed) {
                out.push(format!("{rel}:{line}: {text:?}"));
            }
            i += 3;
            continue;
        }
        match tt {
            TokenTree::Group(g) if !is_doc_attr(tokens, i) => {
                let inner_prefix = inner_scope_prefix(tokens, i, prefix);
                walk(
                    &tokens_of(g),
                    rel,
                    allowed,
                    inner_prefix.as_deref().unwrap_or(prefix),
                    scope,
                    out,
                );
            }
            TokenTree::Literal(l) => {
                // a number can never start with `lk.`; checked anyway through the same decoder
                if let Some(text) = string_value(l)
                    && !is_allowed(&text, allowed)
                {
                    out.push(format!("{rel}:{line}: {text:?}"));
                }
            }
            TokenTree::Ident(_) => {
                if let Some(name) = span_start_name(tokens, i, prefix, scope) {
                    out.push(format!("{rel}:{line}: SpanBuilder::start({name:?})"));
                }
            }
            _ => {}
        }
        i += 1;
    }
}

/// The qualified prefix a brace group at `tokens[i]` opens: `m::` for `mod m { … }`, `T::` for
/// `impl T { … }` / `impl Tr for T { … }`, else `None` (keep the caller's).
fn inner_scope_prefix(tokens: &[TokenTree], i: usize, prefix: &str) -> Option<String> {
    let TokenTree::Group(g) = &tokens[i] else { return None };
    if g.delimiter() != Delimiter::Brace || i < 2 {
        return None;
    }
    let k = (0..i).rev().find(|&k| ident_is(&tokens[k], "mod") || ident_is(&tokens[k], "impl"))?;
    if ident_is(&tokens[k], "mod") && k + 2 == i {
        return ident_text(&tokens[k + 1]).map(|m| format!("{prefix}{m}::"));
    }
    if ident_is(&tokens[k], "impl") {
        let seg = (k + 1..i).rev().find_map(|m| ident_text(&tokens[m]))?;
        return Some(format!("{prefix}{seg}::"));
    }
    None
}

/// The telemetry names `source` hand-writes, judged on its tokens rather than its text. The
/// scan exists to catch an accidental stray telemetry name, not deliberate obfuscation. Its
/// contract is this exact list; no sentence here claims more than the list.
///
/// FLAGGED: a string starting with `lk.`; `SpanBuilder::start(<string>, …)` in any layout, also
/// as `r#start`, also through one direct `type X = SpanBuilder;` or `use … SpanBuilder as X;`
/// alias. "A string" means an expression [`fold`] folds.
///
/// BINDING SITES collected (and nowhere else): `const`, `static` and `let` at the file root,
/// inside `mod <Ident> { … }` bodies (keyed `m::NAME`, nested as `outer::inner::NAME`), inside
/// `impl <Ident> { … }` and `impl <Trait> for <Ident> { … }` blocks (keyed `T::NAME`; the target
/// must be a single identifier, written before any `where`), and `let` in function bodies,
/// including nested tuple destructuring with `mut` / `ref` ignored (a nested `ref n` is read
/// as `n`). For a key bound more than once, the first binding whose value folds at the moment it
/// is visited wins and is never replaced; a binding that does not fold yet is skipped, so a
/// later foldable binding of the same name can take the slot before the earlier one resolves
/// (`let a = B; … let a = ("other",);` with `const B` defined later binds `a` to the tuple).
/// REFERENCE FORMS that resolve: a bare name (the enclosing `mod` / `impl` key first, then the
/// file root), `m::NAME`, `outer::inner::NAME`, `T::NAME`, `Self::NAME` inside that impl, and
/// `self::NAME` resolved against the innermost enclosing key, which is the module at module level
/// but the impl inside an `impl` block (so `self::N` inside `impl Names` looks for `Names::N`,
/// not the module's `N`). Resolution is transitive to a fixed point over those sites.
/// OPERATIONS that fold: string, byte-string, char, byte, integer and float literals;
/// `concat!(…)` of foldables, nesting included; `stringify!(…)`; `format!("literal")` with no
/// placeholder; a parenthesised or `{ … }` block expression holding one foldable; parenthesised
/// tuples element-wise, where every element must fold: the unit tuple `()` does not fold, so a
/// tuple containing `()` does not fold and a binding destructuring it binds nothing.
///
/// NOT CAUGHT, out of scope as obfuscation: alias chains (`type B = A;`), a qualified alias
/// (`type X = crate::…::SpanBuilder;`), `<SpanBuilder>::start`, conditionals (`if c { "a" } else
/// { "b" }`, also behind a `const`).
/// NOT CAUGHT, out of scope by enumeration (the list above does not include them): `super::X`;
/// a bare name that lives in an *enclosing* module when written inside a nested `mod` or `impl`
/// (only the immediate key and the file root are tried); `impl names::Names { … }` (a path
/// target); `impl Names where … { … }`; generic targets `impl<T> Names<T>`; `let ref n` at the
/// top of a pattern and a nested `&` pattern (a nested `ref` IS read, see above); array or slice
/// patterns `let [a, b] = …`; `concat!` of a boolean; `self::NAME` inside an impl meaning the
/// enclosing module's `NAME`; a tuple binding that contains `()`; a rebound `let` whose first
/// binding resolves only later.
/// NOT CAUGHT, runtime: a parameter, a function result, `format!` with a placeholder, a binding
/// in another file, a procedural macro; and never a comment or a `#[doc]` attribute.
fn offenders_in(rel: &str, source: &str, allowed: &[&str]) -> Vec<String> {
    let stream: TokenStream = source.parse().unwrap_or_else(|e| panic!("{rel}: {e}"));
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    let mut scope = Scope::default();
    // to a fixed point, so chains resolve in any order; each pass learns at least one new name
    // or stops, and a pass that learns nothing ends it
    while collect_scope(&tokens, "", &mut scope) {}
    let mut offenders = Vec::new();
    walk(&tokens, rel, allowed, "", &scope, &mut offenders);
    offenders
}

// Catches: a new `"lk.…"` literal or a span started by a string anywhere in the workspace, i.e.
// a telemetry name that bypasses the registry. Binds the emitter sites, not just the constants.
#[test]
fn no_telemetry_name_is_hand_written_at_an_emitter_site() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates dir").to_path_buf();
    let mut files = Vec::new();
    rust_sources(&crates, &mut files);
    assert!(files.len() > 100, "expected the whole workspace, found {} files", files.len());
    let mut offenders = Vec::new();
    for file in files {
        let rel =
            file.strip_prefix(&crates).expect("under crates").to_string_lossy().replace('\\', "/");
        if rel.ends_with("telemetry/semconv.rs") || rel.ends_with("telemetry/semconv_pins.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&file).expect("readable source");
        let allowed =
            LITERAL_ALLOWLIST.iter().find(|(f, _, _)| rel == *f).map(|(_, l, _)| *l).unwrap_or(&[]);
        offenders.extend(offenders_in(&rel, &text, allowed));
    }
    assert!(
        offenders.is_empty(),
        "telemetry names outside the registry:\n{}",
        offenders.join("\n")
    );
}

// CAUGHT: every bypass from Codex rounds 2 to 4 that the contract covers, as must-reject fixtures.
#[test]
fn the_emitter_site_scan_is_not_fooled_by_spelling() {
    for probe in [
        r#"const A: &str = "lk.scan_probe";"#,
        r#"const B: &str = "lk\u{2e}scan_probe";"#,
        r##"const C: &str = r#"lk.scan_probe"#;"##,
        r#"const D: &[u8] = b"lk.scan_probe";"#,
        r##"const D2: &[u8] = br#"lk.scan_probe"#;"##,
        "const E: &str = \"lk.\\\nscan_probe\";",
        r#"macro_rules! m { () => { "lk.scan_probe" }; }"#,
        r#"#[cfg(any())] const A: &str = "lk.scan_probe";"#,
        r#"const F: &str = concat!("l", "k.scan_probe");"#,
        r#"const F2: &str = concat!("lk.", "scan_probe");"#,
        r#"macro_rules! m { () => { concat!("l", "k.scan_probe") }; }"#,
        r#"const G: &str = stringify!(lk.scan_probe);"#,
        r#"const K: &str = concat!('l', "k.scan_probe");"#,
        r#"macro_rules! m { () => { concat!('l', "k.scan_probe") }; }"#,
        "fn f() { SpanBuilder::start(\"scan_probe\", frame, None); }",
        "fn f() { SpanBuilder :: start (\n    \"scan_probe\" , frame , None ) ; }",
        "fn f() { SpanBuilder::start(\"scan\\u{5f}probe\", frame, None); }",
        r##"fn f() { SpanBuilder::start(r#"scan_probe"#, frame, None); }"##,
        r#"#[cfg(any())] fn f() { SpanBuilder::start("scan_probe", frame, None); }"#,
        r#"const N: &str = "scan_probe"; fn f() { SpanBuilder::start(N, frame, None); }"#,
        r#"fn f() { SpanBuilder::start(N, frame, None); } const N: &str = "scan_probe";"#,
        r#"static N: &str = "scan_probe"; fn f() { SpanBuilder::start(N, frame, None); }"#,
        r#"fn f() { let n = "scan_probe"; SpanBuilder::start(n, frame, None); }"#,
        r#"fn f() { let (n,) = ("scan_probe",); SpanBuilder::start(n, frame, None); }"#,
        r#"fn f() { let (a, b) = ("other", "scan_probe"); SpanBuilder::start(b, frame, None); }"#,
        r#"const A: &str = B; const B: &str = C; const C: &str = D; const D: &str = "scan_probe"; fn f() { SpanBuilder::start(A, frame, None); }"#,
        r#"fn f() { SpanBuilder::start(concat!("scan", "_probe"), frame, None); }"#,
        r#"fn f() { SpanBuilder::start(stringify!(scan_probe), frame, None); }"#,
        r#"fn f() { SpanBuilder::start(("scan_probe"), frame, None); }"#,
        r#"fn f() { SpanBuilder::start({ "scan_probe" }, frame, None); }"#,
        r#"fn f() { SpanBuilder::start(format!("scan_probe"), frame, None); }"#,
        r#"fn f() { SpanBuilder::r#start("scan_probe", frame, None); }"#,
        r#"const r#name: &str = "scan_probe"; fn f() { SpanBuilder::r#start(r#name, frame, None); }"#,
        r#"type SB = SpanBuilder; fn f() { SB::start("scan_probe", frame, None); }"#,
        r#"use crate::telemetry::SpanBuilder as SB; fn f() { SB::start("scan_probe", frame, None); }"#,
        r#"#[cfg(feature = "x")] fn f() { SpanBuilder::start(concat!("scan", "_probe"), frame, None); }"#,
        // round 5: one evaluator for bindings and call sites, nested patterns, qualified paths
        r#"fn f() { let (a, (b, c)) = ("other", ("scan_probe", "other")); SpanBuilder::start(b, frame, None); }"#,
        r#"fn f() { let (a, b): (&str, &str) = ("other", "scan_probe"); SpanBuilder::start(b, frame, None); }"#,
        r#"fn f() { let (mut a, mut b) = ("other", "scan_probe"); SpanBuilder::start(b, frame, None); }"#,
        r#"const K: &str = concat!('l', concat!('k', '.', "scan_probe"));"#,
        r#"macro_rules! m { () => { let (a, b) = ("other", "scan_probe"); SpanBuilder::start(b, frame, None); }; }"#,
        r#"static mut A: &str = B; const B: &str = C; const C: &str = "scan_probe"; fn f() { SpanBuilder::start(A, frame, None); }"#,
        r#"mod names { pub const SPAN: &str = "scan_probe"; } fn f() { SpanBuilder::start(names::SPAN, frame, None); }"#,
        r#"struct Names; impl Names { const SPAN: &str = "scan_probe"; } fn f() { SpanBuilder::start(Names::SPAN, frame, None); }"#,
        // own compositions
        r#"struct Names; impl Names { const A: &str = "scan"; const SPAN: &str = concat!(Self::A, "_probe"); } fn f() { SpanBuilder::start(Names::SPAN, frame, None); }"#,
        r#"mod names { pub const PARTS: (&str, &str) = ("scan", "_probe"); } fn f() { let (a, b) = names::PARTS; SpanBuilder::start(concat!(a, b), frame, None); }"#,
        r#"mod outer { pub mod inner { pub const K: &str = "lk.scan_probe"; } } fn f() { let _ = outer::inner::K; }"#,
        // round 6: numeric literals in concat!, self::, and the rest of Codex's in-list forms
        r#"fn f() { SpanBuilder::start(concat!("span_", 1), frame, None); }"#,
        r#"fn f() { SpanBuilder::start(concat!("span_", 1.5), frame, None); }"#,
        r#"const K: &str = concat!("lk.", 1, '.', 2u8);"#,
        r#"mod names { const N: &str = "scan_probe"; fn f() { SpanBuilder::start(self::N, frame, None); } }"#,
        r#"mod outer { pub mod inner { pub const N: &str = "scan_probe"; } } fn f() { SpanBuilder::start(outer::inner::N, frame, None); }"#,
        r#"mod names { pub const A: &str = B; const B: &str = "scan_probe"; } fn f() { SpanBuilder::start(names::A, frame, None); }"#,
        r#"struct Names; impl Names { const A: &str = Self::B; const B: &str = "scan_probe"; fn f() { SpanBuilder::start(Self::A, frame, None); } }"#,
        r#"macro_rules! m { () => { mod names { pub const N: &str = "scan_probe"; } SpanBuilder::start(names::N, frame, None); }; }"#,
        r#"fn f() { let (ref a, (mut b, c)) = ("other", ("scan_probe", "other")); SpanBuilder::start(b, frame, None); }"#,
        r#"fn f() { let (a, (ref n,)) = ("other", ("scan_probe",)); SpanBuilder::start(n, frame, None); }"#,
        r#"fn f() { let (a, (b, c)) = { ("other", ("scan_probe", "other")) }; SpanBuilder::start(b, frame, None); }"#,
    ] {
        assert!(!offenders_in("probe.rs", probe, &[]).is_empty(), "not caught: {probe}");
    }
}

// NOT CAUGHT, out of scope: obfuscation. Listed so the boundary is a fact the suite checks, not
// a claim. A change that starts catching one of these is welcome; a change that starts catching
// nothing from the list above is not.
#[test]
fn the_emitter_site_scan_leaves_obfuscation_out_of_scope() {
    for probe in [
        r#"fn f() { SpanBuilder::start(if true { "scan_probe" } else { "other" }, frame, None); }"#,
        r#"const N: &str = if true { "scan_probe" } else { "other" }; fn f() { SpanBuilder::start(N, frame, None); }"#,
        r#"type A = SpanBuilder; type B = A; fn f() { B::start("scan_probe", frame, None); }"#,
        r#"type SB = crate::telemetry::SpanBuilder; fn f() { SB::start("scan_probe", frame, None); }"#,
        r#"fn f() { <SpanBuilder>::start("scan_probe", frame, None); }"#,
    ] {
        assert!(
            offenders_in("probe.rs", probe, &[]).is_empty(),
            "out of scope, yet caught: {probe}"
        );
    }
}

// NOT CAUGHT, out of scope by enumeration: forms the contract's exact list does not include.
// Each is a fact the suite checks; adding one to the evaluator means moving it to the list above.
#[test]
fn the_emitter_site_scan_leaves_unlisted_forms_out_of_scope() {
    for probe in [
        // out of scope: `super::` is not a reference form
        r#"mod names { const N: &str = "scan_probe"; mod inner { fn f() { SpanBuilder::start(super::N, frame, None); } } }"#,
        // out of scope: a bare name in an enclosing module, read from a nested impl
        r#"mod names { const N: &str = "scan_probe"; struct Names; impl Names { fn f() { SpanBuilder::start(N, frame, None); } } }"#,
        // out of scope: an impl target written as a path, or with a where clause, or generic
        r#"mod names { pub struct Names; } impl names::Names { const N: &str = "scan_probe"; } fn f() { SpanBuilder::start(names::Names::N, frame, None); }"#,
        r#"struct Names; impl Names where Names: Send { const N: &str = "scan_probe"; } fn f() { SpanBuilder::start(Names::N, frame, None); }"#,
        r#"struct Names<T>(T); impl<T> Names<T> { const N: &str = "scan_probe"; } fn f() { SpanBuilder::start(Names::<()>::N, frame, None); }"#,
        // out of scope: `let ref` at the top of a pattern (the `ref` sits where the name is
        // expected) and a nested `&` pattern
        r#"fn f() { let ref n = "scan_probe"; SpanBuilder::start(n, frame, None); }"#,
        r#"fn f() { let ref n = "scan_probe"; SpanBuilder::start(*n, frame, None); }"#,
        r#"fn f() { let (a, &b) = ("other", &"scan_probe"); SpanBuilder::start(b, frame, None); }"#,
        // out of scope: `self::` inside an impl resolves against the impl, not the module
        r#"mod names { const N: &str = "scan_probe"; struct Names; impl Names { fn f() { SpanBuilder::start(self::N, frame, None); } } }"#,
        // out of scope: the unit tuple does not fold, so the whole tuple binding binds nothing
        r#"fn f() { let (n, ()) = ("scan_probe", ()); SpanBuilder::start(n, frame, None); }"#,
        // out of scope: a rebound `let` whose first binding resolves only in a later pass
        r#"fn f() { let a = B; SpanBuilder::start(a, frame, None); let a = ("other",); } const B: &str = "scan_probe";"#,
        // out of scope: array / slice patterns, and `concat!` of a boolean
        r#"fn f() { let [a, b] = ["other", "scan_probe"]; SpanBuilder::start(b, frame, None); }"#,
        r#"const K: &str = concat!('l', 'k', '.', true);"#,
    ] {
        assert!(
            offenders_in("probe.rs", probe, &[]).is_empty(),
            "out of scope, yet caught: {probe}"
        );
    }
}

// NOT CAUGHT, runtime: the name does not exist until the code runs; a comment is not an emitter
// site.
#[test]
fn the_emitter_site_scan_documents_its_runtime_boundary() {
    for probe in [
        "fn f(name: String) { SpanBuilder::start(name, frame, None); }",
        "fn f() { SpanBuilder::start(compute(), frame, None); }",
        r#"fn f() { SpanBuilder::start(format!("s{i}"), frame, None); }"#,
        "use elsewhere::NAME; fn f() { SpanBuilder::start(NAME, frame, None); }",
        "fn f() { SpanBuilder::start(compute_name!(), frame, None); }",
        "fn f() { let n = name_for(x); SpanBuilder::start(n, frame, None); }",
        "// \"lk.scan_probe\"\n/// `\"lk.scan_probe\"`\nfn f() {}",
    ] {
        assert!(offenders_in("probe.rs", probe, &[]).is_empty(), "false positive: {probe}");
    }
}
