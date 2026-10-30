//! The STT-context options as the host writes them, the resolved detection config, and the
//! keyterm-set change records. The resolver lives in `lk-agents-voice/src/keyterm_detection.rs`.

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "keyterms", path = "livekit.agents.voice.agent_session"))]
const _: () = ();

/// `LLM | str`; an instance stays host-side and crosses as "use the one given", distinct from
/// `None`, "none configured".
#[crate::host(handwritten(extract, host = "..llm.LLM | str", py))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionLlm {
    Instance,
    /// Resolved through the inference gateway.
    Model(String),
}

/// Configuration for automatic keyterm detection.
///
/// Lives under the ``keyterm_detection`` key of :class:`STTContextOptions`. Absent or
/// ``{"enabled": False}`` keeps detection off.
//
// The `///` is the reference's docstring verbatim; it is what the docs render. One `Option` per
// key, since default and explicit `None` coincide for all six.
#[crate::host(record(module = "keyterms", name = "KeytermDetectionOptions", typed_dict, extract))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeytermDetectionConfig {
    /// Whether to run the background detector. Defaults to ``False``.
    #[host(default = None)]
    pub enabled: Option<bool>,
    /// LLM used for extraction. An ``LLM`` instance, or a model string (e.g.
    /// ``"google/gemini-3.5-flash"``) resolved via the inference gateway. Defaults to a
    /// built-in detection model; the agent's own LLM is not used.
    #[host(default = None, annotation = "..llm.LLM | str | None")]
    pub llm: Option<DetectionLlm>,
    /// Run a pass once per N user turns. Defaults to ``1``.
    #[host(default = None)]
    pub turn_interval: Option<u32>,
    /// Cap on the confirmed (applied) detected keyterms if provided. Defaults to ``None``.
    #[host(default = None, annotation = "int | None")]
    pub max_keyterms: Option<usize>,
    /// Override the built-in extraction prompt.
    #[host(default = None, annotation = "str | None")]
    pub instructions: Option<String>,
    /// Seconds a single detection pass may run before it is dropped (no keyterm change).
    /// Defaults to ``10.0``. Raise it if a slow detection ``llm`` needs longer.
    #[host(default = None)]
    pub timeout: Option<f64>,
}

/// The resolved detection config as the host receives it: `model` is `None` for a caller-supplied
/// `LLM`, and `instructions` is always the resolved string.
#[crate::host(record(module = "keyterms", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedKeytermDetection {
    pub enabled: bool,
    pub turn_interval: u32,
    pub max_keyterms: Option<usize>,
    pub timeout: f64,
    pub instructions: String,
    /// The model to build, or `None` for "the caller supplied an LLM; use it".
    pub model: Option<String>,
    pub tool_name: String,
}

/// Keyterm detection's resolution.
#[crate::host(functions(pid = false))]
pub trait KeytermFunctions {
    /// `_resolve_detection`: the defaults, then whatever keys the caller set. `turn_interval`
    /// comes back clamped to at least 1, and an empty `instructions` resolves to the default prompt.
    #[host(function)]
    fn resolve_keyterm_detection(
        config: Option<KeytermDetectionConfig>,
    ) -> ResolvedKeytermDetection;
}

/// Conversation-aware context for the STT.
///
/// Can be passed as a plain dict::
///
#[cfg_attr(doctest, doc = "```text")]
///     AgentSession(
///         stt_context_options={
///             "keyterms": ["LiveKit", "Acme Corp"],
///             "keyterm_detection": {"enabled": True, "turn_interval": 1},
///             "forward_chat_context": True,
///         },
///     )
#[cfg_attr(doctest, doc = "```")]
//
// The reference's docstring, verbatim. The `cfg_attr(doctest)` lines fence the Python sample for
// `cargo test --doc` only; the codegen reads `///` alone, so the generated docstring is unchanged.
#[crate::host(record(module = "keyterms", name = "STTContextOptions", typed_dict, extract))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SttContextConfig {
    /// Static keyterms applied wherever the STT accepts a term list; never touched by detection.
    #[host(default = None)]
    pub keyterms: Option<Vec<String>>,
    /// LLM-based keyterm extraction, for STTs that accept a term list.
    #[host(default = None)]
    pub keyterm_detection: Option<KeytermDetectionConfig>,
    /// Forward conversation turns to STTs that consume context directly (e.g. AssemblyAI u3-pro).
    /// Defaults to ``True``; only STTs that advertise the ``chat_context`` capability act on it.
    #[host(default = None)]
    pub forward_chat_context: Option<bool>,
}

/// One LLM answer: three disjoint lists of terms.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeytermUpdate {
    /// Distinctive terms seen but not yet trusted -- tracked, not applied.
    pub pending: Vec<String>,
    /// Pending terms the transcript has now corroborated -- applied.
    pub confirm: Vec<String>,
    /// Only a spelling the user corrected away; applied terms are otherwise sticky.
    pub remove: Vec<String>,
}

/// The effective list changed, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeytermsChanged {
    /// What the STT should now be biased toward.
    pub keyterms: Vec<String>,
    pub added: Vec<String>,
    pub removed: Vec<String>,
}
