//! The STT-context options as the host writes them, the resolved detection config, and the
//! keyterm-set change records. The resolver lives in `lk-agents-voice/src/keyterm_detection.rs`.

use lk_agents_host_derive::Host;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "keyterms", path = "livekit.agents.voice.keyterm_detection"))]
const _: () = ();

/// `LLM | str | None`; an instance stays host-side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionLlm {
    Instance,
    /// Resolved through the inference gateway.
    Model(String),
}

/// An `LLM` instance crosses as "use the one given", distinct from `None`, "none configured".
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for DetectionLlm {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(match obj.extract::<String>() {
            Ok(model) => DetectionLlm::Model(model),
            Err(_) => DetectionLlm::Instance,
        })
    }
}

/// Configuration for automatic keyterm detection.
///
/// Lives under the ``keyterm_detection`` key of :class:`KeytermsOptions`. Absent or
/// ``{"enabled": False}`` keeps detection off.
//
// The `///` is the reference's docstring verbatim; it is what the docs render. `total=False`: one
// `Option` per field, since default and explicit `None` coincide for all six. Each field spells
// `ty`: `Option` here means absent, not nullable.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    name = "KeytermDetectionOptions"
)]
// The read target: `total=False` in the generator's vocabulary.
#[binding(extract, from_items)]
pub struct KeytermDetectionConfig {
    /// Whether to run the background detector. Defaults to ``False``.
    #[python(ty = "bool")]
    pub enabled: Option<bool>,
    /// LLM used for extraction. An ``LLM`` instance, or a model string (e.g.
    /// ``"google/gemini-3.5-flash"``) resolved via the inference gateway. Defaults to a
    /// built-in detection model; the agent's own LLM is not used.
    #[python(ty = "LLM | str | None", import = "..llm:LLM")]
    // An `LLM` instance cannot cross and reads as `DetectionLlm::Instance` ("use the one given",
    // not "none configured").
    pub llm: Option<DetectionLlm>,
    /// Run a pass once per N user turns. Defaults to ``1``.
    #[python(ty = "int")]
    pub turn_interval: Option<u32>,
    /// Cap on the confirmed (applied) detected keyterms if provided. Defaults to ``None``.
    #[python(ty = "int | None")]
    pub max_keyterms: Option<usize>,
    /// Override the built-in extraction prompt.
    #[python(ty = "str | None")]
    pub instructions: Option<String>,
    /// Seconds a single detection pass may run before it is dropped (no keyterm change).
    /// Defaults to ``10.0``. Raise it if a slow detection ``llm`` needs longer.
    #[python(ty = "float")]
    pub timeout: Option<f64>,
}

/// The resolved detection config as the host receives it: `model` is `None` for a caller-supplied
/// `LLM`, and `instructions` is always the resolved string.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit)]
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
// The reference's docstring, verbatim. `total=False`, same `ty`-per-field rule.
// The `cfg_attr(doctest)` lines fence the Python sample for `cargo test --doc` only; the codegen
// reads `///` alone, so the generated docstring is unchanged.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    name = "STTContextOptions"
)]
#[binding(extract, from_items)]
pub struct SttContextConfig {
    /// Static keyterms applied wherever the STT accepts a term list; never touched by detection.
    #[python(ty = "list[str]")]
    pub keyterms: Option<Vec<String>>,
    /// LLM-based keyterm extraction, for STTs that accept a term list.
    #[python(ty = "KeytermDetectionOptions")]
    pub keyterm_detection: Option<KeytermDetectionConfig>,
    /// Forward conversation turns to STTs that consume context directly (e.g. AssemblyAI u3-pro).
    /// Defaults to ``True``; only STTs that advertise the ``chat_context`` capability act on it.
    #[python(ty = "bool")]
    pub forward_chat_context: Option<bool>,
}

/// Keyterm biasing for STTs that accept a term list.
///
/// .. deprecated:: v2.0
///     Use :class:`STTContextOptions` (``AgentSession(stt_context_options=...)``) instead;
///     its ``keyterms`` and ``keyterm_detection`` keys are identical.
//
// The reference's docstring, verbatim. Its two keys are declared beside `STTContextOptions`'.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    name = "KeytermsOptions"
)]
pub struct KeytermsConfig {
    /// Static keyterms applied wherever the STT accepts a term list; never touched by detection.
    #[python(ty = "list[str]")]
    pub keyterms: Option<Vec<String>>,
    /// LLM-based keyterm extraction, for STTs that accept a term list.
    #[python(ty = "KeytermDetectionOptions")]
    pub keyterm_detection: Option<KeytermDetectionConfig>,
}

/// One LLM answer: three disjoint lists of terms.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct KeytermUpdate {
    /// Distinctive terms seen but not yet trusted -- tracked, not applied.
    pub pending: Vec<String>,
    /// Pending terms the transcript has now corroborated -- applied.
    pub confirm: Vec<String>,
    /// Only a spelling the user corrected away; applied terms are otherwise sticky.
    pub remove: Vec<String>,
}

/// The effective list changed, and how.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct KeytermsChanged {
    /// What the STT should now be biased toward.
    pub keyterms: Vec<String>,
    pub added: Vec<String>,
    pub removed: Vec<String>,
}
