//! `voice/run_result.py`'s `RunOutputOptions`. Declaration-only: the retry is host code.

use lk_agents_host_derive::Host;

/// Structured-output behavior for :meth:`AgentSession.run`.
///
/// Can be passed as a plain dict::
///
#[cfg_attr(doctest, doc = "```text")]
///     sess.run(
///         user_input=...,
///         output_type=MyOutput,
///         output_options={"max_retries": 2, "retry_instructions": "Call submit_result."},
///     )
#[cfg_attr(doctest, doc = "```")]
///
/// Pass ``output_options=None`` to disable the retry behavior.
//
// The `///` is the reference's docstring verbatim. `total=False`; each field spells `ty` since
// `Option` here means absent, not nullable.
// The `cfg_attr(doctest)` lines fence the Python sample for `cargo test --doc` only; the codegen
// reads `///` alone, so the generated docstring is unchanged.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "run_result",
    path = "livekit.agents.voice.run_result",
    name = "RunOutputOptions"
)]
pub struct RunOutputOptions {
    /// Re-prompts when a run ends without its ``output_type``, before raising
    /// UnexpectedModelBehavior. Defaults to ``2``.
    #[python(ty = "int")]
    pub max_retries: Option<i64>,
    /// Override the built-in retry prompt.
    #[python(ty = "str")]
    pub retry_instructions: Option<String>,
}
