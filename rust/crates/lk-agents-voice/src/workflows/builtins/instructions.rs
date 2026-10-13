//! A modal substitution uses the matching audio/text variant, falling back to common text when
//! that variant is absent. An explicitly empty variant removes that template section.

use lk_agents_voice::generation::{Instructions, Rendered, safe_render};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateValue {
    Plain(String),
    /// `audio`/`text` substituted into the matching variant, each falling back to `common`.
    Modal(Instructions),
}

impl TemplateValue {
    pub fn plain(s: impl Into<String>) -> Self {
        TemplateValue::Plain(s.into())
    }

    /// The `Instructions(audio=A, text=T)` shape the two callers use.
    pub fn modal(audio: impl Into<String>, text: impl Into<String>) -> Self {
        TemplateValue::Modal(Instructions {
            common: String::new(),
            audio: Some(audio.into()),
            text: Some(text.into()),
        })
    }

    fn for_audio(&self) -> String {
        match self {
            TemplateValue::Plain(s) => s.clone(),
            TemplateValue::Modal(i) => i.audio.clone().unwrap_or_else(|| i.common.clone()),
        }
    }

    fn for_text(&self) -> String {
        match self {
            TemplateValue::Plain(s) => s.clone(),
            TemplateValue::Modal(i) => i.text.clone().unwrap_or_else(|| i.common.clone()),
        }
    }

    fn is_modal(&self) -> bool {
        matches!(self, TemplateValue::Modal(_))
    }
}

/// What [`resolve_template`] produced. `missing` lists unfilled placeholders (the tests assert the
/// built-in templates leave none).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub instructions: Instructions,
    pub missing: Vec<String>,
}

/// `Instructions.resolve_template` (`chat_context.Instructions.render`).
pub fn resolve_template(template: &str, kwargs: &[(&str, TemplateValue)]) -> Resolved {
    if !kwargs.iter().any(|(_, v)| v.is_modal()) {
        let Rendered { text, missing } = safe_render(template, &data_of(kwargs, |v| v.for_audio()));
        return Resolved { instructions: Instructions::plain(text), missing };
    }

    let audio = safe_render(template, &data_of(kwargs, |v| v.for_audio()));
    let text = safe_render(template, &data_of(kwargs, |v| v.for_text()));

    let mut missing = audio.missing;
    for m in text.missing {
        if !missing.contains(&m) {
            missing.push(m);
        }
    }

    // A single common variant, or `render(modality=None)` would return the empty common part.
    if audio.text == text.text {
        return Resolved { instructions: Instructions::plain(audio.text), missing };
    }

    // Fully rendered variants replace common, which `render()` would otherwise prepend.
    Resolved {
        instructions: Instructions {
            common: String::new(),
            audio: Some(audio.text),
            text: Some(text.text),
        },
        missing,
    }
}

fn data_of(kwargs: &[(&str, TemplateValue)], pick: impl Fn(&TemplateValue) -> String) -> Value {
    let map: Map<String, Value> =
        kwargs.iter().map(|(k, v)| ((*k).to_owned(), Value::String(pick(v)))).collect();
    Value::Object(map)
}

/// `WorkflowInstructions`: only `persona` and `extra`, the two fields `resolve` reads (the
/// inherited `audio`/`text`/`common` are write-only there). Literal instructions use
/// [`InstructionsArg::Fixed`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkflowInstructions {
    /// `None` keeps the built-in persona; `Some("")` removes the section.
    pub persona: Option<TemplateValue>,
    pub extra: Option<TemplateValue>,
}

impl WorkflowInstructions {
    /// `utils.WorkflowInstructions.resolve`.
    pub fn resolve(
        &self,
        template: &str,
        default_persona: &str,
        sections: &[(&str, TemplateValue)],
    ) -> Resolved {
        let mut kwargs: Vec<(&str, TemplateValue)> = Vec::with_capacity(sections.len() + 2);
        kwargs.push((
            "persona",
            self.persona.clone().unwrap_or_else(|| TemplateValue::plain(default_persona)),
        ));
        kwargs.push(("extra", self.extra.clone().unwrap_or_else(|| TemplateValue::plain(""))));
        kwargs.extend(sections.iter().cloned());
        resolve_template(template, &kwargs)
    }
}

/// The `instructions: WorkflowInstructions | Instructions | str` parameter; `NOT_GIVEN` is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructionsArg {
    Workflow(WorkflowInstructions),
    /// The `Instructions` and `str` arms, passed through without a template.
    Fixed(Instructions),
}

/// Both a fixed `Instructions` and a `WorkflowInstructions` given; `Display` is the `ValueError` text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "pass either `instructions` (a fixed Instructions) or `sections` (a WorkflowInstructions' \
     persona/extra), never both"
)]
pub struct BothInstructionForms;

impl InstructionsArg {
    /// The parameter as the host classified it (`fixed` or `sections`, both `None` for
    /// `NOT_GIVEN`). A persona is kept `Modal` so an `Instructions(audio=.., text=..)` survives.
    pub fn from_host(
        fixed: Option<Instructions>,
        sections: Option<(Option<Instructions>, Option<Instructions>)>,
    ) -> Result<Option<InstructionsArg>, BothInstructionForms> {
        match (fixed, sections) {
            (Some(_), Some(_)) => Err(BothInstructionForms),
            (Some(i), None) => Ok(Some(InstructionsArg::Fixed(i))),
            (None, Some((persona, extra))) => {
                Ok(Some(InstructionsArg::Workflow(WorkflowInstructions {
                    persona: persona.map(TemplateValue::Modal),
                    extra: extra.map(TemplateValue::Modal),
                })))
            }
            (None, None) => Ok(None),
        }
    }
}

impl From<&str> for InstructionsArg {
    fn from(s: &str) -> Self {
        InstructionsArg::Fixed(Instructions::plain(s))
    }
}

impl From<Instructions> for InstructionsArg {
    fn from(i: Instructions) -> Self {
        InstructionsArg::Fixed(i)
    }
}

impl From<WorkflowInstructions> for InstructionsArg {
    fn from(w: WorkflowInstructions) -> Self {
        InstructionsArg::Workflow(w)
    }
}

/// `str.format` over a built-in template, dropping `missing` ([`render_checked`] proves there are none).
pub fn render(template: &str, data: &[(&str, &str)]) -> String {
    render_checked(template, data).text
}

pub fn render_checked(template: &str, data: &[(&str, &str)]) -> Rendered {
    let map: BTreeMap<&str, &str> = data.iter().copied().collect();
    let obj: Map<String, Value> =
        map.into_iter().map(|(k, v)| (k.to_owned(), Value::String(v.to_owned()))).collect();
    safe_render(template, &Value::Object(obj))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `instructions=` and `sections=` both being accepted.
    #[test]
    fn a_fixed_instruction_and_a_section_override_cannot_both_be_given() {
        let err = InstructionsArg::from_host(Some(Instructions::plain("x")), Some((None, None)))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "pass either `instructions` (a fixed Instructions) or `sections` (a WorkflowInstructions' persona/extra), never both"
        );

        let fixed =
            InstructionsArg::from_host(Some(Instructions::plain("verbatim")), None).unwrap();
        assert!(matches!(fixed, Some(InstructionsArg::Fixed(_))));

        let sections =
            InstructionsArg::from_host(None, Some((Some(Instructions::plain("me")), None)))
                .unwrap();
        assert!(matches!(sections, Some(InstructionsArg::Workflow(_))));

        assert!(
            InstructionsArg::from_host(None, None).unwrap().is_none(),
            "NOT_GIVEN stays absent"
        );
    }

    // Catches: a persona crossing as `TemplateValue::Plain` (same text in both variants).
    #[test]
    fn a_modal_persona_reaches_both_variants_separately() {
        let persona = Instructions {
            common: String::new(),
            audio: Some("spoken".into()),
            text: Some("typed".into()),
        };
        let arg = InstructionsArg::from_host(None, Some((Some(persona.clone()), None))).unwrap();
        let Some(InstructionsArg::Workflow(w)) = arg else { panic!("expected the workflow arm") };
        assert_eq!(w.persona, Some(TemplateValue::Modal(persona)));
        assert_eq!(w.extra, None);
    }
}
