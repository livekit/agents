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

pub use lk_agents_abi::voice::workflows::{InstructionsArg, WorkflowInstructionsArg};

/// The host's sections, substituted modally.
impl From<WorkflowInstructionsArg> for WorkflowInstructions {
    fn from(w: WorkflowInstructionsArg) -> Self {
        WorkflowInstructions {
            persona: w.persona.given().map(TemplateValue::Modal),
            extra: Some(TemplateValue::Modal(w.extra)),
        }
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
