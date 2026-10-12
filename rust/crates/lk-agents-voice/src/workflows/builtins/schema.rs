use lk_agents_abi::{ToolFlags, ToolSpec};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Copy)]
pub struct Param {
    pub name: &'static str,
    pub ty: ParamType,
    /// The `Args:` line from the reference docstring; may be empty.
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub enum ParamType {
    Str,
    Int,
    /// `str | None = None`.
    OptStr,
    /// `list[E]` for a string enum, inlined rather than a `$defs` ref.
    StrEnumList(&'static [&'static str]),
}

impl Param {
    pub const fn string(name: &'static str, description: &'static str) -> Param {
        Param { name, ty: ParamType::Str, description }
    }
    pub const fn integer(name: &'static str, description: &'static str) -> Param {
        Param { name, ty: ParamType::Int, description }
    }
    pub const fn opt_string(name: &'static str, description: &'static str) -> Param {
        Param { name, ty: ParamType::OptStr, description }
    }
    pub const fn str_enum_list(
        name: &'static str,
        values: &'static [&'static str],
        description: &'static str,
    ) -> Param {
        Param { name, ty: ParamType::StrEnumList(values), description }
    }

    fn schema(&self) -> Value {
        let mut node = match self.ty {
            ParamType::Str => json!({ "type": "string" }),
            ParamType::Int => json!({ "type": "integer" }),
            ParamType::OptStr => json!({ "type": ["string", "null"] }),
            ParamType::StrEnumList(values) => json!({
                "type": "array",
                "items": { "type": "string", "enum": values },
            }),
        };
        if !self.description.is_empty() {
            node.as_object_mut()
                .expect("built from a json! object literal")
                .insert("description".into(), Value::String(self.description.into()));
        }
        node
    }
}

/// The arguments schema: an object, every property required, no extras.
pub fn params_schema(params: &[Param]) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::with_capacity(params.len());
    for p in params {
        properties.insert(p.name.to_owned(), p.schema());
        required.push(Value::String(p.name.to_owned()));
    }
    json!({
        "type": "object",
        "properties": Value::Object(properties),
        "required": Value::Array(required),
        "additionalProperties": false,
    })
}

pub fn tool(name: &str, description: &str, params: &[Param], flags: ToolFlags) -> ToolSpec {
    ToolSpec {
        name: name.to_owned(),
        description: description.to_owned(),
        json_schema: params_schema(params),
        flags,
    }
}

/// A no-argument tool: every `confirm_*` in the reference except the three in `credit_card.py`.
pub fn nullary(name: &str, description: &str) -> ToolSpec {
    tool(name, description, &[], ToolFlags::default())
}
