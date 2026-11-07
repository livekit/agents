//! `_inject_schema_defaults` (`llm/utils.py:305`): a strict schema's `null` means "use your default",
//! turned back before pydantic validates. `null` is a sentinel only where it was not already legal.

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::llm::provider_format::StrictError;
use serde_json::{Map, Number, Value};

type Object = Map<String, Value>;

/// Over schema and value depth together. The reference gets a catchable `RecursionError`; Rust
/// would abort. Inside `serde_json`'s 128.
use lk_agents_abi::llm::schema::MAX_DEPTH;

pub use lk_agents_abi::llm::schema::SchemaError;

/// [`inject_schema_defaults`] for the host, where `root` defaults to `schema`.
pub fn inject_schema_defaults_of(
    value: &Value,
    schema: &Value,
    root: Option<&Value>,
) -> Result<Value, SchemaError> {
    inject_schema_defaults(value, schema, root.unwrap_or(schema))
}

/// `schema` governs `value`; `root` is what `$ref`s resolve against.
pub fn inject_schema_defaults(
    value: &Value,
    schema: &Value,
    root: &Value,
) -> Result<Value, SchemaError> {
    let schema = schema.as_object().ok_or(SchemaError::NotAnObject { what: "schema" })?;
    let root = root.as_object().ok_or(SchemaError::NotAnObject { what: "root" })?;
    Walk { root }.inject(value, schema, 0)
}

struct Walk<'a> {
    root: &'a Object,
}

impl Walk<'_> {
    /// Shares [`crate::llm::provider_format::resolve_ref`], as the reference shares `_strict.resolve_ref`.
    fn resolve(&self, reference: &str) -> Result<Object, SchemaError> {
        crate::llm::provider_format::resolve_ref(self.root, reference).map_err(|e| match e {
            StrictError::RefNotRooted { reference } => SchemaError::RefNotRooted { reference },
            StrictError::RefUnresolvable { reference, segment } => {
                SchemaError::RefUnresolvable { reference, segment }
            }
            other => SchemaError::Strict(other),
        })
    }

    /// `{**resolved, **schema}`: the siblings win.
    fn with_ref<'s>(&self, schema: &'s Object) -> Result<Cow<'s, Object>, SchemaError> {
        let Some(Value::String(reference)) = schema.get("$ref") else {
            return Ok(Cow::Borrowed(schema));
        };
        let mut merged = self.resolve(reference)?;
        for (key, value) in schema {
            merged.insert(key.clone(), value.clone());
        }
        Ok(Cow::Owned(merged))
    }

    fn inject(&self, value: &Value, schema: &Object, depth: usize) -> Result<Value, SchemaError> {
        if depth > MAX_DEPTH {
            return Err(SchemaError::TooDeep);
        }
        let schema = self.with_ref(schema)?;
        let schema = schema.as_ref();

        if value.is_null() {
            // A `null` against a defaulted, non-nullable schema is the sentinel.
            if let Some(default) = schema.get("default")
                && !self.allows_null(schema, &BTreeSet::new(), depth)?
            {
                return Ok(default.clone());
            }
            return Ok(Value::Null);
        }

        // `allOf` is conjunctive: every branch runs and sees the previous one's substitutions.
        let mut value = Cow::Borrowed(value);
        if let Some(Value::Array(all_of)) = schema.get("allOf") {
            for variant in all_of {
                if let Some(variant) = variant.as_object() {
                    value = Cow::Owned(self.inject(&value, variant, depth + 1)?);
                }
            }
        }

        // A union: descend into the first matching branch only (`anyOf` before `oneOf`), then stop.
        for union_key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(variants)) = schema.get(union_key) {
                for variant in variants {
                    if let Some(variant) = variant.as_object()
                        && self.matches(&value, variant, depth)?
                    {
                        return self.inject(&value, variant, depth + 1);
                    }
                }
            }
        }

        if let Value::Object(map) = value.as_ref() {
            let properties = schema.get("properties").and_then(Value::as_object);
            let additional = schema.get("additionalProperties").and_then(Value::as_object);
            if properties.is_some() || additional.is_some() {
                let mut out = Object::new();
                for (key, item) in map {
                    // A declared key whose schema is not an object falls through to
                    // `additionalProperties`.
                    let child = properties.and_then(|p| p.get(key)).and_then(Value::as_object);
                    let injected = match child.or(additional) {
                        Some(child) => self.inject(item, child, depth + 1)?,
                        None => item.clone(),
                    };
                    out.insert(key.clone(), injected);
                }
                return Ok(Value::Object(out));
            }
        }

        if let Value::Array(items) = value.as_ref() {
            let prefix_items = schema.get("prefixItems").and_then(Value::as_array);
            let rest = schema.get("items").and_then(Value::as_object);
            if prefix_items.is_some() || rest.is_some() {
                let mut out = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    // `prefixItems` covers positional slots (fixed tuples); `items` the rest.
                    let slot =
                        prefix_items.and_then(|p| p.get(index)).and_then(Value::as_object).or(rest);
                    out.push(match slot {
                        Some(slot) => self.inject(item, slot, depth + 1)?,
                        None => item.clone(),
                    });
                }
                return Ok(Value::Array(out));
            }
        }

        Ok(value.into_owned())
    }

    /// `utils.py:391`. Conjunctive; a `$ref` cycle answers `false`, an unresolvable `$ref` raises.
    fn allows_null(
        &self,
        schema: &Object,
        seen_refs: &BTreeSet<String>,
        depth: usize,
    ) -> Result<bool, SchemaError> {
        if depth > MAX_DEPTH {
            return Err(SchemaError::TooDeep);
        }

        if let Some(Value::String(reference)) = schema.get("$ref") {
            if seen_refs.contains(reference) {
                return Ok(false);
            }
            let resolved = self.resolve(reference)?;
            let mut seen = seen_refs.clone();
            seen.insert(reference.clone());
            if !self.allows_null(&resolved, &seen, depth + 1)? {
                return Ok(false);
            }
        }

        match schema.get("type") {
            Some(Value::String(typ)) if typ != "null" => return Ok(false),
            Some(Value::Array(types)) if !types.iter().any(|t| t == "null") => return Ok(false),
            _ => {}
        }

        if let Some(constant) = schema.get("const")
            && !constant.is_null()
        {
            return Ok(false);
        }
        if let Some(Value::Array(variants)) = schema.get("enum")
            && !variants.iter().any(Value::is_null)
        {
            return Ok(false);
        }

        if let Some(Value::Array(all_of)) = schema.get("allOf") {
            for variant in all_of {
                if let Some(variant) = variant.as_object()
                    && !self.allows_null(variant, seen_refs, depth + 1)?
                {
                    return Ok(false);
                }
            }
        }

        for union_key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(variants)) = schema.get(union_key) {
                let mut any = false;
                for variant in variants {
                    if let Some(variant) = variant.as_object()
                        && self.allows_null(variant, seen_refs, depth + 1)?
                    {
                        any = true;
                        break;
                    }
                }
                if !any {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// `utils.py:427`: decisive, not a validator; checks the discriminating keywords only.
    fn matches(&self, value: &Value, schema: &Object, depth: usize) -> Result<bool, SchemaError> {
        if depth > MAX_DEPTH {
            return Err(SchemaError::TooDeep);
        }
        let schema = self.with_ref(schema)?;
        let schema = schema.as_ref();

        // A null sentinel matches the branch it is a sentinel for, before any type check.
        if value.is_null()
            && schema.contains_key("default")
            && !self.allows_null(schema, &BTreeSet::new(), depth)?
        {
            return Ok(true);
        }

        if let Some(constant) = schema.get("const")
            && !python_eq(value, constant)
        {
            return Ok(false);
        }
        if let Some(Value::Array(variants)) = schema.get("enum")
            && !variants.iter().any(|variant| python_eq(value, variant))
        {
            return Ok(false);
        }

        match schema.get("type") {
            // A list of types returns here, as in the reference: the object checks are skipped.
            Some(Value::Array(types)) => {
                return Ok(types
                    .iter()
                    .any(|t| t.as_str().is_some_and(|t| json_type_matches(value, t))));
            }
            Some(Value::String(typ)) if !json_type_matches(value, typ) => return Ok(false),
            _ => {}
        }

        if let Value::Object(map) = value {
            if let Some(Value::Array(required)) = schema.get("required")
                && !required.iter().all(|key| key.as_str().is_some_and(|key| map.contains_key(key)))
            {
                return Ok(false);
            }

            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                // Strict schemas close objects, so an undeclared key rules the branch out; an
                // absent `additionalProperties` counts as closed.
                let closed = matches!(
                    schema.get("additionalProperties"),
                    None | Some(Value::Null) | Some(Value::Bool(false))
                );
                if closed && map.keys().any(|key| !properties.contains_key(key)) {
                    return Ok(false);
                }
                for (key, item) in map {
                    if let Some(property) = properties.get(key).and_then(Value::as_object)
                        && !self.matches(item, property, depth + 1)?
                    {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }
}

/// `utils.py:479`. An unknown type name matches everything, as there.
fn json_type_matches(value: &Value, typ: &str) -> bool {
    match typ {
        "null" => value.is_null(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        // JSON already separates bools from numbers and `1.0` from integers.
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "string" => value.is_string(),
        _ => true,
    }
}

/// Python's `==` on two JSON values: `1 == 1.0`, unlike `serde_json`. `True == 1` is deliberately
/// not reproduced, to agree with [`json_type_matches`].
fn python_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => number_eq(left, right),
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(l, r)| python_eq(l, r))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, l)| right.get(key).is_some_and(|r| python_eq(l, r)))
        }
        _ => left == right,
    }
}

fn number_eq(left: &Number, right: &Number) -> bool {
    if let (Some(left), Some(right)) = (left.as_i64(), right.as_i64()) {
        return left == right;
    }
    if let (Some(left), Some(right)) = (left.as_u64(), right.as_u64()) {
        return left == right;
    }
    match (left.as_f64(), right.as_f64()) {
        (Some(left), Some(right)) => left == right,
        // Reachable only with serde_json's `arbitrary_precision`; kept exact for that case.
        _ => left == right,
    }
}

#[cfg(test)]
#[path = "defaults_tests.rs"]
mod tests;
