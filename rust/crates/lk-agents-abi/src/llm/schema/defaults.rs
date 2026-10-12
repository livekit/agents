//! `_inject_schema_defaults` (`llm/utils.py:305`): a strict schema's `null` means "use your default",
//! turned back before pydantic validates. `null` is a sentinel only where it was not already legal.

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::llm::provider_format::StrictError;
use serde_json::{Map, Number, Value};

type Object = Map<String, Value>;

/// Over schema and value depth together. The reference gets a catchable `RecursionError`; Rust
/// would abort. Inside `serde_json`'s 128.
const MAX_DEPTH: usize = 100;

/// Everything the defaults walk can refuse. A wrapped strict fault keeps its own kind.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchemaError {
    /// From [`to_strict_json_schema`](crate::llm::provider_format::to_strict_json_schema).
    #[error(transparent)]
    #[host(delegate)]
    Strict(#[from] StrictError),
    #[error("expected a JSON object for {what}")]
    #[host(kind = "TypeError")]
    NotAnObject { what: &'static str },
    /// The reference's `resolve_ref` raises `ValueError` here.
    #[error("unexpected $ref format {reference:?}; does not start with #/")]
    RefNotRooted { reference: String },
    /// The reference's `KeyError` (missing segment) and `AssertionError` (not a dict).
    #[error("$ref {reference:?} does not resolve: no object at segment {segment:?}")]
    #[host(kind = "KeyError")]
    RefUnresolvable { reference: String, segment: String },
    #[error("value or schema nests deeper than {MAX_DEPTH} levels")]
    TooDeep,
}

/// [`inject_schema_defaults`] for the host, where `root` defaults to `schema`.
#[crate::host(export(pid = false, name = "inject_schema_defaults"))]
pub fn inject_schema_defaults_of(value: &Value, schema: &Value, root: Option<&Value>) -> Result<Value, SchemaError> {
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
mod tests {
    use serde_json::json;

    use super::*;

    /// Expected values were produced by running each case through the reference.
    fn inject(value: Value, schema: Value) -> Result<Value, SchemaError> {
        inject_schema_defaults(&value, &schema, &schema)
    }

    // One row per rule of the injector; the label names what a regression would break.
    #[test]
    fn defaults_are_injected_the_way_the_reference_does() {
        let cat_dog = json!({
            "$defs": {
                "Cat": {"type": "object", "required": ["meow"], "additionalProperties": false,
                        "properties": {"meow": {"type": "string"}, "lives": {"type": "integer", "default": 9}}},
                "Dog": {"type": "object", "required": ["woof"], "additionalProperties": false,
                        "properties": {"woof": {"type": "string"}, "good": {"type": "boolean", "default": true}}},
            },
            "type": "object",
            "properties": {"pet": {"anyOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}]}},
        });
        let int_or_num = json!({"anyOf": [
            {"type": "object", "properties": {"n": {"type": "integer"}, "tag": {"type": "string", "default": "INT"}}},
            {"type": "object", "properties": {"n": {"type": "number"}, "tag": {"type": "string", "default": "NUM"}}},
        ]});
        let closed_b = json!({"type": "object", "additionalProperties": false,
            "properties": {"a": {"type": "string"}, "b": {"type": "string", "default": "B"}}});
        let cases = [
            ("null sentinel resolves", json!(null), json!({"type": "integer", "default": 7}), json!(7)),
            ("nullable type keeps null", json!(null), json!({"type": ["integer", "null"], "default": 7}), json!(null)),
            ("no default, no change", json!(null), json!({"type": "integer"}), json!(null)),
            ("type null is nullable", json!(null), json!({"type": "null", "default": 7}), json!(null)),
            ("type list without null", json!(null), json!({"type": ["integer", "string"], "default": 5}), json!(5)),
            ("enum with null is nullable", json!(null), json!({"enum": [null, "a"], "default": "a"}), json!(null)),
            ("const null is nullable", json!(null), json!({"const": null, "default": "a"}), json!(null)),
            ("enum without null", json!(null), json!({"enum": ["a", "b"], "default": "a"}), json!("a")),
            ("allOf nullability is conjunctive", json!(null), json!({"allOf": [{"type": ["integer", "null"]}], "default": 5}), json!(null)),
            ("allOf non-nullable", json!(null), json!({"allOf": [{"type": "integer"}], "default": 5}), json!(5)),
            ("union nullability is disjunctive", json!(null), json!({"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 3}), json!(null)),
            ("union non-nullable", json!(null), json!({"anyOf": [{"type": "integer"}, {"type": "string"}], "default": 3}), json!(3)),
            ("nested objects", json!({"a": null, "b": {"c": null}}),
             json!({"type": "object", "properties": {"a": {"type": "string", "default": "x"},
                    "b": {"type": "object", "properties": {"c": {"type": "boolean", "default": true}}}}}),
             json!({"a": "x", "b": {"c": true}})),
            ("array items", json!([null, null]), json!({"type": "array", "items": {"type": "integer", "default": 9}}), json!([9, 9])),
            ("additionalProperties only for undeclared keys", json!({"declared": null, "other": null}),
             json!({"type": "object", "properties": {"declared": {"type": "string", "default": "d"}},
                    "additionalProperties": {"type": "string", "default": "a"}}),
             json!({"declared": "d", "other": "a"})),
            ("a non-dict property falls back to additionalProperties", json!({"declared": null}),
             json!({"type": "object", "properties": {"declared": true},
                    "additionalProperties": {"type": "string", "default": "ADD"}}),
             json!({"declared": "ADD"})),
            ("no member keywords: object untouched", json!({"x": null}), json!({"type": "object"}), json!({"x": null})),
            ("no member keywords: array untouched", json!([null]), json!({"type": "array"}), json!([null])),
            ("prefixItems slots then items tail", json!([null, null, null]),
             json!({"type": "array", "prefixItems": [{"type": "string", "default": "p0"}, {"type": "integer", "default": 1}],
                    "items": {"type": "boolean", "default": false}}),
             json!(["p0", 1, false])),
            ("allOf applies every branch", json!({"a": null, "b": null}),
             json!({"allOf": [{"type": "object", "properties": {"a": {"type": "string", "default": "from_allof"}}},
                              {"type": "object", "properties": {"b": {"type": "integer", "default": 2}}}]}),
             json!({"a": "from_allof", "b": 2})),
            ("union picks the Dog branch", json!({"pet": {"woof": "hi", "good": null}}), cat_dog.clone(),
             json!({"pet": {"woof": "hi", "good": true}})),
            ("union picks the Cat branch", json!({"pet": {"meow": "hi", "lives": null}}), cat_dog,
             json!({"pet": {"meow": "hi", "lives": 9}})),
            ("an extra key rules a closed variant out", json!({"a": "x", "b": null}),
             json!({"anyOf": [{"type": "object", "additionalProperties": false,
                               "properties": {"a": {"type": "string", "default": "A"}}}, closed_b.clone()]}),
             json!({"a": "x", "b": "B"})),
            ("an open variant swallows the extra key and wins", json!({"a": "x", "b": null}),
             json!({"anyOf": [{"type": "object", "additionalProperties": true,
                               "properties": {"a": {"type": "string", "default": "A"}}}, closed_b]}),
             json!({"a": "x", "b": null})),
            ("a missing required key rules a variant out", json!({"a": null}),
             json!({"anyOf": [
                 {"type": "object", "required": ["b"], "additionalProperties": false,
                  "properties": {"a": {"type": "string", "default": "A"}, "b": {"type": "string"}}},
                 {"type": "object", "additionalProperties": false, "properties": {"a": {"type": "string", "default": "B"}}}]}),
             json!({"a": "B"})),
            ("const compares numerically", json!({"kind": 1, "extra": null}),
             json!({"anyOf": [
                 {"type": "object", "additionalProperties": false,
                  "properties": {"kind": {"const": 1.0}, "extra": {"type": "string", "default": "NUM"}}},
                 {"type": "object", "additionalProperties": false,
                  "properties": {"kind": {"const": 2}, "extra": {"type": "string", "default": "TWO"}}}]}),
             json!({"kind": 1, "extra": "NUM"})),
            ("enum compares numerically", json!({"k": 1, "e": null}),
             json!({"anyOf": [{"type": "object", "additionalProperties": false,
                    "properties": {"k": {"enum": [1.0, 2.0]}, "e": {"type": "string", "default": "E"}}}]}),
             json!({"k": 1, "e": "E"})),
            ("a list-valued type answers the match immediately", json!({"z": null}),
             json!({"anyOf": [{"type": ["object", "null"], "required": ["nope"]},
                              {"type": "object", "properties": {"z": {"type": "integer", "default": 3}}}]}),
             json!({"z": null})),
            ("a float is not an integer", json!({"n": 1.5, "tag": null}), int_or_num.clone(), json!({"n": 1.5, "tag": "NUM"})),
            ("an integer picks the integer branch", json!({"n": 1, "tag": null}), int_or_num, json!({"n": 1, "tag": "INT"})),
            ("a type mismatch rules a branch out", json!({"z": null}),
             json!({"anyOf": [{"type": "string"}, {"type": "object", "properties": {"z": {"type": "integer", "default": 7}}}]}),
             json!({"z": 7})),
            ("the sentinel shortcut needs an actual null", json!({"z": null}),
             json!({"anyOf": [{"type": "string", "default": "S", "properties": {"z": {"type": "boolean", "default": true}}},
                              {"type": "object", "properties": {"z": {"type": "integer", "default": 7}}}]}),
             json!({"z": 7})),
            ("$ref siblings win", json!({"x": null}),
             json!({"$defs": {"D": {"type": "integer", "default": 1}}, "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/D", "default": 42}}}),
             json!({"x": 42})),
            ("a self-referential $ref terminates", json!({"x": null}),
             json!({"$defs": {"C": {"$ref": "#/$defs/C", "default": 5}}, "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/C"}}}),
             json!({"x": 5})),
        ];
        for (label, value, schema, want) in cases {
            assert_eq!(inject(value, schema).unwrap(), want, "{label}");
        }
    }

    // Catches: swallowing a broken `$ref`.
    #[test]
    fn a_ref_that_does_not_resolve_is_an_error_naming_the_segment() {
        let schema = json!({"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}});
        assert_eq!(
            inject(json!({"x": null}), schema),
            Err(SchemaError::RefUnresolvable {
                reference: "#/$defs/Missing".into(),
                segment: "$defs".into(),
            })
        );
        let unrooted = json!({"type": "object", "properties": {"x": {"$ref": "D"}}});
        assert_eq!(
            inject(json!({"x": null}), unrooted),
            Err(SchemaError::RefNotRooted { reference: "D".into() })
        );
    }

    // Catches: a `_json_type_matches` arm dropped into the catch-all (`-1` has no unsigned reading).
    #[test]
    fn every_json_type_name_accepts_exactly_its_own_values() {
        let values =
            [json!(null), json!({}), json!([]), json!(true), json!(1), json!(1.5), json!("s")];
        let expected = [
            ("null", [true, false, false, false, false, false, false]),
            ("object", [false, true, false, false, false, false, false]),
            ("array", [false, false, true, false, false, false, false]),
            ("boolean", [false, false, false, true, false, false, false]),
            ("integer", [false, false, false, false, true, false, false]),
            ("number", [false, false, false, false, true, true, false]),
            ("string", [false, false, false, false, false, false, true]),
            ("unrecognised", [true, true, true, true, true, true, true]),
        ];
        for (typ, row) in expected {
            for (value, want) in values.iter().zip(row) {
                assert_eq!(json_type_matches(value, typ), want, "{typ} vs {value}");
            }
        }
        assert!(json_type_matches(&json!(-1), "integer"), "a negative integer is still an integer");
    }

    // Catches: `const`/`enum` compared by representation, and each half of the structural recursion.
    #[test]
    fn python_equality_compares_numbers_by_value_and_containers_element_by_element() {
        assert!(python_eq(&json!(1), &json!(1.0)));
        assert!(python_eq(&json!(-1), &json!(-1.0)));
        assert!(python_eq(&json!([1, "a"]), &json!([1.0, "a"])));
        assert!(python_eq(&json!({"a": 1}), &json!({"a": 1.0})));

        assert!(!python_eq(&json!(1), &json!(2)));
        assert!(!python_eq(&json!(1), &json!("1")));
        assert!(!python_eq(&json!([1]), &json!([1, 2])), "length matters");
        assert!(!python_eq(&json!([1]), &json!([2])));
        assert!(!python_eq(&json!({"a": 1}), &json!({"a": 1, "b": 2})), "key count matters");
        assert!(!python_eq(&json!({"a": 1}), &json!({"b": 1})), "key names matter");
        assert!(!python_eq(&json!({"a": 1}), &json!({"a": 2})));
        assert!(
            !python_eq(&json!(1), &json!(true)),
            "Python's `True == 1` wart is deliberately not reproduced"
        );

        // u64-only and i64-only readings must meet in the middle.
        assert!(python_eq(&json!(u64::MAX), &json!(u64::MAX)));
        assert!(!python_eq(&json!(u64::MAX), &json!(-1)));
    }

    // Catches: a recursion site that forgets to deepen the count.
    #[test]
    fn every_recursion_site_is_bounded_independently() {
        let nest = |wrap: &dyn Fn(Value) -> Value, leaf: Value| {
            let mut schema = leaf;
            for _ in 0..MAX_DEPTH + 2 {
                schema = wrap(schema);
            }
            schema
        };

        let all_of = nest(&|inner| json!({"allOf": [inner]}), json!({"type": "integer"}));
        assert_eq!(inject(json!(1), all_of), Err(SchemaError::TooDeep));

        let any_of = nest(&|inner| json!({"anyOf": [inner]}), json!({"type": "integer"}));
        assert_eq!(inject(json!(1), any_of), Err(SchemaError::TooDeep));

        let items = nest(&|inner| json!({"type": "array", "items": inner}), json!({}));
        let mut value = json!([]);
        for _ in 0..MAX_DEPTH + 2 {
            value = json!([value]);
        }
        assert_eq!(inject(value, items), Err(SchemaError::TooDeep));

        // The nullability walk is only entered for a null value against a schema with a default.
        let null_all_of = nest(&|inner| json!({"allOf": [inner]}), json!({"type": "integer"}));
        let mut schema = null_all_of.as_object().unwrap().clone();
        schema.insert("default".into(), json!(0));
        assert_eq!(inject(json!(null), Value::Object(schema)), Err(SchemaError::TooDeep));

        let null_any_of = nest(&|inner| json!({"anyOf": [inner]}), json!({"type": "integer"}));
        let mut schema = null_any_of.as_object().unwrap().clone();
        schema.insert("default".into(), json!(0));
        assert_eq!(inject(json!(null), Value::Object(schema)), Err(SchemaError::TooDeep));

        let mut defs = serde_json::Map::new();
        for level in 0..MAX_DEPTH + 2 {
            defs.insert(format!("D{level}"), json!({"$ref": format!("#/$defs/D{}", level + 1)}));
        }
        defs.insert(format!("D{}", MAX_DEPTH + 2), json!({"type": "integer"}));
        let chain = json!({"$defs": defs, "$ref": "#/$defs/D0", "default": 0});
        assert_eq!(inject(json!(null), chain), Err(SchemaError::TooDeep));
    }

    // Catches: an off-by-one in the depth bound, or no bound on the object walk.
    #[test]
    fn the_depth_bound_is_inclusive_at_the_limit_and_exclusive_one_past_it() {
        let nested = |levels: usize| {
            let mut value = json!(null);
            let mut schema = json!({"type": "integer", "default": 1});
            for _ in 0..levels {
                value = json!({"n": value});
                schema = json!({"type": "object", "properties": {"n": schema}});
            }
            (value, schema)
        };

        let (value, schema) = nested(MAX_DEPTH);
        assert!(inject(value, schema).is_ok(), "exactly at the limit is still allowed");

        let (value, schema) = nested(MAX_DEPTH + 1);
        assert_eq!(inject(value, schema), Err(SchemaError::TooDeep));
    }

    // Catches: the branch matcher's own recursion unbounded or off by one (it reaches one deeper).
    #[test]
    fn the_branch_matcher_is_bounded_one_level_deeper_than_the_injector() {
        /// Nests `leaf` inside `levels` objects; the injector reaches it at `depth == levels`.
        fn at_depth(levels: usize, leaf_value: Value, leaf_schema: Value) -> (Value, Value) {
            let (mut value, mut schema) = (leaf_value, leaf_schema);
            for _ in 0..levels {
                value = json!({"n": value});
                schema = json!({"type": "object", "properties": {"n": schema}});
            }
            (value, schema)
        }

        let (value, schema) = at_depth(MAX_DEPTH, json!(1), json!({"anyOf": [{"type": "string"}]}));
        assert_eq!(
            inject(value.clone(), schema).unwrap(),
            value,
            "a branch rejected at exactly the limit is not an error"
        );

        let (value, schema) = at_depth(
            MAX_DEPTH,
            json!({"deep": 5}),
            json!({"anyOf": [{"type": "object", "properties": {"deep": {"type": "string"}}}]}),
        );
        assert_eq!(inject(value, schema), Err(SchemaError::TooDeep));
    }

    // Catches: a fault mapped to the wrong host exception family.
    #[test]
    fn every_schema_fault_maps_to_the_reference_sites_exception_family() {
        let all = [
            (SchemaError::NotAnObject { what: "schema" }, "TypeError"),
            (
                SchemaError::RefUnresolvable { reference: "#/x".into(), segment: "x".into() },
                "KeyError",
            ),
            (SchemaError::RefNotRooted { reference: "x".into() }, "ValueError"),
            (SchemaError::TooDeep, "ValueError"),
            (
                SchemaError::Strict(StrictError::NotAnObject { path: "/".into() }),
                "TypeError",
            ),
            (
                SchemaError::Strict(StrictError::RefUnresolvable {
                    reference: "#/x".into(),
                    segment: "x".into(),
                }),
                "KeyError",
            ),
            (
                SchemaError::Strict(StrictError::NullableUnionBranch { path: "/".into() }),
                "ValueError",
            ),
        ];
        for (fault, kind) in &all {
            assert_eq!(fault.host_kind(), *kind, "{fault}");
        }
    }

    // Catches: a non-object schema treated as empty.
    #[test]
    fn a_schema_or_root_that_is_not_an_object_is_rejected() {
        assert_eq!(
            inject_schema_defaults(&json!(null), &json!("nope"), &json!({})),
            Err(SchemaError::NotAnObject { what: "schema" })
        );
        assert_eq!(
            inject_schema_defaults(&json!(null), &json!({}), &json!([])),
            Err(SchemaError::NotAnObject { what: "root" })
        );
    }
}
