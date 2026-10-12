//! `llm/_strict.py`. Key order is observable (`preserve_order`); `$defs` are made strict before
//! anything resolves them, but `$ref`s are inlined by value, without Python's incidental aliasing.

use serde_json::{Map, Value};

/// Public so `resolve_ref` is usable from the schema walk.
pub type Object = Map<String, Value>;

const DEF_KEYS: [&str; 2] = ["$defs", "definitions"];

/// `anyOf` before `oneOf`: with both, `oneOf` wins, as in the reference.
const UNION_KEYS: [&str; 2] = ["anyOf", "oneOf"];

/// Bounds a self-referential `$ref` with siblings, where the reference raises `RecursionError`.
const MAX_DEPTH: usize = 100;

/// A strict-schema fault; the kind table picks the host exception. Plugins catch the reference's
/// exceptions per site; `AssertionError` (a non-dict `resolve_ref` segment) folds into `KeyError`.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StrictError {
    #[error("expected a JSON object at {path}")]
    #[host(kind = "TypeError")]
    NotAnObject { path: String },
    #[error("non-string $ref at {path}")]
    NonStringRef { path: String },
    #[error("unexpected $ref format {reference:?}; does not start with #/")]
    RefNotRooted { reference: String },
    #[error("$ref {reference:?} does not resolve: no object at segment {segment:?}")]
    #[host(kind = "KeyError")]
    RefUnresolvable { reference: String, segment: String },
    #[error("nullable union at {path} has no non-null object branch")]
    NullableUnionBranch { path: String },
    #[error("schema nests deeper than {MAX_DEPTH} levels at {path}; is a $ref self-referential?")]
    TooDeep { path: String },
}

/// Rewrite a JSON schema into the strict dialect.
///
/// `null_sentinel_for_defaults`: whether a defaulted field also accepts `null` ("use the default").
/// Tool schemas want it; response formats must not, since nothing resolves the sentinel there.
#[crate::host(export(
    pid = false,
    kw_only = "null_sentinel_for_defaults",
    defaults(null_sentinel_for_defaults = false)
))]
pub fn to_strict_json_schema(
    schema: &Value,
    null_sentinel_for_defaults: bool,
) -> Result<Value, StrictError> {
    let obj = schema.as_object().ok_or_else(|| StrictError::NotAnObject { path: "/".into() })?;

    let root = processed_root(obj, null_sentinel_for_defaults)?;
    let mut walk = Walk { root: &root, null_sentinel_for_defaults, path: Vec::new(), depth: 0 };

    let mut top = obj.clone();
        // Splice in the precomputed definitions: a second walk is not idempotent (`enum.append(None)`).
    for key in DEF_KEYS {
        if let Some(defs) = root.get(key) {
            top.insert(key.to_string(), defs.clone());
        }
    }

    let collapsed = walk.body(&mut top)?;
    Ok(Value::Object(collapsed.unwrap_or(top)))
}

/// Definitions made strict up front, in declaration order, so a `$ref` sees them already strict
/// (and a forward `$ref` sees one raw), as in the reference.
fn processed_root(obj: &Object, null_sentinel_for_defaults: bool) -> Result<Object, StrictError> {
    let mut root = obj.clone();

    for key in DEF_KEYS {
        let Some(defs) = root.get(key).and_then(Value::as_object).cloned() else { continue };
        for (name, def) in defs {
            let snapshot = root.clone();
            let mut walk = Walk {
                root: &snapshot,
                null_sentinel_for_defaults,
                path: vec![key.to_string(), name.clone()],
                depth: 0,
            };
            let def = walk.definition(def)?;
            root[key][name.as_str()] = def;
        }
    }

    Ok(root)
}

struct Walk<'a> {
    root: &'a Object,
    null_sentinel_for_defaults: bool,
    path: Vec<String>,
    depth: usize,
}

impl Walk<'_> {
    fn path(&self) -> String {
        format!("/{}", self.path.join("/"))
    }

    /// A definition: walked for its side effects only. The reference discards the return value,
    /// so a top-level `X | None` keeps its `anyOf` while its branch is still widened in place.
    fn definition(&mut self, mut def: Value) -> Result<Value, StrictError> {
        let Some(obj) = def.as_object_mut() else {
            return Err(StrictError::NotAnObject { path: self.path() });
        };
        self.ensure(obj)?;
        Ok(def)
    }

    fn child(&mut self, segments: &[&str], mut node: Value) -> Result<Value, StrictError> {
        let restore = self.path.len();
        self.path.extend(segments.iter().map(|s| (*s).to_string()));

        let out = match node.as_object_mut() {
            None => Err(StrictError::NotAnObject { path: self.path() }),
            Some(obj) => match self.ensure(obj) {
                Err(e) => Err(e),
                Ok(Some(collapsed)) => Ok(Value::Object(collapsed)),
                Ok(None) => Ok(node),
            },
        };

        self.path.truncate(restore);
        out
    }

    fn ensure(&mut self, schema: &mut Object) -> Result<Option<Object>, StrictError> {
        if self.depth >= MAX_DEPTH {
            return Err(StrictError::TooDeep { path: self.path() });
        }
        self.depth += 1;
        self.defs(schema)?;
        let out = self.body(schema);
        self.depth -= 1;
        out
    }

    fn defs(&mut self, schema: &mut Object) -> Result<(), StrictError> {
        for key in DEF_KEYS {
            let Some(defs) = schema.get(key).and_then(Value::as_object).cloned() else { continue };
            let mut out = Object::new();
            for (name, def) in defs {
                let restore = self.path.len();
                self.path.push(key.to_string());
                self.path.push(name.clone());
                let walked = self.definition(def);
                self.path.truncate(restore);
                out.insert(name, walked?);
            }
            schema.insert(key.to_string(), Value::Object(out));
        }
        Ok(())
    }

    fn body(&mut self, schema: &mut Object) -> Result<Option<Object>, StrictError> {
        // Only for a bare `"object"`, and never over an explicit `additionalProperties`.
        if schema.get("type").and_then(Value::as_str) == Some("object")
            && !schema.contains_key("additionalProperties")
        {
            schema.insert("additionalProperties".into(), Value::Bool(false));
        }

        // Every property is required. An empty `properties` gets no `required`: groq rejects
        // `{"required": []}` (#5082).
        if let Some(props) = schema.get("properties").and_then(Value::as_object).cloned()
            && !props.is_empty()
        {
            let names = props.keys().cloned().map(Value::String).collect();
            schema.insert("required".into(), Value::Array(names));

            let mut out = Object::new();
            for (name, prop) in props {
                let walked = self.child(&["properties", &name], prop)?;
                out.insert(name, walked);
            }
            schema.insert("properties".into(), Value::Object(out));
        }

        // Only the single-schema `items`; the draft-4 tuple form is not something pydantic emits.
        if schema.get("items").is_some_and(Value::is_object) {
            let items = schema["items"].take();
            schema["items"] = self.child(&["items"], items)?;
        }

        self.unions(schema)?;
        self.all_of(schema)?;

        // Popped before the default handling so the sentinel wrapper does not carry them. Without
        // `discriminator`, a variant's `const` tag identifies the branch (#6688).
        schema.shift_remove("title");
        schema.shift_remove("discriminator");

        if schema.shift_remove("default").is_some() && self.null_sentinel_for_defaults {
            add_null_sentinel(schema);
        }

        // Strict mode forbids `$ref` siblings: flatten into the target (siblings win) and re-walk.
        // A lone `$ref` is left alone, which stops a recursive model expanding forever.
        let reference = schema.get("$ref").cloned();
        if reference.as_ref().is_some_and(crate::llm::provider_format::is_truthy) && schema.len() > 1 {
            let Some(Value::String(reference)) = reference else {
                return Err(StrictError::NonStringRef { path: self.path() });
            };
            for (key, value) in resolve_ref(self.root, &reference)? {
                schema.entry(key).or_insert(value);
            }
            schema.shift_remove("$ref");
            return self.ensure(schema);
        }

        self.collapse_nullable_union(schema)
    }

    fn unions(&mut self, schema: &mut Object) -> Result<(), StrictError> {
        for union_key in UNION_KEYS {
            let Some(variants) = schema.get(union_key).and_then(Value::as_array) else { continue };

            // Drop `{}`, which strict mode rejects for having no type (`Union[Literal[...], Any]`, #5137).
            let variants: Vec<Value> =
                variants.iter().filter(|v| !is_empty_object(v)).cloned().collect();

            match variants.len() {
                // Hoist the survivor into the parent; its keys beat the parent's.
                1 => {
                    let only =
                        self.child(&[union_key, "0"], variants.into_iter().next().unwrap())?;
                    if let Value::Object(only) = only {
                        schema.extend(only);
                    }
                    schema.shift_remove(union_key);
                }
                0 => {
                    schema.shift_remove(union_key);
                }
                // `oneOf` is not permitted (#5324); `anyOf` is equivalent, and moves to the key end.
                _ => {
                    schema.shift_remove(union_key);
                    let mut out = Vec::with_capacity(variants.len());
                    for (i, variant) in variants.into_iter().enumerate() {
                        out.push(self.child(&["anyOf", &i.to_string()], variant)?);
                    }
                    schema.insert("anyOf".into(), Value::Array(out));
                }
            }
        }
        Ok(())
    }

    /// Hoist a one-element `allOf` (pydantic's `$ref`-with-description wrapper).
    fn all_of(&mut self, schema: &mut Object) -> Result<(), StrictError> {
        let Some(all_of) = schema.get("allOf").and_then(Value::as_array).cloned() else {
            return Ok(());
        };
        if all_of.len() == 1 {
            let only = self.child(&["allOf", "0"], all_of.into_iter().next().unwrap())?;
            if let Value::Object(only) = only {
                schema.extend(only);
            }
            schema.shift_remove("allOf");
        } else {
            let mut out = Vec::with_capacity(all_of.len());
            for (i, entry) in all_of.into_iter().enumerate() {
                out.push(self.child(&["allOf", &i.to_string()], entry)?);
            }
            schema.insert("allOf".into(), Value::Array(out));
        }
        Ok(())
    }

    /// `X | None` becomes `X` with `"null"` admitted by its own `type`, and by its `enum` if any
    /// (#5080). Exactly-two-branch unions only, first of `anyOf`/`oneOf` wins. The widening happens
    /// in place in the variants array, which is observable in `$defs` (see [`Walk::definition`]).
    fn collapse_nullable_union(&self, schema: &mut Object) -> Result<Option<Object>, StrictError> {
        for union_key in UNION_KEYS {
            let Some(variants) = schema.get(union_key).and_then(Value::as_array) else { continue };
            if variants.len() != 2 || !variants.iter().any(is_null_type) {
                continue;
            }

            // Both branches null, or a non-null branch not an object: the reference asserts.
            let index = variants.iter().position(|v| !is_null_type(v));
            let Some(index) = index.filter(|&i| variants[i].is_object()) else {
                return Err(StrictError::NullableUnionBranch { path: self.path() });
            };

            // A branch with no `type` (`{"$ref": ...} | None`) keeps its `anyOf` (#4236); the
            // reference goes on to try `oneOf`.
            if !variants[index].as_object().is_some_and(|b| b.contains_key("type")) {
                continue;
            }

            let non_null = schema[union_key][index].as_object_mut().expect("checked just above");
            if let Some(t) = non_null["type"].as_str() {
                non_null["type"] = Value::Array(vec![t.into(), "null".into()]);
            }
            if let Some(values) = non_null.get_mut("enum").and_then(Value::as_array_mut) {
                // Unconditional, as the reference: an `enum` already listing null gets a second.
                values.push(Value::Null);
            }
            let non_null = non_null.clone();

            let mut out: Object = schema
                .iter()
                .filter(|(k, _)| !UNION_KEYS.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            out.extend(non_null);
            return Ok(Some(out));
        }
        Ok(None)
    }
}

/// A bare `type` gets `"null"` appended; `enum`, a union or a `$ref` gets a null branch instead.
/// A `const` gets nothing: it is a discriminated union's tag (#6688).
fn add_null_sentinel(schema: &mut Object) {
    match schema.get("type") {
        Some(Value::String(t)) if t == "null" => return,
        Some(Value::Array(ts)) if ts.iter().any(|t| t == "null") => return,
        _ => {}
    }
    if schema.contains_key("const") {
        return;
    }

    for union_key in UNION_KEYS {
        if let Some(variants) = schema.get_mut(union_key).and_then(Value::as_array_mut) {
            if !variants.iter().any(is_null_type) {
                variants.push(null_type());
            }
            return;
        }
    }

    if !schema.contains_key("enum") {
        match schema.get("type") {
            Some(Value::String(t)) => {
                schema["type"] = Value::Array(vec![t.clone().into(), "null".into()]);
                return;
            }
            Some(Value::Array(ts)) => {
                let mut ts = ts.clone();
                ts.push("null".into());
                schema["type"] = Value::Array(ts);
                return;
            }
            _ => {}
        }
    }

    // An empty schema already accepts null.
    if schema.is_empty() {
        return;
    }

    let inner = std::mem::take(schema);
    if inner.contains_key("$ref") && inner.len() > 1 {
        // Siblings stay on the wrapper; the ref goes inside alone.
        *schema = inner;
        let reference = schema.shift_remove("$ref").unwrap_or(Value::Null);
        let mut only_ref = Object::new();
        only_ref.insert("$ref".into(), reference);
        schema.insert("anyOf".into(), Value::Array(vec![Value::Object(only_ref), null_type()]));
    } else {
        schema.insert("anyOf".into(), Value::Array(vec![Value::Object(inner), null_type()]));
    }
}

/// Resolve a local `$ref` against the schema root (`_strict.resolve_ref`, shared with
/// `llm/utils.py`). `#/`-rooted pointers only, with no `~0`/`~1` unescaping, as the reference.
#[crate::host(export(pid = false, kw_only))]
pub fn resolve_ref(root: &Object, r#ref: &str) -> Result<Object, StrictError> {
    let reference = r#ref;
    let Some(rest) = reference.strip_prefix("#/") else {
        return Err(StrictError::RefNotRooted { reference: reference.to_string() });
    };
    let mut resolved = root;
    for segment in rest.split('/') {
        let Some(next) = resolved.get(segment).and_then(Value::as_object) else {
            return Err(StrictError::RefUnresolvable {
                reference: reference.to_string(),
                segment: segment.to_string(),
            });
        };
        resolved = next;
    }
    Ok(resolved.clone())
}

fn null_type() -> Value {
    let mut obj = Object::new();
    obj.insert("type".into(), "null".into());
    Value::Object(obj)
}

fn is_null_type(value: &Value) -> bool {
    value.as_object().is_some_and(|o| o.len() == 1 && o.get("type").is_some_and(|t| t == "null"))
}

fn is_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(Object::is_empty)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{StrictError, to_strict_json_schema};

    // Catches: a fault sliding into the wrong host exception family.
    #[test]
    fn every_strict_fault_maps_to_the_reference_sites_exception_family() {
        let p = || "/".to_owned();
        let r = || "#/x".to_owned();
        let all = [
            (StrictError::NotAnObject { path: p() }, "TypeError"),
            (StrictError::RefUnresolvable { reference: r(), segment: "x".into() }, "KeyError"),
            (StrictError::NonStringRef { path: p() }, "ValueError"),
            (StrictError::RefNotRooted { reference: r() }, "ValueError"),
            (StrictError::NullableUnionBranch { path: p() }, "ValueError"),
            (StrictError::TooDeep { path: p() }, "ValueError"),
        ];
        for (fault, kind) in &all {
            assert_eq!(fault.host_kind(), *kind, "{fault}");
        }
    }

    // One row per rewrite, compared as a string: key order is part of the promise. The label
    // names what a regression would break; the issue numbers are the reference's.
    #[test]
    fn each_rewrite_matches_the_reference_byte_for_byte() {
        let cases: &[(&str, Value, bool, &str)] = &[
            (
                "additional_properties_is_added_only_to_a_bare_untouched_object",
                json!({"type": "object", "properties": {"a": {"type": "string"}}}),
                false,
                r##"{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false,"required":["a"]}"##,
            ),
            (
                "additional_properties_is_added_only_to_a_bare_untouched_object",
                json!({"type": "object", "additionalProperties": true}),
                false,
                r##"{"type":"object","additionalProperties":true}"##,
            ),
            (
                "additional_properties_is_added_only_to_a_bare_untouched_object",
                json!({"type": ["object", "null"]}),
                false,
                r##"{"type":["object","null"]}"##,
            ),
            (
                "required_lists_every_property_in_declaration_order",
                json!({ "type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "string"}}, "required": ["a"], }),
                false,
                r##"{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"string"}},"required":["z","a"],"additionalProperties":false}"##,
            ),
            (
                "a_parameterless_object_gets_no_required_at_all",
                json!({"type": "object", "properties": {}}),
                false,
                r##"{"type":"object","properties":{},"additionalProperties":false}"##,
            ),
            (
                "empty_variants_are_stripped_and_a_lone_survivor_is_hoisted",
                json!({"anyOf": [{"enum": ["a"], "type": "string"}, {}]}),
                false,
                r##"{"enum":["a"],"type":"string"}"##,
            ),
            (
                "empty_variants_are_stripped_and_a_lone_survivor_is_hoisted",
                json!({"description": "d", "anyOf": [{}]}),
                false,
                r##"{"description":"d"}"##,
            ),
            (
                "one_of_becomes_any_of_at_the_end_of_the_key_order",
                json!({"oneOf": [{"type": "string"}, {"type": "integer"}], "description": "d"}),
                false,
                r##"{"description":"d","anyOf":[{"type":"string"},{"type":"integer"}]}"##,
            ),
            (
                "title_and_discriminator_are_dropped",
                json!({ "title": "T", "discriminator": {"propertyName": "kind"}, "anyOf": [{"const": "a", "type": "string"}, {"const": "b", "type": "string"}], }),
                false,
                r##"{"anyOf":[{"const":"a","type":"string"},{"const":"b","type":"string"}]}"##,
            ),
            (
                "default_is_always_removed_but_the_null_sentinel_is_opt_in",
                json!({"type": "string", "default": "x"}),
                false,
                r##"{"type":"string"}"##,
            ),
            (
                "default_is_always_removed_but_the_null_sentinel_is_opt_in",
                json!({"type": "string", "default": "x"}),
                true,
                r##"{"type":["string","null"]}"##,
            ),
            (
                "a_collapsed_nullable_enum_admits_null_in_both_type_and_enum",
                json!({"anyOf": [{"enum": ["a", "b"], "type": "string"}, {"type": "null"}]}),
                false,
                r##"{"enum":["a","b",null],"type":["string","null"]}"##,
            ),
            (
                "a_nullable_union_over_an_untyped_branch_is_left_as_a_union",
                json!({ "anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}], "$defs": {"M": {"type": "object", "properties": {"a": {"type": "string"}}}}, }),
                false,
                r##"{"$defs":{"M":{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false,"required":["a"]}},"anyOf":[{"$ref":"#/$defs/M"},{"type":"null"}]}"##,
            ),
            (
                "a_ref_with_siblings_is_inlined_and_the_siblings_win",
                json!({ "$defs": {"M": {"type": "object", "properties": {"a": {"type": "string"}}, "description": "from the model"}}, "$ref": "#/$defs/M", "description": "from the field", }),
                false,
                r##"{"$defs":{"M":{"type":"object","properties":{"a":{"type":"string"}},"description":"from the model","additionalProperties":false,"required":["a"]}},"description":"from the field","type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false,"required":["a"]}"##,
            ),
            (
                "a_lone_ref_is_never_inlined",
                json!({ "type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}, "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}}, }),
                false,
                r##"{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}},"$defs":{"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}},"additionalProperties":false,"required":["next"]}},"additionalProperties":false,"required":["next"]}"##,
            ),
            (
                "the_sentinel_never_nulls_a_const",
                json!({"const": "celsius", "type": "string", "default": "celsius"}),
                true,
                r##"{"const":"celsius","type":"string"}"##,
            ),
            (
                "the_sentinel_wraps_a_constrained_schema_instead_of_widening_its_type",
                json!({"enum": ["a"], "type": "string", "default": "a"}),
                true,
                r##"{"enum":["a",null],"type":["string","null"]}"##,
            ),
            (
                "the_sentinel_wraps_a_constrained_schema_instead_of_widening_its_type",
                json!({"$defs": {"M": {"type": "string"}}, "$ref": "#/$defs/M", "default": "a"}),
                true,
                r##"{"$defs":{"M":{"type":"string"}},"anyOf":[{"$ref":"#/$defs/M"},{"type":"null"}]}"##,
            ),
            (
                "the_sentinel_keeps_ref_siblings_outside_the_wrapper",
                json!({ "$defs": {"M": {"type": "string"}}, "$ref": "#/$defs/M", "description": "d", "default": "a", }),
                true,
                r##"{"$defs":{"M":{"type":"string"}},"description":"d","anyOf":[{"$ref":"#/$defs/M"},{"type":"null"}]}"##,
            ),
            (
                "the_sentinel_is_idempotent_on_something_already_nullable",
                json!({"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}),
                true,
                r##"{"type":["string","null"]}"##,
            ),
            (
                "the_sentinel_is_idempotent_on_something_already_nullable",
                json!({"type": ["string", "null"], "default": null}),
                true,
                r##"{"type":["string","null"]}"##,
            ),
            (
                "the_sentinel_is_idempotent_on_something_already_nullable",
                json!({"default": 1}),
                true,
                "{}",
            ),
            (
                "a_definition_keeps_its_union_but_its_branch_is_still_widened",
                json!({ "$defs": {"Opt": {"anyOf": [{"enum": ["a"], "type": "string"}, {"type": "null"}]}}, "type": "string", }),
                false,
                r##"{"$defs":{"Opt":{"anyOf":[{"enum":["a",null],"type":["string","null"]},{"type":"null"}]}},"type":"string"}"##,
            ),
            (
                "a_single_element_all_of_is_hoisted_but_a_longer_one_is_not",
                json!({"allOf": [{"$ref": "#/$defs/M"}], "description": "d", "$defs": {"M": {"type": "string"}}}),
                false,
                r##"{"description":"d","$defs":{"M":{"type":"string"}},"type":"string"}"##,
            ),
            (
                "a_single_element_all_of_is_hoisted_but_a_longer_one_is_not",
                json!({"allOf": [{"type": "string", "title": "T"}, {"minLength": 1}]}),
                false,
                r##"{"allOf":[{"type":"string"},{"minLength":1}]}"##,
            ),
        ];
        for (label, schema, sentinel, want) in cases {
            assert_eq!(to_strict_json_schema(schema, *sentinel).unwrap().to_string(), *want, "{label}");
        }
    }

    // Catches: a malformed schema panicking, or a `$ref` cycle overflowing the stack.
    #[test]
    fn malformed_input_is_an_error_and_a_ref_cycle_is_bounded() {
        assert!(matches!(
            to_strict_json_schema(&json!({"properties": {"a": "not a schema"}}), false),
            Err(StrictError::NotAnObject { .. })
        ));
        assert!(matches!(
            to_strict_json_schema(&json!({"$ref": "http://x", "description": "d"}), false),
            Err(StrictError::RefNotRooted { .. })
        ));
        assert!(matches!(
            to_strict_json_schema(&json!({"$ref": "#/$defs/Nope", "description": "d"}), false),
            Err(StrictError::RefUnresolvable { .. })
        ));
        assert!(matches!(
            to_strict_json_schema(
                &json!({
                    "$defs": {"A": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A", "description": "d"}}}},
                    "type": "string",
                }),
                false
            ),
            Err(StrictError::TooDeep { .. })
        ));
    }
}
