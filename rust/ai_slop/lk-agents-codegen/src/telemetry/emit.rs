//! The registry's outputs: the Weaver registry (`rust/otel/`), the consumer schema
//! (`rust/otel/schema.json`) and the Python constants (`_generated/semconv.py`). The Weaver files
//! are YAML written in its JSON subset, so one JSON printer writes every output.

use super::{Attr, Content, Encoding, Parent, Registry, Shape, ShapeField, Wire};
use lk_agents_bindgen::telemetry::{InstrumentKind, SpanKind, Unmeasured};
use serde_json::{Map, Value, json};

/// livekit/protocol's commit whose `otel/model/shared` the shared keys are read from (PR #1866).
pub const PROTOCOL_REF: &str = "424a912b1db6878a43a9cbea3cd71cb92e4ad712";
/// OpenTelemetry semantic conventions.
const SEMCONV_REF: &str = "v1.44.0";
/// The GenAI registry has no tags; it is pinned by commit, as protocol pins it.
const GENAI_REF: &str = "547934afa90a34298441caf80c43b27255c3a292";

const SOURCE: &str = "rust/crates/lk-agents-abi/src/telemetry/";

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).expect("a JSON value serializes") + "\n"
}

fn quote(text: &str) -> String {
    serde_json::to_string(text).expect("a string serializes")
}

fn wire_name(wire: &Wire) -> String {
    match wire {
        Wire::String => "string".into(),
        Wire::Int => "int".into(),
        Wire::Double => "double".into(),
        Wire::Bool => "boolean".into(),
        Wire::KvList => "kvlist".into(),
        Wire::Array(inner) => format!("{}[]", wire_name(inner)),
    }
}

fn kind_name(kind: SpanKind) -> &'static str {
    match kind {
        SpanKind::Internal => "internal",
        SpanKind::Client => "client",
        SpanKind::Server => "server",
    }
}

fn instrument_name(kind: InstrumentKind) -> &'static str {
    match kind {
        InstrumentKind::Counter => "counter",
        InstrumentKind::Histogram => "histogram",
    }
}

fn parent_name(p: &Parent) -> &str {
    match p {
        Parent::Span(name) => name,
        Parent::Current => "<current>",
        Parent::External => "<external>",
    }
}

fn encoding(e: &Option<Encoding>) -> Option<&'static str> {
    match e {
        Some(Encoding::Json) => Some("json"),
        Some(Encoding::BoolString) => Some("bool_string"),
        _ => None,
    }
}

// --- rust/otel/schema.json -----------------------------------------------------------------------

fn schema_attr(a: &Attr) -> Value {
    let mut f = Map::new();
    f.insert("key".into(), json!(a.key));
    f.insert("type".into(), json!(wire_name(&a.wire)));
    f.insert("required".into(), json!(a.required));
    if a.registry != "agents" {
        f.insert("registry".into(), json!(a.registry));
    }
    if a.seconds {
        f.insert("unit".into(), json!("s"));
    }
    if let Some(e) = encoding(&a.encoding) {
        f.insert("encoding".into(), json!(e));
    }
    if let Some(Encoding::Enum(members)) = &a.encoding {
        f.insert("members".into(), json!(members));
    }
    match a.unmeasured {
        Some(Unmeasured::Omit) => f.insert("unmeasured".into(), json!("omit")),
        Some(Unmeasured::Zero) => f.insert("unmeasured".into(), json!(0)),
        Some(Unmeasured::MinusOne) => f.insert("unmeasured".into(), json!(-1)),
        None => None,
    };
    match a.content {
        Some(Content::Pii) => f.insert("content".into(), json!("pii")),
        Some(Content::GenAi) => f.insert("content".into(), json!("gen_ai")),
        None => None,
    };
    match &a.label {
        Some(key) if *key == a.key => f.insert("label".into(), json!(true)),
        Some(key) => f.insert("label".into(), json!(key)),
        None => None,
    };
    if let Some(metric) = &a.metric {
        f.insert("metric".into(), json!(metric));
    }
    if let Some(shape) = &a.shape {
        f.insert("shape".into(), schema_shape(shape));
    }
    Value::Object(f)
}

fn schema_shape(shape: &Shape) -> Value {
    let fields = |fields: &[ShapeField]| -> Value {
        fields
            .iter()
            .map(|f| {
                let mut o = json!({
                    "key": f.key,
                    "type": wire_name(&f.wire),
                    "required": f.required,
                });
                if let Some(shape) = &f.shape {
                    o["shape"] = schema_shape(shape);
                }
                o
            })
            .collect()
    };
    match shape {
        Shape::Fields(f) => json!({ "fields": fields(f) }),
        Shape::Tagged { tag, variants } => json!({
            "tag": tag,
            "variants": variants
                .iter()
                .map(|(value, f)| json!({ "value": value, "fields": fields(f) }))
                .collect::<Vec<_>>(),
        }),
    }
}

fn schema_attrs(attrs: &[Attr]) -> Value {
    attrs.iter().map(schema_attr).collect()
}

fn schema(r: &Registry) -> String {
    let mut spans: Vec<_> = r.spans.iter().collect();
    spans.sort_by(|a, b| a.name.cmp(&b.name));
    let mut events: Vec<_> = r.events.iter().collect();
    events.sort_by(|a, b| a.name.cmp(&b.name));
    let mut metrics: Vec<_> = r.metrics.iter().collect();
    metrics.sort_by(|a, b| a.name.cmp(&b.name));
    pretty(&json!({
        "$comment": format!(
            "GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from {SOURCE}."
        ),
        "spans": spans
            .iter()
            .map(|span| json!({
                "name": span.name,
                "brief": span.brief,
                "kind": kind_name(span.kind),
                "parents": span.parents.iter().map(parent_name).collect::<Vec<_>>(),
                "outlives_parent": span.outlives_parent,
                "start": schema_attrs(&span.start),
                "end": schema_attrs(&span.end),
                "events": span.events,
            }))
            .collect::<Vec<_>>(),
        "events": events
            .iter()
            .map(|e| json!({
                "name": e.name,
                "brief": e.brief,
                "attributes": schema_attrs(&e.attrs),
            }))
            .collect::<Vec<_>>(),
        "metrics": metrics
            .iter()
            .map(|m| {
                let mut o = json!({
                    "name": m.name,
                    "brief": m.brief,
                    "kind": instrument_name(m.kind),
                    "unit": m.unit,
                    "registry": if m.upstream { "otel" } else { "agents" },
                    "labels": schema_attrs(&m.labels),
                });
                if !m.bounds.is_empty() {
                    o["bounds"] = json!(m.bounds);
                }
                o
            })
            .collect::<Vec<_>>(),
        "records": r
            .records
            .iter()
            .map(|rec| json!({
                "scope": rec.scope,
                "body": rec.body,
                "brief": rec.brief,
                "span_id": rec.span_id,
                "attributes": schema_attrs(&rec.attrs),
            }))
            .collect::<Vec<_>>(),
        "groups": r
            .groups
            .iter()
            .map(|g| json!({
                "name": g.name,
                "brief": g.brief,
                "attributes": schema_attrs(&g.attrs),
            }))
            .collect::<Vec<_>>(),
    }))
}

// --- rust/otel/manifest.yaml and rust/otel/registry/ ---------------------------------------------

fn yaml_file(doc: Value) -> String {
    format!(
        "# GENERATED FILE -- DO NOT EDIT.\n# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from {SOURCE}.\n{}",
        pretty(&doc)
    )
}

fn manifest() -> String {
    let dep = |schema: &str, path: String| json!({ "schema_url": schema, "registry_path": path });
    yaml_file(json!({
        "file_format": "definition_manifest/2.0",
        "schema_url": "https://github.com/livekit/agents/tree/main/rust/otel",
        "stability": "development",
        "description": "LiveKit Agents telemetry: the framework's spans, events, metrics and report records.",
        "dependencies": [
            dep(
                "https://github.com/livekit/protocol/tree/main/otel",
                format!("https://github.com/livekit/protocol.git@{PROTOCOL_REF}[otel/model/shared]"),
            ),
            dep(
                "https://opentelemetry.io/schemas/1.44.0",
                format!("https://github.com/open-telemetry/semantic-conventions.git@{SEMCONV_REF}[model]"),
            ),
            dep(
                "https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev",
                format!(
                    "https://github.com/open-telemetry/semantic-conventions-genai.git@{GENAI_REF}[model]"
                ),
            ),
        ],
    }))
}

fn weaver_type(a: &Attr) -> Value {
    if let Some(Encoding::Enum(members)) = &a.encoding {
        let members: Vec<Value> = members
            .iter()
            .map(|m| {
                let id = m.to_lowercase().replace('-', "_");
                json!({ "id": id, "value": m, "brief": m, "stability": "development" })
            })
            .collect();
        return json!({ "members": members });
    }
    match &a.wire {
        Wire::KvList => json!("any"),
        Wire::Array(inner) if **inner == Wire::KvList => json!("any"),
        wire => json!(wire_name(wire)),
    }
}

fn attributes_yaml(r: &Registry) -> String {
    let mut modules: Vec<&str> = r.owned.iter().map(|a| a.declared_in.as_str()).collect();
    modules.sort();
    modules.dedup();
    let groups: Vec<Value> = modules
        .iter()
        .map(|module| {
            let mut owned: Vec<_> = r.owned.iter().filter(|a| a.declared_in == *module).collect();
            owned.sort_by(|a, b| a.key.cmp(&b.key));
            let id = module.rsplit("::").next().unwrap_or(module);
            json!({
                "id": format!("registry.livekit.agents.{id}"),
                "type": "attribute_group",
                "brief": format!("Attributes declared in `lk_agents_abi::{module}`."),
                "attributes": owned.iter().map(|a| weaver_attr(a)).collect::<Vec<_>>(),
            })
        })
        .chain([json!({
            "id": "registry.livekit.agents.semconv",
            "type": "attribute_group",
            "brief": "Every dependency key `lk_agents_abi::telemetry::semconv` names.",
            "attributes": r.semconv.iter()
                .map(|k| json!({ "ref": k, "requirement_level": "recommended" }))
                .collect::<Vec<_>>(),
        })])
        .collect();
    yaml_file(json!({ "groups": groups }))
}

fn weaver_attr(a: &Attr) -> Value {
    let mut livekit = Map::new();
    if a.seconds {
        livekit.insert("unit".into(), json!("s"));
    }
    if a.content == Some(Content::Pii) {
        livekit.insert("sensitivity".into(), json!("pii"));
    }
    if let Some(e) = encoding(&a.encoding) {
        livekit.insert("encoding".into(), json!(e));
    }
    let mut f = json!({
        "id": a.key,
        "type": weaver_type(a),
        "stability": "development",
        "brief": a.brief,
    });
    if !livekit.is_empty() {
        f["annotations"] = json!({ "livekit": livekit });
    }
    f
}

/// Attribute references, once per key.
fn refs<'a>(attrs: impl IntoIterator<Item = &'a Attr>) -> Value {
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for a in attrs {
        if seen.contains(&&a.key) {
            continue;
        }
        seen.push(&a.key);
        let level = if a.required { "required" } else { "recommended" };
        out.push(json!({ "ref": a.key, "requirement_level": level }));
    }
    Value::Array(out)
}

fn spans_yaml(r: &Registry) -> String {
    let mut spans: Vec<_> = r.spans.iter().collect();
    spans.sort_by(|a, b| a.name.cmp(&b.name));
    let groups: Vec<Value> = spans
        .iter()
        .map(|span| {
            let mut livekit = Map::new();
            let parents: Vec<_> = span.parents.iter().map(parent_name).collect();
            livekit.insert("parents".into(), json!(parents));
            if let Some(reason) = &span.outlives_parent {
                livekit.insert("outlives_parent".into(), json!(reason));
            }
            if !span.end.is_empty() {
                let keys: Vec<_> = span.end.iter().map(|a| &a.key).collect();
                livekit.insert("end_attributes".into(), json!(keys));
            }
            if !span.events.is_empty() {
                livekit.insert("events".into(), json!(span.events));
            }
            let mut f = json!({
                "id": format!("span.{}", span.name),
                "type": "span",
                "span_kind": kind_name(span.kind),
                "stability": "development",
                "brief": span.brief,
            });
            if !span.start.is_empty() || !span.end.is_empty() {
                f["attributes"] = refs(span.start.iter().chain(&span.end));
            }
            f["annotations"] = json!({ "livekit": livekit });
            f
        })
        .collect();
    yaml_file(json!({ "groups": groups }))
}

fn events_yaml(r: &Registry) -> String {
    let mut events: Vec<_> = r.events.iter().collect();
    events.sort_by(|a, b| a.name.cmp(&b.name));
    let groups: Vec<Value> = events
        .iter()
        .map(|e| {
            let mut f = json!({
                "id": format!("event.{}", e.name),
                "type": "event",
                "name": e.name,
                "stability": "development",
                "brief": e.brief,
            });
            if !e.attrs.is_empty() {
                f["attributes"] = refs(&e.attrs);
            }
            f
        })
        .collect();
    yaml_file(json!({ "groups": groups }))
}

fn metrics_yaml(r: &Registry) -> String {
    let mut metrics: Vec<_> = r.metrics.iter().filter(|m| !m.upstream).collect();
    metrics.sort_by(|a, b| a.name.cmp(&b.name));
    let groups: Vec<Value> = metrics
        .iter()
        .map(|m| {
            json!({
                "id": format!("metric.{}", m.name),
                "type": "metric",
                "metric_name": m.name,
                "instrument": instrument_name(m.kind),
                "unit": m.unit,
                "stability": "development",
                "brief": m.brief,
                "attributes": refs(&m.labels),
            })
        })
        .collect();
    yaml_file(json!({ "groups": groups }))
}

fn imports_yaml(r: &Registry) -> String {
    let mut names: Vec<&String> =
        r.metrics.iter().filter(|m| m.upstream).map(|m| &m.name).collect();
    names.sort();
    yaml_file(json!({ "file_format": "definition/2", "imports": { "metrics": names } }))
}

/// Every file under `rust/otel/`, by repo-relative path.
pub fn otel(r: &Registry) -> Vec<(String, String)> {
    vec![
        ("rust/otel/manifest.yaml".into(), manifest()),
        ("rust/otel/registry/attributes.yaml".into(), attributes_yaml(r)),
        ("rust/otel/registry/spans.yaml".into(), spans_yaml(r)),
        ("rust/otel/registry/events.yaml".into(), events_yaml(r)),
        ("rust/otel/registry/metrics.yaml".into(), metrics_yaml(r)),
        ("rust/otel/registry/imports.yaml".into(), imports_yaml(r)),
        ("rust/otel/schema.json".into(), schema(r)),
    ]
}

// --- _generated/semconv.py ----------------------------------------------------------------------

fn constant(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
        .collect()
}

/// The Python constants: every key, span, event and metric name and record scope and body.
pub fn semconv_py(r: &Registry) -> Result<String, String> {
    let mut sections: Vec<(&str, Vec<(String, String)>)> = Vec::new();
    let mut keys: Vec<&Attr> = r.owned.iter().chain(&r.referenced).collect();
    keys.sort_by(|a, b| a.key.cmp(&b.key));
    sections
        .push(("Attribute keys", keys.iter().map(|a| (constant(&a.key), a.key.clone())).collect()));
    let mut named = |title, prefix: &str, names: Vec<&String>| {
        let mut names = names;
        names.sort();
        names.dedup();
        sections.push((
            title,
            names.into_iter().map(|n| (format!("{prefix}{}", constant(n)), n.clone())).collect(),
        ));
    };
    named("Spans", "SPAN_", r.spans.iter().map(|x| &x.name).collect());
    named("Span events", "EVENT_", r.events.iter().map(|x| &x.name).collect());
    named("Metric instruments", "METRIC_", r.metrics.iter().map(|x| &x.name).collect());
    named("Report record scopes", "LOG_SCOPE_", r.records.iter().map(|x| &x.scope).collect());
    named("Report record bodies", "LOG_BODY_", r.records.iter().map(|x| &x.body).collect());
    let mut out = format!(
        "# GENERATED FILE -- DO NOT EDIT.\n# From the telemetry declarations in {SOURCE}\n\n\"\"\"Telemetry names: attribute keys, span, event and metric names, report record scopes.\"\"\"\n\nfrom __future__ import annotations\n\nfrom typing import Final\n"
    );
    let mut seen = std::collections::BTreeMap::new();
    for (title, constants) in sections {
        out.push_str(&format!("\n# {title}\n"));
        for (name, value) in constants {
            if let Some(other) = seen.insert(name.clone(), value.clone()) {
                return Err(format!("`{value}` and `{other}` are both `{name}` in semconv.py"));
            }
            out.push_str(&format!("{name}: Final = {}\n", quote(&value)));
        }
    }
    Ok(out)
}
