//! The registry's outputs: the Weaver registry (`otel/`), the consumer schema (`otel/schema.json`),
//! the Python constants (`_generated/semconv.py`) and the trace rules (`tests/trace_schema.py`).

use super::{Attr, Content, Encoding, Parent, Registry, Wire};
use lk_agents_bindgen::telemetry::{InstrumentKind, SpanKind, Unmeasured};

/// livekit/protocol's commit whose `otel/model/shared` the shared keys are read from (PR #1866).
pub const PROTOCOL_REF: &str = "424a912b1db6878a43a9cbea3cd71cb92e4ad712";
/// OpenTelemetry semantic conventions.
const SEMCONV_REF: &str = "v1.44.0";
/// The GenAI registry has no tags; it is pinned by commit, as protocol pins it.
const GENAI_REF: &str = "547934afa90a34298441caf80c43b27255c3a292";

const SOURCE: &str = "rust/crates/lk-agents-abi/src/telemetry/";

/// A JSON or YAML value.
enum V {
    S(String),
    N(i64),
    B(bool),
    Null,
    A(Vec<V>),
    O(Vec<(&'static str, V)>),
}

fn s(text: impl Into<String>) -> V {
    V::S(text.into())
}

fn strings<'a>(items: impl IntoIterator<Item = &'a String>) -> V {
    V::A(items.into_iter().map(|i| s(i.as_str())).collect())
}

fn json(v: &V, indent: usize, out: &mut String) {
    let pad = |n: usize| "  ".repeat(n);
    match v {
        V::S(text) => out.push_str(&quote(text)),
        V::N(n) => out.push_str(&n.to_string()),
        V::B(b) => out.push_str(&b.to_string()),
        V::Null => out.push_str("null"),
        V::A(items) if items.is_empty() => out.push_str("[]"),
        V::A(items) if items.iter().all(|i| matches!(i, V::S(_) | V::N(_))) => {
            let mut parts = Vec::new();
            for i in items {
                let mut part = String::new();
                json(i, 0, &mut part);
                parts.push(part);
            }
            out.push_str(&format!("[{}]", parts.join(", ")));
        }
        V::A(items) => {
            out.push_str("[\n");
            for (n, i) in items.iter().enumerate() {
                out.push_str(&pad(indent + 1));
                json(i, indent + 1, out);
                out.push_str(if n + 1 < items.len() { ",\n" } else { "\n" });
            }
            out.push_str(&format!("{}]", pad(indent)));
        }
        V::O(fields) if fields.is_empty() => out.push_str("{}"),
        V::O(fields) => {
            out.push_str("{\n");
            for (n, (k, f)) in fields.iter().enumerate() {
                out.push_str(&format!("{}{}: ", pad(indent + 1), quote(k)));
                json(f, indent + 1, out);
                out.push_str(if n + 1 < fields.len() { ",\n" } else { "\n" });
            }
            out.push_str(&format!("{}}}", pad(indent)));
        }
    }
}

/// Block-style YAML; every string is double-quoted (a JSON string is a YAML string).
fn yaml(fields: &[(&'static str, V)], indent: usize, out: &mut String) {
    for (n, (k, v)) in fields.iter().enumerate() {
        // The first field of a sequence item follows its `- `.
        let pad = if n == 0 && indent > 0 && out.ends_with("- ") {
            String::new()
        } else {
            " ".repeat(indent)
        };
        out.push_str(&format!("{pad}{k}:"));
        match v {
            V::O(inner) if !inner.is_empty() => {
                out.push('\n');
                yaml(inner, indent + 2, out);
            }
            V::A(items) if !items.is_empty() => {
                out.push('\n');
                for item in items {
                    out.push_str(&format!("{}- ", " ".repeat(indent + 2)));
                    match item {
                        V::O(inner) => yaml(inner, indent + 4, out),
                        scalar => {
                            json(scalar, 0, out);
                            out.push('\n');
                        }
                    }
                }
            }
            scalar => {
                out.push(' ');
                json(scalar, 0, out);
                out.push('\n');
            }
        }
    }
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

fn parent_name(p: &Parent) -> String {
    match p {
        Parent::Span(name) => name.clone(),
        Parent::Current => "<current>".into(),
        Parent::External => "<external>".into(),
    }
}

// --- otel/schema.json ---------------------------------------------------------------------------

fn schema_attr(a: &Attr) -> V {
    let mut f =
        vec![("key", s(&a.key)), ("type", s(wire_name(&a.wire))), ("required", V::B(a.required))];
    if a.registry != "agents" {
        f.push(("registry", s(&a.registry)));
    }
    if a.seconds {
        f.push(("unit", s("s")));
    }
    match &a.encoding {
        Some(Encoding::Json) => f.push(("encoding", s("json"))),
        Some(Encoding::BoolString) => f.push(("encoding", s("bool_string"))),
        Some(Encoding::Enum(members)) => f.push(("members", strings(members))),
        None => {}
    }
    match a.unmeasured {
        Some(Unmeasured::Omit) => f.push(("unmeasured", s("omit"))),
        Some(Unmeasured::Zero) => f.push(("unmeasured", V::N(0))),
        Some(Unmeasured::MinusOne) => f.push(("unmeasured", V::N(-1))),
        None => {}
    }
    match a.content {
        Some(Content::Pii) => f.push(("content", s("pii"))),
        Some(Content::GenAi) => f.push(("content", s("gen_ai"))),
        None => {}
    }
    if a.template {
        f.push(("template", V::B(true)));
    }
    match &a.label {
        Some(key) if *key == a.key => f.push(("label", V::B(true))),
        Some(key) => f.push(("label", s(key))),
        None => {}
    }
    if let Some(metric) = &a.metric {
        f.push(("metric", s(metric)));
    }
    V::O(f)
}

fn schema_attrs(attrs: &[Attr]) -> V {
    V::A(attrs.iter().map(schema_attr).collect())
}

fn schema(r: &Registry) -> String {
    let mut spans: Vec<_> = r.spans.iter().collect();
    spans.sort_by(|a, b| a.name.cmp(&b.name));
    let mut events: Vec<_> = r.events.iter().collect();
    events.sort_by(|a, b| a.name.cmp(&b.name));
    let mut metrics: Vec<_> = r.metrics.iter().collect();
    metrics.sort_by(|a, b| a.name.cmp(&b.name));
    let doc = V::O(vec![
        (
            "$comment",
            s(format!(
                "GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from {SOURCE}."
            )),
        ),
        (
            "spans",
            V::A(
                spans
                    .iter()
                    .map(|span| {
                        V::O(vec![
                            ("name", s(&span.name)),
                            ("brief", s(&span.brief)),
                            ("kind", s(kind_name(span.kind))),
                            (
                                "parents",
                                V::A(span.parents.iter().map(|p| s(parent_name(p))).collect()),
                            ),
                            ("outlives_parent", span.outlives_parent.as_ref().map_or(V::Null, s)),
                            ("start", schema_attrs(&span.start)),
                            ("end", schema_attrs(&span.end)),
                            ("events", strings(&span.events)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "events",
            V::A(
                events
                    .iter()
                    .map(|e| {
                        V::O(vec![
                            ("name", s(&e.name)),
                            ("brief", s(&e.brief)),
                            ("attributes", schema_attrs(&e.attrs)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "metrics",
            V::A(
                metrics
                    .iter()
                    .map(|m| {
                        V::O(vec![
                            ("name", s(&m.name)),
                            ("brief", s(&m.brief)),
                            ("kind", s(instrument_name(m.kind))),
                            ("unit", s(&m.unit)),
                            ("registry", s(if m.upstream { "otel" } else { "agents" })),
                            ("labels", schema_attrs(&m.labels)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "records",
            V::A(
                r.records
                    .iter()
                    .map(|rec| {
                        V::O(vec![
                            ("scope", s(&rec.scope)),
                            ("body", s(&rec.body)),
                            ("brief", s(&rec.brief)),
                            ("span_id", V::B(rec.span_id)),
                            ("attributes", schema_attrs(&rec.attrs)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "groups",
            V::A(
                r.groups
                    .iter()
                    .map(|g| {
                        V::O(vec![
                            ("name", s(&g.name)),
                            ("brief", s(&g.brief)),
                            ("attributes", schema_attrs(&g.attrs)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    let mut out = String::new();
    json(&doc, 0, &mut out);
    out.push('\n');
    out
}

// --- otel/manifest.yaml and otel/registry/ ------------------------------------------------------

fn header() -> String {
    format!(
        "# GENERATED FILE -- DO NOT EDIT.\n# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from {SOURCE}.\n\n"
    )
}

fn yaml_file(fields: Vec<(&'static str, V)>) -> String {
    let mut out = header();
    yaml(&fields, 0, &mut out);
    out
}

fn manifest() -> String {
    let dep = |schema: &str, path: String| {
        V::O(vec![("schema_url", s(schema)), ("registry_path", s(path))])
    };
    yaml_file(vec![
        ("file_format", s("definition_manifest/2.0")),
        ("schema_url", s("https://github.com/livekit/agents/tree/main/otel")),
        ("stability", s("development")),
        (
            "description",
            s(
                "LiveKit Agents telemetry: the framework's spans, events, metrics and report records.",
            ),
        ),
        (
            "dependencies",
            V::A(vec![
                dep(
                    "https://github.com/livekit/protocol/tree/main/otel",
                    format!(
                        "https://github.com/livekit/protocol.git@{PROTOCOL_REF}[otel/model/shared]"
                    ),
                ),
                dep(
                    "https://opentelemetry.io/schemas/1.44.0",
                    format!(
                        "https://github.com/open-telemetry/semantic-conventions.git@{SEMCONV_REF}[model]"
                    ),
                ),
                dep(
                    "https://opentelemetry.io/schemas/gen-ai-dev/1.42.0-dev",
                    format!(
                        "https://github.com/open-telemetry/semantic-conventions-genai.git@{GENAI_REF}[model]"
                    ),
                ),
            ]),
        ),
    ])
}

fn weaver_type(a: &Attr) -> V {
    match &a.encoding {
        Some(Encoding::Enum(members)) => V::O(vec![(
            "members",
            V::A(
                members
                    .iter()
                    .map(|m| {
                        V::O(vec![
                            ("id", s(m)),
                            ("value", s(m)),
                            ("brief", s(m)),
                            ("stability", s("development")),
                        ])
                    })
                    .collect(),
            ),
        )]),
        _ => {
            let base = match &a.wire {
                Wire::KvList => "any".to_owned(),
                Wire::Array(inner) if **inner == Wire::KvList => "any".to_owned(),
                wire => wire_name(wire),
            };
            s(if a.template { format!("template[{base}]") } else { base })
        }
    }
}

fn attributes_yaml(r: &Registry) -> String {
    let mut modules: Vec<&str> = r.owned.iter().map(|a| a.declared_in.as_str()).collect();
    modules.sort();
    modules.dedup();
    let groups = modules
        .iter()
        .map(|module| {
            let mut owned: Vec<_> = r.owned.iter().filter(|a| a.declared_in == *module).collect();
            owned.sort_by(|a, b| a.key.cmp(&b.key));
            let id = module.rsplit("::").next().unwrap_or(module);
            V::O(vec![
                ("id", s(format!("registry.livekit.agents.{id}"))),
                ("type", s("attribute_group")),
                ("brief", s(format!("Attributes declared in `lk_agents_abi::{module}`."))),
                ("attributes", V::A(owned.iter().map(|a| weaver_attr(a)).collect())),
            ])
        })
        .collect();
    yaml_file(vec![("groups", V::A(groups))])
}

fn weaver_attr(a: &Attr) -> V {
    let mut livekit = Vec::new();
    if a.seconds {
        livekit.push(("unit", s("s")));
    }
    if a.content == Some(Content::Pii) {
        livekit.push(("sensitivity", s("pii")));
    }
    match a.encoding {
        Some(Encoding::Json) => livekit.push(("encoding", s("json"))),
        Some(Encoding::BoolString) => livekit.push(("encoding", s("bool_string"))),
        _ => {}
    }
    let mut f = vec![
        ("id", s(&a.key)),
        ("type", weaver_type(a)),
        ("stability", s("development")),
        ("brief", s(&a.brief)),
    ];
    if !livekit.is_empty() {
        f.push(("annotations", V::O(vec![("livekit", V::O(livekit))])));
    }
    V::O(f)
}

/// Attribute references, once per key.
fn refs<'a>(attrs: impl IntoIterator<Item = &'a Attr>) -> V {
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for a in attrs {
        if seen.contains(&&a.key) {
            continue;
        }
        seen.push(&a.key);
        let level = if a.required { "required" } else { "recommended" };
        out.push(V::O(vec![("ref", s(&a.key)), ("requirement_level", s(level))]));
    }
    V::A(out)
}

fn spans_yaml(r: &Registry) -> String {
    let mut spans: Vec<_> = r.spans.iter().collect();
    spans.sort_by(|a, b| a.name.cmp(&b.name));
    let groups = spans
        .iter()
        .map(|span| {
            let mut livekit =
                vec![("parents", V::A(span.parents.iter().map(|p| s(parent_name(p))).collect()))];
            if let Some(reason) = &span.outlives_parent {
                livekit.push(("outlives_parent", s(reason)));
            }
            if !span.end.is_empty() {
                livekit
                    .push(("end_attributes", V::A(span.end.iter().map(|a| s(&a.key)).collect())));
            }
            if !span.events.is_empty() {
                livekit.push(("events", strings(&span.events)));
            }
            let mut f = vec![
                ("id", s(format!("span.{}", span.name))),
                ("type", s("span")),
                ("span_kind", s(kind_name(span.kind))),
                ("stability", s("development")),
                ("brief", s(&span.brief)),
            ];
            if !span.start.is_empty() || !span.end.is_empty() {
                f.push(("attributes", refs(span.start.iter().chain(&span.end))));
            }
            f.push(("annotations", V::O(vec![("livekit", V::O(livekit))])));
            V::O(f)
        })
        .collect();
    yaml_file(vec![("groups", V::A(groups))])
}

fn events_yaml(r: &Registry) -> String {
    let mut events: Vec<_> = r.events.iter().collect();
    events.sort_by(|a, b| a.name.cmp(&b.name));
    let groups = events
        .iter()
        .map(|e| {
            let mut f = vec![
                ("id", s(format!("event.{}", e.name))),
                ("type", s("event")),
                ("name", s(&e.name)),
                ("stability", s("development")),
                ("brief", s(&e.brief)),
            ];
            if !e.attrs.is_empty() {
                f.push(("attributes", refs(&e.attrs)));
            }
            V::O(f)
        })
        .collect();
    yaml_file(vec![("groups", V::A(groups))])
}

fn metrics_yaml(r: &Registry) -> String {
    let mut metrics: Vec<_> = r.metrics.iter().filter(|m| !m.upstream).collect();
    metrics.sort_by(|a, b| a.name.cmp(&b.name));
    let groups = metrics
        .iter()
        .map(|m| {
            V::O(vec![
                ("id", s(format!("metric.{}", m.name))),
                ("type", s("metric")),
                ("metric_name", s(&m.name)),
                ("instrument", s(instrument_name(m.kind))),
                ("unit", s(&m.unit)),
                ("stability", s("development")),
                ("brief", s(&m.brief)),
                ("attributes", refs(&m.labels)),
            ])
        })
        .collect();
    yaml_file(vec![("groups", V::A(groups))])
}

fn imports_yaml(r: &Registry) -> String {
    let mut names: Vec<&String> =
        r.metrics.iter().filter(|m| m.upstream).map(|m| &m.name).collect();
    names.sort();
    yaml_file(vec![
        ("file_format", s("definition/2")),
        ("imports", V::O(vec![("metrics", strings(names))])),
    ])
}

/// Every file under `otel/`, by repo-relative path.
pub fn otel(r: &Registry) -> Vec<(String, String)> {
    vec![
        ("otel/manifest.yaml".into(), manifest()),
        ("otel/registry/attributes.yaml".into(), attributes_yaml(r)),
        ("otel/registry/spans.yaml".into(), spans_yaml(r)),
        ("otel/registry/events.yaml".into(), events_yaml(r)),
        ("otel/registry/metrics.yaml".into(), metrics_yaml(r)),
        ("otel/registry/imports.yaml".into(), imports_yaml(r)),
        ("otel/schema.json".into(), schema(r)),
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

// --- tests/trace_schema.py ----------------------------------------------------------------------

const TRACE_SCHEMA: &str = include_str!("trace_schema.py.in");

fn py_parent(p: &Parent) -> String {
    match p {
        Parent::Span(name) => quote(name),
        Parent::Current => "ANY".into(),
        Parent::External => "ROOT".into(),
    }
}

/// The trace rules: the checker with its span parents and outliving edges from the declarations.
pub fn trace_schema(r: &Registry) -> String {
    let mut spans: Vec<_> = r.spans.iter().collect();
    spans.sort_by(|a, b| a.name.cmp(&b.name));
    let mut parents = String::new();
    let mut outlive = String::new();
    for span in &spans {
        let mut allowed: Vec<String> = span.parents.iter().map(py_parent).collect();
        allowed.sort();
        parents.push_str(&format!("        {}: {{{}}},\n", quote(&span.name), allowed.join(", ")));
        let Some(reason) = &span.outlives_parent else { continue };
        for p in &span.parents {
            let parent = match p {
                Parent::External => continue,
                p => py_parent(p),
            };
            let line = format!("    ({}, {parent}): {},", quote(&span.name), quote(reason));
            if line.len() <= 100 {
                outlive.push_str(&line);
                outlive.push('\n');
            } else {
                outlive.push_str(&format!(
                    "    (\n        {},\n        {parent},\n    ): {},\n",
                    quote(&span.name),
                    quote(reason)
                ));
            }
        }
    }
    TRACE_SCHEMA.replace("{span_parents}", &parents).replace("{may_outlive_parent}", &outlive)
}
