//! `_resolve` and `_default_tool_result_resolver`: a `CallToolResult` (wire JSON) to the string the
//! model reads, byte for byte what the reference gets from the SDK's pydantic models (mcp 1.29):
//! one block is `model_dump_json()`, several `json.dumps([b.model_dump() ...])`, and an error joins
//! `part.text` or `str(part)` with newlines. Hence three writers over each model's declared fields:
//! unset fields `null`, `_meta` dumped as `meta`, extra keys last, URIs normalised as `AnyUrl`.

use serde_json::Value;

use super::McpError;

/// `_resolve` then the default resolver.
#[lk_agents_abi::host(export(name = "mcp_render_result", pid = false))]
pub fn render_result(tool_name: &str, result: &Value) -> Result<String, McpError> {
    check_result(result)?;
    match content(result) {
        [] => Err(McpError::EmptyResult(tool_name.to_owned())),
        [one] => Ok(write(&shape(one), Style::Pydantic)),
        many => Ok(write(&P::List(many.iter().map(shape).collect()), Style::Json)),
    }
}

/// `_resolve`'s `isError` arm, checked before the content count.
#[lk_agents_abi::host(export(name = "mcp_check_result", pid = false))]
pub fn check_result(result: &Value) -> Result<(), McpError> {
    if !result["isError"].as_bool().unwrap_or(false) {
        return Ok(());
    }
    let part = |b: &Value| match (&b["text"], shape(b)) {
        (Value::String(t), _) => t.clone(),
        // `str(model)`: the repr's fields, space-separated.
        (_, P::Model(_, fs)) => fs
            .iter()
            .map(|(k, v)| format!("{k}={}", write(v, Style::Repr)))
            .collect::<Vec<_>>()
            .join(" "),
        (_, other) => write(&other, Style::Repr),
    };
    Err(McpError::ToolFailed(content(result).iter().map(part).collect::<Vec<_>>().join("\n")))
}

fn content(result: &Value) -> &[Value] {
    result["content"].as_array().map_or(&[], Vec::as_slice)
}

enum P<'a> {
    V(&'a Value),
    Null,
    Url(String),
    List(Vec<P<'a>>),
    Model(&'static str, Vec<(String, P<'a>)>),
}

/// A model: its name and fields in order; a field's second item names a nested model ("" plain).
type Class = (&'static str, &'static [(&'static str, &'static str)]);
const MEDIA: &[(&str, &str)] =
    &[("type", ""), ("data", ""), ("mimeType", ""), ("annotations", "Annotations"), ("meta", "")];
const CLASSES: &[Class] = &[
    ("text", &[("type", ""), ("text", ""), ("annotations", "Annotations"), ("meta", "")]),
    ("image", MEDIA),
    ("audio", MEDIA),
    (
        "resource_link",
        &[
            ("name", ""),
            ("title", ""),
            ("uri", "url"),
            ("description", ""),
            ("mimeType", ""),
            ("size", ""),
            ("icons", "[Icon]"),
            ("annotations", "Annotations"),
            ("meta", ""),
            ("type", ""),
        ],
    ),
    (
        "resource",
        &[("type", ""), ("resource", "resource"), ("annotations", "Annotations"), ("meta", "")],
    ),
    ("TextResourceContents", &[("uri", "url"), ("mimeType", ""), ("meta", ""), ("text", "")]),
    ("BlobResourceContents", &[("uri", "url"), ("mimeType", ""), ("meta", ""), ("blob", "")]),
    ("Annotations", &[("audience", ""), ("priority", "")]),
    ("Icon", &[("src", ""), ("mimeType", ""), ("sizes", "")]),
];
const NAMES: &[(&str, &str)] = &[
    ("text", "TextContent"),
    ("image", "ImageContent"),
    ("audio", "AudioContent"),
    ("resource_link", "ResourceLink"),
    ("resource", "EmbeddedResource"),
];

/// A content block as its `mcp.types` model; an unknown `type` stays plain JSON.
fn shape(v: &Value) -> P<'_> {
    match v["type"].as_str().and_then(|t| NAMES.iter().find(|(k, _)| *k == t)) {
        Some(&(key, name)) => model(key, name, v),
        None => P::V(v),
    }
}

fn model<'a>(key: &str, name: &'static str, v: &'a Value) -> P<'a> {
    let (Some(obj), Some((_, fields))) = (v.as_object(), CLASSES.iter().find(|(k, _)| *k == key))
    else {
        return P::V(v);
    };
    let mut out: Vec<(String, P<'a>)> = fields
        .iter()
        .map(|&(f, kind)| {
            let p = match (
                obj.get(if f == "meta" { "_meta" } else { f }).filter(|x| !x.is_null()),
                kind,
            ) {
                (None, _) => P::Null,
                (Some(x), "") => P::V(x),
                (Some(x), "url") => {
                    let s = x.as_str().unwrap_or_default();
                    P::Url(url::Url::parse(s).map_or_else(|_| s.to_owned(), |u| u.to_string()))
                }
                (Some(x), "[Icon]") => P::List(
                    x.as_array().into_iter().flatten().map(|i| model("Icon", "Icon", i)).collect(),
                ),
                (Some(x), "resource") => {
                    let c = ["BlobResourceContents", "TextResourceContents"]
                        [x.get("text").is_some() as usize];
                    model(c, c, x)
                }
                (Some(x), nested) => model(nested, nested, x),
            };
            (f.to_owned(), p)
        })
        .collect();
    let known = |k: &str| k == "_meta" || fields.iter().any(|(f, _)| *f == k);
    out.extend(obj.iter().filter(|(k, _)| !known(k)).map(|(k, x)| (k.clone(), P::V(x))));
    P::Model(name, out)
}

/// pydantic's `model_dump_json`, Python's `json.dumps`, or a pydantic `repr`.
#[derive(Clone, Copy, PartialEq)]
enum Style {
    Pydantic,
    Json,
    Repr,
}

fn write(p: &P, s: Style) -> String {
    let mut out = String::new();
    put(p, s, &mut out);
    out
}

fn put(p: &P, s: Style, out: &mut String) {
    let (sep, colon) = if s == Style::Pydantic { (",", ":") } else { (", ", ": ") };
    let entries = |out: &mut String, open: &str, items: Vec<(Option<String>, &P)>, close: &str| {
        out.push_str(open);
        for (i, (k, x)) in items.into_iter().enumerate() {
            out.push_str(if i > 0 { sep } else { "" });
            if let Some(k) = k {
                out.push_str(&k);
            }
            put(x, s, out);
        }
        out.push_str(close);
    };
    let key = |k: &str| Some(format!("{}{colon}", string(k, s)));
    match p {
        P::Null | P::V(Value::Null) => out.push_str(if s == Style::Repr { "None" } else { "null" }),
        P::V(Value::Bool(b)) => out.push_str(match (s == Style::Repr, b) {
            (true, true) => "True",
            (true, false) => "False",
            (false, true) => "true",
            (false, false) => "false",
        }),
        P::V(Value::Number(n)) if n.is_f64() => {
            out.push_str(&float(n.as_f64().unwrap_or_default(), s))
        }
        P::V(Value::Number(n)) => out.push_str(&n.to_string()),
        P::V(Value::String(x)) => out.push_str(&string(x, s)),
        P::Url(u) if s == Style::Repr => out.push_str(&format!("AnyUrl({})", string(u, s))),
        P::Url(u) => out.push_str(&string(u, s)),
        P::V(Value::Array(xs)) => put(&P::List(xs.iter().map(P::V).collect()), s, out),
        P::List(xs) => entries(out, "[", xs.iter().map(|x| (None, x)).collect(), "]"),
        P::V(Value::Object(m)) => {
            let vals: Vec<P> = m.values().map(P::V).collect();
            entries(out, "{", m.keys().map(|k| key(k)).zip(&vals).collect(), "}")
        }
        P::Model(name, fs) if s == Style::Repr => entries(
            out,
            &format!("{name}("),
            fs.iter().map(|(k, x)| (Some(format!("{k}=")), x)).collect(),
            ")",
        ),
        P::Model(_, fs) => entries(out, "{", fs.iter().map(|(k, x)| (key(k), x)).collect(), "}"),
    }
}

/// Positional for exponents -5..16 (pydantic) or -4..16 (Python), else the shortest digits with a
/// signed exponent, zero-padded to two digits by Python.
fn float(f: f64, s: Style) -> String {
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let py = s != Style::Pydantic;
    if !((if py { -4 } else { -5 })..16).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:0w$}", exp.abs(), w = if py { 2 } else { 1 });
    }
    let fixed = f.to_string();
    if fixed.contains('.') { fixed } else { fixed + ".0" }
}

/// serde_json's escaping is pydantic's; `json.dumps` also escapes everything past `~`; `repr`
/// quotes and escapes like Python.
fn string(x: &str, s: Style) -> String {
    let json = || serde_json::to_string(x).unwrap_or_default();
    match s {
        Style::Pydantic => json(),
        Style::Json => json().chars().fold(String::new(), |mut out, c| {
            if c > '~' {
                c.encode_utf16(&mut [0; 2])
                    .iter()
                    .for_each(|u| out.push_str(&format!("\\u{u:04x}")));
            } else {
                out.push(c);
            }
            out
        }),
        Style::Repr => {
            let q = if x.contains('\'') && !x.contains('"') { '"' } else { '\'' };
            let mut out = String::from(q);
            for c in x.chars() {
                match (c, c as u32) {
                    ('\\', _) => out.push_str("\\\\"),
                    ('\n', _) => out.push_str("\\n"),
                    ('\r', _) => out.push_str("\\r"),
                    ('\t', _) => out.push_str("\\t"),
                    _ if c == q => out.extend(['\\', c]),
                    _ if printable(c) => out.push(c),
                    (_, n @ ..=0xff) => out.push_str(&format!("\\x{n:02x}")),
                    (_, n @ ..=0xffff) => out.push_str(&format!("\\u{n:04x}")),
                    (_, n) => out.push_str(&format!("\\U{n:08x}")),
                }
            }
            out.push(q);
            out
        }
    }
}

/// `str.isprintable` for controls, format characters, separators other than space and private use
/// (unassigned code points are not tracked).
fn printable(c: char) -> bool {
    !matches!(c as u32, 0..=0x1f | 0x7f..=0xa0 | 0xad | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x1680 | 0x180e
        | 0x2000..=0x200f | 0x2028..=0x202f | 0x205f..=0x206f | 0x3000 | 0xe000..=0xf8ff | 0xfeff | 0xfff9..=0xfffb
        | 0xe0000..=0xe0fff | 0xf0000..)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Catches: serde's shape or text standing in for pydantic's and Python's -- omitted unset
    // fields, `_meta` under its alias, dropped extras, one escaping or float notation for all
    // three writers -- and `isError` checked after the empty case. Expected strings are the
    // reference's output (mcp 1.29, pydantic 2.13).
    #[test]
    fn results_render_as_the_reference_does() {
        let text = json!({"type": "text", "text": "héllo \"q\"\u{7}", "_meta": {"k": 1}, "x": [1e-7, 1e20, 3.0, 1e-5]});
        let r = |c: Value| render_result("t", &json!({ "content": c }));
        assert_eq!(
            r(json!([text.clone()])).unwrap(),
            r#"{"type":"text","text":"héllo \"q\"\u0007","annotations":null,"meta":{"k":1},"x":[1e-7,1e+20,3.0,0.00001]}"#
        );
        assert_eq!(
            r(json!([text, {"type": "text", "text": "😀"}])).unwrap(),
            r#"[{"type": "text", "text": "h\u00e9llo \"q\"\u0007", "annotations": null, "meta": {"k": 1}, "x": [1e-07, 1e+20, 3.0, 1e-05]}, {"type": "text", "text": "\ud83d\ude00", "annotations": null, "meta": null}]"#
        );
        assert_eq!(
            r(json!([{"type": "resource_link", "name": "n", "uri": "https://h"}])).unwrap(),
            r#"{"name":"n","title":null,"uri":"https://h/","description":null,"mimeType":null,"size":null,"icons":null,"annotations":null,"meta":null,"type":"resource_link"}"#
        );
        assert_eq!(r(json!([])), Err(McpError::EmptyResult("t".into())));

        let err =
            |c: Value| render_result("t", &json!({"isError": true, "content": c})).unwrap_err();
        assert_eq!(err(json!([])), McpError::ToolFailed(String::new()));
        let image = json!({"type": "image", "data": "it's", "mimeType": "a\"b'\u{a0}", "annotations": {"audience": ["user"], "priority": 0.5},
                           "_meta": {"k": [1, 2.0, null, true]}});
        let embedded = json!({"type": "resource", "resource": {"uri": "file:///x", "text": "hi"}});
        assert_eq!(
            err(json!([{"type": "text", "text": "bad"}, image, embedded])).to_string(),
            "bad\ntype='image' data=\"it's\" mimeType='a\"b\\'\\xa0' annotations=Annotations(audience=['user'], priority=0.5) meta={'k': [1, 2.0, None, True]}\n\
             type='resource' resource=TextResourceContents(uri=AnyUrl('file:///x'), mimeType=None, meta=None, text='hi') annotations=None meta=None"
        );
    }
}
