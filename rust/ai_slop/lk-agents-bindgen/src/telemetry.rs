//! The telemetry declaration language, read once for the proc macros and the generator: the role
//! attributes (`#[span]`, `#[event]`, `#[metric]`, `#[record]`) and the field attributes of
//! `#[derive(Attrs)]` (`#[attr]`, `#[attrs]`, `#[span_id]`).

use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream, Parser};

/// The role attributes, by name.
pub const ROLES: [&str; 4] = ["span", "event", "metric", "record"];

/// A key or a name: a literal owned by this registry, or a path to a constant of a dependency
/// registry (`semconv::otel::GEN_AI_REQUEST_MODEL`).
#[derive(Clone)]
pub enum Name {
    Lit(syn::LitStr),
    Path(syn::Path),
}

impl syn::parse::Parse for Name {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(syn::LitStr) {
            Ok(Name::Lit(input.parse()?))
        } else {
            Ok(Name::Path(input.parse()?))
        }
    }
}

impl quote::ToTokens for Name {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Name::Lit(lit) => lit.to_tokens(tokens),
            Name::Path(path) => path.to_tokens(tokens),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    Internal,
    Client,
    Server,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrumentKind {
    Counter,
    Histogram,
}

pub struct Span {
    pub name: syn::LitStr,
    pub parents: Vec<syn::Path>,
    pub end: Option<syn::Type>,
    pub events: Vec<syn::Path>,
    pub kind: SpanKind,
    pub outlives_parent: Option<syn::LitStr>,
    /// `handle = Manual`: an entity-owned `ManualSpan` of a sans-I/O crate, not a `ScopedSpan`.
    pub manual: bool,
}

pub struct Metric {
    pub name: Name,
    pub kind: InstrumentKind,
    pub unit: syn::LitStr,
}

pub enum Role {
    Span(Box<Span>),
    Event { name: syn::LitStr },
    Metric(Metric),
    Record { scope: syn::LitStr, body: syn::LitStr },
}

/// How a `Measured` attribute is written when its value could not be computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unmeasured {
    Omit,
    Zero,
    MinusOne,
}

pub struct Attr {
    pub key: Name,
    pub label: bool,
    /// `label = "key"`: the key the field takes as a metric label, when not its attribute key.
    pub label_key: Option<syn::LitStr>,
    pub metric: Option<syn::LitStr>,
    pub unmeasured: Unmeasured,
}

pub enum Field {
    Attr(Attr),
    /// `#[attrs]`: the field is an attribute group, written in place.
    Flatten,
    /// `#[span_id]`: a record's span, not an attribute.
    SpanId,
}

/// Reads the arguments of the role attribute `role` (one of [`ROLES`]).
pub fn role(role: &str, args: TokenStream) -> syn::Result<Role> {
    let parse = |input: ParseStream| -> syn::Result<Role> {
        let mut seen = Vec::<String>::new();
        let (mut name, mut parents, mut end, mut events) = (None, Vec::new(), None, Vec::new());
        let (mut kind, mut outlives, mut unit, mut scope, mut body) =
            (None, None, None, None, None);
        let mut handle: Option<syn::Ident> = None;
        let mut metric_name = None;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let k = key.to_string();
            if seen.contains(&k) {
                return Err(syn::Error::new_spanned(&key, format!("`{k}` is given twice")));
            }
            seen.push(k.clone());
            input.parse::<syn::Token![=]>()?;
            match (role, k.as_str()) {
                ("metric", "name") => metric_name = Some(input.parse::<Name>()?),
                ("span" | "event", "name") => name = Some(input.parse::<syn::LitStr>()?),
                ("span", "parent") => {
                    parents.push(input.parse::<syn::Path>()?);
                    while input.parse::<Option<syn::Token![|]>>()?.is_some() {
                        parents.push(input.parse()?);
                    }
                }
                ("span", "end") => end = Some(input.parse::<syn::Type>()?),
                ("span", "events") => {
                    let content;
                    syn::bracketed!(content in input);
                    events = content
                        .parse_terminated(syn::Path::parse, syn::Token![,])?
                        .into_iter()
                        .collect();
                }
                ("span" | "metric", "kind") => kind = Some(input.parse::<syn::Ident>()?),
                ("span", "outlives_parent") => outlives = Some(input.parse::<syn::LitStr>()?),
                ("span", "handle") => handle = Some(input.parse::<syn::Ident>()?),
                ("metric", "unit") => unit = Some(input.parse::<syn::LitStr>()?),
                ("record", "scope") => scope = Some(input.parse::<syn::LitStr>()?),
                ("record", "body") => body = Some(input.parse::<syn::LitStr>()?),
                _ => return Err(syn::Error::new_spanned(&key, usage(role))),
            }
            if !input.is_empty() {
                input.parse::<syn::Token![,]>()?;
            }
        }
        let missing = |what: &str| {
            syn::Error::new(input.span(), format!("#[{role}] needs `{what}`; {}", usage(role)))
        };
        Ok(match role {
            "span" => {
                if parents.is_empty() {
                    return Err(missing("parent"));
                }
                let kind = match kind.as_ref().map(|k| k.to_string()).as_deref() {
                    None | Some("Internal") => SpanKind::Internal,
                    Some("Client") => SpanKind::Client,
                    Some("Server") => SpanKind::Server,
                    Some(_) => {
                        return Err(syn::Error::new_spanned(
                            kind,
                            "a span kind is `Internal`, `Client` or `Server`",
                        ));
                    }
                };
                let manual = match handle.as_ref().map(|h| h.to_string()).as_deref() {
                    None | Some("Scoped") => false,
                    Some("Manual") => true,
                    Some(_) => {
                        return Err(syn::Error::new_spanned(
                            handle,
                            "a span handle is `Scoped` or `Manual`",
                        ));
                    }
                };
                Role::Span(Box::new(Span {
                    name: name.ok_or_else(|| missing("name"))?,
                    parents,
                    end,
                    events,
                    kind,
                    outlives_parent: outlives,
                    manual,
                }))
            }
            "event" => Role::Event { name: name.ok_or_else(|| missing("name"))? },
            "metric" => {
                let kind = match kind.as_ref().map(|k| k.to_string()).as_deref() {
                    Some("Counter") => InstrumentKind::Counter,
                    Some("Histogram") => InstrumentKind::Histogram,
                    Some(_) => {
                        return Err(syn::Error::new_spanned(
                            kind,
                            "a metric kind is `Counter` or `Histogram`",
                        ));
                    }
                    None => return Err(missing("kind")),
                };
                Role::Metric(Metric {
                    name: metric_name.ok_or_else(|| missing("name"))?,
                    kind,
                    unit: unit.ok_or_else(|| missing("unit"))?,
                })
            }
            "record" => Role::Record {
                scope: scope.ok_or_else(|| missing("scope"))?,
                body: body.ok_or_else(|| missing("body"))?,
            },
            _ => {
                return Err(syn::Error::new(
                    input.span(),
                    format!("`{role}` is not a telemetry role"),
                ));
            }
        })
    };
    parse.parse2(args)
}

fn usage(role: &str) -> &'static str {
    match role {
        "span" => {
            "#[span(name = \"..\", parent = A | B, end = T, events = [E, ..], kind = Internal | Client | Server, outlives_parent = \"reason\", handle = Scoped | Manual)]"
        }
        "event" => "#[event(name = \"..\")]",
        "metric" => {
            "#[metric(name = \"..\" or a semconv path, kind = Counter | Histogram, unit = \"..\")]"
        }
        _ => "#[record(scope = \"..\", body = \"..\")]",
    }
}

/// The one telemetry attribute of a field of a `#[derive(Attrs)]` struct.
pub fn field(field: &syn::Field) -> syn::Result<Field> {
    let mut found = None;
    for a in &field.attrs {
        let kind = if a.path().is_ident("attr") {
            Field::Attr(a.parse_args_with(attr)?)
        } else if a.path().is_ident("attrs") {
            a.meta.require_path_only()?;
            Field::Flatten
        } else if a.path().is_ident("span_id") {
            a.meta.require_path_only()?;
            Field::SpanId
        } else {
            continue;
        };
        if found.replace(kind).is_some() {
            return Err(syn::Error::new_spanned(
                a,
                "a field takes one of `#[attr(..)]`, `#[attrs]` and `#[span_id]`",
            ));
        }
    }
    found.ok_or_else(|| {
        syn::Error::new_spanned(field, "every field of an `Attrs` struct is an `#[attr(\"key\")]`, an `#[attrs]` group or a record's `#[span_id]`")
    })
}

fn attr(input: ParseStream) -> syn::Result<Attr> {
    let key = input.parse::<Name>()?;
    let (mut label, mut label_key, mut metric, mut unmeasured) =
        (false, None, None, Unmeasured::Omit);
    while input.parse::<Option<syn::Token![,]>>()?.is_some() {
        if input.is_empty() {
            break;
        }
        let word: syn::Ident = input.parse()?;
        match word.to_string().as_str() {
            "label" => {
                label = true;
                if input.parse::<Option<syn::Token![=]>>()?.is_some() {
                    label_key = Some(input.parse::<syn::LitStr>()?);
                }
            }
            "metric" => {
                input.parse::<syn::Token![=]>()?;
                metric = Some(input.parse::<syn::LitStr>()?);
            }
            "unmeasured" => {
                input.parse::<syn::Token![=]>()?;
                let negative = input.parse::<Option<syn::Token![-]>>()?.is_some();
                let n: syn::LitInt = input.parse()?;
                unmeasured = match (negative, n.base10_digits()) {
                    (false, "0") => Unmeasured::Zero,
                    (true, "1") => Unmeasured::MinusOne,
                    _ => {
                        return Err(syn::Error::new_spanned(
                            n,
                            "`unmeasured` is `0` or `-1`; leave it out to omit the attribute",
                        ));
                    }
                };
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    word,
                    "#[attr(\"key\", label [= \"label key\"], metric = \"instrument\", unmeasured = 0 | -1)]",
                ));
            }
        }
    }
    Ok(Attr { key, label, label_key, metric, unmeasured })
}
