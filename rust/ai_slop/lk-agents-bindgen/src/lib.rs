//! `lk-agents-bindgen`: the `#[host(..)]` attribute language, read once for every reader. The proc
//! macros (`lk-agents-host-derive`) and the generator (`lk-agents-codegen`) both read it through
//! [`meta`], so the two cannot read one attribute two ways.

mod error;
pub mod meta;
mod record;
#[cfg(test)]
mod tests;

pub use error::{HostException, ToHost, host_exception, host_kind, raised};
pub use record::RecordDefaults;

pub use meta::{Meta, Metas, Value, doc, is_host, to_snake};
use proc_macro2::TokenStream;

pub const DEFAULT_MODULE: &str = "livekit.agents._core";

/// The host's `__doc__`: the first paragraph, intra-doc links unlinked.
pub fn summary(doc: &[String]) -> Vec<String> {
    let end = doc.iter().position(|l| l.is_empty()).unwrap_or(doc.len());
    doc[..end].iter().map(|l| unlink(l)).collect()
}

/// `` [`x`] `` -> `` `x` ``; a `[text](url)` link is kept.
fn unlink(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(i) = rest.find("[`") {
        let Some(j) = rest[i..].find("`]") else { break };
        let after = &rest[i + j + 2..];
        out.push_str(&rest[..i]);
        if after.starts_with('(') || after.starts_with('[') {
            out.push_str(&rest[i..i + j + 2]);
        } else {
            out.push_str(&rest[i + 1..i + j + 1]);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

pub const PARAM_MARKERS: [&str; 4] = ["label", "lenient", "kwargs", "required"];

pub struct Shape<'a> {
    pub name: &'a str,
    pub option: bool,
    /// Defaults to `NotGiven` unwritten.
    pub maybe: bool,
    pub required: bool,
    /// Has a `defaults(..)` entry.
    pub written: bool,
}

/// `Option<T>` parameters not `required` whose followers all default: `None` unwritten (Python's
/// rule). Read by the codegen's stub.
pub fn implied_none<'a>(host: &[Shape<'a>]) -> Vec<&'a str> {
    let defaulted = |p: &Shape| p.written || p.maybe || (p.option && !p.required);
    let mut out = Vec::new();
    for (i, p) in host.iter().enumerate() {
        if p.option && !p.required && !p.written && host[i + 1..].iter().all(defaulted) {
            out.push(p.name);
        }
    }
    out
}

/// `#[host(method(send = <expr>))]` on a body-less method (`pub fn f(&self, ..) -> R;`): the body
/// is `self.send(<expr>)` written for it, the expression built from the parameters. Returning
/// `Result<_, E>` the refusal is returned (`?`, converted by `From`); returning nothing it is
/// dropped. Run on every `#[host(object)]` impl before anything else reads it -- the expansion,
/// the strip and the codegen -- so each sees an ordinary method. A body-less fn without `send`, and
/// `send` beside a written body, are refused.
pub fn fill_sends(imp: &mut syn::ItemImpl) -> syn::Result<()> {
    struct Bodiless {
        attrs: Vec<syn::Attribute>,
        vis: syn::Visibility,
        sig: syn::Signature,
    }
    impl syn::parse::Parse for Bodiless {
        fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
            let attrs = input.call(syn::Attribute::parse_outer)?;
            let vis = input.parse()?;
            let sig = input.parse()?;
            input.parse::<syn::Token![;]>()?;
            Ok(Bodiless { attrs, vis, sig })
        }
    }
    let send_of = |attrs: &[syn::Attribute]| -> syn::Result<Option<TokenStream>> {
        if !attrs.iter().any(is_host) {
            return Ok(None);
        }
        Metas::of(attrs)?.expr("method.send")
    };
    for item in &mut imp.items {
        match item {
            syn::ImplItem::Fn(f) => {
                if send_of(&f.attrs)?.is_some() {
                    return Err(syn::Error::new_spanned(
                        &f.sig.ident,
                        "`send` writes the body: declare the method without one (`fn f(..) -> R;`)",
                    ));
                }
            }
            syn::ImplItem::Verbatim(tokens) => {
                let Ok(Bodiless { mut attrs, vis, sig }) = syn::parse2::<Bodiless>(tokens.clone())
                else {
                    continue;
                };
                let Some(send) = send_of(&attrs)? else {
                    return Err(syn::Error::new_spanned(
                        &sig.ident,
                        "a method without a body needs #[host(method(send = <expr>))], which writes it",
                    ));
                };
                if !matches!(sig.inputs.first(), Some(syn::FnArg::Receiver(r)) if r.reference.is_some())
                {
                    return Err(syn::Error::new_spanned(
                        &sig.ident,
                        "`send` calls `self.send(..)`: the method takes `&self`",
                    ));
                }
                let block: syn::Block = match &sig.output {
                    syn::ReturnType::Default => syn::parse_quote!({ let _ = self.send(#send); }),
                    syn::ReturnType::Type(_, t) if result_ok(t).is_some() => {
                        // `?` converts the refusal; it is the identity when the two types agree.
                        attrs.push(syn::parse_quote!(#[allow(clippy::needless_question_mark)]));
                        syn::parse_quote!({ ::core::result::Result::Ok(self.send(#send)?) })
                    }
                    syn::ReturnType::Type(..) => {
                        return Err(syn::Error::new_spanned(
                            &sig.output,
                            "a `send` method returns nothing (the refusal dropped) or a `Result` (the refusal returned)",
                        ));
                    }
                };
                *item = syn::ImplItem::Fn(syn::ImplItemFn {
                    attrs,
                    vis,
                    defaultness: None,
                    sig,
                    block,
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn result_ok(t: &syn::Type) -> Option<syn::Type> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    a.args.iter().find_map(|g| match g {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    })
}
