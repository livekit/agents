//! The declared errors' host exceptions and the exported fns and consts: what `_core` raises and
//! holds beside the declared classes.

use super::super::tree::{Crate, Found};
use super::super::{DeclaredError, Export};
use super::voc;
use super::{Resolver, item_attrs};

/// The `#[host(error ..)]` attribute's arguments, unless it is a refusal the host raises.
pub(super) fn error_entry(item: &syn::Item) -> Option<proc_macro2::TokenStream> {
    let (syn::Item::Struct(_) | syn::Item::Enum(_)) = item else { return None };
    host_args(item, "error").filter(|args| {
        !lk_agents_bindgen::Metas::parse(args.clone()).is_ok_and(|m| lk_agents_bindgen::raised(&m))
    })
}

pub(super) fn export_entry(item: &syn::Item) -> Option<proc_macro2::TokenStream> {
    host_args(item, "export")
}

/// The arguments of the item's `#[host(<head> ..)]`.
fn host_args(item: &syn::Item, head: &str) -> Option<proc_macro2::TokenStream> {
    item_attrs(item).iter().filter(|a| voc::is_host(a.path())).find_map(|a| {
        let tokens = a.meta.require_list().ok()?.tokens.clone();
        let first = tokens.clone().into_iter().next()?;
        (first.to_string() == head).then_some(tokens)
    })
}

/// The error's host exception, formatted, from the one reader of the error vocabulary. The enum a
/// `by = field` selects on is named as the binding names it.
pub(super) fn declared_error(
    krate: &Crate,
    module: usize,
    item: &syn::Item,
    args: proc_macro2::TokenStream,
    rust: &str,
    family: String,
) -> Result<DeclaredError, String> {
    let at = format!("{rust}: #[host({args})]");
    let path: syn::Path = syn::parse_str(rust).map_err(|e| format!("{at}: {e}"))?;
    let mut item = item.clone();
    let m = lk_agents_bindgen::Metas::parse(args.clone()).map_err(|e| format!("{at}: {e}"))?;
    if let Some(by) = m.ident("error.by").map_err(|e| format!("{at}: {e}"))?
        && let syn::Item::Struct(st) = &mut item
    {
        for field in st.fields.iter_mut().filter(|f| f.ident.as_ref() == Some(&by)) {
            let syn::Type::Path(ty) = &field.ty else { continue };
            if let Found::Item { module, item } = krate.resolve(module, &ty.path)? {
                let named = krate.extern_path(module, item)?;
                field.ty = syn::parse_str(&named).map_err(|e| format!("{at}: {e}"))?;
            }
        }
    }
    let resolver = Resolver { krate, module, at: &at, assoc: Vec::new() };
    let to_host = |ty: &syn::Type, value: proc_macro2::TokenStream| {
        let fail = |e: String| syn::Error::new(proc_macro2::Span::call_site(), e);
        let ty = resolver.ty(ty, &at).map_err(fail)?;
        let expr = crate::records::emit(&ty, &value.to_string()).map_err(fail)?;
        expr.parse::<proc_macro2::TokenStream>().map_err(|e| fail(e.to_string()))
    };
    let exception = lk_agents_bindgen::host_exception(args, &item, &path, &to_host)
        .map_err(|e| format!("{at}: {e}"))?;
    let capture = if let syn::Item::Enum(en) = &item {
        let mut captures = Vec::new();
        for variant in &en.variants {
            let entries = voc::read(&variant.attrs)?;
            if voc::find(&entries, "capture").is_none() {
                continue;
            }
            let syn::Fields::Unnamed(fields) = &variant.fields else {
                return Err(format!("{at}: a capture variant wraps one host error"));
            };
            if fields.unnamed.len() != 1 {
                return Err(format!("{at}: a capture variant wraps one host error"));
            }
            let field = resolver.ty(&fields.unnamed[0].ty, &at)?;
            let field = field.rust();
            let variant = &variant.ident;
            captures.push(format!(
                "impl crate::py_errors::FromHostErr for {rust} {{
                    fn from_host(py: ::pyo3::Python<'_>, err: &::pyo3::PyErr) -> Self {{
                        Self::{variant}(<{field} as crate::py_errors::FromHostErr>::from_host(py, err))
                    }}
                }}"
            ));
        }
        if captures.len() > 1 {
            return Err(format!("{at}: only one variant captures host failures"));
        }
        captures.concat()
    } else {
        String::new()
    };
    // The generated file is formatted whole.
    Ok(DeclaredError {
        family,
        binding: format!("{}\n{capture}", exception.tokens),
        declared: exception.declared,
        base: m.text("error.declare").map_err(|e| format!("{at}: {e}"))?.map(str::to_owned),
        doc: lk_agents_bindgen::doc(super::item_attrs(&item)),
    })
}

/// `export(name = "..", pid = false)` on a free fn taking nothing, or on a const.
pub(super) fn export_decl(
    krate: &Crate,
    module: usize,
    item: &syn::Item,
    args: proc_macro2::TokenStream,
    rust: String,
    family: String,
) -> Result<Export, String> {
    let at = format!("{rust}: #[host({args})]");
    let m = lk_agents_bindgen::Metas::parse(args).map_err(|e| format!("{at}: {e}"))?;
    m.only("export", &["export", "export.name", "export.pid"]).map_err(|e| format!("{at}: {e}"))?;
    let name = m.text("export.name").map_err(|e| format!("{at}: {e}"))?.map(str::to_owned);
    let pid = m.bool("export.pid").map_err(|e| format!("{at}: {e}"))?.unwrap_or(true);
    let (name, constant, attrs) = match item {
        syn::Item::Const(c) => {
            (name.unwrap_or_else(|| c.ident.to_string().to_lowercase()), true, &c.attrs)
        }
        syn::Item::Fn(f) => {
            if !f.sig.inputs.is_empty() || f.sig.asyncness.is_some() {
                return Err(format!(
                    "{at}: an exported fn takes nothing and is not async; declare a `#[host(functions)]` trait"
                ));
            }
            (name.unwrap_or_else(|| f.sig.ident.to_string()), false, &f.attrs)
        }
        _ => unreachable!("only fns and consts are exported"),
    };
    // An UPPER_CASE const name is a module attribute.
    let attr = constant
        && name.chars().any(|c| c.is_ascii_uppercase())
        && !name.chars().any(|c| c.is_ascii_lowercase());
    let resolver = Resolver { krate, module, at: &at, assoc: Vec::new() };
    let ty = match item {
        syn::Item::Const(item) => resolver.ty(&item.ty, &at)?,
        syn::Item::Fn(item) => match &item.sig.output {
            syn::ReturnType::Default => super::super::Ty::Unit,
            syn::ReturnType::Type(_, ty) => resolver.ty(ty, &at)?,
        },
        _ => unreachable!("checked above"),
    };
    Ok(Export { family, name, rust, doc: lk_agents_bindgen::doc(attrs), constant, attr, pid, ty })
}
