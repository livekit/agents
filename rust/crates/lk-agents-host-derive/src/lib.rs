//! `#[derive(Host)]` -- **a host record, declared once: every host's view of it follows.**
//!
//! Two readers, one declaration:
//!
//! * **The Python binding, in place.** The derive expands (`lk_agents_bindgen::record`), under the
//!   declaring crate's `python` feature, to the record's pyo3 shapes beside it: `binding(emit)` an
//!   `IntoPyObject` dict, `binding(extract)` a `FromPyObject` reader, `binding(class)` the getters,
//!   keyword constructor and `__repr__` of its `_core` class. Without the feature it expands to
//!   nothing.
//! * **The generator** (`lk-agents-codegen`) parses the source with `syn` for what a proc macro
//!   cannot see -- source order, doc comments, `// ---` sections -- and writes the pydantic modules
//!   (`#[python(..)]`), the TypeScript interfaces (`#[node(..)]`) and the napi records.
//!
//! The record's own vocabulary (kinds, tags, field words) is documented where it expands:
//! `lk_agents_bindgen::record`. A value with a host shape of its own -- a `datetime`, a string
//! type, a host object -- is not a word here: it implements `IntoPyObject`/`FromPyObject` in its
//! own crate, and the record's field crosses by it.
//!
//! Type level, one attribute per host:
//!
//! ```ignore
//! #[derive(Host)]
//! #[python(pydantic, module = "events")]
//! #[node(interface)]
//! #[binding(emit)]
//! pub struct AgentStateChanged { .. }
//! ```
//!
//! Field level, for the generator: `#[host(rename = "type")]` (the host's name when the Rust one
//! cannot be it), `#[python(literal = "..")]` (the discriminant), `#[python(default = "..")]` /
//! `#[python(default_factory = "..")]`, `#[python(serializer = "..")]`, `#[python(exclude)]`,
//! `#[python(ty = "..")]` (a type the generator does not declare). A typo in any of them fails
//! `make generated`, and CI's regenerate-and-diff.

use proc_macro::TokenStream;

mod table;

/// See the module docs: the record's host shapes (`lk_agents_bindgen::record`), under the calling
/// crate's `python` feature.
#[proc_macro_derive(Host, attributes(host, python, node, binding, table))]
pub fn host_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    derive_host(&input).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// A unit enum's wire strings (whatever the feature: the core reads them too), then its binding.
fn derive_host(input: &syn::DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut out = proc_macro2::TokenStream::new();
    if let syn::Data::Enum(data) = &input.data
        && data.variants.iter().all(|v| matches!(v.fields, syn::Fields::Unit))
    {
        let m = lk_agents_bindgen::Metas::of(&input.attrs)?;
        m.only(&format!("#[host] on {}", input.ident), &["wire.err", "wire.msg"])?;
        out.extend(wire_strs(&input.ident, &input.generics, data.variants.iter(), &m, "wire.")?);
    }
    out.extend(lk_agents_bindgen::record(input)?);
    out.extend(table::table(input)?);
    Ok(out)
}

/// `#[derive(Vocabulary)]` -- the search space's completeness check (lk-agents-voice/TEST_STRATEGY.md §5).
///
/// For `enum Foo { A, B(..), C { .. } }` it generates, next to the enum:
///
/// * `pub mod foo_vocabulary { pub struct A; pub struct B; pub struct C; pub const NAMES: &[&str]; }`
///   -- one zero-sized type per variant, the *key* a generator implements `Covers<K>` for;
/// * `pub fn assert_foo_is_covered<G>() where G: Covers<A> + Covers<B> + Covers<C> {}` -- referenced
///   from the harness, so a variant nobody has decided about is a compile error whose message
///   (`Covers`'s `#[diagnostic::on_unimplemented]`) says which variant and what to do;
/// * `pub fn foo_coverage<G, S>() -> Vec<(&'static str, Coverage<S>)>` -- every variant, covered or skipped, in declaration
///   order, which is what the §6.1 coverage test counts and what a sampler reads.
///
/// With `#[vocabulary(kind = FooKind)]` it also generates `enum FooKind` (one unit variant per
/// variant, `Copy + Eq + Hash + Debug`) and `Foo::kind(&self) -> FooKind`.
///
/// `Covers` and `Coverage` live in `lk_agents_abi::utils::vocabulary`. Generics on the enum are not supported:
/// a boundary vocabulary is a plain enum.
#[proc_macro_derive(Vocabulary, attributes(vocabulary))]
pub fn vocabulary(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;
    let vis = &input.vis;
    let syn::Data::Enum(data) = &input.data else {
        return syn::Error::new_spanned(name, "#[derive(Vocabulary)] is for enums: a vocabulary is the set of variants someone must cover or skip")
            .to_compile_error()
            .into();
    };
    if !input.generics.params.is_empty() {
        return syn::Error::new_spanned(&input.generics, "#[derive(Vocabulary)] does not support generic enums")
            .to_compile_error()
            .into();
    }
    let snake = to_snake(&name.to_string());
    let module = quote::format_ident!("{snake}_vocabulary");
    let assert_fn = quote::format_ident!("assert_{snake}_is_covered");
    let coverage_fn = quote::format_ident!("{snake}_coverage");
    let variants: Vec<&syn::Ident> = data.variants.iter().map(|v| &v.ident).collect();
    let names: Vec<String> = variants.iter().map(|v| v.to_string()).collect();
    let bounds = variants.iter().map(|v| quote::quote!(::lk_agents_abi::utils::vocabulary::Covers<#module::#v, With = S>));
    let bounds2 = bounds.clone();
    let mut kind = None;
    for a in input.attrs.iter().filter(|a| a.path().is_ident("vocabulary")) {
        if let Err(e) = a.parse_nested_meta(|m| {
            kind = Some(m.value()?.parse::<syn::Ident>()?);
            if m.path.is_ident("kind") { Ok(()) } else { Err(m.error("#[vocabulary] takes `kind = Name`")) }
        }) {
            return e.to_compile_error().into();
        }
    }
    let kind = kind.map(|k| {
        let pats = data.variants.iter().map(|v| {
            let i = &v.ident;
            match v.fields {
                syn::Fields::Unit => quote::quote!(#name::#i),
                syn::Fields::Unnamed(_) => quote::quote!(#name::#i(..)),
                syn::Fields::Named(_) => quote::quote!(#name::#i { .. }),
            }
        });
        quote::quote! {
            #[doc = concat!("The variants of `", stringify!(#name), "`, without their payloads.")]
            #[allow(missing_docs)]
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            #vis enum #k { #( #variants ),* }
            impl #name {
                #[allow(missing_docs)]
                #vis fn kind(&self) -> #k {
                    match self { #( #pats => #k::#variants ),* }
                }
            }
        }
    });
    let expanded = quote::quote! {
        #kind
        #[allow(non_snake_case, missing_docs)]
        #[doc = concat!("One zero-sized key per variant of `", stringify!(#name), "`: what a generator implements `Covers<K>` for.")]
        #vis mod #module {
            #( #[allow(missing_docs)] pub struct #variants; )*
            /// Every variant, in declaration order.
            pub const NAMES: &[&str] = &[ #( #names ),* ];
        }
        #[doc = concat!("Compiles only when `G` covers or skips every variant of `", stringify!(#name), "`. Reference it from the harness.")]
        #[allow(dead_code)]
        #vis fn #assert_fn<G, S>()
        where
            G: #( #bounds )+*,
        {
        }
        #[doc = concat!("Every variant, covered or skipped, in declaration order, for the coverage count and for sampling.")]
        #[allow(dead_code)]
        #vis fn #coverage_fn<G, S>() -> ::std::vec::Vec<(&'static str, ::lk_agents_abi::utils::vocabulary::Coverage<S>)>
        where
            G: #( #bounds2 )+*,
        {
            ::std::vec![ #( (#names, <G as ::lk_agents_abi::utils::vocabulary::Covers<#module::#variants>>::coverage()) ),* ]
        }
    };
    expanded.into()
}

/// `#[host(..)]` on a core item -- **this item is exposed to every host, and says so where it
/// is declared.** In a core crate (`lk_agents_bindgen::IN_PLACE_CRATES`) it expands in place, under
/// the crate's `python` feature, to the item's pyo3 binding (`lk-agents-bindgen`); the item itself
/// comes back unchanged but for the per-item markers rustc does not know (`#[host(getter)]` on a
/// method, `#[host(lenient)]` on a parameter, `#[host(kwarg(..))]` on a field, `#[host(kind =
/// "..")]` on a variant), which are stripped. `lk-agents-codegen` reads the same source for the
/// `.pyi` stub and the napi binding. Reached as `lk_agents_abi::host`.
///
/// | head | item | what it binds |
/// |---|---|---|
/// | `export` | free fn, const, or an associated const in a plain `impl` | a `#[pyfunction]` (an UPPER_CASE const: a module attribute) |
/// | `object` | inherent `impl` | `#[pymethods]` on the type's class, or a cell class over it |
/// | `proxy` | trait | `impl Trait for Py<Trait><B: HostBridge>`: the core calling a host object |
/// | `stream` | the source type of a stream pair | the Rust->host stream class (`recv`/`aclose`/`push`..) |
/// | `client` / `kwargs` | a config struct | a keyword constructor / a keyword-spread parameter |
/// | `error` | an error enum or struct | its `HostException`: every wrapper returning it raises through it |
/// | `wire_enum` | a unit enum | its wire strings (below), and the string both ways |
/// | `wire_strs` | a unit enum | its wire strings only: it crosses by its own impls, or not at all |
/// | `string` | a struct or enum with `Display`, `FromStr` and `as_str()` | the string both ways |
/// | `file` | `const _: () = ();` | nothing: `lk-agents-codegen` reads its `module`/`path` as the default of every record in the file |
/// | `group` | an inline `mod` of host records and literal consts | strips their markers; `lk-agents-codegen` reads each struct/enum as a declaration and each `const X: &[&str]` as a host `Literal`, with the group's defaults |
///
/// Inside an `object`, a fn takes `new`, `method`, `getter` or `setter` (and an associated const
/// `const`); a `method`'s kind is its signature's -- an `async fn` is awaitable, one without a
/// receiver a staticmethod, one returning `Reply<T>` sent now and awaited by the host. A `method`
/// declared without a body takes `send = <expr>`, and its body is `self.send(<expr>)`
/// (`lk_agents_bindgen::fill_sends`), whatever the crate. A parameter
/// takes `lenient`, `label = ".."`, `ambient`, `kwargs` or `required` (an `Option<T>` whose
/// followers all default is `None` unwritten; `required` keeps it without). One value rule
/// (`lk_agents_bindgen::meta`): Rust is written unquoted (`defaults(size = -1)`, `stream(sink =
/// crate::Sink)`), host text as a string (`name = "read"`, `err = "RuntimeError"`).
///
/// `wire_enum`, `wire_strs` and a unit enum's `#[derive(Host)]` write Rust whatever the feature:
/// the wire strings are the enum's, so they are written once, at the variants (`#[host(rename =
/// "..")]`, else the snake-cased identifier), and become `ALL`, `as_str`, `from_wire`, `FromStr` and
/// `Display` (see [`wire_strs`]). `err = Type` / `msg = expr` pick the `FromStr` error (`s` is the
/// refused string): `#[host(wire_enum(err = E, msg = E(s.to_owned())))]`, and on a derive
/// `#[host(wire(err = .., msg = ..))]`.
#[proc_macro_attribute]
pub fn host(attr: TokenStream, item: TokenStream) -> TokenStream {
    let in_place = lk_agents_bindgen::in_place(proc_macro::Span::call_site().local_file().as_deref());
    expand_host(attr.into(), item.into(), in_place).into()
}

use lk_agents_bindgen::to_snake;

/// What each item takes, for a head on the wrong item.
const HEADS: &str = "the `host` attribute macro is `export` on a fn/const/impl, `object` on an impl, `proxy` on a trait, \
    `stream` on a type, `client`/`kwargs` on a struct, `error` on an enum or struct, `wire_enum`/`wire_strs` on a unit enum, `string` on a struct or enum, \
    `file` on `const _`, or `group` on an inline mod";

fn expand_host(attr: proc_macro2::TokenStream, item: proc_macro2::TokenStream, in_place: bool) -> proc_macro2::TokenStream {
    use syn::Item as I;
    let head = attr.clone().into_iter().next().map(|t| t.to_string());
    let mut parsed: syn::Item = match syn::parse2(item) {
        Ok(parsed) => parsed,
        Err(err) => return err.to_compile_error(),
    };
    let fits = matches!(
        (&parsed, head.as_deref()),
        (I::Fn(_) | I::Const(_) | I::Impl(_), Some("export"))
            | (I::Impl(_), Some("object"))
            | (I::Trait(_), Some("proxy"))
            | (I::Type(_) | I::Struct(_), Some("stream"))
            | (I::Struct(_), Some("client" | "kwargs"))
            | (I::Enum(_) | I::Struct(_), Some("error" | "string"))
            | (I::Enum(_), Some("wire_enum" | "wire_strs"))
            | (I::Mod(_), Some("group"))
            | (I::Const(_), Some("file"))
    );
    if !fits {
        return syn::Error::new_spanned(attr, HEADS).to_compile_error();
    }
    // Whatever the crate: a `send` method's body is the core's own.
    if let I::Impl(imp) = &mut parsed
        && head.as_deref() == Some("object")
        && let Err(err) = lk_agents_bindgen::fill_sends(imp)
    {
        return err.to_compile_error();
    }
    // A core crate's file gets the binding here, read from the item before its markers are
    // stripped; any other file keeps the inert markers the codegen reads.
    let mut extra = if in_place {
        match expand(&attr, &parsed, head.as_deref()) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error(),
        }
    } else {
        proc_macro2::TokenStream::new()
    };
    if let (I::Enum(e), Some(head @ ("wire_enum" | "wire_strs"))) = (&mut parsed, head.as_deref()) {
        let tokens = lk_agents_bindgen::Metas::parse(attr.clone()).and_then(|m| {
            let prefix = format!("{head}.");
            m.only(&format!("`host({head})` on {}", e.ident), &[head, &format!("{prefix}err"), &format!("{prefix}msg")])?;
            wire_strs(&e.ident, &e.generics, e.variants.iter(), &m, &prefix)
        });
        match tokens {
            Ok(tokens) => extra.extend(tokens),
            Err(err) => return err.to_compile_error(),
        }
    }
    strip(&mut parsed, head.as_deref() == Some("group"));
    quote::quote!(#parsed #extra)
}

/// The item's in-place binding (`lk-agents-bindgen`); the pairing is checked by the caller.
fn expand(attr: &proc_macro2::TokenStream, parsed: &syn::Item, head: Option<&str>) -> syn::Result<proc_macro2::TokenStream> {
    use syn::Item as I;
    let attr = attr.clone();
    match (parsed, head) {
        (I::Impl(imp), Some("object")) => lk_agents_bindgen::object(attr, imp),
        (I::Fn(f), Some("export")) => lk_agents_bindgen::export_fn(attr, f),
        (I::Const(c), Some("export")) => lk_agents_bindgen::export_const(attr, c),
        (I::Impl(imp), Some("export")) => lk_agents_bindgen::export_impl(attr, imp),
        (I::Enum(e), Some("wire_enum")) => Ok(lk_agents_bindgen::wire_enum(&e.ident)),
        (I::Enum(_) | I::Struct(_), Some("error")) => lk_agents_bindgen::error(attr, parsed),
        (I::Enum(e), Some("string")) => Ok(lk_agents_bindgen::string(&e.ident)),
        (I::Struct(st), Some("string")) => Ok(lk_agents_bindgen::string(&st.ident)),
        (I::Trait(t), Some("proxy")) => lk_agents_bindgen::proxy(attr, t),
        (I::Type(_) | I::Struct(_), Some("stream")) => lk_agents_bindgen::stream(attr, parsed),
        (I::Struct(st), Some("client")) => lk_agents_bindgen::client(attr, st),
        (I::Struct(st), Some("kwargs")) => lk_agents_bindgen::kwargs(attr, st),
        // A group's records are the codegen's (the pydantic modules); nothing crosses in Rust.
        _ => Ok(proc_macro2::TokenStream::new()),
    }
}

/// Every marker inside the item that rustc does not know: `#[host(..)]` on a method, parameter,
/// field, variant or associated const -- and, inside a `group`, the record helpers (`python`,
/// `node`, `binding`) on each struct, enum and const directly in it (a nested group strips its
/// own). The item's own attributes are left alone: a second `#[host(stream)]` is its own call.
fn strip(item: &mut syn::Item, group: bool) {
    let words: &[&str] = if group { &["host", "python", "node", "binding"] } else { &["host"] };
    let clean = |attrs: &mut Vec<syn::Attribute>| {
        attrs.retain(|a| !a.path().segments.last().is_some_and(|s| words.iter().any(|w| s.ident == w)));
    };
    let params = |inputs: &mut syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>| {
        for input in inputs {
            if let syn::FnArg::Typed(t) = input {
                clean(&mut t.attrs);
            }
        }
    };
    let fields = |fields: &mut syn::Fields| fields.iter_mut().for_each(|f| clean(&mut f.attrs));
    let variants = |e: &mut syn::ItemEnum| {
        for v in &mut e.variants {
            clean(&mut v.attrs);
            fields(&mut v.fields);
        }
    };
    match item {
        syn::Item::Fn(f) => params(&mut f.sig.inputs),
        syn::Item::Impl(imp) => {
            for inner in &mut imp.items {
                match inner {
                    syn::ImplItem::Fn(f) => {
                        clean(&mut f.attrs);
                        params(&mut f.sig.inputs);
                    }
                    syn::ImplItem::Const(c) => clean(&mut c.attrs),
                    _ => {}
                }
            }
        }
        syn::Item::Trait(t) => {
            for inner in &mut t.items {
                if let syn::TraitItem::Fn(f) = inner {
                    clean(&mut f.attrs);
                    params(&mut f.sig.inputs);
                }
            }
        }
        syn::Item::Struct(st) => fields(&mut st.fields),
        syn::Item::Enum(e) => variants(e),
        syn::Item::Mod(m) => {
            for inner in m.content.iter_mut().flat_map(|(_, items)| items) {
                match inner {
                    syn::Item::Const(c) => clean(&mut c.attrs),
                    syn::Item::Struct(st) => {
                        clean(&mut st.attrs);
                        fields(&mut st.fields);
                    }
                    syn::Item::Enum(e) => {
                        clean(&mut e.attrs);
                        variants(e);
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// `#[lk_agents_abi::class]` on a struct -- **this type is a host class.** The class half of a
/// `#[host(object)]` whose impl is elsewhere: an attribute on an impl cannot touch the struct, and
/// pyo3 declares a class on the struct.
///
/// `name = ".."` (default: the identifier, and it must match the impl's `object(name)`, which the
/// expansion asserts), `module = ".."` (default `livekit.agents._core`), `mutable` (drop
/// `frozen`, for `&mut self` methods under pyo3's borrow flag), `from_py_object` (an argument
/// may take it by value). Inert outside a core crate (`lk_agents_bindgen::IN_PLACE_CRATES`); see that crate
/// for what it expands to.
#[proc_macro_attribute]
pub fn class(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = match syn::parse(item) {
        Ok(item) => item,
        Err(err) => return err.to_compile_error().into(),
    };
    // Outside a core crate the marker is inert, like `#[host]`.
    if !lk_agents_bindgen::in_place(proc_macro::Span::call_site().local_file().as_deref()) {
        return quote::quote!(#item).into();
    }
    lk_agents_bindgen::class(attr.into(), &item).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// `ALL`, `as_str`, `from_wire`, `FromStr` and `Display` for a unit enum, from its variants' wire
/// strings. The `FromStr` error is `<prefix>err` (default `String`), built by `<prefix>msg` from the
/// refused `s: &str` (default `unknown <Enum> "<s>"`). A caller that owns the item strips the
/// variants' `#[host(rename)]`; a derive cannot, and the markers are its helpers.
fn wire_strs<'a>(
    name: &syn::Ident,
    generics: &syn::Generics,
    variants: impl Iterator<Item = &'a syn::Variant>,
    m: &lk_agents_bindgen::Metas,
    prefix: &str,
) -> syn::Result<proc_macro2::TokenStream> {
    if !generics.params.is_empty() {
        return Err(syn::Error::new_spanned(generics, "wire strings are for a plain enum"));
    }
    let mut arms: Vec<(String, syn::Ident)> = Vec::new();
    for v in variants {
        if !matches!(v.fields, syn::Fields::Unit) {
            return Err(syn::Error::new_spanned(&v.ident, "a wire-string variant carries no data: the host sees a string"));
        }
        let vm = lk_agents_bindgen::Metas::of(&v.attrs)?;
        vm.only(&format!("{name}::{}", v.ident), &["rename"])?;
        let wire = vm.text("rename")?.map(str::to_owned).unwrap_or_else(|| to_snake(&v.ident.to_string()));
        if arms.iter().any(|(w, _)| w == &wire) {
            return Err(syn::Error::new_spanned(&v.ident, format!("two variants have the wire string {wire:?}")));
        }
        arms.push((wire, v.ident.clone()));
    }
    let err = m.ty(&format!("{prefix}err"))?.map(|t| quote::quote!(#t)).unwrap_or_else(|| quote::quote!(::std::string::String));
    let label = name.to_string();
    let msg = m.expr(&format!("{prefix}msg"))?.unwrap_or_else(|| quote::quote!(::std::format!("unknown {} {:?}", #label, s)));
    let n = arms.len();
    let (wires, idents): (Vec<_>, Vec<_>) = arms.into_iter().unzip();
    Ok(quote::quote! {
        impl #name {
            /// Every variant, in declaration order.
            pub const ALL: [#name; #n] = [#( #name::#idents ),*];
            /// The wire string.
            pub const fn as_str(self) -> &'static str {
                match self { #( #name::#idents => #wires, )* }
            }
            /// `None` for a string no variant has.
            pub fn from_wire(s: &str) -> ::core::option::Option<Self> {
                match s { #( #wires => ::core::option::Option::Some(#name::#idents), )* _ => ::core::option::Option::None }
            }
        }
        impl ::core::str::FromStr for #name {
            type Err = #err;
            fn from_str(s: &str) -> ::core::result::Result<Self, Self::Err> {
                Self::from_wire(s).ok_or_else(|| #msg)
            }
        }
        impl ::core::fmt::Display for #name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use quote::quote;

    // Catches: `#[host(object)]` leaving a per-method marker on the impl (rustc would reject the
    // unknown attribute), or `#[host(..)]` accepting an item/word pair the codegen never reads.
    #[test]
    fn host_strips_method_markers_and_refuses_other_shapes() {
        let out = super::expand_host(
            quote!(object(cell = "mutex")),
            quote! { impl Foo { #[host(getter)] #[inline] pub fn a(&self) -> u8 { 1 } } },
            false,
        )
        .to_string();
        assert!(!out.contains("getter") && out.contains("inline"), "{out}");
        let item = quote! { pub fn f(x: u8) -> u8 { x } };
        assert_eq!(super::expand_host(quote!(export), item.clone(), false).to_string(), item.to_string());
        let out = super::expand_host(quote!(object), quote! { pub fn f() {} }, false).to_string();
        assert!(out.contains("compile_error"), "{out}");
        let out = super::expand_host(quote!(export), quote! { struct S; }, false).to_string();
        assert!(out.contains("compile_error"), "{out}");
    }

    // Catches: a trait method's `#[host(soon = ..)]`, a parameter's `#[host(lenient)]` or a config
    // field's `#[host(kwarg(..))]` reaching rustc (an unknown attribute there is a hard error), or
    // the strip taking an unrelated attribute with it.
    #[test]
    fn host_strips_trait_parameter_and_field_markers() {
        let out = super::expand_host(
            quote!(proxy(name = "P")),
            quote! { pub trait T { #[host(soon = "flush")] #[doc = "kept"] fn flush(&self, #[host(lenient)] x: u8); } },
            false,
        )
        .to_string();
        assert!(!out.contains("soon") && !out.contains("lenient") && out.contains("kept"), "{out}");
        let out = super::expand_host(
            quote!(export),
            quote! { pub fn f(#[host(lenient)] x: u8) -> u8 { x } },
            false,
        )
        .to_string();
        assert!(!out.contains("lenient"), "{out}");
        let out = super::expand_host(
            quote!(client(class = "C")),
            quote! { pub struct Cfg { #[host(kwarg(default = "1"))] #[serde(skip)] pub a: u8 } },
            false,
        )
        .to_string();
        assert!(!out.contains("kwarg") && out.contains("serde"), "{out}");
    }

    // Catches: the wire string taken from anything but `#[host(rename)]` or the snake-cased
    // identifier, `as_str` and `from_wire` disagreeing (they are generated from one list), the
    // `FromStr` error ignoring `err`/`msg`, `wire_strs` binding anything, a derive's unit enum
    // getting no strings, or a variant with data / two variants on one string being accepted.
    #[test]
    fn wire_strings_are_generated_once_from_the_variants() {
        let out = super::expand_host(
            quote!(wire_enum),
            quote! { pub enum Mode { #[host(rename = "on-air")] OnAir, SlowStart } },
            true,
        )
        .to_string();
        assert!(out.contains("Mode :: OnAir => \"on-air\""), "{out}");
        assert!(out.contains("Mode :: SlowStart => \"slow_start\""), "{out}");
        assert!(out.contains("\"on-air\" => :: core :: option :: Option :: Some (Mode :: OnAir)"), "{out}");
        assert!(out.contains("pub const ALL : [Mode ; 2usize]"), "{out}");
        assert!(out.contains("unknown {} {:?}") && out.contains("IntoPyObject"), "{out}");
        assert!(!out.contains("# [host"), "the variant markers are stripped: {out}");
        let out = super::expand_host(
            quote!(wire_strs(err = Bad, msg = Bad(s.to_owned()))),
            quote! { pub enum Mode { A } },
            true,
        )
        .to_string();
        assert!(out.contains("type Err = Bad") && out.contains("ok_or_else (|| Bad (s . to_owned ()))"), "{out}");
        assert!(!out.contains("IntoPyObject"), "wire_strs binds nothing: {out}");
        let input: syn::DeriveInput = syn::parse_quote! {
            // Spaced so the layout check does not read the fixture as a declaration.
            # [host(wire(msg = format!("unknown mode {s:?}")))]
            enum Mode { #[host(rename = "a")] A }
        };
        let out = super::derive_host(&input).expect("expands").to_string();
        assert!(out.contains("Mode :: A => \"a\"") && out.contains("unknown mode {s:?}"), "{out}");
        let out = super::expand_host(quote!(wire_enum), quote! { enum E { A(u8) } }, false).to_string();
        assert!(out.contains("compile_error") && out.contains("carries no data"), "{out}");
        let out = super::expand_host(
            quote!(wire_enum),
            quote! { enum E { #[host(rename = "a")] A, #[host(rename = "a")] B } },
            false,
        )
        .to_string();
        assert!(out.contains("compile_error") && out.contains("two variants"), "{out}");
    }

}
