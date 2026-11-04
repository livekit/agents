//! Makes `#[host(..)]` legal Rust. `lk-agents-codegen` reads the attributes for every host's view;
//! this crate strips the per-item markers rustc does not know and writes the Rust a declaration
//! implies whatever the host (wire strings, declared defaults, `model_dump`, `send` bodies), plus
//! the `Table` and `Vocabulary` derives.

use proc_macro::TokenStream;

mod table;
mod telemetry;

/// `#[derive(Table)]`: a column declaration, `#[table("name", remote = path)]`, for a row record
/// defined in another crate (see [`table`]).
#[proc_macro_derive(Table, attributes(table))]
pub fn table_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    table::table(&input).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// `#[derive(Attrs)]`: a telemetry attribute set (`lk_agents_abi::telemetry::decl::Attrs`). Each
/// field is `#[attr("key", label, metric = "..", unmeasured = 0 | -1)]`, an `#[attrs]` group written
/// in place, or a record's `#[span_id]`.
#[proc_macro_derive(Attrs, attributes(attr, attrs, span_id, at))]
pub fn attrs_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    telemetry::attrs(&input).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// `#[derive(SpanValue)]`: a closed vocabulary written as its snake_case variant names.
#[proc_macro_derive(SpanValue)]
pub fn span_value_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    telemetry::span_value(&input).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// `#[span(name, parent = A | B, end = T, events = [E, ..], kind, outlives_parent)]` on an `Attrs`
/// struct: a span declaration and its start attributes. `#[span(Decl { .. })]` on a function
/// taking `cx: &telemetry::Cx`: the function is that one span (`#[telemetry::span]`,
/// `lk_agents_net::telemetry`).
#[proc_macro_attribute]
pub fn span(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Ok(function) = syn::parse::<syn::ItemFn>(item.clone()) {
        return telemetry::span_fn(attr.into(), function)
            .unwrap_or_else(|e| e.to_compile_error())
            .into();
    }
    telemetry_role("span", attr, item)
}

/// `#[event(name)]` on an `Attrs` struct: a span event.
#[proc_macro_attribute]
pub fn event(attr: TokenStream, item: TokenStream) -> TokenStream {
    telemetry_role("event", attr, item)
}

/// `#[metric(name, kind = Counter | Histogram, unit)]` on an `Attrs` struct: an instrument and its
/// labels.
#[proc_macro_attribute]
pub fn metric(attr: TokenStream, item: TokenStream) -> TokenStream {
    telemetry_role("metric", attr, item)
}

/// `#[record(scope, body)]` on an `Attrs` struct: a report record exported as an OTLP log.
#[proc_macro_attribute]
pub fn record(attr: TokenStream, item: TokenStream) -> TokenStream {
    telemetry_role("record", attr, item)
}

fn telemetry_role(role: &str, attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: proc_macro2::TokenStream = item.into();
    match telemetry::role(role, attr.into(), item.clone()) {
        Ok(out) => out.into(),
        Err(e) => {
            let e = e.to_compile_error();
            quote::quote!(#item #e).into()
        }
    }
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
        return syn::Error::new_spanned(
            &input.generics,
            "#[derive(Vocabulary)] does not support generic enums",
        )
        .to_compile_error()
        .into();
    }
    let snake = to_snake(&name.to_string());
    let module = quote::format_ident!("{snake}_vocabulary");
    let assert_fn = quote::format_ident!("assert_{snake}_is_covered");
    let coverage_fn = quote::format_ident!("{snake}_coverage");
    let variants: Vec<&syn::Ident> = data.variants.iter().map(|v| &v.ident).collect();
    let names: Vec<String> = variants.iter().map(|v| v.to_string()).collect();
    let bounds = variants
        .iter()
        .map(|v| quote::quote!(::lk_agents_abi::utils::vocabulary::Covers<#module::#v, With = S>));
    let bounds2 = bounds.clone();
    let mut kind = None;
    for a in input.attrs.iter().filter(|a| a.path().is_ident("vocabulary")) {
        if let Err(e) = a.parse_nested_meta(|m| {
            kind = Some(m.value()?.parse::<syn::Ident>()?);
            if m.path.is_ident("kind") {
                Ok(())
            } else {
                Err(m.error("#[vocabulary] takes `kind = Name`"))
            }
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

/// `#[host(..)]` on an ABI item -- **this item is exposed to every host, and says so where it is
/// declared.** `lk-agents-codegen` reads the declaration and writes the binding into `lk-agents-py`;
/// here the item comes back unchanged but for the per-item markers rustc does not know
/// (`#[host(getter)]` on a method, `#[host(lenient)]` on a parameter, `#[host(kind = "..")]` on a
/// variant), which are stripped, plus the Rust the declaration implies whatever the host. Reached
/// as `lk_agents_abi::host`.
///
/// | head | item | what it means |
/// |---|---|---|
/// | `export` | free fn or const | a `_core` function (an UPPER_CASE const name: a module attribute) |
/// | `proxy` | trait | the core calling a host object; its methods declare their calls (`await`/`call`/`notify`..) and the proxy is generated (`lk-agents-codegen`'s `decl`) |
/// | `native` | trait | a core trait only Rust implements: the host hands back a native object |
/// | `class` / `functions` | trait | a declaration-model class or module functions (`lk-agents-codegen`'s `decl`), adapters generated |
/// | `sink` | trait | a duplex stream class's input half, its `call`s methods of that class (`lk-agents-codegen`'s `decl`) |
/// | `native_enum` | a unit enum | a generated native Python enum with typed conversions |
/// | `codec` | a type | it crosses by its own conversions |
/// | `record` | a struct or enum | a schema-defined record; `lk-agents-codegen` writes its model and its conversions |
/// | `error` | an error enum or struct | its `host_kind` here; its host exception in the binding. `error(raised = "..")`: a refusal a generated proxy reads |
/// | `wire_enum` | a unit enum | its wire strings (below), crossing as that string both ways |
/// | `wire_strs` | a unit enum | its wire strings only: it crosses by its own conversions, or not at all |
/// | `string` | a struct or enum with `Display` and `FromStr` | the string both ways |
/// | `file` | `const _: () = ();` | `lk-agents-codegen` reads its `module`/`path` as the default of every record in the file |
/// | `interface` | a backend-local unit struct | generates inherent import, member-access and capability-check helpers |
/// | `exceptions` | `const _: () = ();` | ordinary host exception classes, their inheritance and constructors |
/// | `constant` | a `pub const` | written into its file's Python module |
///
/// One value rule
/// (`lk_agents_bindgen::meta`): Rust is written unquoted (`defaults(size = -1)`), host text as a
/// string (`name = "read"`, `err = "RuntimeError"`).
///
/// `wire_enum` and `wire_strs` write the wire strings once, at the variants (`#[host(rename =
/// "..")]`, else the snake-cased identifier), and they become `ALL`, `as_str`, `from_wire`,
/// `FromStr` and `Display` (see [`wire_strs`]). `err = Type` / `msg = expr` pick the `FromStr`
/// error (`s` is the refused string): `#[host(wire_enum(err = E, msg = E(s.to_owned())))]`.
#[proc_macro_attribute]
pub fn host(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_host(attr.into(), item.into()).into()
}

use lk_agents_bindgen::to_snake;

/// What each item takes, for a head on the wrong item.
const HEADS: &str = "the `host` attribute macro is `export` on a fn/const, `proxy`/`native`/`class`/`functions`/`sink` on a trait, `codec`/`record` on a type, \
    `error` on an enum or struct, `wire_enum`/`wire_strs`/`native_enum` on a unit enum, `string` on a struct or enum, \
    `interface` on a unit struct, `file`/`exceptions` on `const _`, or `constant` on a `pub const` (written into its file's Python module)";

fn expand_host(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    use syn::Item as I;
    let head = attr.clone().into_iter().next().map(|t| t.to_string());
    let mut parsed: syn::Item = match syn::parse2(item) {
        Ok(parsed) => parsed,
        Err(err) => return err.to_compile_error(),
    };
    let fits = matches!(
        (&parsed, head.as_deref()),
        (I::Fn(_) | I::Const(_), Some("export"))
            | (I::Trait(_), Some("proxy" | "native" | "class" | "functions" | "sink"))
            | (I::Struct(_), Some("interface"))
            | (I::Enum(_) | I::Struct(_) | I::Type(_), Some("codec"))
            | (I::Enum(_) | I::Struct(_), Some("record"))
            | (I::Enum(_) | I::Struct(_), Some("error" | "string"))
            | (I::Enum(_), Some("wire_enum" | "wire_strs" | "native_enum"))
            | (I::Const(_), Some("file" | "constant" | "exceptions"))
    );
    if !fits {
        return syn::Error::new_spanned(attr, HEADS).to_compile_error();
    }
    let mut extra = match (&parsed, head.as_deref()) {
        (I::Enum(_) | I::Struct(_), Some("error")) => {
            match lk_agents_bindgen::Metas::parse(attr.clone()).and_then(|m| {
                if lk_agents_bindgen::raised(&m) {
                    return Ok(proc_macro2::TokenStream::new());
                }
                lk_agents_bindgen::host_kind(attr.clone(), &parsed)
            }) {
                Ok(tokens) => tokens,
                Err(err) => return err.to_compile_error(),
            }
        }
        (I::Trait(t), Some("proxy")) if !declared_proxy(t) => {
            return syn::Error::new_spanned(
                &t.ident,
                "a `#[host(proxy)]` trait declares how each method is called (`#[host(await)]`, `call`, `notify`, `read_once` or `bind`)",
            )
            .to_compile_error();
        }
        _ => proc_macro2::TokenStream::new(),
    };
    if let (I::Enum(e), Some(head @ ("wire_enum" | "wire_strs"))) = (&mut parsed, head.as_deref()) {
        let tokens = lk_agents_bindgen::Metas::parse(attr.clone()).and_then(|m| {
            let prefix = format!("{head}.");
            m.only(
                &format!("`host({head})` on {}", e.ident),
                &[head, &format!("{prefix}err"), &format!("{prefix}msg")],
            )?;
            wire_strs(&e.ident, &e.generics, e.variants.iter(), &m, &prefix)
        });
        match tokens {
            Ok(tokens) => extra.extend(tokens),
            Err(err) => return err.to_compile_error(),
        }
    }
    if let (I::Struct(st), Some("record")) = (&parsed, head.as_deref())
        && lk_agents_bindgen::Metas::parse(attr.clone()).is_ok_and(|m| m.has("record.default"))
    {
        match declared_default(st) {
            Ok(tokens) => extra.extend(tokens),
            Err(err) => return err.to_compile_error(),
        }
    }
    if let (I::Struct(st), Some("record")) = (&parsed, head.as_deref())
        && let Ok(m) = lk_agents_bindgen::Metas::parse(attr.clone())
        && m.flag("record.model_dump")
    {
        match declared_model_dump(st, &m) {
            Ok(tokens) => extra.extend(tokens),
            Err(err) => return err.to_compile_error(),
        }
    }
    // Record properties, string classes and export forwarding metadata are read by
    // `lk-agents-codegen` only.
    match (&mut parsed, head.as_deref()) {
        (I::Struct(st), Some("string" | "record")) => {
            st.attrs.retain(|a| !a.path().is_ident("python"))
        }
        (I::Fn(f), Some("export")) => f.attrs.retain(|a| !a.path().is_ident("python")),
        _ => {}
    }
    // A later `host` attribute on the item reads the inner markers too (`wire_strs` reads the
    // variants' renames): only the item's last expansion strips them.
    if !item_attrs(&parsed).iter().any(lk_agents_bindgen::is_host) {
        strip(&mut parsed);
    }
    if let I::Trait(trait_) = &mut parsed {
        send_futures(trait_);
    }
    quote::quote!(#parsed #extra)
}

/// Host operations may cross executor threads. Keep the declaration's async syntax,
/// but make the returned future's contract explicit in the compiled Rust trait.
fn send_futures(trait_: &mut syn::ItemTrait) {
    for item in &mut trait_.items {
        let syn::TraitItem::Fn(method) = item else { continue };
        if method.sig.asyncness.take().is_none() {
            continue;
        }
        let output = match &method.sig.output {
            syn::ReturnType::Default => syn::parse_quote!(()),
            syn::ReturnType::Type(_, ty) => *ty.clone(),
        };
        method.sig.output = syn::parse_quote!(
            -> impl ::core::future::Future<Output = #output> + Send
        );
        if let Some(body) = method.default.take() {
            method.default = Some(syn::parse_quote!({ async move #body }));
        }
    }
}

fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Enum(e) => &e.attrs,
        syn::Item::Struct(s) => &s.attrs,
        syn::Item::Trait(t) => &t.attrs,
        syn::Item::Impl(i) => &i.attrs,
        syn::Item::Fn(f) => &f.attrs,
        syn::Item::Const(c) => &c.attrs,
        _ => &[],
    }
}

/// `record(.., default)`: `Default` from the fields' declared `default`s (a `Maybe` field: not
/// given), so the host's defaults and Rust's are one list.
fn declared_default(st: &syn::ItemStruct) -> syn::Result<proc_macro2::TokenStream> {
    let syn::Fields::Named(named) = &st.fields else { return Ok(proc_macro2::TokenStream::new()) };
    let mut inits = Vec::new();
    for f in &named.named {
        let m = lk_agents_bindgen::Metas::of(&f.attrs)?;
        let value = match m.0.iter().rev().find(|x| x.key == "default").map(|x| &x.value) {
            Some(lk_agents_bindgen::Value::Text(s)) => quote::quote!(#s.into()),
            Some(lk_agents_bindgen::Value::Rust(r)) => r.parse::<proc_macro2::TokenStream>()?,
            // Required of the host; Rust callers still start from the type's own default.
            _ if is_maybe(&f.ty) || m.0.iter().any(|x| x.key == "required") => {
                quote::quote!(::core::default::Default::default())
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    f,
                    "`record(.., default)`: every field declares its `default`",
                ));
            }
        };
        let name = &f.ident;
        inits.push(quote::quote!(#name: #value));
    }
    let (name, (imp, ty, wh)) = (&st.ident, st.generics.split_for_impl());
    Ok(quote::quote! {
        impl #imp ::core::default::Default for #name #ty #wh {
            fn default() -> Self { #name { #(#inits),* } }
        }
    })
}

/// `record(.., model_dump)`: `ModelDump` (`model_dump(mode="json", exclude_none=True)`), the
/// model's fields in its order: constants first or after `constants_after`, private and skipped
/// fields left out, a field's `dump_with = path` dumping it as `path(self, cx)`.
fn declared_model_dump(
    st: &syn::ItemStruct,
    m: &lk_agents_bindgen::Metas,
) -> syn::Result<proc_macro2::TokenStream> {
    use lk_agents_bindgen::Value;
    let text = |key: &str| {
        m.0.iter().find(|x| x.key == key).and_then(|x| match &x.value {
            Value::Text(t) => Some(t.clone()),
            _ => None,
        })
    };
    let name = text("record.name").unwrap_or_else(|| st.ident.to_string());
    let constants: Vec<proc_macro2::TokenStream> = m
        .0
        .iter()
        .filter_map(|x| match (x.key.strip_prefix("record.constant."), &x.value) {
            (Some(key), Value::Text(v)) => Some(quote::quote!(
                cx.insert(&mut d, #key, ::std::option::Option::Some(::std::convert::From::from(#v)));
            )),
            _ => None,
        })
        .collect();
    let after = text("record.constants_after");
    let mut sets = Vec::new();
    if after.is_none() {
        sets.extend(constants.iter().cloned());
    }
    let syn::Fields::Named(named) = &st.fields else { return Ok(proc_macro2::TokenStream::new()) };
    for f in &named.named {
        let fm = lk_agents_bindgen::Metas::of(&f.attrs)?;
        let ident = f.ident.as_ref().expect("named");
        let key = ident.to_string();
        if !fm.has("private") && !fm.has("skip") {
            let at = format!("{name}.{key}");
            sets.push(match fm.0.iter().find(|x| x.key == "dump_with").map(|x| &x.value) {
                Some(Value::Rust(path)) => {
                    let path = path.parse::<proc_macro2::TokenStream>()?;
                    quote::quote!(cx.insert(&mut d, #key, #path(self, cx)?);)
                }
                _ => quote::quote!(
                    cx.insert(&mut d, #key, ::lk_agents_abi::ModelDump::model_dump(&self.#ident, cx, #at)?);
                ),
            });
        }
        if after.as_deref() == Some(key.as_str()) {
            sets.extend(constants.iter().cloned());
        }
    }
    let ident = &st.ident;
    Ok(quote::quote! {
        impl ::lk_agents_abi::ModelDump for #ident {
            fn model_dump(&self, cx: &::lk_agents_abi::DumpCx<'_>, _at: &'static str) -> ::lk_agents_abi::Dumped {
                let mut d = ::lk_agents_abi::DumpMap::new();
                #(#sets)*
                ::std::result::Result::Ok(::std::option::Option::Some(::lk_agents_abi::DumpValue::Object(d)))
            }
        }
    })
}

fn is_maybe(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Maybe"))
}

/// The method keys of a declared proxy (`lk-agents-codegen`'s `decl::vocabulary`).
const PROXY_CALLS: &[&str] = &["await", "call", "notify", "read_once", "bind"];

/// Some method declares how it is called: the trait is a declaration-model proxy.
fn declared_proxy(t: &syn::ItemTrait) -> bool {
    t.items.iter().any(|item| {
        let syn::TraitItem::Fn(f) = item else { return false };
        f.attrs.iter().filter(|a| a.path().segments.last().is_some_and(|s| s.ident == "host")).any(
            |a| {
                let mut declared = false;
                // `await` is a keyword: read the keys as `parse_nested_meta` does.
                let _ = a.parse_nested_meta(|m| {
                    declared |= PROXY_CALLS.iter().any(|k| m.path.is_ident(k));
                    if !m.input.is_empty() && !m.input.peek(syn::Token![,]) {
                        m.input.parse::<proc_macro2::TokenStream>()?;
                    }
                    Ok(())
                });
                declared
            },
        )
    })
}

/// Every marker inside the item that rustc does not know: `#[host(..)]` on a method, parameter,
/// field, variant or associated const. A proxy trait's Python face (`#[python(..)]` on the trait, its methods and parameters,
/// read by `lk-agents-codegen`) goes too. The item's other attributes are left alone: a second
/// `#[host]` is its own call.
fn strip(item: &mut syn::Item) {
    let words: &[&str] = match item {
        syn::Item::Trait(_) => &["host", "python"],
        _ => &["host"],
    };
    let clean = |attrs: &mut Vec<syn::Attribute>| {
        attrs.retain(|a| {
            !a.path().segments.last().is_some_and(|s| words.iter().any(|w| s.ident == w))
        });
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
            t.attrs.retain(|a| !a.path().is_ident("python"));
            for inner in &mut t.items {
                match inner {
                    syn::TraitItem::Fn(f) => {
                        clean(&mut f.attrs);
                        params(&mut f.sig.inputs);
                    }
                    syn::TraitItem::Type(t) => clean(&mut t.attrs),
                    _ => {}
                }
            }
        }
        syn::Item::Struct(st) => fields(&mut st.fields),
        syn::Item::Enum(e) => variants(e),
        _ => {}
    }
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
            return Err(syn::Error::new_spanned(
                &v.ident,
                "a wire-string variant carries no data: the host sees a string",
            ));
        }
        let vm = lk_agents_bindgen::Metas::of(&v.attrs)?;
        vm.only(&format!("{name}::{}", v.ident), &["rename"])?;
        let wire =
            vm.text("rename")?.map(str::to_owned).unwrap_or_else(|| to_snake(&v.ident.to_string()));
        if arms.iter().any(|(w, _)| w == &wire) {
            return Err(syn::Error::new_spanned(
                &v.ident,
                format!("two variants have the wire string {wire:?}"),
            ));
        }
        arms.push((wire, v.ident.clone()));
    }
    let err = m
        .ty(&format!("{prefix}err"))?
        .map(|t| quote::quote!(#t))
        .unwrap_or_else(|| quote::quote!(::std::string::String));
    let label = name.to_string();
    let msg = m
        .expr(&format!("{prefix}msg"))?
        .unwrap_or_else(|| quote::quote!(::std::format!("unknown {} {:?}", #label, s)));
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

    /// `item`'s outer `host` attributes expanded one at a time, first to last, as rustc does.
    fn expand_all(item: proc_macro2::TokenStream) -> String {
        let mut item: syn::Item = syn::parse2(item).unwrap();
        let mut impls: Vec<syn::Item> = Vec::new();
        while let Some(i) = super::item_attrs(&item).iter().position(lk_agents_bindgen::is_host) {
            let syn::Item::Enum(e) = &mut item else { unreachable!("an enum") };
            let args = e.attrs.remove(i).meta.require_list().unwrap().tokens.clone();
            // The expansion is the item followed by its impls; the item expands on.
            let file: syn::File = syn::parse2(super::expand_host(args, quote!(#item))).unwrap();
            let mut items = file.items.into_iter();
            item = items.next().unwrap();
            impls.extend(items);
        }
        quote!(#item #(#impls)*).to_string()
    }

    // Catches: an earlier expansion stripping the variants' `#[host(rename)]` before `wire_strs`
    // reads them, so the wire string silently falls back to `machine_vm` (AMDCategory, 74b0adf1a).
    #[test]
    fn a_renamed_variant_keeps_its_wire_string_in_either_attribute_order() {
        for item in [
            quote! {
                #[host(record(module = "m", emit))]
                #[host(wire_strs)]
                enum E { #[host(rename = "machine-vm")] MachineVm }
            },
            quote! {
                #[host(wire_strs)]
                #[host(record(module = "m", emit))]
                enum E { #[host(rename = "machine-vm")] MachineVm }
            },
        ] {
            let out = expand_all(item);
            assert!(out.contains("\"machine-vm\" =>"), "{out}");
            assert!(!out.contains("rename"), "a marker outlived the last expansion: {out}");
        }
    }

    // Catches: `#[host(..)]` accepting an item/word pair the codegen never reads, or an export
    // coming back changed.
    #[test]
    fn host_refuses_shapes_no_binding_reads() {
        let item = quote! { pub fn f(x: u8) -> u8 { x } };
        assert_eq!(super::expand_host(quote!(export), item.clone()).to_string(), item.to_string());
        let out = super::expand_host(quote!(object), quote! { impl Foo {} }).to_string();
        assert!(out.contains("compile_error"), "{out}");
        let out = super::expand_host(quote!(export), quote! { struct S; }).to_string();
        assert!(out.contains("compile_error"), "{out}");
    }

    // Catches: a trait method's `#[host(soon = ..)]`, a parameter's `#[host(lenient)]` or a config
    // field's `#[host(kwarg(..))]` reaching rustc (an unknown attribute there is a hard error), or
    // the strip taking an unrelated attribute with it.
    // Catches: a declared proxy emitting host bindings, an undeclared one accepted, or a
    // raised refusal given a `host_kind` it has no table for.
    #[test]
    fn declared_proxies_emit_only_traits_and_refusals_no_host_kind() {
        let out = super::expand_host(
            quote!(proxy(host = "a.B")),
            quote! { pub trait T { #[host(await)] async fn f(&self, #[host(context)] a: &A) -> Result<(), E>; #[host(notify)] fn g(&self, #[host(context)] a: &A); } },
        )
        .to_string();
        let file: syn::File = syn::parse_str(&out).unwrap();
        assert!(matches!(file.items.as_slice(), [syn::Item::Trait(_)]), "{out}");
        assert!(!out.contains("host"), "{out}");
        let out = super::expand_host(
            quote!(proxy(host = "a.B")),
            quote! { pub trait T { fn f(&self); } },
        )
        .to_string();
        assert!(out.contains("compile_error"), "{out}");
        let out = super::expand_host(
            quote!(error(raised = "a.Refused")),
            quote! { pub struct Refused { pub message: String } },
        )
        .to_string();
        assert!(!out.contains("impl"), "{out}");
    }

    #[test]
    fn host_strips_trait_parameter_and_field_markers() {
        let out = super::expand_host(
            quote!(proxy(name = "P")),
            quote! { pub trait T { #[host(call, soon = "flush")] #[doc = "kept"] fn flush(&self, #[host(lenient)] x: u8); } },
        )
        .to_string();
        assert!(!out.contains("soon") && !out.contains("lenient") && out.contains("kept"), "{out}");
        let out = super::expand_host(
            quote!(proxy(host = "a.B")),
            quote! { #[python(provider(module = "b"))] pub trait T { #[python(hook)] #[host(call)] fn f(&self, #[python(attr)] x: u8); } },
        )
        .to_string();
        assert!(!out.contains("python"), "the Python face is the codegen's: {out}");
        let out = super::expand_host(
            quote!(export),
            quote! { pub fn f(#[host(lenient)] x: u8) -> u8 { x } },
        )
        .to_string();
        assert!(!out.contains("lenient"), "{out}");
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
        )
        .to_string();
        assert!(out.contains("Mode :: OnAir => \"on-air\""), "{out}");
        assert!(out.contains("Mode :: SlowStart => \"slow_start\""), "{out}");
        assert!(
            out.contains("\"on-air\" => :: core :: option :: Option :: Some (Mode :: OnAir)"),
            "{out}"
        );
        assert!(out.contains("pub const ALL : [Mode ; 2usize]"), "{out}");
        assert!(out.contains("unknown {} {:?}") && !out.contains("IntoPyObject"), "{out}");
        assert!(!out.contains("# [host"), "the variant markers are stripped: {out}");
        let out = super::expand_host(
            quote!(wire_strs(err = Bad, msg = Bad(s.to_owned()))),
            quote! { pub enum Mode { A } },
        )
        .to_string();
        assert!(
            out.contains("type Err = Bad") && out.contains("ok_or_else (|| Bad (s . to_owned ()))"),
            "{out}"
        );
        let out = super::expand_host(quote!(wire_enum), quote! { enum E { A(u8) } }).to_string();
        assert!(out.contains("compile_error") && out.contains("carries no data"), "{out}");
        let out = super::expand_host(
            quote!(wire_enum),
            quote! { enum E { #[host(rename = "a")] A, #[host(rename = "a")] B } },
        )
        .to_string();
        assert!(out.contains("compile_error") && out.contains("two variants"), "{out}");
    }
    #[test]
    fn declared_async_operations_have_send_futures() {
        let output = super::expand_host(
            quote!(class),
            quote! { pub trait Api: Send + Sync { #[host(await)] async fn ping(&self) -> u32 { 9 } } },
        );
        let file: syn::File = syn::parse2(output).unwrap();
        let syn::Item::Trait(trait_) = &file.items[0] else { panic!("a trait") };
        let syn::TraitItem::Fn(method) = &trait_.items[0] else { panic!("a method") };
        assert!(method.sig.asyncness.is_none());
        let syn::ReturnType::Type(_, ty) = &method.sig.output else { panic!("a return type") };
        let syn::Type::ImplTrait(future) = ty.as_ref() else { panic!("a future contract") };
        assert!(future.bounds.iter().any(|bound| matches!(bound, syn::TypeParamBound::Trait(bound) if bound.path.is_ident("Send"))));
        assert!(matches!(
            method.default.as_ref().unwrap().stmts.last(),
            Some(syn::Stmt::Expr(syn::Expr::Async(_), None))
        ));
    }
}
