//! The telemetry declarations (`lk_agents_abi::telemetry::decl`): `#[derive(Attrs)]`,
//! `#[derive(SpanValue)]` and the role attributes `#[span]`, `#[event]`, `#[metric]` and
//! `#[record]`. The attribute language is read by `lk_agents_bindgen::telemetry`, which the
//! generator shares.

use lk_agents_bindgen::telemetry::{self as t, Field, InstrumentKind, Role, SpanKind, Unmeasured};
use proc_macro2::TokenStream;
use quote::quote;

/// The declarations' module, as every crate names it.
fn decl() -> TokenStream {
    quote!(::lk_agents_abi::telemetry::decl)
}

/// `#[derive(Attrs)]`: writes each `#[attr]` field under its key, flattens each `#[attrs]` group,
/// and collects the `label` and `metric` fields.
pub fn attrs(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let d = decl();
    let syn::Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(&input.ident, "#[derive(Attrs)] is for structs"));
    };
    let (mut write, mut labels, mut measure) = (Vec::new(), Vec::new(), Vec::new());
    for (i, field) in data.fields.iter().enumerate() {
        let member = match &field.ident {
            Some(ident) => quote!(#ident),
            None => {
                let index = syn::Index::from(i);
                quote!(#index)
            }
        };
        match t::field(field)? {
            Field::SpanId => {}
            Field::Flatten => {
                write.push(quote!(#d::Attrs::write(&self.#member, out);));
                labels.push(quote!(#d::Attrs::labels(&self.#member, out);));
                measure.push(quote!(#d::Attrs::measure(&self.#member, out);));
            }
            Field::Attr(attr) => {
                let key = &attr.key;
                if is_template(&field.ty) {
                    if attr.label || attr.metric.is_some() || attr.unmeasured != Unmeasured::Omit {
                        return Err(syn::Error::new_spanned(
                            field,
                            "a `Template` attribute is neither a label nor a measurement",
                        ));
                    }
                    write.push(quote!(self.#member.write(#key, out);));
                    continue;
                }
                let value = match attr.unmeasured {
                    Unmeasured::Omit => quote!(#d::SpanValue::value(&self.#member)),
                    Unmeasured::Zero => quote!(self.#member.value_or(#d::Unmeasured::Zero)),
                    Unmeasured::MinusOne => quote!(self.#member.value_or(#d::Unmeasured::MinusOne)),
                };
                write.push(quote!(#d::put(out, #key, #value);));
                if attr.label {
                    let label = match &attr.label_key {
                        Some(label) => quote!(#label),
                        None => quote!(#key),
                    };
                    labels.push(quote!(#d::put(out, #label, #d::SpanValue::value(&self.#member));));
                }
                if let Some(metric) = &attr.metric {
                    measure.push(quote! {
                        if let Some(n) = #d::measurement(&self.#member) {
                            out.push((#metric, n));
                        }
                    });
                }
            }
        }
    }
    let ident = &input.ident;
    let (imp, ty, wh) = input.generics.split_for_impl();
    let unused = write.is_empty().then(|| quote!(let _ = out;));
    Ok(quote! {
        impl #imp #d::Attrs for #ident #ty #wh {
            fn write(&self, out: &mut #d::KeyValues) { #unused #( #write )* }
            fn labels(&self, out: &mut #d::KeyValues) { #unused #( #labels )* }
            fn measure(&self, out: &mut ::std::vec::Vec<(&'static str, #d::Number)>) { #unused #( #measure )* }
        }
    })
}

fn is_template(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Template"))
}

/// `#[derive(SpanValue)]`: a closed vocabulary written as its snake_case variant names.
pub fn span_value(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let d = decl();
    let syn::Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(SpanValue)] is for enums of unit variants",
        ));
    };
    let ident = &input.ident;
    let arms = data
        .variants
        .iter()
        .map(|v| {
            if !matches!(v.fields, syn::Fields::Unit) {
                return Err(syn::Error::new_spanned(v, "a `SpanValue` variant carries no data"));
            }
            let (name, wire) = (&v.ident, lk_agents_bindgen::to_snake(&v.ident.to_string()));
            Ok(quote!(#ident::#name => #wire))
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        impl #ident {
            /// The wire string.
            pub const fn as_str(&self) -> &'static str {
                match self { #( #arms ),* }
            }
        }
        impl #d::SpanValue for #ident {
            fn value(&self) -> ::core::option::Option<#d::Value> {
                ::core::option::Option::Some(#d::Value::String(self.as_str().into()))
            }
        }
    })
}

/// A role attribute on an `Attrs` struct: the item, unchanged, and the role's impls.
pub fn role(role: &str, args: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let d = decl();
    let st: syn::ItemStruct = syn::parse2(item.clone())
        .map_err(|e| syn::Error::new(e.span(), format!("#[{role}] is for an `Attrs` struct")))?;
    let ident = &st.ident;
    let impls = match t::role(role, args)? {
        Role::Span(span) => {
            let name = &span.name;
            let kind = match span.kind {
                SpanKind::Internal => quote!(Internal),
                SpanKind::Client => quote!(Client),
                SpanKind::Server => quote!(Server),
            };
            let end = span.end.as_ref().map_or_else(|| quote!(()), |t| quote!(#t));
            let outlives = match &span.outlives_parent {
                Some(reason) => quote!(::core::option::Option::Some(#reason)),
                None => quote!(::core::option::Option::None),
            };
            let parents = &span.parents;
            let events = &span.events;
            let manual = span.manual;
            quote! {
                impl #d::SpanDecl for #ident {
                    const NAME: &'static str = #name;
                    const KIND: #d::SpanKind = #d::SpanKind::#kind;
                    type End = #end;
                    const OUTLIVES_PARENT: ::core::option::Option<&'static str> = #outlives;
                    const MANUAL: bool = #manual;
                    const PARENTS: &'static [&'static str] =
                        &[#( <#parents as #d::SpanParent>::PARENT_NAME ),*];
                }
                #( impl #d::ParentOf<#ident> for #parents {} )*
                #( impl #d::HasEvent<#events> for #ident {} )*
            }
        }
        Role::Event { name } => quote! {
            impl #d::EventDecl for #ident { const NAME: &'static str = #name; }
        },
        Role::Metric(metric) => {
            let (name, unit) = (&metric.name, &metric.unit);
            let kind = match metric.kind {
                InstrumentKind::Counter => quote!(Counter),
                InstrumentKind::Histogram => quote!(Histogram),
            };
            quote! {
                impl #d::MetricDecl for #ident {
                    const NAME: &'static str = #name;
                    const KIND: #d::InstrumentKind = #d::InstrumentKind::#kind;
                    const UNIT: &'static str = #unit;
                }
            }
        }
        Role::Record { scope, body } => {
            let mut span_ids = Vec::new();
            for field in &st.fields {
                if let Field::SpanId = t::field(field)? {
                    span_ids.push(field);
                }
            }
            let span = match span_ids.as_slice() {
                [] => quote!(::core::option::Option::None),
                [field] => {
                    let f = field.ident.as_ref().ok_or_else(|| {
                        syn::Error::new_spanned(field, "a `#[span_id]` field is named")
                    })?;
                    quote!(self.#f)
                }
                [_, extra, ..] => {
                    return Err(syn::Error::new_spanned(
                        extra,
                        "a record has at most one `#[span_id]`",
                    ));
                }
            };
            quote! {
                impl #d::RecordDecl for #ident {
                    const SCOPE: &'static str = #scope;
                    const BODY: &'static str = #body;
                    fn span(&self) -> ::core::option::Option<#d::SpanRef> { #span }
                }
            }
        }
    };
    Ok(quote!(#item #impls))
}

/// `#[span(Decl { .. })]` on a function taking `cx: &Cx` and returning a `Result`: the body runs
/// with `cx` as the span's context, its result goes through `check`, and an `Ok` value ends the
/// span through its `SpanEnd<Decl>` implementation. An `Err` ends it as that error.
pub fn span_fn(args: TokenStream, mut function: syn::ItemFn) -> syn::Result<TokenStream> {
    let decl: syn::ExprStruct = syn::parse2(args).map_err(|e| {
        syn::Error::new(e.span(), "#[span] on a function takes the span's value: `Decl { .. }`")
    })?;
    let ty = &decl.path;
    let takes_cx = function.sig.inputs.iter().any(|input| {
        matches!(input, syn::FnArg::Typed(t)
            if matches!(&*t.pat, syn::Pat::Ident(p) if p.ident == "cx"))
    });
    if !takes_cx {
        return Err(syn::Error::new_spanned(&function.sig, "a `#[span]` function takes `cx: &Cx`"));
    }
    let syn::ReturnType::Type(_, output) = &function.sig.output else {
        return Err(syn::Error::new_spanned(
            &function.sig,
            "a `#[span]` function returns a `Result`",
        ));
    };
    let t = quote!(::lk_agents_net::telemetry);
    let body = &function.block;
    let run = if function.sig.asyncness.is_some() {
        quote!(async { let __result: #output = #body; __result }.await)
    } else {
        quote!((|| -> #output { #body })())
    };
    function.block = syn::parse2(quote! {{
        let __span = #t::Cx::start_span(cx, #decl);
        let __result: #output = {
            let cx: &#t::Cx = __span.cx();
            #run
        };
        let __result = __span.check(__result);
        match &__result {
            ::core::result::Result::Ok(value) => {
                __span.end(<_ as #t::SpanEnd<#ty>>::end(value));
            }
            ::core::result::Result::Err(_) => ::core::mem::drop(__span),
        }
        __result
    }})?;
    Ok(quote!(#function))
}

#[cfg(test)]
mod tests {
    use quote::quote;

    fn derive(
        f: fn(&syn::DeriveInput) -> syn::Result<proc_macro2::TokenStream>,
        item: proc_macro2::TokenStream,
    ) -> String {
        match f(&syn::parse2(item).unwrap()) {
            Ok(out) => out.to_string(),
            Err(e) => format!("error: {e}"),
        }
    }

    fn role(role: &str, args: proc_macro2::TokenStream, item: proc_macro2::TokenStream) -> String {
        match super::role(role, args, item) {
            Ok(out) => out.to_string(),
            Err(e) => format!("error: {e}"),
        }
    }

    // Catches: a field written under the wrong key or encoding, a flattened group dropped from
    // labels or measurements, or `unmeasured` ignored.
    #[test]
    fn attrs_write_labels_and_measure_each_field() {
        let out = derive(
            super::attrs,
            quote! {
                struct S {
                    #[attr("lk.a", label)] a: String,
                    #[attr(semconv::otel::GEN_AI_REQUEST_MODEL)] b: Option<String>,
                    #[attr("lk.c", unmeasured = 0, metric = "lk.agents.c")] c: Measured<Seconds>,
                    #[attr("lk.d", unmeasured = -1)] d: Measured<Seconds>,
                    #[attrs] g: Group,
                    #[attr("lk.sip")] sip: Template<String>,
                }
            },
        );
        assert!(out.contains("put (out , \"lk.a\" , :: lk_agents_abi :: telemetry :: decl :: SpanValue :: value (& self . a))"), "{out}");
        assert!(out.contains("put (out , semconv :: otel :: GEN_AI_REQUEST_MODEL"), "{out}");
        assert!(
            out.contains(
                "self . c . value_or (:: lk_agents_abi :: telemetry :: decl :: Unmeasured :: Zero)"
            ),
            "{out}"
        );
        assert!(out.contains("Unmeasured :: MinusOne"), "{out}");
        assert!(out.contains("out . push ((\"lk.agents.c\" , n))"), "{out}");
        assert!(
            out.contains("Attrs :: labels (& self . g , out)")
                && out.contains("Attrs :: measure (& self . g , out)"),
            "{out}"
        );
        assert!(out.contains("self . sip . write (\"lk.sip\" , out)"), "{out}");
        let unit = derive(super::attrs, quote! { struct Marker; });
        assert!(unit.contains("let _ = out ;"), "{unit}");
    }

    // Catches: a field with no telemetry attribute, two of them, or a malformed one accepted.
    #[test]
    fn attrs_refuse_unmarked_ambiguous_and_malformed_fields() {
        for (item, message) in [
            (quote! { struct S { a: String } }, "every field"),
            (quote! { struct S { #[attr("k")] #[attrs] a: G } }, "one of"),
            (quote! { struct S { #[attr("k", unmeasured = 2)] a: Measured<i64> } }, "`0` or `-1`"),
            (
                quote! { struct S { #[attr("k", lable)] a: String } },
                "label [= \"label key\"], metric",
            ),
            (quote! { struct S { #[attr("k", label)] a: Template<String> } }, "neither a label"),
            (quote! { enum E { A } }, "for structs"),
        ] {
            let out = derive(super::attrs, item);
            assert!(out.starts_with("error:") && out.contains(message), "{out}");
        }
    }

    // Catches: a variant spelled other than snake_case, or a data-carrying variant accepted.
    #[test]
    fn span_values_are_snake_case_variants() {
        let out = derive(super::span_value, quote! { enum Origin { ServerSpeech, Text } });
        assert!(out.contains("Origin :: ServerSpeech => \"server_speech\""), "{out}");
        let out = derive(super::span_value, quote! { enum E { A(u8) } });
        assert!(out.contains("carries no data"), "{out}");
    }

    // Catches: the body escaping the span's context or `check`, an `Ok` not ended through
    // `SpanEnd`, or a function without `cx` accepted.
    #[test]
    fn span_functions_run_their_body_in_the_span() {
        let function = syn::parse2(quote! {
            async fn complete(&self, cx: &Cx, request: Request) -> Result<Response, Error> {
                self.send(cx, request).await
            }
        })
        .unwrap();
        let out = super::span_fn(quote!(LlmRequest { model: request.model() }), function)
            .unwrap()
            .to_string();
        assert!(
            out.contains(
                "let __span = :: lk_agents_net :: telemetry :: Cx :: start_span (cx , LlmRequest { model : request . model () }) ;"
            ),
            "{out}"
        );
        assert!(
            out.contains("let cx : & :: lk_agents_net :: telemetry :: Cx = __span . cx () ;"),
            "{out}"
        );
        assert!(out.contains("async { let __result : Result < Response , Error > ="), "{out}");
        assert!(out.contains("__span . check (__result)"), "{out}");
        assert!(out.contains("SpanEnd < LlmRequest > > :: end (value)"), "{out}");
        let sync = syn::parse2(quote! { fn f(cx: &Cx) -> Result<u8, E> { Ok(1) } }).unwrap();
        let out = super::span_fn(quote!(Marker {}), sync).unwrap().to_string();
        assert!(out.contains("(| | -> Result < u8 , E > { { Ok (1) } }) ()"), "{out}");
        let no_cx = syn::parse2(quote! { fn f(x: u8) -> Result<u8, E> { Ok(x) } }).unwrap();
        let err = super::span_fn(quote!(Marker {}), no_cx).unwrap_err().to_string();
        assert!(err.contains("takes `cx: &Cx`"), "{err}");
    }

    // Catches: a parent, an event, the end type, the kind or the outlive reason lost, or a role
    // attribute accepting an unknown or missing argument.
    #[test]
    fn roles_declare_their_impls() {
        let item = quote! { struct LlmRequest { #[attr("k")] a: String } };
        let out = role(
            "span",
            quote!(
                name = "llm_request",
                parent = LlmNode | Current,
                end = LlmRequestEnd,
                events = [Generation],
                kind = Client,
                outlives_parent = "probe"
            ),
            item.clone(),
        );
        assert!(out.starts_with("struct LlmRequest"), "the item comes back first: {out}");
        assert!(out.contains("const NAME : & 'static str = \"llm_request\""), "{out}");
        assert!(
            out.contains("SpanKind :: Client") && out.contains("type End = LlmRequestEnd"),
            "{out}"
        );
        assert!(out.contains("Some (\"probe\")"), "{out}");
        assert!(
            out.contains("ParentOf < LlmRequest > for LlmNode")
                && out.contains("ParentOf < LlmRequest > for Current"),
            "{out}"
        );
        assert!(out.contains("& [< LlmNode as :: lk_agents_abi :: telemetry :: decl :: SpanParent > :: PARENT_NAME , < Current as"), "{out}");
        assert!(out.contains("HasEvent < Generation > for LlmRequest"), "{out}");
        let out = role("span", quote!(name = "x", parent = A), item.clone());
        assert!(
            out.contains("type End = ()")
                && out.contains("SpanKind :: Internal")
                && out.contains(":: None"),
            "{out}"
        );
        for (r, args, message) in [
            ("span", quote!(name = "x"), "needs `parent`"),
            (
                "span",
                quote!(name = "x", parent = A, kind = Producer),
                "`Internal`, `Client` or `Server`",
            ),
            ("span", quote!(name = "x", parent = A, colour = "red"), "#[span(name"),
            ("span", quote!(name = "x", name = "y", parent = A), "twice"),
            ("metric", quote!(name = "m", unit = "s"), "needs `kind`"),
            ("metric", quote!(name = "m", kind = Gauge, unit = "s"), "`Counter` or `Histogram`"),
            ("record", quote!(scope = "chat_history"), "needs `body`"),
        ] {
            let out = role(r, args, item.clone());
            assert!(out.starts_with("error:") && out.contains(message), "{r}: {out}");
        }
        let out = role(
            "metric",
            quote!(name = semconv::otel::GEN_AI_X, kind = Histogram, unit = "s"),
            item.clone(),
        );
        assert!(
            out.contains("const NAME : & 'static str = semconv :: otel :: GEN_AI_X")
                && out.contains("InstrumentKind :: Histogram"),
            "{out}"
        );
        let out = role(
            "record",
            quote!(scope = "chat_history", body = "chat item"),
            quote! { struct R { #[attr("chat.item")] item: KvList<JsonObject>, #[span_id] turn: Option<SpanRef> } },
        );
        assert!(
            out.contains("const BODY : & 'static str = \"chat item\"")
                && out.contains("{ self . turn }"),
            "{out}"
        );
        let out = role("event", quote!(name = "x"), quote! { fn f() {} });
        assert!(out.contains("is for an `Attrs` struct"), "{out}");
    }
}
