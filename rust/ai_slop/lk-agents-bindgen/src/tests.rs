//! The declaration readers as text. What Python observes of the generated code (names, exceptions)
//! is pinned end to end by `lk-agents-py`'s export-table test and the `_core` surface diff; these
//! pin the choices that diff cannot localise.

use quote::quote;

fn exception(attr: proc_macro2::TokenStream, item: &syn::Item) -> syn::Result<String> {
    let ty: syn::Path = syn::parse_quote!(::abi::E);
    crate::host_exception(attr, item, &ty, &|_, value| Ok(value)).map(|e| e.tokens.to_string())
}

fn kind(attr: proc_macro2::TokenStream, item: &syn::Item) -> String {
    crate::host_kind(attr, item).unwrap().to_string()
}

// Catches: a variant's kind not reaching its arm, a dotted kind not importing the host class, a
// `delegate` not asking the wrapped error's own table, or the default kind lost for unmarked
// variants (in `to_py_err` or `host_kind`).
#[test]
fn an_error_is_a_host_exception_by_its_kind_table() {
    let e: syn::Item = syn::parse2(quote! {
        pub enum SchemaError {
            #[host(delegate)] Strict(StrictError),
            #[host(kind = "TypeError")] NotAnObject { path: String },
            #[host(kind = "asyncio.CancelledError")] Cancelled,
            Other(String),
        }
    })
    .unwrap();
    let attr = quote!(error(kind = "RuntimeError"));
    let out = exception(attr.clone(), &e).unwrap();
    assert!(out.contains("impl crate :: python :: HostException for :: abi :: E"), "{out}");
    assert!(
        out.contains(
            "Self :: Strict (inner) => crate :: python :: HostException :: to_py_err (inner)"
        ),
        "{out}"
    );
    assert!(
        out.contains(
            "Self :: NotAnObject { .. } => :: pyo3 :: exceptions :: PyTypeError :: new_err"
        ),
        "{out}"
    );
    assert!(out.contains("Self :: Cancelled => crate :: python :: host_exception (\"asyncio\" , \"CancelledError\""), "{out}");
    assert!(out.contains("_ => :: pyo3 :: exceptions :: PyRuntimeError :: new_err"), "{out}");
    // The same table, as the class name a non-Python binding raises.
    let out = kind(attr, &e);
    assert!(out.contains("impl SchemaError"), "{out}");
    assert!(out.contains("Self :: Strict (inner) => inner . host_kind ()"), "{out}");
    assert!(out.contains("Self :: NotAnObject { .. } => \"TypeError\""), "{out}");
    assert!(out.contains("Self :: Cancelled => \"asyncio.CancelledError\""), "{out}");
    assert!(out.contains("_ => \"RuntimeError\""), "{out}");
}

// Catches: a kind-selected struct raising one class for every kind, a listed field not passed as
// that class's keyword argument (or an unlisted one passed), a named variant's `delegate = field`
// not raising through that field, a kind table missing a variant accepted, or a `cause` not set
// as the exception's `__cause__`.
#[test]
fn a_struct_error_selects_its_class_by_a_field() {
    let e: syn::Item = syn::parse2(quote! {
        pub struct ApiError { pub kind: Kind, pub message: String, pub retryable: bool, pub code: Option<i64>, pub cause: Option<Box<E>> }
    })
    .unwrap();
    let attr = quote!(error(
        by = kind,
        cause = cause,
        kinds(A(class = "m.A", kwargs(retryable)), B(class = "m.sub.B", kwargs(retryable, code)))
    ));
    let out = exception(attr.clone(), &e).unwrap();
    assert!(out.contains("match self . kind"), "{out}");
    assert!(
        out.contains("< Kind > :: A => crate :: python :: host_exception_with (\"m\" , \"A\""),
        "{out}"
    );
    assert!(
        out.contains("< Kind > :: B => crate :: python :: host_exception_with (\"m.sub\" , \"B\""),
        "{out}"
    );
    assert_eq!(out.matches("\"code\"").count(), 1, "{out}");
    assert_eq!(out.matches("\"retryable\"").count(), 2, "{out}");
    assert!(
        out.contains("if let :: std :: option :: Option :: Some (cause) = & self . cause"),
        "{out}"
    );
    assert!(out.contains("err . set_cause (py , :: std :: option :: Option :: Some (crate :: python :: HostException :: to_py_err (& * * cause)))"), "{out}");
    assert!(kind(attr, &e).contains("< Kind > :: B => \"m.sub.B\""));
    assert!(exception(quote!(error(by = kind, kinds(A(kwargs(retryable))))), &e).is_err());
    assert!(exception(quote!(error(by = nope, kinds(A(class = "m.A")))), &e).is_err());
    let e: syn::Item = syn::parse2(quote! {
        pub enum ProviderError { #[host(delegate = error)] Host { error: HostError, retryable: bool }, #[host(delegate)] Local(ApiError) }
    })
    .unwrap();
    let out = exception(quote!(error), &e).unwrap();
    assert!(
        out.contains(
            "Self :: Host { error , .. } => crate :: python :: HostException :: to_py_err (error)"
        ),
        "{out}"
    );
    assert!(
        kind(quote!(error), &e).contains("Self :: Host { error , .. } => error . host_kind ()")
    );
    let bad: syn::Item =
        syn::parse2(quote! { pub enum E { #[host(delegate = nope)] Host { error: HostError } } })
            .unwrap();
    assert!(exception(quote!(error), &bad).is_err());
}

// Catches: a declared class not created from its base, not raised, or not reported for
// registration; a dotted declared class accepted.
#[test]
fn a_declared_error_is_a_core_exception_class_of_its_base() {
    let item: syn::Item = syn::parse2(quote! { pub enum IdleRefused { Closing } }).unwrap();
    let ty: syn::Path = syn::parse_quote!(::abi::IdleRefused);
    let attr = quote!(error(kind = "SessionClosing", declare = "RuntimeError"));
    let e = crate::host_exception(attr, &item, &ty, &|_, value| Ok(value)).unwrap();
    let out = e.tokens.to_string();
    assert!(
        out.contains(
            "create_exception ! (_core , SessionClosing , :: pyo3 :: exceptions :: PyRuntimeError"
        ),
        "{out}"
    );
    assert!(out.contains("SessionClosing :: new_err"), "{out}");
    assert_eq!(e.declared.as_deref(), Some("SessionClosing"));
    assert!(exception(quote!(error(kind = "a.B", declare = "RuntimeError")), &item).is_err());
}

fn filled(item: proc_macro2::TokenStream) -> syn::Result<syn::ItemImpl> {
    let mut imp: syn::ItemImpl = syn::parse2(item).unwrap();
    crate::fill_sends(&mut imp).map(|()| imp)
}

// Catches: a `send` method's body not sending its expression, its refusal swallowed where the
// signature returns it (or returned where it returns nothing), the declaration's docs, markers or
// visibility lost on the way.
#[test]
fn a_send_method_is_written_as_a_send_of_its_expression() {
    let imp = filled(quote! { impl S {
        /// Drop the turn.
        #[host(method(send = TurnCommand::Clear))]
        pub fn clear(&self) -> Result<(), Refused>;
        #[host(method(send = Hold { id: HoldId(hold) }))]
        pub fn release(&self, hold: u64);
        #[host(method)]
        pub fn plain(&self) -> u8 { 0 }
    } })
    .unwrap();
    let text = quote!(#imp).to_string();
    assert!(text.contains("# [doc = r\" Drop the turn.\"] # [host (method (send = TurnCommand :: Clear))] # [allow (clippy :: needless_question_mark)] pub fn clear (& self) -> Result < () , Refused > { :: core :: result :: Result :: Ok (self . send (TurnCommand :: Clear) ?) }"), "{text}");
    assert!(text.contains("pub fn release (& self , hold : u64) { let _ = self . send (Hold { id : HoldId (hold) }) ; }"), "{text}");
    assert!(text.contains("pub fn plain (& self) -> u8 { 0 }"), "{text}");
}

// Catches: a body-less method compiled with no body at all, `send` beside a body it would silently
// replace, or a `send` method whose return cannot carry the refusal.
#[test]
fn a_body_is_written_by_send_or_by_hand_never_both_nor_neither() {
    let err = |item| filled(item).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        err(quote! { impl S { #[host(method)] pub fn f(&self); } })
            .contains("needs #[host(method(send = <expr>))]")
    );
    assert!(err(quote! { impl S { #[host(method(send = C::X))] pub fn f(&self) -> Result<(), E> { todo!() } } }).contains("`send` writes the body"));
    assert!(
        err(quote! { impl S { #[host(method(send = C::X))] pub fn f(&self) -> bool; } })
            .contains("returns nothing")
    );
    assert!(
        err(quote! { impl S { #[host(method(send = C::X))] pub fn f() -> Result<(), E>; } })
            .contains("takes `&self`")
    );
}
