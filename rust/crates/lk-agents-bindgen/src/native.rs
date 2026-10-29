//! `#[host(native(..))]` on a core trait only Rust implements: the host hands back a native object.

use proc_macro2::TokenStream;
use quote::quote;

use crate::Metas;

/// `#[host(native(..))]` on a trait no host object implements directly: a registered native class
/// (`as_dyn`) crosses as itself; anything else is refused with `refused = ".."`.
pub fn native(attr: TokenStream, tr: &syn::ItemTrait) -> syn::Result<TokenStream> {
    let m = Metas::parse(attr)?;
    let at = &tr.ident;
    for x in &m.0 {
        if !(x.key == "native" || x.key == "native.refused") {
            return Err(syn::Error::new_spanned(at, format!("native({}) has no meaning", x.key.trim_start_matches("native."))));
        }
    }
    // A bare marker is declaration metadata: its generated Python reader uses HostNative.
    let Some(refused) = m.text("native.refused")? else {
        return Ok(TokenStream::new());
    };
    let rt = quote!(::lk_agents_abi::python);
    Ok(quote! {
        #[cfg(feature = "python")]
        impl #rt::HostProxy for dyn #at {
            fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
                #rt::native::<dyn #at>(obj).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(#refused))
            }
        }
    })
}
