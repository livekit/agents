//! `#[host(native(..))]` on a core trait only Rust implements: the host hands back a native object.

use proc_macro2::TokenStream;

use crate::Metas;

/// `#[host(native(..))]` on a trait no host object implements directly: declaration metadata the
/// generated reader (`HostNative`) uses; only its keys are checked here.
pub fn native(attr: TokenStream, tr: &syn::ItemTrait) -> syn::Result<TokenStream> {
    for x in &Metas::parse(attr)?.0 {
        if !matches!(x.key.as_str(), "native" | "native.refused" | "native.class") {
            return Err(syn::Error::new_spanned(
                &tr.ident,
                format!("native({}) has no meaning", x.key.trim_start_matches("native.")),
            ));
        }
    }
    Ok(TokenStream::new())
}
