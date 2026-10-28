//! Grandfathered ABI proxy conversions for remaining in-place consumers, listed in the shrink-only
//! `in-place-proxies-baseline.json` (its history is checked with the other baselines by
//! `rust/architecture/check.py`). New declarations extract their proxies in generated Python code
//! and need no ABI implementation.

use crate::decl::Declarations;

const BASELINE: &str = include_str!("../../in-place-proxies-baseline.json");

/// A declared proxy whose `HostProxy` bridge stays in ABI until its in-place consumers are gone.
pub(super) fn proxy_in_place(path: &str) -> bool {
    in_place().iter().any(|entry| entry == path)
}

fn in_place() -> Vec<String> {
    // A JSON array of strings is also a Rust array expression.
    let array: syn::ExprArray = syn::parse_str(BASELINE).expect("in-place-proxies-baseline.json");
    array
        .elems
        .into_iter()
        .map(|expression| match expression {
            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(value), .. }) => value.value(),
            _ => panic!("in-place-proxies-baseline.json holds string identities"),
        })
        .collect()
}

/// Every baseline entry is still a declared proxy: a retired one leaves the baseline.
pub(super) fn check_proxies(declarations: &Declarations) -> Result<(), String> {
    for path in in_place() {
        if !declarations.proxies.iter().any(|proxy| proxy.trait_path == path) {
            return Err(format!(
                "in-place-proxies-baseline.json: `{path}` is no longer declared; remove its legacy bridge"
            ));
        }
    }
    Ok(())
}
